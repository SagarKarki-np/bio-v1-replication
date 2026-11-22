
"""
Figure 10: Firing-rate calibration pilot.

Compares two target-rate conditions:
    Baseline: E=2 Hz, I=4 Hz
    Higher:   E=5 Hz, I=10 Hz

Three calibration batches, eight images per batch.

The experiment:
- Uses the same initial synaptic weights.
- Uses identical LGN spike trains for both conditions.
- Matches stochastic simulation inputs.
- Updates neuronal thresholds only.
- Does not overwrite original calibration results.
"""

from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

ORIGINAL = PROJECT.parent / "V1_network_learning"

OUTPUT = (
    PROJECT / "results"
    / "figure10_robustness"
    / "calibration_pilot"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

from config import (
    DT, N_STEPS, TAU_M,
    NOISE_STD, V_MIN, V_RESET,
    N_E, N_I
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch

# ------------------------------------------------------------
# 2. EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

SEED = 42
N_BATCHES = 3
PILOT_BATCH_SIZE = 8

INITIAL_THRESHOLD = 10.0
ETA_E = 1.0
ETA_I = 1.0

CONDITIONS = {
    "E02_I04": (2.0, 4.0),
    "E05_I10": (5.0, 10.0)
}

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)

# ------------------------------------------------------------
# 3. LOAD INITIAL WEIGHTS AND IMAGES
# ------------------------------------------------------------

weights_path = (
    ORIGINAL / "results"
    / "untrained" / "batch_001"
    / "weights.npz"
)

images_path = (
    ORIGINAL / "data"
    / "van_hateren_dog.npy"
)

with np.load(weights_path, allow_pickle=False) as data:
    weights = tuple(
        data[name].copy()
        for name in WEIGHT_NAMES
    )

images = np.load(images_path, mmap_mode="r")

print("Images:", images.shape)
print("Weights:", [W.shape for W in weights])

# ------------------------------------------------------------
# 4. HOMEOSTATIC THRESHOLD UPDATE
# ------------------------------------------------------------

def update_thresholds(
    thresholds,
    observed_counts,
    target_counts,
    eta
):
    """
    Same update rule as original plasticity.py.

    Threshold increase is capped at eta * target_counts.
    Threshold decrease has no symmetric cap.
    """

    delta = eta * (
        observed_counts - target_counts
    )

    delta = np.minimum(
        delta,
        eta * target_counts
    )

    return thresholds + delta


# ------------------------------------------------------------
# 5. INITIALIZE EACH EXPERIMENTAL CONDITION
# ------------------------------------------------------------

states = {}

for name, (target_E, target_I) in CONDITIONS.items():

    states[name] = {
        "target_E": target_E,
        "target_I": target_I,

        "threshold_E": np.full(
            (N_E, 1),
            INITIAL_THRESHOLD
        ),

        "threshold_I": np.full(
            (N_I, 1),
            INITIAL_THRESHOLD
        ),

        "observed_E": [],
        "observed_I": [],

        "threshold_mean_E": [],
        "threshold_mean_I": []
    }

# ------------------------------------------------------------
# 6. RUN MATCHED CALIBRATION BATCHES
# ------------------------------------------------------------

np.random.seed(SEED)

duration = N_STEPS * DT

for batch in range(N_BATCHES):

    print(f"\nCALIBRATION BATCH {batch + 1}")

    LGN_spikes = generate_lgn_batch(
        images,
        batch_size=PILOT_BATCH_SIZE,
        n_steps=N_STEPS,
        dt=DT
    )

    # Capture RNG state after generating LGN spikes.
    # Both conditions will start simulation from this state.
    noise_state = np.random.get_state()
    next_state = None

    for name, state in states.items():

        np.random.set_state(noise_state)

        E_spikes, I_spikes = simulate_v1_batch(
            LGN_spikes,
            weights,
            dt=DT,
            tau_m=TAU_M,
            noise_std=NOISE_STD,
            v_min=V_MIN,
            v_reset=V_RESET,
            threshold_E=state["threshold_E"],
            threshold_I=state["threshold_I"]
        )

        # Preserve the RNG progression from one simulation,
        # rather than executing the random stream twice.
        if next_state is None:
            next_state = np.random.get_state()

        # Average counts for each neuron over presentations.
        observed_E = (
            E_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        observed_I = (
            I_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        mean_E = float(observed_E.mean())
        mean_I = float(observed_I.mean())

        target_counts_E = (
            state["target_E"] * duration
        )

        target_counts_I = (
            state["target_I"] * duration
        )

        # Threshold adaptation only.
        state["threshold_E"] = update_thresholds(
            state["threshold_E"],
            observed_E,
            target_counts_E,
            ETA_E
        )

        state["threshold_I"] = update_thresholds(
            state["threshold_I"],
            observed_I,
            target_counts_I,
            ETA_I
        )

        state["observed_E"].append(
            mean_E / duration
        )

        state["observed_I"].append(
            mean_I / duration
        )

        state["threshold_mean_E"].append(
            float(state["threshold_E"].mean())
        )

        state["threshold_mean_I"].append(
            float(state["threshold_I"].mean())
        )

        assert np.isfinite(
            state["threshold_E"]
        ).all()

        assert np.isfinite(
            state["threshold_I"]
        ).all()

        print(
            f"{name} | "
            f"E: {mean_E / duration:.3f} Hz | "
            f"I: {mean_I / duration:.3f} Hz | "
            f"theta E: "
            f"{state['threshold_E'].mean():.3f} | "
            f"theta I: "
            f"{state['threshold_I'].mean():.3f}"
        )

    # Advance to the next independent batch.
    np.random.set_state(next_state)

# ------------------------------------------------------------
# 7. SAVE RESULTS FOR EACH CONDITION
# ------------------------------------------------------------

for name, state in states.items():

    condition_dir = OUTPUT / name
    condition_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        condition_dir / "pilot_results.npz",
        observed_E=np.array(
            state["observed_E"]
        ),
        observed_I=np.array(
            state["observed_I"]
        ),
        threshold_mean_E=np.array(
            state["threshold_mean_E"]
        ),
        threshold_mean_I=np.array(
            state["threshold_mean_I"]
        ),
        final_threshold_E=state["threshold_E"],
        final_threshold_I=state["threshold_I"],
        target_E=state["target_E"],
        target_I=state["target_I"]
    )

# ------------------------------------------------------------
# 8. PLOT THRESHOLD ADAPTATION
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.5),
    constrained_layout=True
)

batches = np.arange(
    1, N_BATCHES + 1
)

for name, state in states.items():

    axes[0].plot(
        batches,
        state["threshold_mean_E"],
        marker="o",
        label=name
    )

    axes[1].plot(
        batches,
        state["threshold_mean_I"],
        marker="o",
        label=name
    )

axes[0].set_title(
    "Excitatory threshold adaptation"
)

axes[1].set_title(
    "Inhibitory threshold adaptation"
)

for ax in axes:
    ax.set_xlabel("Calibration batch")
    ax.set_ylabel("Mean threshold")
    ax.grid(alpha=0.3)
    ax.legend()

fig.savefig(
    OUTPUT / "calibration_pilot.png",
    dpi=250
)

plt.close(fig)

# ------------------------------------------------------------
# 9. METADATA
# ------------------------------------------------------------

metadata = {
    "experiment": "figure10_calibration_pilot",
    "seed": SEED,
    "batches": N_BATCHES,
    "batch_size": PILOT_BATCH_SIZE,
    "conditions": CONDITIONS,
    "initial_threshold": INITIAL_THRESHOLD,
    "synaptic_learning": False,
    "initial_weights": str(weights_path),
    "matching": (
        "Same LGN spikes and RNG state "
        "for both conditions within each batch"
    ),
    "note": (
        "Technical pilot only. "
        "Not a full Figure 10 replication."
    )
}

with open(
    OUTPUT / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nFIGURE 10 CALIBRATION PILOT COMPLETED")
print("Results saved to:", OUTPUT)
