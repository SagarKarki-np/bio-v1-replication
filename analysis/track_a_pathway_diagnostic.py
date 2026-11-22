
"""
FINAL DIAGNOSTIC
================

Track A: Individual synaptic-pathway substitution.

Compare the original V1 network against the
batch-5 trained network using eight conditions:

1. All original weights
2-7. One trained pathway substituted at a time
8. All trained weights

Controls:
- Same LGN spike trains
- Same replayed noise
- Same initial thresholds
- No STDP or threshold adaptation
- No output files modified

Purpose:
Determine which pathway substitutions influence
the elevated firing observed after five
synaptic-learning batches.
"""

from pathlib import Path
import numpy as np

import figure10_config

from analysis import figure10_full_calibration as fc
from analysis import figure10_train as tr


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

NETWORK_SEED = 3002
TRAINING_SEED = 303002
DIAGNOSTIC_SEED = 404002

BATCH_SIZE = 100
EXPECTED_TRAINING_BATCH = 5

ROOT = (
    PROJECT
    / "results"
    / "paper_protocol"
    / f"seed_{NETWORK_SEED}"
)

TRAINING_DIR = (
    ROOT
    / "synaptic_training"
    / "track_a"
)

THRESHOLD_PATH = (
    ROOT
    / "calibration"
    / "E02_I04"
    / "calibrated_thresholds.npz"
)

WEIGHT_NAMES = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E"
)


# ============================================================
# 2. LOAD ORIGINAL WEIGHTS
# ============================================================

initial_tuple, _ = fc.load_initial_weights(
    parameterization="paper",
    network_seed=NETWORK_SEED
)

initial_weights = {
    name: W.copy()
    for name, W in zip(
        WEIGHT_NAMES,
        initial_tuple
    )
}


# ============================================================
# 3. LOAD ORIGINAL THRESHOLDS
# ============================================================

with np.load(
    THRESHOLD_PATH,
    allow_pickle=False
) as data:

    theta_E = data["threshold_E"].copy()
    theta_I = data["threshold_I"].copy()


# ============================================================
# 4. LOAD BATCH-5 TRAINED WEIGHTS
# ============================================================

condition = figure10_config.make_condition(
    2.0,
    parameterization="paper"
)

(
    completed,
    trained_weights,
    trained_theta_E,
    trained_theta_I,
    history_E,
    history_I
) = tr.load_latest_checkpoint(
    TRAINING_DIR,
    condition,
    BATCH_SIZE,
    TRAINING_SEED
)

if completed != EXPECTED_TRAINING_BATCH:
    raise ValueError(
        "Expected batch-5 checkpoint. "
        f"Loaded batch {completed} instead."
    )

for name in WEIGHT_NAMES:

    if initial_weights[name].shape != (
        trained_weights[name].shape
    ):
        raise ValueError(
            f"Shape mismatch: {name}"
        )


# ============================================================
# 5. GENERATE MATCHED LGN INPUT
# ============================================================

images = fc.PaperImageDataset(
    str(
        PROJECT
        / "data"
        / "paper_van_hateren"
    )
)

np.random.seed(DIAGNOSTIC_SEED)

LGN_spikes = fc.generate_lgn_batch(
    images,
    batch_size=BATCH_SIZE,
    n_steps=fc.N_STEPS,
    dt=fc.DT
)

# Reset to this exact RNG state before
# every V1 simulation.
shared_noise_state = np.random.get_state()


# ============================================================
# 6. SIMULATE ONE WEIGHT CONDITION
# ============================================================

def measure_firing(weights):

    np.random.set_state(
        shared_noise_state
    )

    weight_tuple = tuple(
        weights[name]
        for name in tr.WEIGHT_ORDER
    )

    E_spikes, I_spikes = fc.simulate_v1_batch(
        LGN_spikes,
        weight_tuple,
        dt=fc.DT,
        tau_m=fc.TAU_M,
        noise_std=fc.NOISE_STD,
        v_min=fc.V_MIN,
        v_reset=fc.V_RESET,
        threshold_E=theta_E.copy(),
        threshold_I=theta_I.copy(),
        record_inputs=False
    )

    duration = fc.N_STEPS * fc.DT

    E_rate = float(
        E_spikes.sum()
        / (
            E_spikes.shape[0]
            * BATCH_SIZE
            * duration
        )
    )

    I_rate = float(
        I_spikes.sum()
        / (
            I_spikes.shape[0]
            * BATCH_SIZE
            * duration
        )
    )

    return E_rate, I_rate


# ============================================================
# 7. BASELINE: ALL ORIGINAL WEIGHTS
# ============================================================

print("=" * 75)
print("FINAL DIAGNOSTIC — PATHWAY SUBSTITUTION")
print("=" * 75)

print("Network seed:", NETWORK_SEED)
print("Training checkpoint:", completed)
print("Diagnostic presentations:", BATCH_SIZE)

baseline_E, baseline_I = measure_firing(
    initial_weights
)

print("\nBASELINE")
print(
    f"E: {baseline_E:.4f} Hz | "
    f"I: {baseline_I:.4f} Hz"
)


# ============================================================
# 8. SUBSTITUTE ONE PATHWAY AT A TIME
# ============================================================

results = []

for name in WEIGHT_NAMES:

    # Start with all original weights.
    test_weights = {
        key: W.copy()
        for key, W in initial_weights.items()
    }

    # Replace only the selected pathway
    # with its batch-5 trained version.
    test_weights[name] = (
        trained_weights[name].copy()
    )

    E_rate, I_rate = measure_firing(
        test_weights
    )

    results.append({
        "pathway": name,
        "E": E_rate,
        "I": I_rate,
        "delta_E": E_rate - baseline_E,
        "delta_I": I_rate - baseline_I
    })


# ============================================================
# 9. ALL TRAINED WEIGHTS
# ============================================================

all_E, all_I = measure_firing(
    trained_weights
)


# ============================================================
# 10. PRINT COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("PATHWAY SUBSTITUTION RESULTS")
print("=" * 75)

print(
    f"{'Pathway':<12}"
    f"{'E rate':>12}"
    f"{'I rate':>12}"
    f"{'Delta E':>12}"
    f"{'Delta I':>12}"
)

print("-" * 75)

print(
    f"{'Original':<12}"
    f"{baseline_E:>12.3f}"
    f"{baseline_I:>12.3f}"
    f"{0.0:>+12.3f}"
    f"{0.0:>+12.3f}"
)

for result in results:

    print(
        f"{result['pathway']:<12}"
        f"{result['E']:>12.3f}"
        f"{result['I']:>12.3f}"
        f"{result['delta_E']:>+12.3f}"
        f"{result['delta_I']:>+12.3f}"
    )

print("-" * 75)

print(
    f"{'All trained':<12}"
    f"{all_E:>12.3f}"
    f"{all_I:>12.3f}"
    f"{all_E - baseline_E:>+12.3f}"
    f"{all_I - baseline_I:>+12.3f}"
)


# ============================================================
# 11. RANK INDIVIDUAL PATHWAYS
# ============================================================

print("\n" + "=" * 75)
print("LARGEST INDIVIDUAL FIRING-RATE EFFECTS")
print("=" * 75)

ranked_E = sorted(
    results,
    key=lambda x: abs(x["delta_E"]),
    reverse=True
)

ranked_I = sorted(
    results,
    key=lambda x: abs(x["delta_I"]),
    reverse=True
)

print("\nExcitatory population:")

for rank, result in enumerate(
    ranked_E, start=1
):
    print(
        f"{rank}. {result['pathway']}: "
        f"{result['delta_E']:+.4f} Hz"
    )

print("\nInhibitory population:")

for rank, result in enumerate(
    ranked_I, start=1
):
    print(
        f"{rank}. {result['pathway']}: "
        f"{result['delta_I']:+.4f} Hz"
    )


# ============================================================
# 12. INTERACTION DIAGNOSTIC
# ============================================================

sum_effect_E = sum(
    result["delta_E"]
    for result in results
)

sum_effect_I = sum(
    result["delta_I"]
    for result in results
)

total_effect_E = all_E - baseline_E
total_effect_I = all_I - baseline_I

print("\n" + "=" * 75)
print("NONLINEAR PATHWAY INTERACTIONS")
print("=" * 75)

print(
    "Sum of individual E effects:",
    f"{sum_effect_E:+.4f} Hz"
)

print(
    "Actual combined E effect:",
    f"{total_effect_E:+.4f} Hz"
)

print(
    "Combined minus individual E sum:",
    f"{total_effect_E - sum_effect_E:+.4f} Hz"
)

print()

print(
    "Sum of individual I effects:",
    f"{sum_effect_I:+.4f} Hz"
)

print(
    "Actual combined I effect:",
    f"{total_effect_I:+.4f} Hz"
)

print(
    "Combined minus individual I sum:",
    f"{total_effect_I - sum_effect_I:+.4f} Hz"
)

print("\n" + "=" * 75)
print("FINAL DIAGNOSTIC COMPLETE")
print("=" * 75)

print("Synaptic plasticity performed: False")
print("Existing files modified: None")
