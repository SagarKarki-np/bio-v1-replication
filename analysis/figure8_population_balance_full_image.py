
"""
Figure 8E-H: Population excitation-inhibition balance.

Paper-protocol evaluation for Track A or Track B.

Uses:
- PaperImageDataset (1000 natural images)
- 100,000-stimulus STA receptive fields
- Final batch_1100 trained checkpoints
- 40 excitatory neurons
- 100 candidate natural-image patches per neuron

No training or plasticity.
Original checkpoints remain unchanged.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import (
    DT,
    N_STEPS,
    TAU_M,
    V_MIN,
    V_RESET,
    PATCH_SIZE,
    LGN_RATE_SCALE,
    LGN_MAX_RATE
)

from preprocessing import extract_random_patch
from paper_image_dataset import PaperImageDataset

from simulation import (
    generate_lgn_population_spikes,
    simulate_v1_batch
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--track",
    choices=("a", "b"),
    required=True
)

TRACK = parser.parse_args().track

BASE = (
    PROJECT
    / "results"
    / "paper_protocol"
    / "seed_3002"
)

CHECKPOINT = (
    BASE
    / "synaptic_training"
    / f"track_{TRACK}"
    / "checkpoints"
    / "batch_1100.npz"
)

STA_FILE = (
    BASE
    / "functional_sta"
    / f"track_{TRACK}"
    / "full_image_100000"
    / "STA_100000_results.npz"
)

IMAGE_DIR = (
    PROJECT
    / "data"
    / "paper_van_hateren"
)

OUTPUT = (
    BASE
    / "figure8_population_balance_full_image"
    / "paper_1000_source"
    / f"track_{TRACK}"
)

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E"
)

N_NEURONS = 40
N_CANDIDATES = 100
BATCH_SIZE = 8

SEED = 2028
NOISE_STD = 0.0

LABELS = [
    "Anti-correlated",
    "Uncorrelated",
    "Correlated"
]

rng = np.random.default_rng(SEED)


# ==================================================
# 2. OUTPUT PROTECTION
# ==================================================

if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"Results already exist: {OUTPUT}"
    )


# ==================================================
# 3. LOAD NETWORK CHECKPOINT
# ==================================================

with np.load(
    CHECKPOINT,
    allow_pickle=False
) as data:

    weights = tuple(
        data[name].copy()
        for name in WEIGHT_NAMES
    )

    threshold_E = data["threshold_E"].copy()
    threshold_I = data["threshold_I"].copy()

    assert int(data["completed_batches"]) == 1100

    assert str(data["parameterization"]) == "paper"


# ==================================================
# 4. LOAD RECEPTIVE FIELDS
# ==================================================

with np.load(
    STA_FILE,
    allow_pickle=False
) as data:

    STA_E = data["STA_E"].copy()

    assert int(data["n_stimuli"]) == 100000


assert STA_E.shape == (400, 256)
assert threshold_E.shape == (400, 1)
assert threshold_I.shape == (100, 1)


# ==================================================
# 5. LOAD PAPER DATASET
# ==================================================

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(IMAGE_DIR)

images = PaperImageDataset(str(IMAGE_DIR))

if tuple(images.shape) != (1024, 1536, 1000):
    raise RuntimeError(
        f"WRONG IMAGE DATASET: {images.shape}"
    )

print("=" * 65)
print("FIGURE 8E-H: POPULATION BALANCE")
print("=" * 65)

print("PROTOCOL: paper_1000_source")
print("IMAGE_DIR:", IMAGE_DIR)
print("Images:", images.shape)
print("Track:", TRACK)
print("Checkpoint:", CHECKPOINT)
print("STA:", STA_FILE)


# ==================================================
# 6. SELECT 40 EXCITATORY NEURONS
# ==================================================

neurons = rng.choice(
    400,
    size=N_NEURONS,
    replace=False
)

print("\nSelected neurons:", neurons)


# ==================================================
# 7. SELECT NATURAL-IMAGE PATCHES
# ==================================================

selected_patches = np.zeros(
    (N_NEURONS, 3, PATCH_SIZE, PATCH_SIZE),
    dtype=float
)

selected_correlations = np.zeros(
    (N_NEURONS, 3),
    dtype=float
)

for i, neuron in enumerate(neurons):

    candidates = np.empty(
        (N_CANDIDATES, PATCH_SIZE, PATCH_SIZE)
    )

    for j in range(N_CANDIDATES):

        image_index = rng.integers(
            images.shape[2]
        )

        # Existing extractor uses the NumPy global RNG.
        np.random.seed(
            int(rng.integers(0, 2**32 - 1))
        )

        candidates[j] = extract_random_patch(
            images[:, :, image_index],
            PATCH_SIZE,
            20
        )

    # MATLAB-compatible column-major flattening.
    vectors = np.stack([
        patch.flatten(order="F")
        for patch in candidates
    ])

    sta = STA_E[neuron]
    sta = sta - sta.mean()

    sta_norm = np.linalg.norm(sta)

    centered = vectors - vectors.mean(
        axis=1,
        keepdims=True
    )

    norms = np.linalg.norm(
        centered,
        axis=1
    )

    correlations = np.full(
        N_CANDIDATES,
        np.nan
    )

    valid = (
        np.isfinite(norms)
        & (norms > 0)
        & (sta_norm > 0)
    )

    correlations[valid] = (
        centered[valid] @ sta
    ) / (
        norms[valid] * sta_norm
    )

    if not np.any(valid):
        raise ValueError(
            f"No valid patches for neuron {neuron}"
        )

    chosen = [
        np.nanargmin(correlations),
        np.nanargmin(np.abs(correlations)),
        np.nanargmax(correlations)
    ]

    selected_patches[i] = candidates[chosen]

    selected_correlations[i] = (
        correlations[chosen]
    )


print("\nSTIMULUS SELECTION COMPLETED")

for k, label in enumerate(LABELS):

    print(
        label,
        "mean STA correlation:",
        selected_correlations[:, k].mean()
    )


# ==================================================
# 8. PREPARE VOLTAGE RECORDINGS
# ==================================================

mean_exc = np.zeros((N_NEURONS, 3))
mean_inh = np.zeros((N_NEURONS, 3))
mean_total = np.zeros((N_NEURONS, 3))

voltage_exc = np.zeros(
    (N_NEURONS, 3, N_STEPS)
)

voltage_inh = np.zeros(
    (N_NEURONS, 3, N_STEPS)
)

voltage_total = np.zeros(
    (N_NEURONS, 3, N_STEPS)
)

presentations = [
    (i, condition)
    for i in range(N_NEURONS)
    for condition in range(3)
]

decay = np.exp(-DT / TAU_M)


# ==================================================
# 9. SIMULATE SELECTED PRESENTATIONS
# ==================================================

for start in range(
    0,
    len(presentations),
    BATCH_SIZE
):

    group = presentations[
        start:start + BATCH_SIZE
    ]

    n_batch = len(group)

    LGN_spikes = np.zeros(
        (512, n_batch, N_STEPS),
        dtype=bool
    )

    for j, (i, condition) in enumerate(group):

        patch = selected_patches[i, condition]

        vector = patch.flatten(order="F")

        scaled = vector * LGN_RATE_SCALE

        ON = np.maximum(scaled, 0)
        OFF = np.maximum(-scaled, 0)

        rates = np.concatenate([ON, OFF])

        rates = np.minimum(
            rates,
            LGN_MAX_RATE
        )

        LGN_spikes[:, j, :] = (
            generate_lgn_population_spikes(
                rates,
                DT,
                N_STEPS
            )
        )

    E_spikes, I_spikes, traces = simulate_v1_batch(
        LGN_spikes,
        weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I,
        record_inputs=True
    )

    for j, (i, condition) in enumerate(group):

        neuron = int(neurons[i])

        exc_input = (
            traces["LGN_E"][neuron, j, :]
            + traces["E_E"][neuron, j, :]
        )

        inh_input = (
            traces["I_E"][neuron, j, :]
        )

        total_input = (
            traces["total_E"][neuron, j, :]
        )

        ve = 0.0
        vi = 0.0
        vt = 0.0

        for t in range(N_STEPS):

            ve = decay * ve + exc_input[t]

            vi = decay * vi - inh_input[t]

            vt = decay * vt + total_input[t]

            voltage_exc[i, condition, t] = ve

            voltage_inh[i, condition, t] = vi

            voltage_total[i, condition, t] = vt

        mean_exc[i, condition] = (
            voltage_exc[i, condition].mean()
        )

        mean_inh[i, condition] = (
            voltage_inh[i, condition].mean()
        )

        mean_total[i, condition] = (
            voltage_total[i, condition].mean()
        )

    print(
        f"Processed {start + n_batch}"
        f"/{len(presentations)} presentations",
        flush=True
    )


# ==================================================
# 10. VALIDATE VOLTAGE RECONSTRUCTION
# ==================================================

error = np.max(
    np.abs(
        voltage_total
        - (voltage_exc + voltage_inh)
    )
)

print(
    "\nVoltage decomposition maximum error:",
    error
)

assert error < 1e-8

inh_magnitude = -mean_inh


# ==================================================
# 11. FIGURE 8E: POPULATION MEAN VOLTAGES
# ==================================================

fig = plt.figure(
    figsize=(15, 10),
    constrained_layout=True
)

gs = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs[0, :])

x = np.arange(3)

series = [
    (mean_exc, "Excitation only", "green"),
    (mean_inh, "Inhibition only", "purple"),
    (mean_total, "Total", "goldenrod")
]

for measurements, label, color in series:

    means = measurements.mean(axis=0)

    stds = measurements.std(
        axis=0,
        ddof=1
    )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        capsize=5,
        label=label,
        color=color
    )

ax.set_xticks(x)
ax.set_xticklabels(LABELS)

ax.set_title(
    "E. Population voltage balance (40 neurons)"
)

ax.set_ylabel("Mean voltage (model units)")

ax.grid(alpha=0.2)
ax.legend()


# ==================================================
# 12. FIGURE 8F-H: VOLTAGE VS INHIBITION
# ==================================================

for condition in range(3):

    ax = fig.add_subplot(
        gs[1, condition]
    )

    inhibitor = inh_magnitude[:, condition]

    excitation = mean_exc[:, condition]

    total = mean_total[:, condition]

    ax.scatter(
        inhibitor,
        excitation,
        color="green",
        alpha=0.7,
        label="Excitation only"
    )

    ax.scatter(
        inhibitor,
        total,
        color="goldenrod",
        alpha=0.7,
        label="Total"
    )

    threshold_mean = (
        threshold_E[neurons, 0].mean()
    )

    ax.axhline(
        threshold_mean,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Mean spiking threshold"
    )

    ax.set_title(
        f"{'FGH'[condition]}. {LABELS[condition]}"
    )

    ax.set_xlabel(
        "Mean inhibitory voltage magnitude"
    )

    if condition == 0:
        ax.set_ylabel(
            "Mean voltage (model units)"
        )

    ax.grid(alpha=0.2)

    if condition == 0:
        ax.legend(fontsize=8)

fig.suptitle(
    "Figure 8: Population-level loose balance",
    fontsize=15
)


# ==================================================
# 13. PRINT POPULATION STATISTICS
# ==================================================

def pearson(x, y):

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


print("\nFIGURE 8 POPULATION RESULTS")
print("=" * 65)

for condition, label in enumerate(LABELS):

    exc = mean_exc[:, condition]

    inh = inh_magnitude[:, condition]

    total = mean_total[:, condition]

    print(f"\n{label}")
    print("-" * 45)

    print("Mean excitation:", exc.mean())
    print("SD excitation:", exc.std(ddof=1))

    print("Mean inhibition magnitude:", inh.mean())
    print("SD inhibition:", inh.std(ddof=1))

    print("Mean total voltage:", total.mean())
    print("SD total voltage:", total.std(ddof=1))

    print(
        "Excitation vs inhibition r:",
        pearson(inh, exc)
    )

    print(
        "Total vs inhibition r:",
        pearson(inh, total)
    )

    print(
        "Mean learned threshold:",
        threshold_E[neurons, 0].mean()
    )


# ==================================================
# 14. SAVE RESULTS AND PROVENANCE
# ==================================================

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

figure_path = (
    OUTPUT
    / "figure8_population_balance.png"
)

fig.savefig(
    figure_path,
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)

statistics_path = (
    OUTPUT
    / "figure8_population_statistics.npz"
)

np.savez_compressed(
    statistics_path,

    neurons=neurons,

    selected_patches=selected_patches,

    selected_correlations=selected_correlations,

    mean_exc=mean_exc,
    mean_inh=mean_inh,
    mean_total=mean_total,

    voltage_exc=voltage_exc,
    voltage_inh=voltage_inh,
    voltage_total=voltage_total,

    threshold_E=threshold_E[neurons],

    seed=SEED,
    n_candidates=N_CANDIDATES,
    noise_std=NOISE_STD,

    checkpoint_source=str(CHECKPOINT),
    sta_source=str(STA_FILE),

    evaluation_method="offline_voltage_replay",

    stimulus_protocol="paper_1000_source",
    stimulus_path=str(IMAGE_DIR),
    stimulus_shape=np.array(images.shape),

    track=TRACK,

    presentation_duration=DT * N_STEPS,
    n_source_images=1000
)

print("\n" + "=" * 65)
print("FIGURE 8 POPULATION ANALYSIS COMPLETED")
print("=" * 65)

print("Figure:", figure_path)
print("Statistics:", statistics_path)

