
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import (
    DT, N_STEPS, TAU_M, V_MIN, V_RESET,
    PATCH_SIZE, LGN_RATE_SCALE, LGN_MAX_RATE
)
from preprocessing import extract_random_patch
from simulation import (
    generate_lgn_population_spikes,
    simulate_v1_batch
)

ORIGINAL = PROJECT.parent / "V1_network_learning"

CHECKPOINT = (
    ORIGINAL / "results" / "learning"
    / "training_test_25" / "batch_1100.npz"
)

STA_FILE = (
    PROJECT / "results"
    / "STA_100000_training_test_25"
    / "STA_100000_results.npz"
)

IMAGE_FILE = ORIGINAL / "data" / "van_hateren_dog.npy"

OUTPUT = PROJECT / "results" / "figure8_population_balance"
OUTPUT.mkdir(parents=True, exist_ok=True)

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I", "W_E_E",
    "W_E_I", "W_I_I", "W_I_E"
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

# Local generator for stimulus and neuron selection.
rng = np.random.default_rng(SEED)


# --------------------------------------------------
# 1. Load trained network and STAs
# --------------------------------------------------

with np.load(CHECKPOINT, allow_pickle=False) as data:
    weights = tuple(
        data[name].copy() for name in WEIGHT_NAMES
    )
    threshold_E = data["threshold_E"].copy()
    threshold_I = data["threshold_I"].copy()

with np.load(STA_FILE, allow_pickle=False) as data:
    STA_E = data["STA_E"].copy()

images = np.load(IMAGE_FILE, mmap_mode="r")

assert STA_E.shape == (400, 256)

neurons = rng.choice(
    400, size=N_NEURONS, replace=False
)

print("Selected neurons:", neurons)
print("Checkpoint:", CHECKPOINT)


# --------------------------------------------------
# 2. Select 3 patches for each neuron
# --------------------------------------------------

selected_patches = np.zeros(
    (N_NEURONS, 3, PATCH_SIZE, PATCH_SIZE),
    dtype=float
)

selected_correlations = np.zeros((N_NEURONS, 3))

for i, neuron in enumerate(neurons):

    candidates = np.empty(
        (N_CANDIDATES, PATCH_SIZE, PATCH_SIZE)
    )

    for j in range(N_CANDIDATES):

        image_index = rng.integers(images.shape[2])

        # Use a reproducible seed with the existing
        # NumPy-global-RNG patch extractor.
        np.random.seed(
            int(rng.integers(0, 2**32 - 1))
        )

        candidates[j] = extract_random_patch(
            images[:, :, image_index],
            PATCH_SIZE,
            20
        )

    vectors = np.stack([
        p.flatten(order="F") for p in candidates
    ])

    sta = STA_E[neuron]
    sta = sta - sta.mean()
    sta_norm = np.linalg.norm(sta)

    centered = vectors - vectors.mean(
        axis=1, keepdims=True
    )

    norms = np.linalg.norm(centered, axis=1)

    correlations = np.full(
        N_CANDIDATES, np.nan
    )

    valid = (
        np.isfinite(norms)
        & (norms > 0)
        & (sta_norm > 0)
    )

    correlations[valid] = (
        centered[valid] @ sta
    ) / (norms[valid] * sta_norm)

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
    selected_correlations[i] = correlations[chosen]

print("\nSTIMULUS SELECTION COMPLETED")

for k, label in enumerate(LABELS):
    print(
        label,
        "mean STA correlation:",
        selected_correlations[:, k].mean()
    )


# --------------------------------------------------
# 3. Prepare population voltage measurements
# --------------------------------------------------

# Each neuron has one observation for each condition.
# Shape: (40 neurons, 3 conditions)

mean_exc = np.zeros((N_NEURONS, 3))
mean_inh = np.zeros((N_NEURONS, 3))
mean_total = np.zeros((N_NEURONS, 3))

# Also retain the complete voltage trajectories.
voltage_exc = np.zeros((N_NEURONS, 3, N_STEPS))
voltage_inh = np.zeros((N_NEURONS, 3, N_STEPS))
voltage_total = np.zeros((N_NEURONS, 3, N_STEPS))

# Flatten the selected stimulus index:
# neuron 0: conditions 0,1,2
# neuron 1: conditions 0,1,2, etc.
presentations = [
    (i, condition)
    for i in range(N_NEURONS)
    for condition in range(3)
]

decay = np.exp(-DT / TAU_M)


# --------------------------------------------------
# 4. Evaluate small batches
# --------------------------------------------------

for start in range(0, len(presentations), BATCH_SIZE):

    group = presentations[start:start+BATCH_SIZE]
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
        rates = np.minimum(rates, LGN_MAX_RATE)

        LGN_spikes[:, j, :] = (
            generate_lgn_population_spikes(
                rates, DT, N_STEPS
            )
        )

    # Use normal network spiking to obtain
    # recurrent excitatory/inhibitory activity.
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

        inh_input = traces["I_E"][neuron, j, :]

        total_input = traces["total_E"][neuron, j, :]

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
        f"/{len(presentations)} presentations"
    )


# --------------------------------------------------
# 5. Validate decomposition
# --------------------------------------------------

error = np.max(
    np.abs(
        voltage_total - (voltage_exc + voltage_inh)
    )
)

print("\nVoltage decomposition maximum error:", error)

assert error < 1e-8

# Inhibitory magnitude is positive for scatter plots.
inh_magnitude = -mean_inh


# --------------------------------------------------
# 6. Figure 8E: mean and standard deviation
# --------------------------------------------------

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
    stds = measurements.std(axis=0, ddof=1)

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


# --------------------------------------------------
# 7. Figure 8F-H: voltage versus inhibition
# --------------------------------------------------

for condition in range(3):

    ax = fig.add_subplot(gs[1, condition])

    inhibitor = inh_magnitude[:, condition]
    excitation = mean_exc[:, condition]
    total = mean_total[:, condition]

    ax.scatter(
        inhibitor, excitation,
        color="green",
        alpha=0.7,
        label="Excitation only"
    )

    ax.scatter(
        inhibitor, total,
        color="goldenrod",
        alpha=0.7,
        label="Total"
    )

    # Per-neuron learned firing thresholds vary,
    # so plot their population mean as a reference.
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

    ax.set_xlabel("Mean inhibitory voltage magnitude")

    if condition == 0:
        ax.set_ylabel("Mean voltage (model units)")

    ax.grid(alpha=0.2)

    if condition == 0:
        ax.legend(fontsize=8)

fig.suptitle(
    "Figure 8: Population-level loose balance",
    fontsize=15
)

fig.savefig(
    OUTPUT / "figure8_population_balance.png",
    dpi=250
)

plt.close(fig)


# --------------------------------------------------
# 8. Population statistics
# --------------------------------------------------

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


# --------------------------------------------------
# 9. Save all measurements and provenance
# --------------------------------------------------

np.savez_compressed(
    OUTPUT / "figure8_population_statistics.npz",

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
    evaluation_method="offline_voltage_replay"
)

print("\nFIGURE 8 POPULATION ANALYSIS COMPLETED")
print("Results:", OUTPUT)
