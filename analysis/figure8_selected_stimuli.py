
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

OUTPUT = PROJECT / "results" / "figure8_selected_stimuli"
OUTPUT.mkdir(parents=True, exist_ok=True)

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I", "W_E_E",
    "W_E_I", "W_I_I", "W_I_E"
)

NEURON = 0
N_CANDIDATES = 1000
SEED = 2027
NOISE_STD = 0.0

np.random.seed(SEED)


# --------------------------------------------------
# 1. Load learned network, STA, and natural images
# --------------------------------------------------

with np.load(CHECKPOINT, allow_pickle=False) as data:
    weights = tuple(data[k].copy() for k in WEIGHT_NAMES)
    threshold_E = data["threshold_E"].copy()
    threshold_I = data["threshold_I"].copy()

with np.load(STA_FILE, allow_pickle=False) as data:
    STA = data["STA_E"][NEURON].copy()

images = np.load(IMAGE_FILE, mmap_mode="r")

print("Images:", images.shape)
print("STA:", STA.shape)


# --------------------------------------------------
# 2. Generate 1000 candidate natural-image patches
# --------------------------------------------------

patches = np.zeros(
    (N_CANDIDATES, PATCH_SIZE, PATCH_SIZE)
)

for i in range(N_CANDIDATES):
    image_index = np.random.randint(images.shape[2])
    image = images[:, :, image_index]

    # Reuse original patch extraction and rotation
    patches[i] = extract_random_patch(
        image, PATCH_SIZE, 20
    )


# --------------------------------------------------
# 3. Correlate candidate patches with the STA
# --------------------------------------------------

patch_vectors = np.stack([
    patch.flatten(order="F")
    for patch in patches
])

sta_centered = STA - STA.mean()
sta_norm = np.linalg.norm(sta_centered)

if sta_norm == 0:
    raise ValueError("Selected neuron's STA is constant.")

patch_centered = (
    patch_vectors
    - patch_vectors.mean(axis=1, keepdims=True)
)

patch_norms = np.linalg.norm(patch_centered, axis=1)

correlations = np.full(N_CANDIDATES, np.nan)

valid = patch_norms > 0

correlations[valid] = (
    patch_centered[valid] @ sta_centered
) / (patch_norms[valid] * sta_norm)

if not np.any(valid):
    raise ValueError("No valid image patches.")

lowest = int(np.nanargmin(correlations))
neutral = int(np.nanargmin(np.abs(correlations)))
highest = int(np.nanargmax(correlations))

selected_indices = np.array(
    [lowest, neutral, highest]
)

labels = [
    "Anti-correlated",
    "Approximately uncorrelated",
    "Correlated"
]

selected_patches = patches[selected_indices]
selected_correlations = correlations[selected_indices]

print("\nSELECTED PATCHES")
for label, index, corr in zip(
    labels, selected_indices, selected_correlations
):
    print(
        f"{label}: candidate {index}, "
        f"STA correlation = {corr:.5f}"
    )


# --------------------------------------------------
# 4. Convert selected patches to LGN spike trains
# --------------------------------------------------

LGN_spikes = np.zeros(
    (512, 3, N_STEPS), dtype=bool
)

for i, patch in enumerate(selected_patches):

    vector = patch.flatten(order="F")
    scaled = vector * LGN_RATE_SCALE

    ON = np.maximum(scaled, 0)
    OFF = np.maximum(-scaled, 0)

    rates = np.concatenate([ON, OFF])
    rates = np.minimum(rates, LGN_MAX_RATE)

    LGN_spikes[:, i, :] = (
        generate_lgn_population_spikes(
            rates, DT, N_STEPS
        )
    )

print("LGN spikes:", LGN_spikes.shape)


# --------------------------------------------------
# 5. Run normal spiking network
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Reconstruct voltage components
# --------------------------------------------------

excitation = (
    traces["LGN_E"][NEURON]
    + traces["E_E"][NEURON]
)

inhibition = traces["I_E"][NEURON]
total_input = traces["total_E"][NEURON]

decay = np.exp(-DT / TAU_M)

V_exc = np.zeros((3, N_STEPS))
V_inh = np.zeros((3, N_STEPS))
V_total = np.zeros((3, N_STEPS))

ve = np.zeros(3)
vi = np.zeros(3)
vt = np.zeros(3)

for t in range(N_STEPS):

    ve = decay * ve + excitation[:, t]
    vi = decay * vi - inhibition[:, t]
    vt = decay * vt + total_input[:, t]

    V_exc[:, t] = ve
    V_inh[:, t] = vi
    V_total[:, t] = vt

print(
    "\nVoltage reconstruction error:",
    np.max(np.abs(V_total - V_exc - V_inh))
)


# --------------------------------------------------
# 7. Plot Figure 8 A-D reconstruction
# --------------------------------------------------

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

ax_rf = fig.add_subplot(gs[0, 0])

# STA is stored in MATLAB/Fortran pixel ordering
ax_rf.imshow(
    STA.reshape(PATCH_SIZE, PATCH_SIZE, order="F"),
    cmap="coolwarm"
)
ax_rf.set_title(f"A. E neuron {NEURON}: STA")
ax_rf.axis("off")

ax_images = fig.add_subplot(gs[0, 1:])

image_strip = np.concatenate(
    selected_patches, axis=1
)

ax_images.imshow(image_strip, cmap="gray")
ax_images.set_title(
    "Selected natural-image patches\n"
    + " | ".join(
        f"{label}: r={corr:.3f}"
        for label, corr in zip(
            labels, selected_correlations
        )
    )
)
ax_images.axis("off")

time_ms = np.arange(N_STEPS) * DT * 1000

for i in range(3):
    ax = fig.add_subplot(gs[1, i])

    ax.plot(
        time_ms, V_exc[i],
        color="green", label="Excitation"
    )
    ax.plot(
        time_ms, V_inh[i],
        color="purple", label="Inhibition"
    )
    ax.plot(
        time_ms, V_total[i],
        color="goldenrod", label="Total"
    )

    ax.axhline(
        threshold_E[NEURON, 0],
        linestyle="--",
        color="black",
        alpha=0.5
    )

    ax.set_title(
        f"{labels[i]}\n"
        f"STA correlation = {selected_correlations[i]:.3f}"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (model units)")
    ax.grid(alpha=0.2)

    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle(
    "Figure 8: RF-selected natural images and voltage balance",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    OUTPUT / "figure8_selected_stimuli.png",
    dpi=250
)
plt.close(fig)


# --------------------------------------------------
# 8. Numerical summaries and saved results
# --------------------------------------------------

print("\nMEAN VOLTAGE BY STIMULUS")

for i, label in enumerate(labels):
    print(
        label,
        "Excitation:", V_exc[i].mean(),
        "Inhibition:", V_inh[i].mean(),
        "Total:", V_total[i].mean()
    )

np.savez_compressed(
    OUTPUT / "figure8_selected_stimuli.npz",
    selected_indices=selected_indices,
    selected_patches=selected_patches,
    selected_correlations=selected_correlations,
    candidate_correlations=correlations,
    V_exc=V_exc,
    V_inh=V_inh,
    V_total=V_total,
    excitation=excitation,
    inhibition=inhibition,
    total_input=total_input,
    neuron=NEURON,
    seed=SEED,
    sta_source=str(STA_FILE),
    checkpoint_source=str(CHECKPOINT),
    noise_std=NOISE_STD
)

print("\nFIGURE 8 SELECTED-STIMULUS ANALYSIS COMPLETED")
print("Results:", OUTPUT)
