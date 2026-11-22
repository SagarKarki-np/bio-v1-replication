
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import DT, N_STEPS, TAU_M, V_MIN, V_RESET
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch

ORIGINAL = PROJECT.parent / "V1_network_learning"

CHECKPOINT = (
    ORIGINAL / "results" / "learning"
    / "training_test_25" / "batch_1100.npz"
)

IMAGE_PATH = ORIGINAL / "data" / "van_hateren_dog.npy"

OUTPUT = PROJECT / "results" / "figure8_voltage_pilot"
OUTPUT.mkdir(parents=True, exist_ok=True)

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E"
)

NEURON = 0
N_IMAGES = 3

# Zero noise makes the voltage decomposition exact
# for this initial computational validation.
NOISE_STD = 0.0

np.random.seed(2027)


# -------------------------------------------------
# 1. Load trained network
# -------------------------------------------------

with np.load(CHECKPOINT, allow_pickle=False) as data:

    weights = tuple(
        data[name].copy()
        for name in WEIGHT_NAMES
    )

    threshold_E = data["threshold_E"].copy()
    threshold_I = data["threshold_I"].copy()

images = np.load(IMAGE_PATH, mmap_mode="r")

print("Images:", images.shape)
print("Checkpoint:", CHECKPOINT)


# -------------------------------------------------
# 2. Generate three LGN presentations
# -------------------------------------------------

LGN_spikes = generate_lgn_batch(
    images,
    batch_size=N_IMAGES,
    n_steps=N_STEPS,
    dt=DT
)

print("LGN spikes:", LGN_spikes.shape)


# -------------------------------------------------
# 3. Run normal spiking network
# -------------------------------------------------

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

print("E spikes:", E_spikes.shape)
print("I spikes:", I_spikes.shape)


# -------------------------------------------------
# 4. Extract input for one excitatory neuron
# -------------------------------------------------

excitation = (
    traces["LGN_E"][NEURON]
    + traces["E_E"][NEURON]
)

inhibition = traces["I_E"][NEURON]

total_input = traces["total_E"][NEURON]

print(
    "Input identity maximum error:",
    np.max(
        np.abs(
            excitation - inhibition - total_input
        )
    )
)


# -------------------------------------------------
# 5. Reconstruct unbounded membrane voltages
# -------------------------------------------------

decay = np.exp(-DT / TAU_M)

V_exc = np.zeros((N_IMAGES, N_STEPS))
V_inh = np.zeros((N_IMAGES, N_STEPS))
V_total = np.zeros((N_IMAGES, N_STEPS))

previous_exc = np.zeros(N_IMAGES)
previous_inh = np.zeros(N_IMAGES)
previous_total = np.zeros(N_IMAGES)

for t in range(N_STEPS):

    # Excitatory-only contribution
    previous_exc = (
        decay * previous_exc + excitation[:, t]
    )

    # Inhibitory-only contribution, negative by definition
    previous_inh = (
        decay * previous_inh - inhibition[:, t]
    )

    # Combined voltage without resets or clipping
    previous_total = (
        decay * previous_total + total_input[:, t]
    )

    V_exc[:, t] = previous_exc
    V_inh[:, t] = previous_inh
    V_total[:, t] = previous_total


print(
    "Voltage decomposition maximum error:",
    np.max(
        np.abs(V_total - (V_exc + V_inh))
    )
)


# -------------------------------------------------
# 6. Plot three illustrative presentations
# -------------------------------------------------

time_ms = np.arange(N_STEPS) * DT * 1000

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 4),
    sharey=True,
    constrained_layout=True
)

for i, ax in enumerate(axes):

    ax.plot(
        time_ms, V_exc[i],
        color="green", label="Excitation only"
    )

    ax.plot(
        time_ms, V_inh[i],
        color="purple", label="Inhibition only"
    )

    ax.plot(
        time_ms, V_total[i],
        color="goldenrod", label="Total"
    )

    ax.axhline(
        float(threshold_E[NEURON, 0]),
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Original threshold"
    )

    ax.set_title(f"Presentation {i + 1}")
    ax.set_xlabel("Time (ms)")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Membrane-voltage model units")
axes[0].legend(fontsize=8)

fig.suptitle(
    f"Figure 8 pilot: E neuron {NEURON}\n"
    "Unbounded voltage replay using normal network spiking"
)

fig.savefig(
    OUTPUT / "figure8_voltage_pilot.png",
    dpi=250
)

plt.close(fig)


# -------------------------------------------------
# 7. Save numerical results
# -------------------------------------------------

np.savez_compressed(
    OUTPUT / "figure8_voltage_pilot.npz",
    excitation=excitation,
    inhibition=inhibition,
    total_input=total_input,
    V_exc=V_exc,
    V_inh=V_inh,
    V_total=V_total,
    time_ms=time_ms,
    neuron=NEURON,
    noise_std=NOISE_STD
)

print("\nFIGURE 8 VOLTAGE PILOT COMPLETED")
print("Output:", OUTPUT)
