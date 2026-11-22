
"""
Track A — Matched-input synaptic weight diagnostic.

Compare the initial and batch-5 trained V1 networks.

Four conditions:
A: Initial weights + initial thresholds
B: Trained weights + initial thresholds
C: Initial weights + trained thresholds
D: Trained weights + trained thresholds

All conditions receive identical LGN spikes and
replayed NumPy random noise.

No STDP updates are performed.
No files are modified or saved.
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
EXPECTED_CHECKPOINT_BATCH = 5

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

INITIAL_THRESHOLD_PATH = (
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
# 2. LOAD INITIAL NETWORK
# ============================================================

initial_tuple, initial_path = (
    fc.load_initial_weights(
        parameterization="paper",
        network_seed=NETWORK_SEED
    )
)

initial_weights = {
    name: W.copy()
    for name, W in zip(
        WEIGHT_NAMES,
        initial_tuple
    )
}

with np.load(
    INITIAL_THRESHOLD_PATH,
    allow_pickle=False
) as data:

    initial_theta_E = data["threshold_E"].copy()
    initial_theta_I = data["threshold_I"].copy()


# ============================================================
# 3. LOAD BATCH-5 TRAINED NETWORK
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

if completed != EXPECTED_CHECKPOINT_BATCH:
    raise ValueError(
        f"Expected checkpoint at batch 5, "
        f"found batch {completed}"
    )

print("=" * 65)
print("TRACK A — MATCHED-INPUT DIAGNOSTIC")
print("=" * 65)

print("Initial weights:", initial_path)
print("Trained batch:", completed)

print("\nINITIAL THRESHOLDS")

print("E:", initial_theta_E.mean())
print("I:", initial_theta_I.mean())

print("\nTRAINED THRESHOLDS")

print("E:", trained_theta_E.mean())
print("I:", trained_theta_I.mean())


# ============================================================
# 4. GENERATE ONE SHARED LGN INPUT BATCH
# ============================================================

images = fc.PaperImageDataset(
    str(PROJECT / "data" / "paper_van_hateren")
)

# Independent diagnostic seed.
# The same LGN input is reused for all conditions.
np.random.seed(DIAGNOSTIC_SEED)

LGN_spikes = fc.generate_lgn_batch(
    images,
    batch_size=BATCH_SIZE,
    n_steps=fc.N_STEPS,
    dt=fc.DT
)

assert LGN_spikes.shape == (
    512,
    BATCH_SIZE,
    fc.N_STEPS
)

# Save the random state immediately after LGN
# generation so each V1 simulation uses the
# same noise sequence.
shared_noise_state = np.random.get_state()

print("\nShared LGN spikes:", LGN_spikes.shape)


# ============================================================
# 5. SIMULATION FUNCTION
# ============================================================

def run_condition(label, weights, theta_E, theta_I):

    # Rewind random noise before each simulation.
    np.random.set_state(shared_noise_state)

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

    rate_E = float(
        E_spikes.sum()
        / (
            E_spikes.shape[0]
            * BATCH_SIZE
            * duration
        )
    )

    rate_I = float(
        I_spikes.sum()
        / (
            I_spikes.shape[0]
            * BATCH_SIZE
            * duration
        )
    )

    print(
        f"{label:35s} | "
        f"E: {rate_E:8.4f} Hz | "
        f"I: {rate_I:8.4f} Hz"
    )

    return rate_E, rate_I


# ============================================================
# 6. FOUR MATCHED CONDITIONS
# ============================================================

print("\n" + "=" * 65)
print("MATCHED FIRING-RATE COMPARISON")
print("=" * 65)

A = run_condition(
    "A: Initial W + Initial theta",
    initial_weights,
    initial_theta_E,
    initial_theta_I
)

B = run_condition(
    "B: Trained W + Initial theta",
    trained_weights,
    initial_theta_E,
    initial_theta_I
)

C = run_condition(
    "C: Initial W + Trained theta",
    initial_weights,
    trained_theta_E,
    trained_theta_I
)

D = run_condition(
    "D: Trained W + Trained theta",
    trained_weights,
    trained_theta_E,
    trained_theta_I
)


# ============================================================
# 7. CALCULATE DIFFERENCES
# ============================================================

print("\n" + "=" * 65)
print("EFFECTS RELATIVE TO BASELINE")
print("=" * 65)

print("\nChanging only synaptic weights (B - A):")

print(f"E: {B[0] - A[0]:+.4f} Hz")
print(f"I: {B[1] - A[1]:+.4f} Hz")

print("\nChanging only thresholds (C - A):")

print(f"E: {C[0] - A[0]:+.4f} Hz")
print(f"I: {C[1] - A[1]:+.4f} Hz")

print("\nChanging both (D - A):")

print(f"E: {D[0] - A[0]:+.4f} Hz")
print(f"I: {D[1] - A[1]:+.4f} Hz")

print("\nWeight-threshold interaction (D - B - C + A):")

print(
    f"E: {D[0] - B[0] - C[0] + A[0]:+.4f} Hz"
)

print(
    f"I: {D[1] - B[1] - C[1] + A[1]:+.4f} Hz"
)


# ============================================================
# 8. WEIGHT MATRIX DIFFERENCES
# ============================================================

print("\n" + "=" * 65)
print("WEIGHT MATRIX COMPARISON")
print("=" * 65)

for name in tr.WEIGHT_ORDER:

    W0 = initial_weights[name]
    W5 = trained_weights[name]

    difference = W5 - W0

    print(
        f"{name:6s} | "
        f"mean abs change: "
        f"{np.abs(difference).mean():.6f} | "
        f"max abs change: "
        f"{np.abs(difference).max():.6f} | "
        f"row L1 mean: "
        f"{W5.sum(axis=1).mean():.4f}"
    )


# ============================================================
# 9. FINISH
# ============================================================

print("\n" + "=" * 65)
print("DIAGNOSTIC COMPLETE")
print("=" * 65)

print("Synaptic learning performed: False")
print("Files modified: None")
