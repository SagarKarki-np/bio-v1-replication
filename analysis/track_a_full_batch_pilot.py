
"""
Track A — Full-batch synaptic learning pilot

Ruslim et al. (2025) LGN-V1 replication.

Purpose
-------
Test ONE synaptic-learning batch containing
100 image presentations.

Starting conditions:
    Network seed: 3002
    Thresholds: original 100-batch calibration
    Parameterization: paper
    Threshold learning rate: 0.1

The script does not save updated weights or
modify existing calibration results.
"""

from pathlib import Path

import numpy as np

import figure10_config

from analysis import figure10_full_calibration as fc
from analysis import figure10_train as tr


# ============================================================
# 1. CONFIGURATION
# ============================================================

NETWORK_SEED = 3002
PILOT_SEED = 303002

BATCH_SIZE = 100

TARGET_E_HZ = 2.0
TARGET_I_HZ = 4.0

ROOT = Path("results/paper_protocol")

THRESHOLD_PATH = (
    ROOT
    / f"seed_{NETWORK_SEED}"
    / "calibration"
    / "E02_I04"
    / "calibrated_thresholds.npz"
)


# ============================================================
# 2. LOAD PAPER PARAMETERIZATION
# ============================================================

condition = figure10_config.make_condition(
    TARGET_E_HZ,
    parameterization="paper"
)

print("=" * 60)
print("TRACK A — FULL-BATCH PILOT")
print("=" * 60)

print("Network seed:", NETWORK_SEED)
print("Pilot seed:", PILOT_SEED)
print("Batch size:", BATCH_SIZE)

print(
    "Threshold learning rate:",
    condition.eta_threshold_E
)

print(
    "Triplet depression:",
    condition.triplet_depression
)


# ============================================================
# 3. LOAD INITIAL NETWORK WEIGHTS
# ============================================================

initial_weights, weights_path = (
    fc.load_initial_weights(
        parameterization="paper",
        network_seed=NETWORK_SEED
    )
)

# Full-paper initial weights are returned in this order.
INITIAL_WEIGHT_NAMES = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E"
)

weights = {
    name: W.copy()
    for name, W in zip(
        INITIAL_WEIGHT_NAMES,
        initial_weights
    )
}

print("\nInitial weights loaded:")
print(weights_path)


# ============================================================
# 4. LOAD ORIGINAL 100-BATCH THRESHOLDS
# ============================================================

if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(
        THRESHOLD_PATH
    )

with np.load(
    THRESHOLD_PATH,
    allow_pickle=False
) as data:

    theta_E = data["threshold_E"].copy()
    theta_I = data["threshold_I"].copy()

assert theta_E.shape == (fc.N_E, 1)
assert theta_I.shape == (fc.N_I, 1)

print("\nStarting thresholds:")
print("E:", theta_E.mean())
print("I:", theta_I.mean())


# ============================================================
# 5. LOAD LOCKED PAPER DATASET
# ============================================================

images = fc.PaperImageDataset(
    "data/paper_van_hateren"
)

print("\nPaper image dataset loaded.")


# ============================================================
# 6. GENERATE LGN SPIKE TRAINS
# ============================================================

np.random.seed(PILOT_SEED)

LGN_spikes = fc.generate_lgn_batch(
    images,
    batch_size=BATCH_SIZE,
    n_steps=fc.N_STEPS,
    dt=fc.DT
)

print("\nLGN spike shape:")
print(LGN_spikes.shape)

assert LGN_spikes.shape == (
    512,
    BATCH_SIZE,
    fc.N_STEPS
)


# ============================================================
# 7. SIMULATE V1 ACTIVITY
# ============================================================

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
    threshold_E=theta_E,
    threshold_I=theta_I,
    record_inputs=False
)

duration = fc.N_STEPS * fc.DT

rate_E = E_spikes.sum() / (
    E_spikes.shape[0] * BATCH_SIZE * duration
)

rate_I = I_spikes.sum() / (
    I_spikes.shape[0] * BATCH_SIZE * duration
)

print("\nV1 firing rates:")
print(f"E: {rate_E:.4f} Hz")
print(f"I: {rate_I:.4f} Hz")


# ============================================================
# 8. LEARNING-RATE SCHEDULE
# ============================================================

# First synaptic-learning batch follows
# the original 100 calibration batches.
overall_batch = 101

multiplier = tr.get_learning_multiplier(
    overall_batch
)

base_rates = tr.condition_learning_rates(
    condition
)

learning_rates = {
    name: rate * multiplier
    for name, rate in base_rates.items()
}

eta_E = (
    condition.eta_threshold_E
    * multiplier
)

eta_I = (
    condition.eta_threshold_I
    * multiplier
)

print("\nLearning-rate multiplier:", multiplier)

print("\nEffective synaptic learning rates:")

for name, rate in learning_rates.items():
    print(f"{name}: {rate}")


# ============================================================
# 9. APPLY SYNAPTIC PLASTICITY
# ============================================================

updated_weights, scaled_dW = (
    tr.learn_from_spikes_figure10(
        LGN_spikes,
        E_spikes,
        I_spikes,
        weights,
        learning_rates,
        tr.L1_LIMITS,
        A_p=condition.triplet_potentiation,
        A_d=condition.triplet_depression,
        A_i=tr.SYMMETRIC_STDP_COEFFICIENT,
        dt=fc.DT,
        tau_short=tr.TAU_STDP_SHORT,
        tau_long=tr.TAU_STDP_LONG
    )
)


# ============================================================
# 10. APPLY HOMEOSTATIC THRESHOLD ADAPTATION
# ============================================================

new_theta_E, new_theta_I = (
    tr.update_figure10_thresholds(
        E_spikes,
        I_spikes,
        theta_E,
        theta_I,
        target_E_Hz=TARGET_E_HZ,
        target_I_Hz=TARGET_I_HZ,
        eta_theta_E=eta_E,
        eta_theta_I=eta_I,
        dt=fc.DT
    )
)


# ============================================================
# 11. VALIDATE UPDATED PARAMETERS
# ============================================================

tr.validate_parameters(
    updated_weights,
    new_theta_E,
    new_theta_I
)

print("\nUPDATED THRESHOLDS")
print("=" * 60)

print(
    "E:",
    theta_E.mean(),
    "->",
    new_theta_E.mean()
)

print(
    "I:",
    theta_I.mean(),
    "->",
    new_theta_I.mean()
)


# ============================================================
# 12. FULL-BATCH WEIGHT DIAGNOSTICS
# ============================================================

print("\nFULL-BATCH WEIGHT DIAGNOSTICS")
print("=" * 60)

for name in tr.WEIGHT_ORDER:

    W_before = weights[name]
    W_after = updated_weights[name]

    difference = np.abs(
        W_after - W_before
    )

    row_sums = W_after.sum(axis=1)

    print(
        f"{name:6s} | "
        f"mean dW: {difference.mean():.6f} | "
        f"max dW: {difference.max():.6f} | "
        f"row L1 min: {row_sums.min():.4f} | "
        f"row L1 max: {row_sums.max():.4f}"
    )

    assert np.isfinite(W_after).all(), name
    assert np.all(W_after >= 0), name

    # Existing normalization routine determines
    # whether row sums equal or remain below
    # their allowed L1 limits.
    assert np.all(
        row_sums <= tr.L1_LIMITS[name] + 1e-6
    ), name


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRACK A — FULL-BATCH PILOT SUMMARY")
print("=" * 60)

print("Training batches executed: 1")
print("Presentations:", BATCH_SIZE)

print(f"E firing: {rate_E:.4f} Hz")
print(f"I firing: {rate_I:.4f} Hz")

print(
    "Final mean E threshold:",
    new_theta_E.mean()
)

print(
    "Final mean I threshold:",
    new_theta_I.mean()
)

print("\nFULL-BATCH PILOT PASSED")
print("No model parameters were saved.")
print("Original calibration files remain unchanged.")
