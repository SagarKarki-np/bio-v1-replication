
"""
Figure 10: Baseline training-pipeline pilot.

Purpose
-------
Validate that the parameterized Figure 10 learning
function reproduces the original baseline learning
calculations over three consecutive training batches.

This is NOT full Figure 10 training.

No original checkpoints are modified.
"""

from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import (
    DT,
    N_STEPS,
    TAU_STDP_SHORT,
    TAU_STDP_LONG,
    N_E,
    N_I,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
    ETA_THRESHOLD_E,
    ETA_THRESHOLD_I,
    SYMMETRIC_STDP_COEFFICIENT
)

from figure10_config import (
    SOURCE_PROJECT,
    make_condition,
    condition_output_dir
)

from train_v1 import (
    BASE_LEARNING_RATES,
    L1_LIMITS,
    update_homeostatic_thresholds,
    get_learning_multiplier
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch
from learning_batch import learn_from_spikes

from analysis.figure10_learning_batch import (
    learn_from_spikes_figure10
)


# ============================================================
# 1. PILOT CONFIGURATION
# ============================================================

SEED = 2026

N_PILOT_BATCHES = 3
PILOT_BATCH_SIZE = 8

condition = make_condition(
    2.0,
    parameterization="matlab_demo"
)

assert condition.target_E_Hz == 2.0
assert condition.target_I_Hz == 4.0
assert condition.triplet_depression == 2.0

OUTPUT = (
    condition_output_dir(condition)
    / "training_pilot"
)

WEIGHT_KEYS = {
    "LGN_E": "W_LGN_E",
    "LGN_I": "W_LGN_I",
    "E_E": "W_E_E",
    "E_I": "W_E_I",
    "I_I": "W_I_I",
    "I_E": "W_I_E"
}

WEIGHT_ORDER = tuple(WEIGHT_KEYS)


# ============================================================
# 2. LOAD INITIAL WEIGHTS
# ============================================================

weights_path = (
    SOURCE_PROJECT
    / "results"
    / "untrained"
    / "batch_001"
    / "weights.npz"
)

with np.load(weights_path, allow_pickle=False) as data:
    weights = {
        name: data[key].copy()
        for name, key in WEIGHT_KEYS.items()
    }


# ============================================================
# 3. LOAD CONDITION-SPECIFIC THRESHOLDS
# ============================================================

calibration_path = (
    condition_output_dir(condition)
    / "calibration"
    / "calibrated_thresholds.npz"
)

with np.load(calibration_path, allow_pickle=False) as data:
    assert int(data["completed_batches"]) == 100
    assert float(data["target_E_Hz"]) == 2.0
    assert float(data["target_I_Hz"]) == 4.0

    threshold_E = data["threshold_E"].copy()
    threshold_I = data["threshold_I"].copy()

assert threshold_E.shape == (N_E, 1)
assert threshold_I.shape == (N_I, 1)


# ============================================================
# 4. LOAD IMAGE DATA
# ============================================================

images = np.load(
    SOURCE_PROJECT
    / "data"
    / "van_hateren_dog.npy",
    mmap_mode="r"
)


# ============================================================
# 5. PREPARE LEARNING RATES
# ============================================================

learning_rates = BASE_LEARNING_RATES.copy()

learning_rates["LGN_E"] *= (
    condition.lgn_to_e_lr_multiplier
)

learning_rates["E_E"] *= (
    condition.e_to_e_lr_multiplier
)

# For this baseline both multipliers are 1.
# All remaining rates use their verified baseline values.

print("\nFIGURE 10: BASELINE TRAINING PILOT")
print("=" * 65)
print("Condition:", condition.condition_name)
print("Training batches:", N_PILOT_BATCHES)
print("Presentations/batch:", PILOT_BATCH_SIZE)
print("Triplet depression:", condition.triplet_depression)
print("Mean E threshold:", threshold_E.mean())
print("Mean I threshold:", threshold_I.mean())


# ============================================================
# 6. RUN THREE TRAINING BATCHES
# ============================================================

np.random.seed(SEED)

history_E = []
history_I = []

for batch in range(1, N_PILOT_BATCHES + 1):

    # --------------------------------------------------------
    # A. Generate natural-image LGN spikes
    # --------------------------------------------------------

    LGN_spikes = generate_lgn_batch(
        images,
        batch_size=PILOT_BATCH_SIZE,
        n_steps=N_STEPS,
        dt=DT
    )

    # --------------------------------------------------------
    # B. Simulate current network
    # --------------------------------------------------------

    weight_tuple = tuple(
        weights[name]
        for name in WEIGHT_ORDER
    )

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        weight_tuple,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    duration = N_STEPS * DT

    rate_E = (
        E_spikes.sum()
        / (N_E * PILOT_BATCH_SIZE * duration)
    )

    rate_I = (
        I_spikes.sum()
        / (N_I * PILOT_BATCH_SIZE * duration)
    )

    history_E.append(rate_E)
    history_I.append(rate_I)

    # --------------------------------------------------------
    # C. Apply baseline learning-rate schedule
    # --------------------------------------------------------

    overall_batch = 100 + batch

    multiplier = get_learning_multiplier(
        overall_batch
    )

    current_learning_rates = {
        name: value * multiplier
        for name, value in learning_rates.items()
    }

    # --------------------------------------------------------
    # D. Original synaptic-learning function
    # --------------------------------------------------------

    original_weights, original_dW = learn_from_spikes(
        LGN_spikes,
        E_spikes,
        I_spikes,
        {k: v.copy() for k, v in weights.items()},
        current_learning_rates,
        L1_LIMITS,
        dt=DT,
        tau_short=TAU_STDP_SHORT,
        tau_long=TAU_STDP_LONG
    )

    # --------------------------------------------------------
    # E. Figure 10 synaptic-learning function
    # --------------------------------------------------------

    new_weights, new_dW = learn_from_spikes_figure10(
        LGN_spikes,
        E_spikes,
        I_spikes,
        {k: v.copy() for k, v in weights.items()},
        current_learning_rates,
        L1_LIMITS,
        A_p=condition.triplet_potentiation,
        A_d=condition.triplet_depression,
        A_i=SYMMETRIC_STDP_COEFFICIENT,
        dt=DT,
        tau_short=TAU_STDP_SHORT,
        tau_long=TAU_STDP_LONG
    )

    # --------------------------------------------------------
    # F. Check exact equivalence across six pathways
    # --------------------------------------------------------

    for name in WEIGHT_ORDER:

        if not np.array_equal(
            original_weights[name],
            new_weights[name]
        ):
            raise AssertionError(
                f"Weight mismatch: batch {batch}, {name}"
            )

        if not np.array_equal(
            original_dW[name],
            new_dW[name]
        ):
            raise AssertionError(
                f"STDP mismatch: batch {batch}, {name}"
            )

    # --------------------------------------------------------
    # G. Update homeostatic thresholds
    # --------------------------------------------------------

    threshold_E, threshold_I = (
        update_homeostatic_thresholds(
            E_spikes,
            I_spikes,
            threshold_E,
            threshold_I,
            ETA_THRESHOLD_E * multiplier,
            ETA_THRESHOLD_I * multiplier
        )
    )

    # --------------------------------------------------------
    # H. Continue using the updated Figure 10 weights
    # --------------------------------------------------------

    weights = new_weights

    print(
        f"Batch {batch} | "
        f"E: {rate_E:.3f} Hz | "
        f"I: {rate_I:.3f} Hz | "
        f"theta E: {threshold_E.mean():.3f} | "
        f"theta I: {threshold_I.mean():.3f} | "
        f"EXACT MATCH",
        flush=True
    )


# ============================================================
# 7. VALIDATE FINAL PARAMETERS
# ============================================================

for name, W in weights.items():

    if not np.isfinite(W).all():
        raise AssertionError(
            f"Nonfinite weights: {name}"
        )

    if np.any(W < 0):
        raise AssertionError(
            f"Negative weights: {name}"
        )

    if np.any(
        W.sum(axis=1) > L1_LIMITS[name] + 1e-8
    ):
        raise AssertionError(
            f"L1 limit exceeded: {name}"
        )

assert np.isfinite(threshold_E).all()
assert np.isfinite(threshold_I).all()


# ============================================================
# 8. SAVE PILOT RESULTS
# ============================================================

OUTPUT.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT / "training_pilot_results.npz",
    **{
        WEIGHT_KEYS[name]: W
        for name, W in weights.items()
    },
    threshold_E=threshold_E,
    threshold_I=threshold_I,
    observed_E_Hz=np.asarray(history_E),
    observed_I_Hz=np.asarray(history_I),
    completed_batches=N_PILOT_BATCHES,
    seed=SEED,
    batch_size=PILOT_BATCH_SIZE,
    parameterization=condition.parameterization,
    target_E_Hz=condition.target_E_Hz,
    target_I_Hz=condition.target_I_Hz,
    triplet_depression=condition.triplet_depression,
    baseline_equivalence=True
)

print("\n" + "=" * 65)
print("EXACT THREE-BATCH LEARNING MATCH")
print("All six synaptic pathways passed.")
print("Results saved to:", OUTPUT)
