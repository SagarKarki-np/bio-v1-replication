
"""
Figure 10 — Homeostatic learning-rate sensitivity

Full-paper replication, network seed 3002.

Compare eta = 0.05, 0.10 and 0.20 using:
- Identical initial synaptic weights
- Identical initial thresholds
- Matched random stimuli and noise
- 100 calibration batches
- Fixed synaptic weights
- Original count-based threshold rule

Existing calibration results are not modified.
"""

import json
from pathlib import Path

import numpy as np

from analysis import figure10_full_calibration as fc


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

NETWORK_SEED = 3002
CALIBRATION_SEED = NETWORK_SEED + 100000

ETA_VALUES = (0.5,)

N_BATCHES = 100
BATCH_SIZE = 100

TARGET_E_HZ = 2.0
TARGET_I_HZ = 4.0

DURATION = fc.N_STEPS * fc.DT

TARGET_E_COUNT = TARGET_E_HZ * DURATION
TARGET_I_COUNT = TARGET_I_HZ * DURATION

RESULTS_DIR = (
    Path("results")
    / "paper_protocol"
    / f"seed_{NETWORK_SEED}"
    / "eta_sensitivity"
)


# ============================================================
# 2. LOAD LOCKED NETWORK AND IMAGE DATA
# ============================================================

weights, weights_path = fc.load_initial_weights(
    parameterization="paper",
    network_seed=NETWORK_SEED
)

images = fc.PaperImageDataset(
    "data/paper_van_hateren"
)


# ============================================================
# 3. RUN ONE LEARNING-RATE CONDITION
# ============================================================

def run_condition(eta):

    label = f"eta_{eta:.2f}"
    output_dir = RESULTS_DIR / label

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    output_dir.mkdir(parents=True)

    # Reset random sequence for each condition.
    np.random.seed(CALIBRATION_SEED)

    threshold_E = np.full(
        (fc.N_E, 1),
        fc.INITIAL_THRESHOLD_E,
        dtype=float
    )

    threshold_I = np.full(
        (fc.N_I, 1),
        fc.INITIAL_THRESHOLD_I,
        dtype=float
    )

    history_E = []
    history_I = []

    history_theta_E = []
    history_theta_I = []

    print("\n" + "=" * 60, flush=True)
    print(f"LEARNING RATE: {eta}", flush=True)
    print("=" * 60, flush=True)

    for batch in range(N_BATCHES):

        # Generate matching random LGN inputs.
        LGN_spikes = fc.generate_lgn_batch(
            images,
            batch_size=BATCH_SIZE,
            n_steps=fc.N_STEPS,
            dt=fc.DT
        )

        # Run the same frozen network.
        E_spikes, I_spikes = fc.simulate_v1_batch(
            LGN_spikes,
            weights,
            dt=fc.DT,
            tau_m=fc.TAU_M,
            noise_std=fc.NOISE_STD,
            v_min=fc.V_MIN,
            v_reset=fc.V_RESET,
            threshold_E=threshold_E,
            threshold_I=threshold_I,
            record_inputs=False
        )

        # Spike counts per neuron per presentation.
        counts_E = (
            E_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        counts_I = (
            I_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        rate_E = float(
            counts_E.mean() / DURATION
        )

        rate_I = float(
            counts_I.mean() / DURATION
        )

        # Original count-based threshold rule.
        threshold_E = fc.update_thresholds(
            threshold_E,
            counts_E,
            TARGET_E_COUNT,
            eta
        )

        threshold_I = fc.update_thresholds(
            threshold_I,
            counts_I,
            TARGET_I_COUNT,
            eta
        )

        if not (
            np.isfinite(threshold_E).all()
            and np.isfinite(threshold_I).all()
        ):
            raise RuntimeError(
                "Nonfinite threshold detected."
            )

        history_E.append(rate_E)
        history_I.append(rate_I)

        history_theta_E.append(
            float(threshold_E.mean())
        )

        history_theta_I.append(
            float(threshold_I.mean())
        )

        completed = batch + 1

        if completed % 10 == 0:

            print(
                f"Batch {completed:3d}/{N_BATCHES} | "
                f"E: {rate_E:.3f} Hz | "
                f"I: {rate_I:.3f} Hz | "
                f"theta E: {threshold_E.mean():.3f} | "
                f"theta I: {threshold_I.mean():.3f}",
                flush=True
            )

            # Preserve progress every ten batches.
            np.savez_compressed(
                output_dir / "latest_checkpoint.npz",
                completed_batches=completed,
                threshold_E=threshold_E,
                threshold_I=threshold_I,
                observed_E_Hz=np.asarray(history_E),
                observed_I_Hz=np.asarray(history_I),
                mean_threshold_E=np.asarray(history_theta_E),
                mean_threshold_I=np.asarray(history_theta_I)
            )

    # ========================================================
    # 4. SAVE RESULTS
    # ========================================================

    mean_E = float(np.mean(history_E[-10:]))
    mean_I = float(np.mean(history_I[-10:]))

    metadata = {
        "experiment": "eta_sensitivity",
        "network_seed": NETWORK_SEED,
        "calibration_seed": CALIBRATION_SEED,
        "eta_E": eta,
        "eta_I": eta,
        "batches": N_BATCHES,
        "batch_size": BATCH_SIZE,
        "target_E_Hz": TARGET_E_HZ,
        "target_I_Hz": TARGET_I_HZ,
        "update_rule": "count",
        "weights_path": str(weights_path),
        "synaptic_learning": False
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    np.savez_compressed(
        output_dir / "calibration_history.npz",
        observed_E_Hz=np.asarray(history_E),
        observed_I_Hz=np.asarray(history_I),
        mean_threshold_E=np.asarray(history_theta_E),
        mean_threshold_I=np.asarray(history_theta_I)
    )

    np.savez_compressed(
        output_dir / "calibrated_thresholds.npz",
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    print(
        f"\nETA {eta:.2f} COMPLETE | "
        f"Last 10 E: {mean_E:.3f} Hz | "
        f"Last 10 I: {mean_I:.3f} Hz",
        flush=True
    )


# ============================================================
# 5. EXECUTE ALL THREE CONDITIONS
# ============================================================

def main():

    # Check that all destinations are unused before starting.
    for eta in ETA_VALUES:
        path = RESULTS_DIR / f"eta_{eta:.2f}"
        if path.exists():
            raise FileExistsError(
                f"Existing experiment directory: {path}"
            )

    for eta in ETA_VALUES:
        run_condition(eta)

    print("\nALL LEARNING-RATE EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
