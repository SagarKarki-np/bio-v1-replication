
"""
Figure 10: Resumable synaptic training.

Ruslim et al. (2025), PLOS Computational Biology.

Current implementation:
    MATLAB-demo parameterization.

Features:
    - Condition-specific firing-rate targets
    - Condition-specific triplet STDP
    - Learning-rate schedule
    - Homeostatic threshold adaptation
    - Checkpoint saving and exact RNG restoration
    - Independent output directories

Learning-rate assumption:
    Only LGN_E and E_E receive the target-rate
    multipliers currently verified for Figure 10.
    The other four rates remain at baseline values.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    DT, N_STEPS, BATCH_SIZE, N_E, N_I,
    TAU_M, NOISE_STD, V_MIN, V_RESET,
    TAU_STDP_SHORT, TAU_STDP_LONG,
    ETA_THRESHOLD_E, ETA_THRESHOLD_I,
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
    get_learning_multiplier
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch

from analysis.figure10_learning_batch import (
    learn_from_spikes_figure10
)

from analysis.figure10_homeostasis import (
    update_figure10_thresholds
)


WEIGHT_KEYS = {
    "LGN_E": "W_LGN_E",
    "LGN_I": "W_LGN_I",
    "E_E": "W_E_E",
    "E_I": "W_E_I",
    "I_I": "W_I_I",
    "I_E": "W_I_E"
}

WEIGHT_ORDER = (
    "LGN_E", "LGN_I", "E_E",
    "E_I", "I_I", "I_E"
)

CALIBRATION_BATCHES = 100
TOTAL_LEARNING_BATCHES = 1100
CHECKPOINT_INTERVAL = 10


# ============================================================
# 1. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-e", type=float, required=True
    )

    parser.add_argument(
        "--parameterization",
        choices=["matlab_demo"],
        default="matlab_demo"
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=TOTAL_LEARNING_BATCHES,
        help="Total learning batches to complete."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Optional separate output directory."
 )

    return parser.parse_args()


# ============================================================
# 2. LOAD INITIAL NETWORK
# ============================================================

def load_initial_weights():

    path = (
        Path(SOURCE_PROJECT)
        / "results"
        / "untrained"
        / "batch_001"
        / "weights.npz"
    )

    with np.load(path, allow_pickle=False) as data:
        weights = {
            name: data[key].copy()
            for name, key in WEIGHT_KEYS.items()
        }

    return weights


def load_calibrated_thresholds(condition):

    path = (
        condition_output_dir(condition)
        / "calibration"
        / "calibrated_thresholds.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:

        if int(data["completed_batches"]) != 100:
            raise ValueError(
                "Calibration is not complete."
            )

        if not np.isclose(
            float(data["target_E_Hz"]),
            condition.target_E_Hz
        ):
            raise ValueError("E target mismatch.")

        if not np.isclose(
            float(data["target_I_Hz"]),
            condition.target_I_Hz
        ):
            raise ValueError("I target mismatch.")

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

    if theta_E.shape != (N_E, 1):
        raise ValueError("Incorrect E threshold shape.")

    if theta_I.shape != (N_I, 1):
        raise ValueError("Incorrect I threshold shape.")

    return theta_E, theta_I


# ============================================================
# 3. LEARNING RATES
# ============================================================

def condition_learning_rates(condition):

    rates = BASE_LEARNING_RATES.copy()

    rates["LGN_E"] *= (
        condition.lgn_to_e_lr_multiplier
    )

    rates["E_E"] *= (
        condition.e_to_e_lr_multiplier
    )

    # Other four pathways retain baseline rates.
    # This is an explicit Figure 10 assumption.

    return rates


# ============================================================
# 4. CHECKPOINT MANAGEMENT
# ============================================================

def checkpoint_path(output_dir, batch):

    return (
        output_dir
        / "checkpoints"
        / f"batch_{batch:04d}.npz"
    )


def save_checkpoint(
    output_dir,
    batch,
    weights,
    theta_E,
    theta_I,
    history_E,
    history_I,
    condition,
    batch_size,
    seed
):
    path = checkpoint_path(output_dir, batch)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.get_state()

    payload = {
        WEIGHT_KEYS[name]: weights[name]
        for name in WEIGHT_ORDER
    }

    payload.update({
        "threshold_E": theta_E,
        "threshold_I": theta_I,
        "completed_batches": batch,
        "target_E_Hz": condition.target_E_Hz,
        "target_I_Hz": condition.target_I_Hz,
        "triplet_depression":
            condition.triplet_depression,
        "parameterization":
            condition.parameterization,
        "batch_size": batch_size,
        "seed": seed,
        "history_E_Hz": np.asarray(history_E),
        "history_I_Hz": np.asarray(history_I),

        # Complete legacy NumPy RandomState.
        "rng_name": rng[0],
        "rng_keys": rng[1],
        "rng_pos": rng[2],
        "rng_has_gauss": rng[3],
        "rng_cached_gaussian": rng[4]
    })

    # Save to a temporary file first, then replace
    # the destination atomically.
    temporary = path.with_suffix(".tmp")

    with open(temporary, "wb") as file:
        np.savez_compressed(file, **payload)

    os.replace(temporary, path)

    return path


def load_latest_checkpoint(
    output_dir, condition, batch_size, seed
):

    folder = output_dir / "checkpoints"

    files = sorted(folder.glob("batch_*.npz"))

    if not files:
        raise FileNotFoundError(
            "No training checkpoint available."
        )

    path = files[-1]

    with np.load(path, allow_pickle=False) as data:

        completed = int(data["completed_batches"])

        if not np.isclose(
            float(data["target_E_Hz"]),
            condition.target_E_Hz
        ):
            raise ValueError("Checkpoint E mismatch.")

        if not np.isclose(
            float(data["target_I_Hz"]),
            condition.target_I_Hz
        ):
            raise ValueError("Checkpoint I mismatch.")

        if str(data["parameterization"]) != (
            condition.parameterization
        ):
            raise ValueError(
                "Checkpoint parameterization mismatch."
            )

        if int(data["batch_size"]) != batch_size:
            raise ValueError(
                "Checkpoint batch size mismatch."
            )

        if int(data["seed"]) != seed:
            raise ValueError(
                "Checkpoint seed mismatch."
            )

        if not np.isclose(
            float(data["triplet_depression"]),
            condition.triplet_depression
        ):
            raise ValueError(
                "Checkpoint STDP mismatch."
            )

        weights = {
            name: data[key].copy()
            for name, key in WEIGHT_KEYS.items()
        }

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

        history_E = data["history_E_Hz"].tolist()
        history_I = data["history_I_Hz"].tolist()

        rng_state = (
            str(data["rng_name"]),
            data["rng_keys"].copy(),
            int(data["rng_pos"]),
            int(data["rng_has_gauss"]),
            float(data["rng_cached_gaussian"])
        )

    np.random.set_state(rng_state)

    return (
        completed,
        weights,
        theta_E,
        theta_I,
        history_E,
        history_I
    )


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_parameters(weights, theta_E, theta_I):

    for name in WEIGHT_ORDER:

        W = weights[name]

        if not np.isfinite(W).all():
            raise ValueError(
                f"Nonfinite weights: {name}"
            )

        if np.any(W < 0):
            raise ValueError(
                f"Negative weights: {name}"
            )

        if np.any(
            W.sum(axis=1) > L1_LIMITS[name] + 1e-8
        ):
            raise ValueError(
                f"L1 limit exceeded: {name}"
            )

    if not np.isfinite(theta_E).all():
        raise ValueError("Invalid E thresholds.")

    if not np.isfinite(theta_I).all():
        raise ValueError("Invalid I thresholds.")


# ============================================================
# 6. MAIN TRAINING
# ============================================================

def main():

    args = parse_args()

    if args.batches < 1 or args.batches > 1100:
        raise ValueError(
            "Batches must be between 1 and 1100."
        )

    if args.batch_size < 1:
        raise ValueError("Invalid batch size.")

    condition = make_condition(
        args.target_e,
        parameterization=args.parameterization
    )

    output_dir = (
    Path(args.output_dir).resolve()
    if args.output_dir is not None
    else condition_output_dir(condition) / "training"
    )

    checkpoint_folder = (
        output_dir / "checkpoints"
    )

    images = np.load(
        Path(SOURCE_PROJECT)
        / "data"
        / "van_hateren_dog.npy",
        mmap_mode="r"
    )

    base_rates = condition_learning_rates(condition)

    if args.resume:

        (
            completed,
            weights,
            theta_E,
            theta_I,
            history_E,
            history_I
        ) = load_latest_checkpoint(
            output_dir,
            condition,
            args.batch_size,
            args.seed
        )

        start_batch = completed + 1

        print("Resuming after batch:", completed)

    else:

        if (
            checkpoint_folder.exists()
            and any(checkpoint_folder.glob("batch_*.npz"))
        ):
            raise FileExistsError(
                "Existing checkpoints found. "
                "Use --resume to continue."
            )

        weights = load_initial_weights()

        theta_E, theta_I = (
            load_calibrated_thresholds(condition)
        )

        np.random.seed(args.seed)

        history_E = []
        history_I = []

        start_batch = 1

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"

    if not metadata_path.exists():

        metadata = {
            "experiment": "figure10_synaptic_training",
            "condition": condition.condition_name,
            "parameterization":
                condition.parameterization,
            "target_E_Hz": condition.target_E_Hz,
            "target_I_Hz": condition.target_I_Hz,
            "calibration_batches": 100,
            "planned_learning_batches": 1100,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "triplet_depression":
                condition.triplet_depression,
            "learning_rate_multipliers": {
                "LGN_E":
                    condition.lgn_to_e_lr_multiplier,
                "E_E":
                    condition.e_to_e_lr_multiplier,
                "LGN_I": 1.0,
                "E_I": 1.0,
                "I_E": 1.0,
                "I_I": 1.0
            },
            "learning_rate_assumption":
                "Only LGN_E and E_E are target-scaled; "
                "remaining pathways retain baseline rates."
        }

        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)

    print("\nFIGURE 10 SYNAPTIC TRAINING")
    print("=" * 60)
    print("Condition:", condition.condition_name)
    print("Parameterization:", args.parameterization)
    print("Starting batch:", start_batch)
    print("Target completed batches:", args.batches)
    print("Presentations per batch:", args.batch_size)
    print("Initial E threshold:", theta_E.mean())
    print("Initial I threshold:", theta_I.mean())

    if start_batch > args.batches:
        print("Requested batches already completed.")
        return

    for batch in range(start_batch, args.batches + 1):

        # ----------------------------------------------------
        # A. Learning-rate schedule
        # ----------------------------------------------------

        overall_batch = CALIBRATION_BATCHES + batch

        multiplier = get_learning_multiplier(
            overall_batch
        )

        learning_rates = {
            name: value * multiplier
            for name, value in base_rates.items()
        }

        eta_E = condition.eta_threshold_E * multiplier
        eta_I = condition.eta_threshold_I * multiplier

        # ----------------------------------------------------
        # B. Generate LGN spike trains
        # ----------------------------------------------------

        LGN_spikes = generate_lgn_batch(
            images,
            batch_size=args.batch_size,
            n_steps=N_STEPS,
            dt=DT
        )

        # ----------------------------------------------------
        # C. Simulate V1
        # ----------------------------------------------------

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
            threshold_E=theta_E,
            threshold_I=theta_I
        )

        duration = N_STEPS * DT

        rate_E = (
            E_spikes.sum()
            / (N_E * args.batch_size * duration)
        )

        rate_I = (
            I_spikes.sum()
            / (N_I * args.batch_size * duration)
        )

        # ----------------------------------------------------
        # D. Synaptic learning
        # ----------------------------------------------------

        new_weights, _ = learn_from_spikes_figure10(
            LGN_spikes,
            E_spikes,
            I_spikes,
            weights,
            learning_rates,
            L1_LIMITS,
            A_p=condition.triplet_potentiation,
            A_d=condition.triplet_depression,
            A_i=SYMMETRIC_STDP_COEFFICIENT,
            dt=DT,
            tau_short=TAU_STDP_SHORT,
            tau_long=TAU_STDP_LONG
        )

        # ----------------------------------------------------
        # E. Homeostatic adaptation
        # ----------------------------------------------------

        new_theta_E, new_theta_I = (
            update_figure10_thresholds(
                E_spikes,
                I_spikes,
                theta_E,
                theta_I,
                target_E_Hz=condition.target_E_Hz,
                target_I_Hz=condition.target_I_Hz,
                eta_theta_E=eta_E,
                eta_theta_I=eta_I,
                dt=DT
            )
        )

        # ----------------------------------------------------
        # F. Validate and commit updates
        # ----------------------------------------------------

        validate_parameters(
            new_weights,
            new_theta_E,
            new_theta_I
        )

        weights = new_weights
        theta_E = new_theta_E
        theta_I = new_theta_I

        history_E.append(float(rate_E))
        history_I.append(float(rate_I))

        print(
            f"Batch {batch:4d}/{args.batches} | "
            f"E: {rate_E:7.3f} Hz | "
            f"I: {rate_I:7.3f} Hz | "
            f"theta E: {theta_E.mean():7.3f} | "
            f"theta I: {theta_I.mean():7.3f} | "
            f"LR: {multiplier:.2f}",
            flush=True
        )

        # ----------------------------------------------------
        # G. Save checkpoint
        # ----------------------------------------------------

        if (
            batch % CHECKPOINT_INTERVAL == 0
            or batch == args.batches
        ):

            saved = save_checkpoint(
                output_dir,
                batch,
                weights,
                theta_E,
                theta_I,
                history_E,
                history_I,
                condition,
                args.batch_size,
                args.seed
            )

            print("Checkpoint saved:", saved.name)

    print("\nTRAINING STOPPED AT BATCH:", args.batches)
    print("Final mean E threshold:", theta_E.mean())
    print("Final mean I threshold:", theta_I.mean())

    print(
        "Last 10 mean E rate:",
        np.mean(history_E[-10:])
    )

    print(
        "Last 10 mean I rate:",
        np.mean(history_I[-10:])
    )

    print("Results:", output_dir)


if __name__ == "__main__":
    main()
