
"""
Figure 10A-C: Paper-protocol firing-rate sweep training.

Conditions:
    E05_I10
    E10_I20
    E20_I40

Protocol:
    100 calibration batches (already completed separately)
    1100 synaptic training batches

Reuses the existing paper training functions.
No changes to previous training scripts or checkpoints.

Run from project root:
    python -m analysis.figure10_paper_sweep_train
"""

import argparse
import json
from pathlib import Path

import numpy as np

import figure10_config

from analysis import figure10_full_calibration as fc
from analysis import figure10_train as tr


ROOT = Path(__file__).resolve().parents[1]

NETWORK_SEED = 3002
TOTAL_BATCHES = 1100
BATCH_SIZE = 100

WEIGHT_NAMES = (
    "LGN_E", "LGN_I", "E_E",
    "E_I", "I_I", "I_E"
)

BASE = (
    ROOT / "results" / "paper_protocol"
    / f"seed_{NETWORK_SEED}"
)


def calibration_source(condition):
    return (
        BASE / "calibration"
        / condition.condition_name
        / "calibrated_thresholds.npz"
    )


def output_directory(condition):
    return (
        BASE / "figure10_firing_rate_sweep"
        / "paper"
        / condition.condition_name
        / "training"
    )


def load_calibrated_thresholds(path, condition):

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        required = (
            "threshold_E",
            "threshold_I",
            "completed_batches",
            "target_E_Hz",
            "target_I_Hz",
        )

        for key in required:
            if key not in data.files:
                raise KeyError(
                    f"Missing calibration field: {key}"
                )

        if int(data["completed_batches"]) != 100:
            raise ValueError(
                "Calibration must have exactly 100 batches."
            )

        if not np.isclose(
            float(data["target_E_Hz"]),
            condition.target_E_Hz
        ):
            raise ValueError(
                "Incorrect E calibration target."
            )

        if not np.isclose(
            float(data["target_I_Hz"]),
            condition.target_I_Hz
        ):
            raise ValueError(
                "Incorrect I calibration target."
            )

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

    if theta_E.shape != (400, 1):
        raise ValueError("Wrong E threshold shape.")

    if theta_I.shape != (100, 1):
        raise ValueError("Wrong I threshold shape.")

    return theta_E, theta_I


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-e",
        type=int,
        choices=(5, 10, 20),
        required=True
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=303002
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    if not 1 <= args.batches <= TOTAL_BATCHES:
        raise ValueError("Invalid training batch count.")

    if args.checkpoint_every < 1:
        raise ValueError("Invalid checkpoint interval.")

    condition = figure10_config.make_condition(
        args.target_e,
        parameterization="paper"
    )

    source = calibration_source(condition)
    output = output_directory(condition)

    metadata_path = output / "metadata.json"

    expected_metadata = {
        "experiment": "figure10_paper_firing_rate_sweep",
        "network_seed": NETWORK_SEED,
        "threshold_source": str(source.resolve()),
        "parameterization": "paper",
        "condition": condition.condition_name,
        "batch_size": BATCH_SIZE,
        "training_seed": args.seed,
        "learning_schedule":
            "overall_batch_100_plus_training_batch",
        "calibration_batches": 100,
        "triplet_depression":
            condition.triplet_depression,
        "target_E_Hz": condition.target_E_Hz,
        "target_I_Hz": condition.target_I_Hz,
    }

    # --------------------------------------------
    # 1. Initialize or resume the training network
    # --------------------------------------------

    if args.resume:

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        existing = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        for key, value in expected_metadata.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"Training metadata mismatch: {key}"
                )

        (
            completed,
            weights,
            theta_E,
            theta_I,
            history_E,
            history_I
        ) = tr.load_latest_checkpoint(
            output,
            condition,
            BATCH_SIZE,
            args.seed
        )

        start_batch = completed + 1

    else:

        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Training output already exists: {output}"
            )

        # Load the same network initialization
        # used by the baseline paper experiments.

        initial, weights_path = (
            fc.load_initial_weights(
                parameterization="paper",
                network_seed=NETWORK_SEED
            )
        )

        weights = {
            name: W.copy()
            for name, W in zip(
                WEIGHT_NAMES,
                initial
            )
        }

        theta_E, theta_I = (
            load_calibrated_thresholds(
                source,
                condition
            )
        )

        np.random.seed(args.seed)

        history_E = []
        history_I = []

        start_batch = 1

        output.mkdir(
            parents=True,
            exist_ok=False
        )

        metadata_path.write_text(
            json.dumps(
                expected_metadata,
                indent=4
            ),
            encoding="utf-8"
        )

    tr.validate_parameters(
        weights,
        theta_E,
        theta_I
    )

    # --------------------------------------------
    # 2. Paper-specific natural-image dataset
    # --------------------------------------------

    images = fc.PaperImageDataset(
        str(ROOT / "data" / "paper_van_hateren")
    )

    if tuple(images.shape) != (1024, 1536, 1000):
        raise RuntimeError(
            f"Incorrect paper dataset: {images.shape}"
        )

    base_rates = tr.condition_learning_rates(
        condition
    )

    print("=" * 65)
    print("FIGURE 10 — PAPER FIRING-RATE SWEEP TRAINING")
    print("=" * 65)

    print("PROTOCOL: paper_1000_source")
    print("Condition:", condition.condition_name)
    print("Images:", images.shape)
    print("Calibration:", source)
    print("Output:", output)

    print("\nTARGET PARAMETERS")
    print("Target E:", condition.target_E_Hz)
    print("Target I:", condition.target_I_Hz)

    print(
        "Triplet potentiation:",
        condition.triplet_potentiation
    )

    print(
        "Triplet depression:",
        condition.triplet_depression
    )

    print("Base learning rates:", base_rates)

    print("\nTRAINING STATE")
    print("Starting batch:", start_batch)
    print("Target batch:", args.batches)
    print("Batch size:", BATCH_SIZE)

    print(
        "Mean E threshold:",
        float(theta_E.mean())
    )

    print(
        "Mean I threshold:",
        float(theta_I.mean())
    )

    if start_batch > args.batches:
        print("Requested training already complete.")
        return

    # --------------------------------------------
    # 3. Synaptic training
    # --------------------------------------------

    for batch in range(
        start_batch,
        args.batches + 1
    ):

        # Published 100 calibration batches precede
        # synaptic training.

        overall_batch = 100 + batch

        multiplier = tr.get_learning_multiplier(
            overall_batch
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

        # Generate natural-image LGN spike trains.

        LGN = fc.generate_lgn_batch(
            images,
            batch_size=BATCH_SIZE,
            n_steps=fc.N_STEPS,
            dt=fc.DT
        )

        # Simulate the V1 network.

        E, I = fc.simulate_v1_batch(
            LGN,
            tuple(
                weights[name]
                for name in tr.WEIGHT_ORDER
            ),
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

        rate_E = float(
            E.sum() / (
                E.shape[0]
                * BATCH_SIZE
                * duration
            )
        )

        rate_I = float(
            I.sum() / (
                I.shape[0]
                * BATCH_SIZE
                * duration
            )
        )

        # Apply the condition-specific STDP rule.

        new_weights, _ = (
            tr.learn_from_spikes_figure10(
                LGN,
                E,
                I,
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

        # Continue threshold homeostasis during
        # the entire synaptic-training period.

        new_theta_E, new_theta_I = (
            tr.update_figure10_thresholds(
                E,
                I,
                theta_E,
                theta_I,
                target_E_Hz=condition.target_E_Hz,
                target_I_Hz=condition.target_I_Hz,
                eta_theta_E=eta_E,
                eta_theta_I=eta_I,
                dt=fc.DT
            )
        )

        tr.validate_parameters(
            new_weights,
            new_theta_E,
            new_theta_I
        )

        weights = new_weights
        theta_E = new_theta_E
        theta_I = new_theta_I

        history_E.append(rate_E)
        history_I.append(rate_I)

        print(
            f"Batch {batch:4d}/{args.batches} | "
            f"E: {rate_E:7.3f} Hz | "
            f"I: {rate_I:7.3f} Hz | "
            f"theta E: {theta_E.mean():7.3f} | "
            f"theta I: {theta_I.mean():7.3f} | "
            f"LR: {multiplier:.2f}",
            flush=True
        )

        # Save a resumable checkpoint, including
        # the exact NumPy random state.

        if (
            batch % args.checkpoint_every == 0
            or batch == args.batches
        ):

            path = tr.save_checkpoint(
                output,
                batch,
                weights,
                theta_E,
                theta_I,
                history_E,
                history_I,
                condition,
                BATCH_SIZE,
                args.seed
            )

            print(
                "Checkpoint saved:",
                path.name,
                flush=True
            )

    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETED")
    print("=" * 65)

    print("Condition:", condition.condition_name)
    print("Completed batches:", args.batches)

    print(
        "Last 10 mean E:",
        np.mean(history_E[-10:])
    )

    print(
        "Last 10 mean I:",
        np.mean(history_I[-10:])
    )

    print("Results:", output)


if __name__ == "__main__":
    main()
