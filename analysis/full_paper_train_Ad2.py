
"""
SINGLE-FACTOR EXPERIMENT: identical to full_paper_train.py except
A_d = 2.0 (1/tau-normalised traces) and a separate output folder.

Full-paper LGN-V1 synaptic training.

Track A: original batch-100 thresholds.
Track B: validated batch-680 thresholds.

Both tracks share:
- Seed-3002 initial weights
- Paper-mode STDP coefficients
- Identical training learning-rate schedule
- The same training random seed
- Frozen calibration sources

Run from project root using python -m.
"""

import argparse
import dataclasses
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
TARGET_E_HZ = 2.0
A_D_OVERRIDE = 2.0   # MATLAB-literal depression under 1/tau traces

WEIGHT_NAMES = (
    "LGN_E", "LGN_I", "E_E",
    "E_I", "I_I", "I_E"
)


def threshold_source(track):
    base = (
        ROOT / "results" / "paper_protocol"
        / f"seed_{NETWORK_SEED}"
    )

    if track == "A":
        return (
            base / "calibration" / "E02_I04"
            / "calibrated_thresholds.npz"
        )

    return (
        base / "extended_calibration" / "eta_0.10"
        / "calibrated_thresholds.npz"
    )


def output_directory(track):
    return (
        ROOT / "results" / "paper_protocol"
        / f"seed_{NETWORK_SEED}"
        / "synaptic_training"
        / f"track_{track.lower()}_Ad{A_D_OVERRIDE:g}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track", choices=("A", "B"), required=True
    )

    parser.add_argument(
        "--batches", type=int, default=5
    )

    parser.add_argument(
        "--seed", type=int, default=303002
    )

    parser.add_argument(
        "--checkpoint-every", type=int, default=25
    )

    parser.add_argument(
        "--resume", action="store_true"
    )

    args = parser.parse_args()

    if not 1 <= args.batches <= TOTAL_BATCHES:
        raise ValueError("Invalid training batch count")

    if args.checkpoint_every < 1:
        raise ValueError("Invalid checkpoint interval")

    condition = figure10_config.make_condition(
        TARGET_E_HZ,
        parameterization="paper"
    )
    condition = dataclasses.replace(
        condition, triplet_depression=A_D_OVERRIDE
    )

    output = output_directory(args.track)
    metadata_path = output / "metadata.json"

    source = threshold_source(args.track)

    expected_metadata = {
        "experiment": "full_paper_synaptic_training",
        "track": args.track,
        "network_seed": NETWORK_SEED,
        "threshold_source": str(source.resolve()),
        "parameterization": "paper",
        "condition": condition.condition_name,
        "batch_size": BATCH_SIZE,
        "training_seed": args.seed,
        "learning_schedule": "overall_batch_100_plus_training_batch",
        "calibration_batches": (
            100 if args.track == "A" else 680
        ),
        "triplet_depression": (
            condition.triplet_depression
        )
    }

    # --------------------------------------------------
    # 1. Load initial or resumed state.
    # --------------------------------------------------

    if args.resume:

        if not metadata_path.exists():
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

        initial, _ = fc.load_initial_weights(
            parameterization="paper",
            network_seed=NETWORK_SEED
        )

        weights = {
            name: W.copy()
            for name, W in zip(WEIGHT_NAMES, initial)
        }

        if not source.exists():
            raise FileNotFoundError(source)

        with np.load(
            source, allow_pickle=False
        ) as data:
            theta_E = data["threshold_E"].copy()
            theta_I = data["threshold_I"].copy()

        np.random.seed(args.seed)

        history_E = []
        history_I = []

        start_batch = 1

        output.mkdir(parents=True, exist_ok=True)

        metadata_path.write_text(
            json.dumps(expected_metadata, indent=4),
            encoding="utf-8"
        )

    tr.validate_parameters(
        weights, theta_E, theta_I
    )

    # --------------------------------------------------
    # 2. Load paper image dataset.
    # --------------------------------------------------

    images = fc.PaperImageDataset(
        str(ROOT / "data" / "paper_van_hateren")
    )

    base_rates = tr.condition_learning_rates(
        condition
    )

    print("=" * 60)
    print("FULL-PAPER SYNAPTIC TRAINING")
    print("=" * 60)
    print("Track:", args.track)
    print("Starting batch:", start_batch)
    print("Target batch:", args.batches)
    print("Batch size:", BATCH_SIZE)
    print("Mean E threshold:", theta_E.mean())
    print("Mean I threshold:", theta_I.mean())
    print("Output:", output)

    if start_batch > args.batches:
        print("Requested training already complete.")
        return

    # --------------------------------------------------
    # 3. Synaptic learning loop.
    # --------------------------------------------------

    for batch in range(
        start_batch, args.batches + 1
    ):

        # Both tracks use the same learning schedule.
        overall_batch = 100 + batch

        multiplier = tr.get_learning_multiplier(
            overall_batch
        )

        learning_rates = {
            name: rate * multiplier
            for name, rate in base_rates.items()
        }

        eta_E = (
            condition.eta_threshold_E * multiplier
        )

        eta_I = (
            condition.eta_threshold_I * multiplier
        )

        # Generate natural-image LGN spikes.
        LGN = fc.generate_lgn_batch(
            images,
            batch_size=BATCH_SIZE,
            n_steps=fc.N_STEPS,
            dt=fc.DT
        )

        # Simulate V1 with current parameters.
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
                E.shape[0] * BATCH_SIZE * duration
            )
        )

        rate_I = float(
            I.sum() / (
                I.shape[0] * BATCH_SIZE * duration
            )
        )

        # Apply parameterized STDP.
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

        # Continue homeostatic adaptation.
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
            new_weights, new_theta_E, new_theta_I
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

        # Use the existing RNG-preserving checkpoint.
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
                "Checkpoint saved:", path.name,
                flush=True
            )

    print("\nTRAINING RUN COMPLETED")
    print("Track:", args.track)
    print("Completed batches:", args.batches)
    print("Last 10 mean E:", np.mean(history_E[-10:]))
    print("Last 10 mean I:", np.mean(history_I[-10:]))
    print("Results:", output)


if __name__ == "__main__":
    main()
