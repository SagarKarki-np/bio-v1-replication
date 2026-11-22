
"""
Figure 10: Resumable homeostatic calibration.

Ruslim et al. (2025) LGN-V1 replication.

Calibration:
    100 batches
    100 natural-image presentations per batch
    400 ms per presentation
    Fixed synaptic weights

Supported configurations:
    matlab_demo
    paper

Outputs are isolated by configuration, target rates,
and experiment mode.

This script does NOT perform synaptic learning.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import (
    DT, N_STEPS, TAU_M, NOISE_STD,
    V_MIN, V_RESET, N_E, N_I
)

from figure10_config import (
    SOURCE_PROJECT,
    make_condition,
    condition_output_dir,
    CALIBRATION_SEED,
    INITIAL_THRESHOLD_E,
    INITIAL_THRESHOLD_I,
    FULL_CALIBRATION_BATCHES,
    FULL_BATCH_SIZE,
    CALIBRATION_CHECKPOINT_EVERY,
)

from paper_image_dataset import PaperImageDataset
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)


# ============================================================
# 1. HOMEOSTATIC THRESHOLD UPDATE
# ============================================================

def update_thresholds(
    thresholds,
    observed_counts,
    target_counts,
    eta
):
    """
    Original Ruslim MATLAB-style calibration rule.

    Identical mathematical form to the original
    plasticity.update_thresholds implementation.
    """

    delta = eta * (
        observed_counts - target_counts
    )

    delta = np.minimum(
        delta,
        eta * target_counts
    )

    return thresholds + delta


# ============================================================
# 2. SAVE AND RESTORE NUMPY RNG STATE
# ============================================================

def pack_rng_state():

    name, keys, pos, has_gauss, cached = (
        np.random.get_state()
    )

    return {
        "rng_name": np.array(name),
        "rng_keys": keys.copy(),
        "rng_pos": np.int64(pos),
        "rng_has_gauss": np.int64(has_gauss),
        "rng_cached_gauss": np.float64(cached)
    }


def restore_rng_state(data):

    np.random.set_state((
        str(data["rng_name"].item()),
        data["rng_keys"].copy(),
        int(data["rng_pos"]),
        int(data["rng_has_gauss"]),
        float(data["rng_cached_gauss"])
    ))


# ============================================================
# 3. LOAD INITIAL WEIGHTS
# ============================================================

def load_initial_weights(
    parameterization="matlab_demo",
    network_seed=None
):
    """
    Load saved initial weights for demo or paper mode.

    Paper mode verifies the saved network against its
    initialization metadata before returning the matrices.
    """

    if parameterization == "matlab_demo":

        if network_seed is not None:
            raise ValueError(
                "network_seed is not used in demo mode."
            )

        path = (
            SOURCE_PROJECT
            / "results"
            / "untrained"
            / "batch_001"
            / "weights.npz"
        )

    elif parameterization == "paper":

        if network_seed is None:
            raise ValueError(
                "Paper mode requires network_seed."
            )

        folder = (
            PROJECT
            / "results"
            / "paper_protocol"
            / f"seed_{int(network_seed)}"
            / "initialization"
        )

        path = folder / "weights.npz"
        metadata_path = folder / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if int(metadata["seed"]) != int(network_seed):
            raise ValueError("Initialization seed mismatch.")

        import hashlib

        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if digest != metadata["weights_sha256"]:
            raise ValueError(
                "Initial weight SHA256 mismatch."
            )

    else:
        raise ValueError(
            f"Unknown parameterization: {parameterization}"
        )

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

    expected_shapes = (
        (400, 512),
        (100, 512),
        (400, 400),
        (100, 400),
        (100, 100),
        (400, 100)
    )

    for name, W, shape in zip(
        WEIGHT_NAMES, weights, expected_shapes
    ):
        if W.shape != shape:
            raise ValueError(f"Invalid shape: {name}")

        if not np.isfinite(W).all():
            raise ValueError(f"Non-finite weights: {name}")

    return weights, path


# ============================================================
# 4. CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    path,
    completed_batches,
    threshold_E,
    threshold_I,
    history_E,
    history_I,
    history_threshold_E,
    history_threshold_I
):

    state = pack_rng_state()

    np.savez_compressed(
        path,
        completed_batches=completed_batches,
        threshold_E=threshold_E,
        threshold_I=threshold_I,
        observed_E_Hz=np.asarray(history_E),
        observed_I_Hz=np.asarray(history_I),
        mean_threshold_E=np.asarray(history_threshold_E),
        mean_threshold_I=np.asarray(history_threshold_I),
        **state
    )


def load_checkpoint(path):

    with np.load(path, allow_pickle=False) as data:

        completed = int(
            data["completed_batches"]
        )

        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

        history_E = (
            data["observed_E_Hz"].tolist()
        )

        history_I = (
            data["observed_I_Hz"].tolist()
        )

        history_threshold_E = (
            data["mean_threshold_E"].tolist()
        )

        history_threshold_I = (
            data["mean_threshold_I"].tolist()
        )

        restore_rng_state(data)

    return (
        completed,
        threshold_E,
        threshold_I,
        history_E,
        history_I,
        history_threshold_E,
        history_threshold_I
    )


# ============================================================
# 5. FULL CALIBRATION
# ============================================================

def run_calibration(
    target_E_Hz,
    parameterization="matlab_demo",
    resume=False,
    network_seed=None,
    update_rule="count"
):

    condition = make_condition(
        target_E_Hz,
        parameterization
    )

    if update_rule not in ("count", "rate_candidate"):
        raise ValueError("Unknown threshold-update rule.")

    if update_rule == "rate_candidate" and parameterization != "paper":
        raise ValueError(
            "Rate candidate is restricted to paper mode."
        )

    n_batches = FULL_CALIBRATION_BATCHES
    batch_size = FULL_BATCH_SIZE

    if parameterization == "paper":
        if network_seed is None:
            raise ValueError("Paper mode requires network_seed.")

        network_seed = int(network_seed)

        output_dir = (
            PROJECT / "results" / "paper_protocol"
            / f"seed_{network_seed}"
            / "calibration"
            / condition.condition_name
        )

        calibration_seed = network_seed + 100000

    else:
        if network_seed is not None:
            raise ValueError(
                "Demo mode does not accept network_seed."
            )

        output_dir = condition_output_dir(condition) / "calibration"
        calibration_seed = CALIBRATION_SEED

    if update_rule == "rate_candidate":
        output_dir = (
            output_dir.parent
            / f"{condition.condition_name}_rate_candidate"
        )

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    latest_path = (
        checkpoint_dir / "latest.npz"
    )

    metadata_path = (
        output_dir / "metadata.json"
    )

    weights, weights_path = load_initial_weights(
        parameterization=parameterization,
        network_seed=network_seed
    )

    import hashlib

    weights_sha256 = hashlib.sha256(
        weights_path.read_bytes()
    ).hexdigest()

    dataset_sha256 = None

    if parameterization == "paper":
        selection_path = (
            PROJECT / "data" / "paper_van_hateren"
            / "metadata" / "selection_manifest.json"
        )

        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )

        dataset_sha256 = selection["selection_sha256"]

    # --------------------------------------------------------
    # A. Initialize or resume
    # --------------------------------------------------------

    if resume:

        if not latest_path.exists():
            raise FileNotFoundError(
                "No calibration checkpoint found: "
                + str(latest_path)
            )

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        with open(
            metadata_path,
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        expected = {
            "target_E_Hz": condition.target_E_Hz,
            "target_I_Hz": condition.target_I_Hz,
            "parameterization": parameterization,
            "batches": n_batches,
            "batch_size": batch_size,
            "seed": calibration_seed,
            "eta_E": condition.eta_threshold_E,
            "eta_I": condition.eta_threshold_I,
            "weights_path": str(weights_path),
            "network_seed": network_seed,
            "weights_sha256": weights_sha256,
            "dataset_selection_sha256": dataset_sha256,
            "threshold_update_rule": update_rule,
            "dt": DT,
            "n_steps": N_STEPS,
        }

        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Resume metadata mismatch: {key}"
                )

        (
            completed,
            threshold_E,
            threshold_I,
            history_E,
            history_I,
            history_threshold_E,
            history_threshold_I
        ) = load_checkpoint(latest_path)

        start_batch = completed

        print("Resuming after batch:", completed)

    else:

        if output_dir.exists() and any(
            output_dir.iterdir()
        ):
            raise FileExistsError(
                "Output directory is not empty. "
                "Use --resume or preserve/rename "
                "the existing results: "
                + str(output_dir)
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.random.seed(calibration_seed)

        start_batch = 0

        threshold_E = np.full(
            (N_E, 1),
            INITIAL_THRESHOLD_E
        )

        threshold_I = np.full(
            (N_I, 1),
            INITIAL_THRESHOLD_I
        )

        history_E = []
        history_I = []

        history_threshold_E = []
        history_threshold_I = []

        metadata = {
            "experiment": "figure10_full_calibration",
            "parameterization": parameterization,
            "target_E_Hz": condition.target_E_Hz,
            "target_I_Hz": condition.target_I_Hz,
            "batches": n_batches,
            "batch_size": batch_size,
            "seed": calibration_seed,
            "eta_E": condition.eta_threshold_E,
            "eta_I": condition.eta_threshold_I,
            "weights_path": str(weights_path),
            "network_seed": network_seed,
            "weights_sha256": weights_sha256,
            "dataset_selection_sha256": dataset_sha256,
            "threshold_update_rule": update_rule,
            "dt": DT,
            "n_steps": N_STEPS,
            "noise_std": NOISE_STD,
            "initial_threshold_E": INITIAL_THRESHOLD_E,
            "initial_threshold_I": INITIAL_THRESHOLD_I,
            "synaptic_learning": False,
            "triplet_depression_configured":
                condition.triplet_depression,
            "note": (
                "Triplet depression is metadata only. "
                "No STDP occurs during calibration."
            )
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4
            )

    # --------------------------------------------------------
    # B. Load natural-image data
    # --------------------------------------------------------

    if parameterization == "paper":
        images = PaperImageDataset(
            PROJECT / "data" / "paper_van_hateren"
        )
    else:
        images_path = (
            SOURCE_PROJECT / "data" / "van_hateren_dog.npy"
        )

        images = np.load(images_path, mmap_mode="r")

    print("\nFIGURE 10 FULL CALIBRATION")
    print("=" * 60)
    print("Condition:", condition.condition_name)
    print("Parameterization:", parameterization)
    print("Target E:", condition.target_E_Hz, "Hz")
    print("Target I:", condition.target_I_Hz, "Hz")
    print("Images shape:", images.shape)
    print("Batch size:", batch_size)
    print("Total batches:", n_batches)

    duration = N_STEPS * DT

    target_counts_E = (
        condition.target_E_Hz * duration
    )

    target_counts_I = (
        condition.target_I_Hz * duration
    )

    # --------------------------------------------------------
    # C. Calibration loop
    # --------------------------------------------------------

    for batch in range(
        start_batch,
        n_batches
    ):

        LGN_spikes = generate_lgn_batch(
            images,
            batch_size=batch_size,
            n_steps=N_STEPS,
            dt=DT
        )

        E_spikes, I_spikes = simulate_v1_batch(
            LGN_spikes,
            weights,
            dt=DT,
            tau_m=TAU_M,
            noise_std=NOISE_STD,
            v_min=V_MIN,
            v_reset=V_RESET,
            threshold_E=threshold_E,
            threshold_I=threshold_I
        )

        counts_E = (
            E_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        counts_I = (
            I_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        rate_E = float(
            counts_E.mean() / duration
        )

        rate_I = float(
            counts_I.mean() / duration
        )

        if update_rule == "rate_candidate":
            observed_E_for_update = counts_E / duration
            observed_I_for_update = counts_I / duration
            target_E_for_update = condition.target_E_Hz
            target_I_for_update = condition.target_I_Hz
        else:
            observed_E_for_update = counts_E
            observed_I_for_update = counts_I
            target_E_for_update = target_counts_E
            target_I_for_update = target_counts_I

        threshold_E = update_thresholds(
            threshold_E,
            observed_E_for_update,
            target_E_for_update,
            condition.eta_threshold_E
        )

        threshold_I = update_thresholds(
            threshold_I,
            observed_I_for_update,
            target_I_for_update,
            condition.eta_threshold_I
        )

        if not (
            np.isfinite(threshold_E).all()
            and np.isfinite(threshold_I).all()
        ):
            raise RuntimeError(
                f"Nonfinite threshold at batch {batch + 1}"
            )

        history_E.append(rate_E)
        history_I.append(rate_I)

        history_threshold_E.append(
            float(threshold_E.mean())
        )

        history_threshold_I.append(
            float(threshold_I.mean())
        )

        completed = batch + 1

        print(
            f"Batch {completed:3d}/{n_batches} | "
            f"E: {rate_E:7.3f} Hz | "
            f"I: {rate_I:7.3f} Hz | "
            f"theta E: {threshold_E.mean():8.3f} | "
            f"theta I: {threshold_I.mean():8.3f}",
            flush=True
        )

        # Save checkpoints every 10 batches.
        if (
            completed % CALIBRATION_CHECKPOINT_EVERY == 0
            or completed == n_batches
        ):

            numbered_path = (
                checkpoint_dir
                / f"checkpoint_{completed:03d}.npz"
            )

            save_checkpoint(
                numbered_path,
                completed,
                threshold_E,
                threshold_I,
                history_E,
                history_I,
                history_threshold_E,
                history_threshold_I
            )

            save_checkpoint(
                latest_path,
                completed,
                threshold_E,
                threshold_I,
                history_E,
                history_I,
                history_threshold_E,
                history_threshold_I
            )

            print(
                "Checkpoint saved:",
                completed,
                flush=True
            )

    # --------------------------------------------------------
    # D. Final summary
    # --------------------------------------------------------

    last_n = min(
        10,
        len(history_E)
    )

    if last_n == 0:
        raise RuntimeError(
            "No calibration history available."
        )

    mean_final_E = float(
        np.mean(history_E[-last_n:])
    )

    mean_final_I = float(
        np.mean(history_I[-last_n:])
    )

    print("\nCALIBRATION SUMMARY")
    print("=" * 60)

    print(
        "Last 10 batches mean E rate:",
        mean_final_E
    )

    print(
        "Last 10 batches mean I rate:",
        mean_final_I
    )

    print(
        "E target:",
        condition.target_E_Hz
    )

    print(
        "I target:",
        condition.target_I_Hz
    )

    print(
        "Final mean E threshold:",
        threshold_E.mean()
    )

    print(
        "Final mean I threshold:",
        threshold_I.mean()
    )

    print(
        "E target error (Hz):",
        mean_final_E - condition.target_E_Hz
    )

    print(
        "I target error (Hz):",
        mean_final_I - condition.target_I_Hz
    )

    # --------------------------------------------------------
    # E. Save final results
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "calibrated_thresholds.npz",
        threshold_E=threshold_E,
        threshold_I=threshold_I,
        target_E_Hz=condition.target_E_Hz,
        target_I_Hz=condition.target_I_Hz,
        completed_batches=n_batches
    )

    np.savez_compressed(
        output_dir / "calibration_history.npz",
        observed_E_Hz=np.asarray(history_E),
        observed_I_Hz=np.asarray(history_I),
        mean_threshold_E=np.asarray(history_threshold_E),
        mean_threshold_I=np.asarray(history_threshold_I)
    )

    print("\nCalibration completed.")
    print("Results:", output_dir)


# ============================================================
# 6. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-e",
        type=float,
        required=True,
        choices=[1.0, 2.0, 5.0, 10.0, 20.0]
    )

    parser.add_argument(
        "--parameterization",
        choices=["matlab_demo", "paper"],
        default="matlab_demo"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--network-seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--update-rule",
        choices=["count", "rate_candidate"],
        default="count"
    )

    args = parser.parse_args()

    run_calibration(
        target_E_Hz=args.target_e,
        parameterization=args.parameterization,
        resume=args.resume,
        network_seed=args.network_seed,
        update_rule=args.update_rule
    )
