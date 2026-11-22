
"""
Train the biologically grounded LGN-V1 spiking network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Training procedure
------------------
1. Load initial synaptic weights and calibrated thresholds.
2. Generate a batch of natural-image-driven LGN spike trains.
3. Simulate excitatory and inhibitory V1 responses.
4. Calculate STDP updates for six pathways.
5. Apply homeostatic threshold adaptation.
6. Record training diagnostics.
7. Save resumable checkpoints.

The 100 calibration batches are completed separately.

This script implements up to 1100 subsequent learning batches,
matching the released MATLAB demonstration schedule.

Weight convention
-----------------
W[postsynaptic_neuron, presynaptic_neuron]

Checkpoint files also preserve the NumPy legacy RNG state,
allowing deterministic continuation when the same software
environment and execution sequence are used.
"""

from pathlib import Path
import argparse
import csv

import numpy as np

from config import (
    N_E,
    N_I,
    BATCH_SIZE,
    N_STEPS,
    DT,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
    TARGET_RATE_E,
    TARGET_RATE_I,
    ETA_THRESHOLD_E,
    ETA_THRESHOLD_I,
    TAU_STDP_SHORT,
    TAU_STDP_LONG,
    ETA_LGN_TO_E,
    ETA_LGN_TO_I,
    ETA_E_TO_E,
    ETA_E_TO_I,
    ETA_I_TO_E,
    ETA_I_TO_I,
    L1_LGN_TO_E,
    L1_LGN_TO_I,
    L1_E_TO_E,
    L1_E_TO_I,
    L1_I_TO_E,
    L1_I_TO_I,
    CALIBRATION_BATCHES,
    DEMO_MAIN_LEARNING_BATCHES,
    DEMO_RATE_HALVING_INDICES,
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch
from learning_batch import learn_from_spikes


# ============================================================
# 1. TRAINING CONFIGURATION
# ============================================================

TOTAL_LEARNING_BATCHES = DEMO_MAIN_LEARNING_BATCHES

TAU_SHORT = TAU_STDP_SHORT
TAU_LONG = TAU_STDP_LONG

BASE_LEARNING_RATES = {
    "LGN_E": ETA_LGN_TO_E,
    "E_E": ETA_E_TO_E,
    "LGN_I": ETA_LGN_TO_I,
    "E_I": ETA_E_TO_I,
    "I_E": ETA_I_TO_E,
    "I_I": ETA_I_TO_I,
}

L1_LIMITS = {
    "LGN_E": L1_LGN_TO_E,
    "E_E": L1_E_TO_E,
    "LGN_I": L1_LGN_TO_I,
    "E_I": L1_E_TO_I,
    "I_E": L1_I_TO_E,
    "I_I": L1_I_TO_I,
}

WEIGHT_ORDER = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E",
)


# ============================================================
# 2. VALIDATE SYNAPTIC WEIGHTS
# ============================================================

def validate_weights(weights):
    """
    Check that all synaptic matrices contain valid weights.

    Requirements:
        - All values are finite.
        - All weights are nonnegative.
        - Incoming row sums do not exceed L1 limits.
        - Recurrent self-connections remain zero.
    """
    for name, W in weights.items():

        assert np.isfinite(W).all(), (
            f"Non-finite weights in {name}"
        )

        assert np.all(W >= 0), (
            f"Negative weights in {name}"
        )

        row_sums = W.sum(axis=1)

        assert np.all(
            row_sums <= L1_LIMITS[name] + 1e-8
        ), f"L1 limit exceeded in {name}"

        if name in ("E_E", "I_I"):
            assert np.allclose(
                np.diag(W), 0
            ), f"Self-connections found in {name}"


# ============================================================
# 3. LOAD INITIAL NETWORK
# ============================================================

def load_initial_network(source_project):
    """
    Load initial connectivity and calibrated thresholds.

    Parameters
    ----------
    source_project : pathlib.Path
        Project directory containing the archived
        initialization and calibration results.

    Returns
    -------
    weights : dict
        Six synaptic weight matrices.
    threshold_E, threshold_I : numpy.ndarray
        Calibrated neuronal thresholds.
    """
    weights_path = (
        source_project
        / "results"
        / "untrained"
        / "batch_001"
        / "weights.npz"
    )

    with np.load(weights_path) as data:
        weights = {
            name: data[f"W_{name}"].copy()
            for name in WEIGHT_ORDER
        }

    threshold_path = (
        source_project
        / "results"
        / "calibration"
        / "full_100_batches"
        / "calibrated_thresholds.npz"
    )

    with np.load(threshold_path) as data:
        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

    return weights, threshold_E, threshold_I


# ============================================================
# 4. LEARNING-RATE SCHEDULE
# ============================================================

def get_learning_multiplier(overall_batch):
    """
    Return learning-rate multiplier for a MATLAB batch index.

    Overall batches:
        101-400  : 1.00
        401-800  : 0.50
        801-1200 : 0.25
    """
    first_halving, second_halving = (
        DEMO_RATE_HALVING_INDICES
    )

    if overall_batch > second_halving:
        return 0.25

    if overall_batch > first_halving:
        return 0.5

    return 1.0


# ============================================================
# 5. HOMEOSTATIC THRESHOLD UPDATES
# ============================================================

def update_homeostatic_thresholds(
    E_spikes,
    I_spikes,
    threshold_E,
    threshold_I,
    eta_theta_E,
    eta_theta_I
):
    """
    Apply homeostatic threshold adaptation.

    Target firing rates are converted to expected
    spike counts per stimulus presentation.

    Observed spike counts are averaged across the batch.
    """
    duration = N_STEPS * DT

    target_E = TARGET_RATE_E * duration
    target_I = TARGET_RATE_I * duration

    E_counts = E_spikes.sum(axis=2).mean(
        axis=1,
        keepdims=True
    )

    I_counts = I_spikes.sum(axis=2).mean(
        axis=1,
        keepdims=True
    )

    delta_E = np.minimum(
        eta_theta_E * (E_counts - target_E),
        eta_theta_E * target_E
    )

    delta_I = np.minimum(
        eta_theta_I * (I_counts - target_I),
        eta_theta_I * target_I
    )

    new_threshold_E = threshold_E + delta_E
    new_threshold_I = threshold_I + delta_I

    return new_threshold_E, new_threshold_I


# ============================================================
# 6. SAVE TRAINING CHECKPOINT
# ============================================================

def save_checkpoint(
    results_dir,
    learning_batch,
    weights,
    threshold_E,
    threshold_I
):
    """
    Save model parameters and the NumPy RNG state.

    The saved RNG state allows training to resume
    with the same random sequence.
    """
    checkpoint_path = (
        results_dir
        / f"batch_{learning_batch:04d}.npz"
    )

    save_data = {
        f"W_{name}": W
        for name, W in weights.items()
    }

    save_data["threshold_E"] = threshold_E
    save_data["threshold_I"] = threshold_I
    save_data["learning_batch"] = learning_batch

    rng_state = np.random.get_state()

    save_data["rng_algorithm"] = rng_state[0]
    save_data["rng_keys"] = rng_state[1]
    save_data["rng_position"] = rng_state[2]
    save_data["rng_has_gauss"] = rng_state[3]
    save_data["rng_cached_gaussian"] = rng_state[4]

    np.savez_compressed(
        checkpoint_path,
        **save_data
    )

    print(
        "Checkpoint saved:",
        checkpoint_path.name
    )


# ============================================================
# 7. LOAD TRAINING CHECKPOINT
# ============================================================

def load_checkpoint(checkpoint_path):
    """
    Restore weights, thresholds, batch index and RNG state.
    """
    with np.load(checkpoint_path) as data:

        weights = {
            name: data[f"W_{name}"].copy()
            for name in WEIGHT_ORDER
        }

        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

        learning_batch = int(
            data["learning_batch"]
        )

        rng_state = (
            str(data["rng_algorithm"].item()),
            data["rng_keys"].copy(),
            int(data["rng_position"]),
            int(data["rng_has_gauss"]),
            float(data["rng_cached_gaussian"])
        )

    np.random.set_state(rng_state)

    validate_weights(weights)

    assert np.isfinite(threshold_E).all()
    assert np.isfinite(threshold_I).all()

    print("Checkpoint loaded:", checkpoint_path)
    print("Completed learning batch:", learning_batch)
    print("Mean E threshold:", threshold_E.mean())
    print("Mean I threshold:", threshold_I.mean())

    return (
        weights,
        threshold_E,
        threshold_I,
        learning_batch
    )


# ============================================================
# 8. MAIN TRAINING PROCEDURE
# ============================================================

def main(
    n_batches=3,
    checkpoint_every=25,
    resume=None,
    seed=2026,
    source_project=None
):
    """
    Run or resume V1 synaptic learning.

    Parameters
    ----------
    n_batches : int
        Final learning batch to execute (1-1100).
    checkpoint_every : int
        Interval for checkpoint saving.
    resume : str or None
        Path to a previously saved checkpoint.
    seed : int
        Random seed for a new learning trajectory.
    source_project : str or None
        Directory containing the original data,
        initial weights and calibration results.

        Defaults to the directory containing this script.

    Notes
    -----
    For resumed training, the checkpoint's RNG state
    takes precedence over the seed argument.

    Training output is always written to this repository's
    results/learning directory for new runs, or to the
    checkpoint directory for resumed runs.
    """

    if not 1 <= n_batches <= TOTAL_LEARNING_BATCHES:
        raise ValueError(
            "n_batches must be between 1 and 1100."
        )

    if checkpoint_every < 1:
        raise ValueError(
            "checkpoint_every must be at least 1."
        )

    project_dir = Path(__file__).resolve().parent

    if source_project is None:
        source_project = project_dir
    else:
        source_project = Path(
            source_project
        ).expanduser().resolve()

    # --------------------------------------------------------
    # A. START NEW RUN OR RESUME CHECKPOINT
    # --------------------------------------------------------

    if resume is None:

        if n_batches == TOTAL_LEARNING_BATCHES:
            run_name = f"training_full_seed_{seed}"
        else:
            run_name = (
                f"training_test_{n_batches}_seed_{seed}"
            )

        results_dir = (
            project_dir
            / "results"
            / "learning"
            / run_name
        )

        results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Prevent accidentally replacing a previous run.
        if list(results_dir.iterdir()):
            raise FileExistsError(
                "Output folder is not empty. "
                "Use a new seed, move the existing folder, "
                "or resume from an existing checkpoint: "
                f"{results_dir}"
            )

        np.random.seed(seed)

        weights, threshold_E, threshold_I = (
            load_initial_network(source_project)
        )

        start_batch = 1
        history = []

    else:

        checkpoint_path = Path(
            resume
        ).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        results_dir = checkpoint_path.parent

        (
            weights,
            threshold_E,
            threshold_I,
            completed_batch
        ) = load_checkpoint(checkpoint_path)

        start_batch = completed_batch + 1

        history_path = (
            results_dir / "training_history.csv"
        )

        if not history_path.is_file():
            raise FileNotFoundError(
                f"Missing training history: {history_path}"
            )

        with open(
            history_path,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            history = list(reader)

        history = [
            record
            for record in history
            if int(record["learning_batch"]) <= completed_batch
        ]

        if len(history) != completed_batch:
            raise ValueError(
                "Training history does not match checkpoint."
            )

        print(
            "Resuming from learning batch:",
            completed_batch
        )

    validate_weights(weights)

    duration = N_STEPS * DT

    # Load image data after RNG restoration.
    images = np.load(
        source_project / "data" / "van_hateren_dog.npy",
        mmap_mode="r"
    )

    print("Images:", images.shape)
    print("Training through batch:", n_batches)
    print("Starting at batch:", start_batch)
    print("Initial E threshold:", threshold_E.mean())
    print("Initial I threshold:", threshold_I.mean())

    if start_batch > n_batches:
        print("Requested batches already completed.")
        return

    # --------------------------------------------------------
    # B. TRAINING LOOP
    # --------------------------------------------------------

    for learning_batch in range(
        start_batch,
        n_batches + 1
    ):

        overall_batch = (
            CALIBRATION_BATCHES + learning_batch
        )

        multiplier = get_learning_multiplier(
            overall_batch
        )

        learning_rates = {
            name: rate * multiplier
            for name, rate in BASE_LEARNING_RATES.items()
        }

        eta_theta_E = (
            ETA_THRESHOLD_E * multiplier
        )

        eta_theta_I = (
            ETA_THRESHOLD_I * multiplier
        )

        # ----------------------------------------------------
        # 1. Generate new LGN spike trains
        # ----------------------------------------------------

        LGN_spikes = generate_lgn_batch(
            images,
            BATCH_SIZE,
            N_STEPS,
            DT
        )

        # ----------------------------------------------------
        # 2. Convert weight dictionary into simulation tuple
        # ----------------------------------------------------

        weights_tuple = tuple(
            weights[name]
            for name in WEIGHT_ORDER
        )

        # ----------------------------------------------------
        # 3. Simulate V1 neuronal activity
        # ----------------------------------------------------

        E_spikes, I_spikes = simulate_v1_batch(
            LGN_spikes,
            weights_tuple,
            DT,
            TAU_M,
            NOISE_STD,
            V_MIN,
            V_RESET,
            threshold_E=threshold_E,
            threshold_I=threshold_I
        )

        # ----------------------------------------------------
        # 4. Calculate observed firing rates
        # ----------------------------------------------------

        rate_E = E_spikes.sum() / (
            N_E * BATCH_SIZE * duration
        )

        rate_I = I_spikes.sum() / (
            N_I * BATCH_SIZE * duration
        )

        # ----------------------------------------------------
        # 5. Apply STDP and weight normalization
        # ----------------------------------------------------

        new_weights, scaled_dW = learn_from_spikes(
            LGN_spikes,
            E_spikes,
            I_spikes,
            weights,
            learning_rates,
            L1_LIMITS,
            dt=DT,
            tau_short=TAU_SHORT,
            tau_long=TAU_LONG
        )

        # ----------------------------------------------------
        # 6. Apply homeostatic threshold adaptation
        # ----------------------------------------------------

        (
            new_threshold_E,
            new_threshold_I
        ) = update_homeostatic_thresholds(
            E_spikes,
            I_spikes,
            threshold_E,
            threshold_I,
            eta_theta_E,
            eta_theta_I
        )

        # ----------------------------------------------------
        # 7. Validate updated parameters
        # ----------------------------------------------------

        validate_weights(new_weights)

        assert np.isfinite(
            new_threshold_E
        ).all()

        assert np.isfinite(
            new_threshold_I
        ).all()

        # ----------------------------------------------------
        # 8. Measure changes to weights
        # ----------------------------------------------------

        mean_weight_changes = {
            name: np.abs(
                new_weights[name] - weights[name]
            ).mean()
            for name in WEIGHT_ORDER
        }

        # ----------------------------------------------------
        # 9. Commit new parameters
        # ----------------------------------------------------

        weights = new_weights
        threshold_E = new_threshold_E
        threshold_I = new_threshold_I

        # ----------------------------------------------------
        # 10. Record training diagnostics
        # ----------------------------------------------------

        record = {
            "learning_batch": learning_batch,
            "overall_batch": overall_batch,
            "rate_E": float(rate_E),
            "rate_I": float(rate_I),
            "threshold_E_mean": float(
                threshold_E.mean()
            ),
            "threshold_I_mean": float(
                threshold_I.mean()
            ),
            "lr_multiplier": multiplier
        }

        for name in WEIGHT_ORDER:
            record[f"mean_dW_{name}"] = float(
                mean_weight_changes[name]
            )

        history.append(record)

        print(
            f"Batch {learning_batch:4d} | "
            f"E: {rate_E:.3f} Hz | "
            f"I: {rate_I:.3f} Hz | "
            f"LR: {multiplier:.2f}"
        )

        # ----------------------------------------------------
        # 11. Save checkpoint and training history
        # ----------------------------------------------------

        if (
            learning_batch % checkpoint_every == 0
            or learning_batch == n_batches
        ):

            save_checkpoint(
                results_dir,
                learning_batch,
                weights,
                threshold_E,
                threshold_I
            )

            history_path = (
                results_dir / "training_history.csv"
            )

            with open(
                history_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=history[0].keys()
                )

                writer.writeheader()
                writer.writerows(history)

    print("\nTraining completed!")
    print("Results saved to:", results_dir)


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or resume the LGN-V1 network."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    parser.add_argument(
        "--batches",
        type=int,
        default=3
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    parser.add_argument(
        "--source-project",
        type=str,
        default=None,
        help=(
            "Path to archived project containing "
            "natural-image data and initial parameters."
        )
    )

    args = parser.parse_args()

    main(
        n_batches=args.batches,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        seed=args.seed,
        source_project=args.source_project
    )
