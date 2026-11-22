
"""
FILE 29: FIRST LEARNING BATCH EVALUATION

Reproduces:
    evaluate_first_learning_batch.py

Compares the V1 network before and after its
first learning batch.

Original network:
    Initial weights
    Calibrated thresholds

Learned network:
    First-batch learned weights
    First-batch updated thresholds

Evaluation:
    LGN seed = 123
    Membrane-noise seed = 456

Reuses the numerically validated functions
from File 28 without changing their equations.

Original archive:
    results/learning/first_batch/
        first_batch_evaluation.npz

Saved fields:
    original_rates_E
    original_rates_I
    learned_rates_E
    learned_rates_I

Original project data are read-only.
"""

from pathlib import Path
import argparse

import numpy as np

from analysis.plasticity_contributions import (
    load_network_states,
    generate_shared_lgn_inputs,
    evaluate_network,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "learning"
    / "first_batch"
)

ARCHIVE_NAME = "first_batch_evaluation.npz"

ARCHIVE_FIELDS = (
    "original_rates_E",
    "original_rates_I",
    "learned_rates_E",
    "learned_rates_I",
)


# ============================================================
# 2. COMPARE INDIVIDUAL NEURON ACTIVITY
# ============================================================

def compare_neuron_rates(original, learned):
    """
    Reproduce the original individual-neuron
    firing-rate comparison.

    For each E/I population, calculate:

    - Mean firing rate before learning
    - Mean firing rate after learning
    - Mean absolute firing-rate change
    - Maximum absolute firing-rate change
    - Number of neurons with changed spike counts

    No numerical approximations or thresholds are
    introduced in the change-count calculation.
    """

    statistics = {}

    for population in ("E", "I"):

        key = f"neuron_rates_{population}"

        before = original[key]
        after = learned[key]

        difference = after - before

        statistics[population] = {
            "mean_before": before.mean(),
            "mean_after": after.mean(),
            "mean_absolute_change": (
                np.abs(difference).mean()
            ),
            "maximum_absolute_change": (
                np.abs(difference).max()
            ),
            "number_changed": (
                np.count_nonzero(difference)
            ),
        }

        print(f"\n{population} NEURON STATISTICS")

        print(
            "Mean rate before:",
            statistics[population]["mean_before"]
        )

        print(
            "Mean rate after:",
            statistics[population]["mean_after"]
        )

        print(
            "Mean absolute rate change:",
            statistics[population]["mean_absolute_change"]
        )

        print(
            "Maximum absolute rate change:",
            statistics[population]["maximum_absolute_change"]
        )

        print(
            "Neurons with changed spike counts:",
            statistics[population]["number_changed"]
        )

    return statistics


# ============================================================
# 3. SAVE ORIGINAL FOUR-FIELD ARCHIVE
# ============================================================

def save_first_batch_evaluation(
    original,
    learned,
    output_dir
):
    """
    Save exactly the four original neuron-rate arrays.

    Preserves original field names and uncompressed
    np.savez() behavior.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / ARCHIVE_NAME
    )

    np.savez(
        output_path,
        original_rates_E=original["neuron_rates_E"],
        original_rates_I=original["neuron_rates_I"],
        learned_rates_E=learned["neuron_rates_E"],
        learned_rates_I=learned["neuron_rates_I"]
    )

    print(
        "\nFirst learning batch evaluation completed!"
    )

    print("Saved:", output_path)

    return output_path


# ============================================================
# 4. COMPLETE FIRST-BATCH EVALUATION
# ============================================================

def run_first_learning_batch(
    source_project,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Execute the original two-network experiment.

    Steps
    -----
    1. Load initial and first-batch learned states.
    2. Generate identical shared LGN input.
    3. Evaluate original network with noise seed 456.
    4. Evaluate learned network with noise seed 456.
    5. Compare population firing rates.
    6. Compare individual neuron firing rates.
    7. Save the four original arrays.

    Both networks receive the same LGN spikes.
    Each simulation resets the legacy RNG to 456.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    # --------------------------------------------------------
    # Load initial and learned network parameters
    # --------------------------------------------------------

    states = load_network_states(
        source_project
    )

    # --------------------------------------------------------
    # Generate shared evaluation inputs
    # --------------------------------------------------------

    LGN_spikes = generate_shared_lgn_inputs(
        source_project
    )

    # --------------------------------------------------------
    # Original network before learning
    # --------------------------------------------------------

    original = evaluate_network(
        LGN_spikes,
        states["initial_weights"],
        states["calibrated_E"],
        states["calibrated_I"],
        noise_seed=456
    )

    # --------------------------------------------------------
    # Network after first learning batch
    # --------------------------------------------------------

    learned = evaluate_network(
        LGN_spikes,
        states["learned_weights"],
        states["updated_E"],
        states["updated_I"],
        noise_seed=456
    )

    # --------------------------------------------------------
    # Original population firing-rate comparison
    # --------------------------------------------------------

    print("\nPOPULATION FIRING RATES")

    print(
        f"E: {original['rate_E']:.4f} Hz"
        f" -> {learned['rate_E']:.4f} Hz"
    )

    print(
        f"I: {original['rate_I']:.4f} Hz"
        f" -> {learned['rate_I']:.4f} Hz"
    )

    print("\nCHANGE IN FIRING RATE")

    print(
        "E:",
        learned["rate_E"] - original["rate_E"],
        "Hz"
    )

    print(
        "I:",
        learned["rate_I"] - original["rate_I"],
        "Hz"
    )

    # --------------------------------------------------------
    # Original per-neuron statistics
    # --------------------------------------------------------

    statistics = compare_neuron_rates(
        original,
        learned
    )

    # --------------------------------------------------------
    # Save original four fields
    # --------------------------------------------------------

    output_path = save_first_batch_evaluation(
        original,
        learned,
        output_dir
    )

    return original, learned, statistics, output_path


# ============================================================
# 5. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the original first learning "
            "batch network evaluation."
        )
    )

    parser.add_argument(
        "--source-project",
        type=str,
        required=True,
        help="Original V1_network_learning directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    args = parser.parse_args()

    run_first_learning_batch(
        source_project=args.source_project,
        output_dir=args.output_dir
    )
