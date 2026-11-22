
"""
First-batch synaptic plasticity and threshold-homeostasis audit.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.

Reproduces the original:
    evaluate_plasticity_contributions.py

The network-evaluation helper reproduces:
    evaluate_first_learning_batch.py

Experiment
----------
Four conditions:

A_baseline:
    Initial weights + calibrated thresholds

B_STDP_only:
    First-batch learned weights + calibrated thresholds

C_homeostasis_only:
    Initial weights + first-batch updated thresholds

D_both:
    First-batch learned weights + updated thresholds

All conditions receive:
    Identical LGN spike trains (seed 123)
    Matched simulation noise (seed 456)

The original simulation call, firing-rate equations,
condition ordering, and 16 saved result fields
are preserved.

Original project inputs are read-only.

Output
------
results/learning/first_batch/
    plasticity_contributions.npz

The original condition names are maintained, although
"B_STDP_only" represents substitution of all saved
first-batch weight changes, not necessarily STDP alone.
"""

from pathlib import Path
import argparse

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
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


# ============================================================
# 1. PATHS AND EXPERIMENT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "learning"
    / "first_batch"
)

LGN_SEED = 123
NOISE_SEED = 456

DURATION = N_STEPS * DT

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)

WEIGHT_SHAPES = (
    (N_E, 512),
    (N_I, 512),
    (N_E, N_E),
    (N_I, N_E),
    (N_I, N_I),
    (N_E, N_I),
)

CONDITION_NAMES = (
    "A_baseline",
    "B_STDP_only",
    "C_homeostasis_only",
    "D_both",
)

RESULT_NAMES = (
    "rate_E",
    "rate_I",
    "neuron_rates_E",
    "neuron_rates_I",
)


# ============================================================
# 2. LOAD THE SIX WEIGHT MATRICES
# ============================================================

def load_weights(path):
    """
    Load six weight matrices in simulator order.

    Reproduces load_weights() from the original
    evaluate_first_learning_batch.py.

    Returns
    -------
    tuple of ndarray
        Six weight matrices in original order.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

    for name, matrix, expected in zip(
        WEIGHT_NAMES,
        weights,
        WEIGHT_SHAPES
    ):

        if matrix.shape != expected:
            raise ValueError(
                f"{name}: expected {expected}, "
                f"received {matrix.shape}"
            )

    return weights


# ============================================================
# 3. LOAD THE FOUR PARAMETER STATES
# ============================================================

def load_network_states(source_project):
    """
    Load initial and first-batch learned network states.

    Inputs
    ------
    Initial weights:
        results/untrained/batch_001/weights.npz

    First-batch weights:
        results/learning/first_batch/
            learned_weights.npz

    Calibrated thresholds:
        results/calibration/full_100_batches/
            calibrated_thresholds.npz

    Updated thresholds:
        results/learning/first_batch/
            learned_thresholds.npz

    Threshold shapes are preserved as:
        E: (400, 1)
        I: (100, 1)
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    initial_weights_path = (
        source_project
        / "results"
        / "untrained"
        / "batch_001"
        / "weights.npz"
    )

    learning_dir = (
        source_project
        / "results"
        / "learning"
        / "first_batch"
    )

    learned_weights_path = (
        learning_dir
        / "learned_weights.npz"
    )

    calibrated_thresholds_path = (
        source_project
        / "results"
        / "calibration"
        / "full_100_batches"
        / "calibrated_thresholds.npz"
    )

    learned_thresholds_path = (
        learning_dir
        / "learned_thresholds.npz"
    )

    initial_weights = load_weights(
        initial_weights_path
    )

    learned_weights = load_weights(
        learned_weights_path
    )

    with np.load(
        calibrated_thresholds_path,
        allow_pickle=False
    ) as data:

        calibrated_E = data[
            "threshold_E"
        ].copy()

        calibrated_I = data[
            "threshold_I"
        ].copy()

    with np.load(
        learned_thresholds_path,
        allow_pickle=False
    ) as data:

        updated_E = data[
            "threshold_E"
        ].copy()

        updated_I = data[
            "threshold_I"
        ].copy()

    for name, array, expected in (
        ("calibrated_E", calibrated_E, (N_E, 1)),
        ("calibrated_I", calibrated_I, (N_I, 1)),
        ("updated_E", updated_E, (N_E, 1)),
        ("updated_I", updated_I, (N_I, 1)),
    ):

        if array.shape != expected:
            raise ValueError(
                f"{name}: expected {expected}, "
                f"received {array.shape}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name}: non-finite thresholds"
            )

    print("Loaded initial weights:", initial_weights_path)
    print("Loaded learned weights:", learned_weights_path)

    print(
        "Initial thresholds:",
        calibrated_E.shape,
        calibrated_I.shape
    )

    print(
        "Updated thresholds:",
        updated_E.shape,
        updated_I.shape
    )

    return {
        "initial_weights": initial_weights,
        "learned_weights": learned_weights,
        "calibrated_E": calibrated_E,
        "calibrated_I": calibrated_I,
        "updated_E": updated_E,
        "updated_I": updated_I,
    }


# ============================================================
# 4. GENERATE SHARED LGN INPUT
# ============================================================

def generate_shared_lgn_inputs(source_project):
    """
    Reproduce the original shared LGN evaluation input.

    The original script:
        loads van_hateren_dog.npy with mmap_mode='r'
        calls np.random.seed(123)
        calls generate_lgn_batch(images, BATCH_SIZE,
                                 N_STEPS, DT)

    No new preprocessing or random-number generator
    is introduced.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    image_path = (
        source_project
        / "data"
        / "van_hateren_dog.npy"
    )

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    images = np.load(
        image_path,
        mmap_mode="r"
    )

    # Preserve legacy RNG and original seed.
    np.random.seed(LGN_SEED)

    LGN_spikes = generate_lgn_batch(
        images,
        BATCH_SIZE,
        N_STEPS,
        DT
    )

    expected_shape = (
        512,
        BATCH_SIZE,
        N_STEPS
    )

    if LGN_spikes.shape != expected_shape:
        raise ValueError(
            f"Expected LGN shape {expected_shape}, "
            f"received {LGN_spikes.shape}"
        )

    print("\nEvaluation LGN shape:", LGN_spikes.shape)

    return LGN_spikes


# ============================================================
# 5. CALCULATE FIRING RATES FROM SPIKE TRAINS
# ============================================================

def calculate_firing_rates(E_spikes, I_spikes):
    """
    Reproduce the original firing-rate calculations.

    E_spikes:
        shape (N_E, BATCH_SIZE, N_STEPS)

    I_spikes:
        shape (N_I, BATCH_SIZE, N_STEPS)

    Returns
    -------
    dict
        rate_E : scalar Hz
        rate_I : scalar Hz
        neuron_rates_E : (N_E,) Hz
        neuron_rates_I : (N_I,) Hz

    The original summation axes and denominators
    are preserved exactly.
    """

    expected_E = (
        N_E,
        BATCH_SIZE,
        N_STEPS
    )

    expected_I = (
        N_I,
        BATCH_SIZE,
        N_STEPS
    )

    if E_spikes.shape != expected_E:
        raise ValueError(
            f"Unexpected E spikes: {E_spikes.shape}"
        )

    if I_spikes.shape != expected_I:
        raise ValueError(
            f"Unexpected I spikes: {I_spikes.shape}"
        )

    duration = N_STEPS * DT

    # Original population-averaged rates.
    rate_E = E_spikes.sum() / (
        N_E * BATCH_SIZE * duration
    )

    rate_I = I_spikes.sum() / (
        N_I * BATCH_SIZE * duration
    )

    # Original per-neuron mean firing rates.
    neuron_rates_E = E_spikes.sum(
        axis=(1, 2)
    ) / (
        BATCH_SIZE * duration
    )

    neuron_rates_I = I_spikes.sum(
        axis=(1, 2)
    ) / (
        BATCH_SIZE * duration
    )

    return {
        "rate_E": rate_E,
        "rate_I": rate_I,
        "neuron_rates_E": neuron_rates_E,
        "neuron_rates_I": neuron_rates_I,
    }


# ============================================================
# 6. REPRODUCE ORIGINAL NETWORK EVALUATION
# ============================================================

def evaluate_network(
    LGN_spikes,
    weights,
    threshold_E,
    threshold_I,
    noise_seed=NOISE_SEED
):
    """
    Evaluate one frozen V1 network.

    Reproduces evaluate_network() from the original
    evaluate_first_learning_batch.py.

    The legacy RNG is reset before every simulation.
    This preserves matched random-number draws.

    No learning or weight updates are performed.
    """

    np.random.seed(noise_seed)

    # Preserve the original positional parameter order.
    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        weights,
        DT,
        TAU_M,
        NOISE_STD,
        V_MIN,
        V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    return calculate_firing_rates(
        E_spikes,
        I_spikes
    )


# ============================================================
# 7. BUILD ORIGINAL FOUR CONDITIONS
# ============================================================

def create_conditions(states):
    """
    Construct the original four parameter combinations.

    A: Initial weights, calibrated thresholds
    B: Learned weights, calibrated thresholds
    C: Initial weights, updated thresholds
    D: Learned weights, updated thresholds

    Dictionary insertion order is preserved.
    """

    conditions = {
        "A_baseline": (
            states["initial_weights"],
            states["calibrated_E"],
            states["calibrated_I"]
        ),

        "B_STDP_only": (
            states["learned_weights"],
            states["calibrated_E"],
            states["calibrated_I"]
        ),

        "C_homeostasis_only": (
            states["initial_weights"],
            states["updated_E"],
            states["updated_I"]
        ),

        "D_both": (
            states["learned_weights"],
            states["updated_E"],
            states["updated_I"]
        ),
    }

    return conditions


# ============================================================
# 8. CALCULATE CONTRIBUTIONS AND INTERACTION
# ============================================================

def calculate_contributions(results):
    """
    Calculate original effects relative to baseline.

    For each E/I population:
        Weight effect:     B - A
        Threshold effect:  C - A
        Combined effect:   D - A
        Interaction:       D - B - C + A

    Returns
    -------
    dict
        E and I effect summaries.

    These effects are printed but not added to the
    original 16-field output archive.
    """

    A = results["A_baseline"]
    B = results["B_STDP_only"]
    C = results["C_homeostasis_only"]
    D = results["D_both"]

    effects = {}

    print("\nEFFECTS RELATIVE TO BASELINE")

    for pop in ("E", "I"):

        key = f"rate_{pop}"

        weight_effect = (
            B[key] - A[key]
        )

        threshold_effect = (
            C[key] - A[key]
        )

        combined_effect = (
            D[key] - A[key]
        )

        interaction = (
            D[key] - B[key] - C[key] + A[key]
        )

        effects[pop] = {
            "weight_effect": weight_effect,
            "threshold_effect": threshold_effect,
            "combined_effect": combined_effect,
            "interaction": interaction,
        }

        print(f"\n{pop} population:")

        print(
            "STDP only:",
            weight_effect,
            "Hz"
        )

        print(
            "Homeostasis only:",
            threshold_effect,
            "Hz"
        )

        print(
            "Both mechanisms:",
            combined_effect,
            "Hz"
        )

        print(
            "Interaction:",
            interaction,
            "Hz"
        )

    return effects


# ============================================================
# 9. SAVE ORIGINAL 16-FIELD ARCHIVE
# ============================================================

def save_plasticity_contributions(
    results,
    output_dir
):
    """
    Save the exact original 16 output fields.

    Four conditions multiplied by:
        rate_E
        rate_I
        neuron_rates_E
        neuron_rates_I

    No additional fields are inserted.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_data = {}

    for name, result in results.items():

        for metric in RESULT_NAMES:

            save_data[
                f"{name}_{metric}"
            ] = result[metric]

    output_path = (
        output_dir
        / "plasticity_contributions.npz"
    )

    # Preserve original uncompressed np.savez.
    np.savez(
        output_path,
        **save_data
    )

    print(
        "\nFour-condition evaluation completed!"
    )

    print("Saved:", output_path)

    return output_path


# ============================================================
# 10. COMPLETE EXPERIMENT
# ============================================================

def run_plasticity_contributions(
    source_project,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Execute the original four-condition experiment.

    Execution order:
        1. Load natural-image dataset and parameters.
        2. Generate shared LGN spike trains.
        3. Construct original four conditions.
        4. Evaluate each with noise seed 456.
        5. Calculate E/I firing-rate effects.
        6. Save 16 original result fields.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    states = load_network_states(
        source_project
    )

    LGN_spikes = generate_shared_lgn_inputs(
        source_project
    )

    conditions = create_conditions(
        states
    )

    results = {}

    for name, (
        weights,
        theta_E,
        theta_I
    ) in conditions.items():

        print("\nEvaluating:", name)

        results[name] = evaluate_network(
            LGN_spikes,
            weights,
            theta_E,
            theta_I,
            noise_seed=NOISE_SEED
        )

        print(
            "E rate:",
            results[name]["rate_E"],
            "Hz"
        )

        print(
            "I rate:",
            results[name]["rate_I"],
            "Hz"
        )

    effects = calculate_contributions(
        results
    )

    output_path = save_plasticity_contributions(
        results,
        output_dir
    )

    return results, effects, output_path


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce first-batch plasticity "
            "and homeostasis contribution audit."
        )
    )

    parser.add_argument(
        "--source-project",
        type=str,
        required=True,
        help="Original V1_network_learning project"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    args = parser.parse_args()

    run_plasticity_contributions(
        source_project=args.source_project,
        output_dir=args.output_dir
    )
