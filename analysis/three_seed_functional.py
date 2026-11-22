
"""
Three-seed functional reproducibility audit.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.

Reproduces:
    audit_three_seed_functional.py

Purpose
-------
Evaluate three independently trained V1 networks
using identical natural-image-driven LGN spikes
and matched membrane-noise random states.

Original protocol
-----------------
Networks:
    Reference - training_test_25
    Seed 3001 - training_full_seed_3001
    Seed 3002 - training_full_seed_3002

Checkpoint: batch 1100
Stimuli: 100
Evaluation seed: 9012
Simulation duration: N_STEPS * DT

Metrics
-------
1. Mean excitatory firing rate
2. Mean inhibitory firing rate
3. Population Treves-Rolls sparseness
4. Lifetime Treves-Rolls sparseness
5. Fraction of active neuron-stimulus responses
6. Mean signed pairwise response correlation
7. Mean absolute pairwise response correlation
8. Number of valid correlation neurons

The original equations, RNG sequence and
saved output field names are preserved.

Original project files are read-only.

Outputs
-------
results/audit/three_seed/
    three_seed_functional.npz
    three_seed_functional.png
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import (
    DT,
    N_STEPS,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


# ============================================================
# 1. PATHS AND ORIGINAL CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "audit"
    / "three_seed"
)

N_STIMULI = 100
EVAL_SEED = 9012

EXPECTED_LEARNING_BATCH = 1100

DURATION = N_STEPS * DT

NETWORK_RUNS = {
    "Reference": "training_test_25",
    "Seed 3001": "training_full_seed_3001",
    "Seed 3002": "training_full_seed_3002",
}

WEIGHT_ORDER = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E",
)

METRICS = (
    "E_mean",
    "I_mean",
    "population_sparsity",
    "lifetime_sparsity",
    "active_fraction",
    "mean_corr",
    "mean_abs_corr",
    "valid_corr_neurons",
)

PLOT_METRICS = (
    ("population_sparsity", "Population sparseness"),
    ("lifetime_sparsity", "Lifetime sparseness"),
    ("mean_corr", "Mean response correlation"),
    ("active_fraction", "Active neuron fraction"),
)


# ============================================================
# 2. LOAD TRAINED NETWORKS
# ============================================================

def load_trained_networks(source_project):
    """
    Load the three original trained V1 networks.

    The original order is preserved:
        Reference, Seed 3001, Seed 3002.

    Each checkpoint must correspond to batch 1100.

    Returns
    -------
    networks : dict
        Each entry contains:
            weights : tuple of six arrays
            theta_E : ndarray
            theta_I : ndarray
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    networks = {}

    for name, run_dir in NETWORK_RUNS.items():

        path = (
            source_project
            / "results"
            / "learning"
            / run_dir
            / "batch_1100.npz"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(
            path,
            allow_pickle=False
        ) as data:

            learning_batch = int(
                data["learning_batch"]
            )

            if learning_batch != EXPECTED_LEARNING_BATCH:
                raise ValueError(
                    f"{name}: incomplete run "
                    f"({learning_batch})"
                )

            weights = tuple(
                data[f"W_{key}"].copy()
                for key in WEIGHT_ORDER
            )

            theta_E = data["threshold_E"].copy()
            theta_I = data["threshold_I"].copy()

        if len(weights) != 6:
            raise ValueError(
                f"{name}: expected six weight matrices."
            )

        # Preserve original threshold shapes.
        if theta_E.shape != (400, 1):
            raise ValueError(
                f"{name}: E threshold shape "
                f"{theta_E.shape}"
            )

        if theta_I.shape != (100, 1):
            raise ValueError(
                f"{name}: I threshold shape "
                f"{theta_I.shape}"
            )

        networks[name] = {
            "weights": weights,
            "theta_E": theta_E,
            "theta_I": theta_I,
        }

        print(f"Loaded {name}")
        print("Checkpoint:", path)
        print("Learning batch:", learning_batch)

    return networks


# ============================================================
# 3. GENERATE MATCHED NATURAL-IMAGE LGN INPUTS
# ============================================================

def generate_matched_lgn_inputs(source_project):
    """
    Generate one fixed 100-presentation LGN batch.

    The original image data are memory-mapped,
    as in the archived script.

    The original RNG seed is set immediately before
    calling generate_lgn_batch().

    The NumPy random state is saved immediately
    after LGN generation. Each network simulation
    later restores this same state.

    Returns
    -------
    LGN_spikes : ndarray
        Shape (512, N_STIMULI, N_STEPS).

    noise_state : tuple
        NumPy legacy RNG state after LGN generation.
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

    # Do not switch to default_rng().
    # The original experiment uses np.random.seed().
    np.random.seed(EVAL_SEED)

    LGN_spikes = generate_lgn_batch(
        images,
        N_STIMULI,
        N_STEPS,
        DT
    )

    # Preserve precise RNG sequence.
    noise_state = np.random.get_state()

    expected_shape = (
        512,
        N_STIMULI,
        N_STEPS
    )

    if LGN_spikes.shape != expected_shape:
        raise ValueError(
            f"LGN shape {LGN_spikes.shape}; "
            f"expected {expected_shape}"
        )

    print("\nMATCHED LGN INPUT")
    print("Dataset:", image_path)
    print("LGN shape:", LGN_spikes.shape)
    print("Evaluation seed:", EVAL_SEED)

    print(
        "Mean LGN spikes per presentation:",
        LGN_spikes.sum(axis=(0, 2)).mean()
    )

    return LGN_spikes, noise_state


# ============================================================
# 4. ORIGINAL TREVES-ROLLS SPARSENESS
# ============================================================

def treves_rolls_sparseness(x, axis):
    """
    Calculate normalized Treves-Rolls sparseness.

    Preserves the original script's formula
    and silent-response handling.

    Parameters
    ----------
    x : ndarray
        Neuronal firing-rate matrix.

    axis : int
        axis=0: population sparseness
        axis=1: lifetime sparseness

    Returns
    -------
    result : ndarray
        Sparseness scores, with NaN for
        entirely silent response vectors.
    """

    x = np.asarray(x, dtype=float)

    n = x.shape[axis]

    mean = np.mean(x, axis=axis)

    mean_square = np.mean(
        x**2,
        axis=axis
    )

    ratio = np.divide(
        mean**2,
        mean_square,
        out=np.ones_like(mean),
        where=mean_square > 0
    )

    result = (
        (1 - ratio)
        / (1 - 1 / n)
    )

    # The original script uses NaN for
    # completely silent response vectors.
    result = np.where(
        mean_square > 0,
        result,
        np.nan
    )

    return result


# ============================================================
# 5. ORIGINAL PAIRWISE RESPONSE CORRELATION
# ============================================================

def pairwise_response_correlation(responses):
    """
    Calculate pairwise Pearson correlations.

    Rows represent neurons.
    Columns represent stimulus presentations.

    Neurons with zero response variance are excluded.

    Returns
    -------
    dict
        mean : signed mean correlation
        mean_abs : mean absolute correlation
        valid_neurons : number of nonconstant neurons
    """

    centered = (
        responses
        - responses.mean(
            axis=1,
            keepdims=True
        )
    )

    norm = np.linalg.norm(
        centered,
        axis=1
    )

    valid = norm > 0

    normalized = (
        centered[valid]
        / norm[valid, None]
    )

    corr = normalized @ normalized.T

    upper = corr[
        np.triu_indices(
            corr.shape[0],
            k=1
        )
    ]

    return {
        "mean": float(np.mean(upper)),
        "mean_abs": float(
            np.mean(np.abs(upper))
        ),
        "valid_neurons": int(
            np.sum(valid)
        ),
    }


# ============================================================
# 6. SIMULATE ONE TRAINED NETWORK
# ============================================================

def evaluate_network(
    LGN_spikes,
    network,
    noise_state
):
    """
    Simulate one frozen trained V1 network.

    Each network receives:
        the same LGN spike batch
        the same initial membrane-noise RNG state

    The original simulator call is preserved.
    """

    # Matched simulation noise across networks.
    np.random.set_state(noise_state)

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        network["weights"],
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=network["theta_E"],
        threshold_I=network["theta_I"]
    )

    # Shape: neurons x stimulus presentations.
    E_counts = E_spikes.sum(axis=2)
    I_counts = I_spikes.sum(axis=2)

    E_rates = E_counts / DURATION
    I_rates = I_counts / DURATION

    population_sparsity = treves_rolls_sparseness(
        E_rates,
        axis=0
    )

    lifetime_sparsity = treves_rolls_sparseness(
        E_rates,
        axis=1
    )

    corr = pairwise_response_correlation(
        E_rates
    )

    result = {
        "E_rates": E_rates,
        "I_rates": I_rates,

        "E_mean": float(
            E_rates.mean()
        ),

        "I_mean": float(
            I_rates.mean()
        ),

        "population_sparsity": float(
            np.nanmean(population_sparsity)
        ),

        "lifetime_sparsity": float(
            np.nanmean(lifetime_sparsity)
        ),

        "active_fraction": float(
            np.mean(E_counts > 0)
        ),

        "mean_corr": corr["mean"],

        "mean_abs_corr": corr["mean_abs"],

        "valid_corr_neurons": corr[
            "valid_neurons"
        ],
    }

    return result


# ============================================================
# 7. PRINT ORIGINAL METRICS
# ============================================================

def print_three_seed_results(results):
    """
    Print per-network metrics and cross-network
    mean/sample-standard-deviation summaries.
    """

    print("\nMATCHED THREE-SEED FUNCTIONAL EVALUATION")

    for name, result in results.items():

        print(f"\n{'=' * 45}")
        print(name)
        print(f"{'=' * 45}")

        for metric in METRICS:
            print(
                f"{metric}: {result[metric]}"
            )

    print("\nTHREE-SEED SUMMARY")

    # Original script excludes valid_corr_neurons
    # from the mean/SD summary.
    for metric in METRICS[:-1]:

        values = np.array([
            results[name][metric]
            for name in NETWORK_RUNS
        ])

        print(
            f"{metric}: "
            f"mean={values.mean():.6f}, "
            f"SD={values.std(ddof=1):.6f}"
        )


# ============================================================
# 8. ORIGINAL COMPARISON FIGURE
# ============================================================

def plot_three_seed_results(
    results,
    output_dir,
    show=False
):
    """
    Reproduce the original four-panel comparison.
    """

    names = list(NETWORK_RUNS)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8)
    )

    for ax, (metric, title) in zip(
        axes.flat,
        PLOT_METRICS
    ):

        values = [
            results[name][metric]
            for name in names
        ]

        ax.bar(
            names,
            values
        )

        ax.set_title(title)

        ax.tick_params(
            axis="x",
            rotation=15
        )

    fig.suptitle(
        "Functional reproducibility across three seeds"
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / "three_seed_functional.png"
    )

    fig.savefig(
        output_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)

    return output_path


# ============================================================
# 9. SAVE ORIGINAL 32-FIELD ARCHIVE
# ============================================================

def save_three_seed_results(
    results,
    output_dir
):
    """
    Save the same 32 numerical fields as the
    original three-seed functional audit.

    Original prefixes:
        reference
        seed_3001
        seed_3002
    """

    save_data = {
        "eval_seed": EVAL_SEED,
        "n_stimuli": N_STIMULI,
    }

    for name, result in results.items():

        prefix = (
            name.lower()
            .replace(" ", "_")
        )

        for metric, value in result.items():

            save_data[
                f"{prefix}_{metric}"
            ] = value

    output_path = (
        output_dir
        / "three_seed_functional.npz"
    )

    np.savez_compressed(
        output_path,
        **save_data
    )

    return output_path


# ============================================================
# 10. COMPLETE THREE-SEED AUDIT
# ============================================================

def run_three_seed_functional(
    source_project,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Reproduce the original three-seed functional audit.

    Original order:
        load three networks
        generate one LGN input batch
        save post-LGN RNG state
        simulate each network with matched noise
        calculate metrics
        print metrics
        plot
        save numerical archive
    """

    output_dir = Path(
        output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # A. Load the original checkpoints.
    networks = load_trained_networks(
        source_project
    )

    # B. Generate matched LGN inputs.
    LGN_spikes, noise_state = (
        generate_matched_lgn_inputs(
            source_project
        )
    )

    # C. Evaluate the three networks.
    results = {}

    for name, network in networks.items():

        print(f"\nSimulating {name}...")

        results[name] = evaluate_network(
            LGN_spikes,
            network,
            noise_state
        )

    # D. Print original evaluation statistics.
    print_three_seed_results(
        results
    )

    # E. Save original comparison figure.
    figure_path = plot_three_seed_results(
        results,
        output_dir,
        show=show
    )

    # F. Save original numerical archive.
    output_path = save_three_seed_results(
        results,
        output_dir
    )

    print("\nTHREE-SEED FUNCTIONAL AUDIT COMPLETE")
    print("Numerical results:", output_path)
    print("Figure:", figure_path)

    return results


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce matched three-seed "
            "functional V1 evaluation."
        )
    )

    parser.add_argument(
        "--source-project",
        type=str,
        required=True,
        help="Path to the original V1 project"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the comparison figure"
    )

    args = parser.parse_args()

    run_three_seed_functional(
        source_project=args.source_project,
        output_dir=args.output_dir,
        show=args.show
    )
