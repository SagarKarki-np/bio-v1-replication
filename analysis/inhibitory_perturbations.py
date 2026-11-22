
"""
Inhibitory-feedback perturbation experiment.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Purpose
-------
Determine how weakening I-to-E inhibitory feedback
changes activity in a trained recurrent V1 network.

Conditions
----------
Normal : W_I_E multiplied by 1.0
Half   : W_I_E multiplied by 0.5
None   : W_I_E multiplied by 0.0

Original evaluation protocol
----------------------------
Checkpoint: training_test_25/batch_1100.npz
Random seed: 2035
Presentations: 100
Duration: N_STEPS * DT
Noise: NOISE_STD from config.py

All conditions use identical LGN spike trains and
restore the same NumPy random state before simulation.

Measurements
------------
1. Mean excitatory firing rate
2. Mean inhibitory firing rate
3. Excitatory active fraction
4. Treves-Rolls population sparseness
5. Mean signed E response correlation
6. Mean absolute E response correlation

Outputs
-------
inhibition_intervention.npz
inhibition_summary.npz
inhibition_intervention.png

The analysis does not modify the source checkpoint.
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
    N_E,
    N_I,
    N_LGN,
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "inhibitory_perturbations"
)

EVALUATION_SEED = 2035
EVALUATION_BATCH_SIZE = 100

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)

# Maintain the exact condition ordering of the original.
CONDITIONS = {
    "Normal": 1.0,
    "Half": 0.5,
    "None": 0.0,
}


# ============================================================
# 2. LOAD TRAINED NETWORK
# ============================================================

def load_trained_network(checkpoint_path):
    """
    Load six weight matrices and trained thresholds.

    Returns
    -------
    weights : tuple of ndarray
        Same pathway ordering as simulation.py.

    threshold_E : ndarray
    threshold_I : ndarray
    """

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

    expected_shapes = (
        (N_E, N_LGN),
        (N_I, N_LGN),
        (N_E, N_E),
        (N_I, N_E),
        (N_I, N_I),
        (N_E, N_I),
    )

    for name, W, expected in zip(
        WEIGHT_NAMES,
        weights,
        expected_shapes
    ):
        if W.shape != expected:
            raise ValueError(
                f"{name}: expected {expected}, "
                f"got {W.shape}"
            )

        if not np.isfinite(W).all():
            raise ValueError(
                f"{name} contains nonfinite values."
            )

    if threshold_E.shape != (N_E, 1):
        raise ValueError(
        f"Expected excitatory threshold shape {(N_E, 1)}, "
        f"got {threshold_E.shape}"
    )


    if threshold_I.shape != (N_I, 1):
         raise ValueError(
        f"Expected inhibitory threshold shape {(N_I, 1)}, "
        f"got {threshold_I.shape}"
    )

    print("Excitatory threshold shape:", threshold_E.shape)
    print("Inhibitory threshold shape:", threshold_I.shape)
    if (
        not np.isfinite(threshold_E).all()
        or not np.isfinite(threshold_I).all()
    ):
        raise ValueError(
            "Thresholds must contain finite values."
        )

    return weights, threshold_E, threshold_I


# ============================================================
# 3. MODIFY I -> E INHIBITORY WEIGHTS
# ============================================================

def scale_inhibitory_feedback(weights, scale):
    """
    Return copies of all six weight matrices,
    multiplying ONLY W_I_E by the supplied scale.

    The original matrices are not modified.
    """

    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            "Inhibitory scaling must be finite "
            "and nonnegative."
        )

    modified_weights = [
        W.copy() for W in weights
    ]

    # Sixth matrix: inhibitory neurons -> E neurons.
    modified_weights[5] *= scale

    return tuple(modified_weights)


# ============================================================
# 4. EVALUATE ONE INHIBITION CONDITION
# ============================================================

def evaluate_inhibition(
    scale,
    weights,
    LGN_spikes,
    threshold_E,
    threshold_I,
    noise_state
):
    """
    Simulate the trained network with modified I->E input.

    The RNG state is restored before each simulation
    to reproduce the original matched-noise protocol.

    Returns
    -------
    E_counts : ndarray
        E spike counts, shape (N_E, n_presentations).

    I_counts : ndarray
        I spike counts, shape (N_I, n_presentations).
    """

    modified_weights = scale_inhibitory_feedback(
        weights,
        scale
    )

    np.random.set_state(noise_state)

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        modified_weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    # Sum along time: neurons x presentations.
    E_counts = E_spikes.sum(axis=2)
    I_counts = I_spikes.sum(axis=2)

    return E_counts, I_counts


# ============================================================
# 5. TREVES-ROLLS POPULATION SPARSENESS
# ============================================================

def population_sparseness(counts):
    """
    Calculate Treves-Rolls population sparseness.

    Parameters
    ----------
    counts : ndarray
        Shape (neurons, presentations).

    Returns
    -------
    ndarray
        One sparseness value per presentation.

    Formula
    -------
    S = [1 - mean(r)^2 / mean(r^2)]
        / [1 - 1/N]

    Silent presentations produce NaN, matching
    the original intervention analysis.
    """

    counts = np.asarray(counts)

    if counts.ndim != 2:
        raise ValueError(
            "Counts must have shape "
            "(neurons, presentations)."
        )

    n_neurons = counts.shape[0]

    if n_neurons < 2:
        raise ValueError(
            "At least two neurons are required."
        )

    # Preserve the original calculation order.
    mean_r = counts.mean(axis=0)
    mean_r2 = (counts ** 2).mean(axis=0)

    valid = mean_r2 > 0

    sparseness = np.full(
        counts.shape[1],
        np.nan
    )

    sparseness[valid] = (
        1 - mean_r[valid] ** 2 / mean_r2[valid]
    ) / (1 - 1 / n_neurons)

    return sparseness


# ============================================================
# 6. NEURONAL RESPONSE CORRELATIONS
# ============================================================

def response_correlation(counts):
    """
    Calculate excitatory response correlations.

    Each row represents a neuron.
    Each column represents an image presentation.

    Neurons with zero response variance are excluded,
    matching the original intervention script.

    Returns
    -------
    mean_corr : float
    mean_abs_corr : float
    n_pairs : int
    n_valid_neurons : int
    """

    counts = np.asarray(counts)

    if counts.ndim != 2:
        raise ValueError(
            "Counts must be a 2D array."
        )

    # Original validity rule.
    valid = counts.std(axis=1) > 0

    X = counts[valid]

    n_valid_neurons = X.shape[0]

    if n_valid_neurons < 2:
        return np.nan, np.nan, 0, n_valid_neurons

    # Retain the original NumPy Pearson calculation.
    C = np.corrcoef(X)

    pairs = C[
        np.triu_indices(len(X), k=1)
    ]

    return (
        pairs.mean(),
        np.abs(pairs).mean(),
        len(pairs),
        n_valid_neurons
    )


# ============================================================
# 7. SUMMARIZE ONE CONDITION
# ============================================================

def summarize_condition(
    name,
    E_counts,
    I_counts,
    duration
):
    """
    Calculate original intervention summary metrics.

    Additional diagnostics:
        Number of valid correlation neurons
        Number of silent presentations
        Number of valid correlation pairs
    """

    if duration <= 0:
        raise ValueError(
            "Presentation duration must be positive."
        )

    if (
        E_counts.ndim != 2
        or I_counts.ndim != 2
        or E_counts.shape[1] != I_counts.shape[1]
    ):
        raise ValueError(
            "Incompatible spike-count shapes."
        )

    E_rate = E_counts.mean() / duration
    I_rate = I_counts.mean() / duration

    active_fraction = np.mean(E_counts > 0)

    sparseness = population_sparseness(E_counts)

    (
        mean_corr,
        abs_corr,
        n_pairs,
        n_valid_neurons
    ) = response_correlation(E_counts)

    n_silent = int(np.isnan(sparseness).sum())

    mean_sparseness = (
        float(np.nanmean(sparseness))
        if n_silent < sparseness.size
        else np.nan
    )

    summary = {
        "E_rate": float(E_rate),
        "I_rate": float(I_rate),
        "active_fraction": float(active_fraction),
        "sparseness": mean_sparseness,
        "mean_corr": float(mean_corr),
        "abs_corr": float(abs_corr),
        "n_pairs": int(n_pairs),
        "n_valid_neurons": int(n_valid_neurons),
        "n_silent_presentations": n_silent,
    }

    print(f"\n{name.upper()}")

    print("E firing rate:", summary["E_rate"])
    print("I firing rate:", summary["I_rate"])
    print("Active fraction:", summary["active_fraction"])

    print(
        "Population sparseness:",
        summary["sparseness"]
    )

    print(
        "Silent image presentations:",
        summary["n_silent_presentations"]
    )

    print(
        "Mean response correlation:",
        summary["mean_corr"]
    )

    print(
        "Mean absolute correlation:",
        summary["abs_corr"]
    )

    print(
        "Valid correlation neurons:",
        summary["n_valid_neurons"]
    )

    print(
        "Valid correlation pairs:",
        summary["n_pairs"]
    )

    return summary


# ============================================================
# 8. PLOT INTERVENTION SUMMARY
# ============================================================

def plot_intervention_summary(
    summary,
    save_path,
    show=False
):
    """
    Reproduce the original four-panel bar plot.
    """

    labels = list(summary.keys())

    metrics = [
        ("E_rate", "E firing rate (Hz)"),
        ("active_fraction", "Active E fraction"),
        ("sparseness", "Population sparseness"),
        ("abs_corr", "Mean absolute correlation")
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        constrained_layout=True
    )

    for ax, (key, title) in zip(
        axes.flat,
        metrics
    ):
        values = [
            summary[name][key]
            for name in labels
        ]

        ax.bar(labels, values)

        ax.set_title(title)
        ax.set_xlabel("Inhibition condition")
        ax.grid(axis="y", alpha=0.3)

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. SAVE SUMMARY STATISTICS
# ============================================================

def save_summary(summary, output_path):
    """
    Save all condition metrics as NumPy arrays.

    String labels and numeric arrays avoid pickle-based
    serialization and are easy to inspect later.
    """

    labels = list(summary.keys())

    metric_names = (
        "E_rate",
        "I_rate",
        "active_fraction",
        "sparseness",
        "mean_corr",
        "abs_corr",
        "n_pairs",
        "n_valid_neurons",
        "n_silent_presentations",
    )

    arrays = {
        "conditions": np.array(labels),
        "scales": np.array([
            CONDITIONS[name] for name in labels
        ]),
    }

    for metric in metric_names:
        arrays[metric] = np.array([
            summary[name][metric]
            for name in labels
        ])

    np.savez_compressed(
        output_path,
        **arrays
    )


# ============================================================
# 10. COMPLETE INTERVENTION EXPERIMENT
# ============================================================

def analyze_inhibitory_perturbations(
    source_project,
    learned_run="training_test_25",
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Evaluate normal, half, and zero I->E feedback.

    Read-only inputs come from the original project.
    Outputs are written to the new repository.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Locate archived inputs
    # --------------------------------------------------------

    checkpoint_path = (
        source_project
        / "results"
        / "learning"
        / learned_run
        / "batch_1100.npz"
    )

    image_path = (
        source_project
        / "data"
        / "van_hateren_dog.npy"
    )

    # --------------------------------------------------------
    # B. Load trained weights and thresholds
    # --------------------------------------------------------

    (
        weights,
        threshold_E,
        threshold_I
    ) = load_trained_network(
        checkpoint_path
    )

    print("Loaded trained network.")
    print("Checkpoint:", checkpoint_path)
    print("I -> E matrix:", weights[5].shape)

    # --------------------------------------------------------
    # C. Generate identical LGN input
    # --------------------------------------------------------

    images = np.load(image_path)

    np.random.seed(EVALUATION_SEED)

    LGN_spikes = generate_lgn_batch(
        images,
        EVALUATION_BATCH_SIZE,
        N_STEPS,
        DT
    )

    print("\nLGN shape:", LGN_spikes.shape)

    expected_shape = (
        N_LGN,
        EVALUATION_BATCH_SIZE,
        N_STEPS
    )

    if LGN_spikes.shape != expected_shape:
        raise ValueError(
            f"Expected LGN shape {expected_shape}, "
            f"got {LGN_spikes.shape}"
        )

    # Preserve the random state immediately after
    # LGN spike generation, exactly as in the original.
    noise_state = np.random.get_state()

    # --------------------------------------------------------
    # D. Simulate each intervention condition
    # --------------------------------------------------------

    results = {}

    for name, scale in CONDITIONS.items():

        print(
            f"\nEvaluating {name} inhibition..."
        )

        E_counts, I_counts = evaluate_inhibition(
            scale=scale,
            weights=weights,
            LGN_spikes=LGN_spikes,
            threshold_E=threshold_E,
            threshold_I=threshold_I,
            noise_state=noise_state
        )

        results[name] = {
            "E": E_counts,
            "I": I_counts
        }

    # --------------------------------------------------------
    # E. Calculate statistics
    # --------------------------------------------------------

    duration = N_STEPS * DT

    summary = {}

    print("\nINHIBITION INTERVENTION RESULTS")

    for name, data in results.items():

        summary[name] = summarize_condition(
            name=name,
            E_counts=data["E"],
            I_counts=data["I"],
            duration=duration
        )

    # --------------------------------------------------------
    # F. Save outputs
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Keep all original response-array key names.
    np.savez_compressed(
        output_dir / "inhibition_intervention.npz",

        normal_E=results["Normal"]["E"],
        half_E=results["Half"]["E"],
        none_E=results["None"]["E"],

        normal_I=results["Normal"]["I"],
        half_I=results["Half"]["I"],
        none_I=results["None"]["I"],

        duration=duration
    )

    save_summary(
        summary,
        output_dir / "inhibition_summary.npz"
    )

    plot_intervention_summary(
        summary,
        output_dir / "inhibition_intervention.png",
        show=show
    )

    print(
        "\nInhibition intervention completed!"
    )

    print(
        "Results saved to:", output_dir
    )

    return results, summary


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the effects of reducing or removing "
            "I-to-E inhibitory feedback in trained V1."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to the original V1 project"
    )

    parser.add_argument(
        "--learned-run",
        type=str,
        default="training_test_25",
        help="Original learning run directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for outputs in the new project"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the summary figure"
    )

    args = parser.parse_args()

    analyze_inhibitory_perturbations(
        source_project=args.source_project,
        learned_run=args.learned_run,
        output_dir=args.output_dir,
        show=args.show
    )
