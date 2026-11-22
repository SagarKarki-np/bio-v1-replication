
"""
Pairwise neuronal response-correlation analysis.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This script compares excitatory V1 response correlations
before and after synaptic learning.

Input
-----
matched_responses.npz containing:

    initial_E : (n_neurons, n_presentations)
    learned_E : (n_neurons, n_presentations)

The same neurons and stimuli must be used in both arrays.

Analysis
--------
1. Calculate Pearson correlations between neuron pairs.
2. Exclude undefined correlations from summary statistics.
3. Compare initial and learned correlation distributions.
4. Plot complete neuron-by-neuron correlation matrices.
5. Calculate paired changes for valid neuron pairs.
6. Save numerical results and figures.

Constant-response neurons have undefined Pearson
correlations, represented by NaN.

Outputs
-------
neuronal_correlations.npz
neuronal_correlations.png
neuronal_correlation_matrices.png
neuronal_correlations_paired.png
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_DIR
    / "results"
    / "sparsity"
    / "matched_responses.npz"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "response_correlations"
)


# ============================================================
# 2. LOAD MATCHED RESPONSES
# ============================================================

def load_matched_responses(input_path):
    """
    Load excitatory responses before and after learning.

    Both matrices must have shape:
        (n_neurons, n_presentations)

    Returns
    -------
    initial_E, learned_E : numpy.ndarray
    """

    with np.load(
        input_path,
        allow_pickle=False
    ) as data:

        initial_E = data["initial_E"].copy()
        learned_E = data["learned_E"].copy()

    if initial_E.ndim != 2 or learned_E.ndim != 2:
        raise ValueError(
            "Response arrays must be two-dimensional."
        )

    if initial_E.shape != learned_E.shape:
        raise ValueError(
            "Initial and learned response shapes differ."
        )

    if min(initial_E.shape) < 2:
        raise ValueError(
            "At least two neurons and two presentations "
            "are required."
        )

    if (
        not np.isfinite(initial_E).all()
        or not np.isfinite(learned_E).all()
    ):
        raise ValueError(
            "Response arrays contain non-finite values."
        )

    return initial_E, learned_E


# ============================================================
# 3. CALCULATE NEURON-PAIR CORRELATIONS
# ============================================================

def calculate_pairwise_correlations(R):
    """
    Calculate Pearson correlations between neuron pairs.

    Parameters
    ----------
    R : numpy.ndarray
        Response matrix:
            (n_neurons, n_presentations)

    Returns
    -------
    correlation_matrix : numpy.ndarray
        Square neuron-by-neuron correlation matrix.

    pair_correlations : numpy.ndarray
        Unique pairwise correlations from the upper
        triangle, excluding the diagonal.

    Notes
    -----
    Neurons with constant responses have undefined
    Pearson correlations.

    Those correlations are stored as NaN.

    Pair ordering follows np.triu_indices(n_neurons, k=1),
    matching the original analysis.
    """

    R = np.asarray(R, dtype=float)

    if R.ndim != 2:
        raise ValueError(
            "Response matrix must be two-dimensional."
        )

    n_neurons, n_presentations = R.shape

    if n_neurons < 2 or n_presentations < 2:
        raise ValueError(
            "At least two neurons and two presentations "
            "are required."
        )

    if not np.isfinite(R).all():
        raise ValueError(
            "Response matrix contains non-finite values."
        )

    # Center each neuron's responses.
    centered = R - R.mean(
        axis=1,
        keepdims=True
    )

    # Calculate each neuron's centered response norm.
    norms = np.linalg.norm(
        centered,
        axis=1
    )

    # Constant neurons have no defined correlation.
    valid_neurons = norms > 0

    # Normalize valid response vectors.
    normalized = np.zeros_like(
        centered,
        dtype=float
    )

    normalized[valid_neurons] = (
        centered[valid_neurons]
        / norms[valid_neurons, None]
    )

    # Pearson correlation is the dot product of
    # centered, unit-normalized response vectors.
    correlation_matrix = (
        normalized @ normalized.T
    )

    # Numerical rounding can produce values
    # marginally outside [-1, 1].
    correlation_matrix = np.clip(
        correlation_matrix,
        -1.0,
        1.0
    )

    # Pairs involving constant-response neurons
    # have undefined correlations.
    pair_valid = (
        valid_neurons[:, None]
        & valid_neurons[None, :]
    )

    correlation_matrix[~pair_valid] = np.nan

    # Diagonal entries for valid neurons equal 1.
    np.fill_diagonal(
        correlation_matrix,
        np.where(valid_neurons, 1.0, np.nan)
    )

    # Extract the unique upper-triangular pairs.
    upper_indices = np.triu_indices(
        n_neurons,
        k=1
    )

    pair_correlations = correlation_matrix[
        upper_indices
    ]

    return correlation_matrix, pair_correlations


# ============================================================
# 4. SUMMARY STATISTICS
# ============================================================

def summarize_correlations(name, pairs):
    """
    Print statistics for defined pairwise correlations.

    Undefined correlations are excluded.

    Returns
    -------
    dict
        Summary statistics for the valid neuron pairs.
    """

    valid = pairs[np.isfinite(pairs)]

    print(f"\n{name.upper()}")

    print("Total neuron pairs:", pairs.size)
    print("Valid pairs:", valid.size)
    print("Undefined pairs:", pairs.size - valid.size)

    if valid.size == 0:
        print("No valid pairwise correlations.")

        return {
            "n_valid": 0,
            "mean": np.nan,
            "median": np.nan,
            "mean_absolute": np.nan,
        }

    mean_corr = valid.mean()
    median_corr = np.median(valid)
    mean_abs_corr = np.abs(valid).mean()

    print("Mean correlation:", mean_corr)
    print("Median correlation:", median_corr)

    print(
        "Mean absolute correlation:",
        mean_abs_corr
    )

    print("Minimum:", valid.min())
    print("Maximum:", valid.max())

    return {
        "n_valid": int(valid.size),
        "mean": float(mean_corr),
        "median": float(median_corr),
        "mean_absolute": float(mean_abs_corr),
    }


# ============================================================
# 5. PAIRED CORRELATION COMPARISON
# ============================================================

def compare_paired_correlations(
    pairs_initial,
    pairs_learned
):
    """
    Compare correlations for the same neuron pairs.

    A pair is included only if its correlation is
    defined both before and after learning.
    """

    if pairs_initial.shape != pairs_learned.shape:
        raise ValueError(
            "Pairwise correlation arrays must match."
        )

    valid = (
        np.isfinite(pairs_initial)
        & np.isfinite(pairs_learned)
    )

    changes = (
        pairs_learned[valid]
        - pairs_initial[valid]
    )

    print("\nPAIRED CORRELATION COMPARISON")
    print("Valid paired comparisons:", valid.sum())

    if changes.size:
        print("Mean signed change:", changes.mean())
        print("Median signed change:", np.median(changes))

        print(
            "Pairs with decreased signed correlation:",
            np.sum(changes < 0)
        )

        print(
            "Pairs with increased signed correlation:",
            np.sum(changes > 0)
        )

        absolute_changes = (
            np.abs(pairs_learned[valid])
            - np.abs(pairs_initial[valid])
        )

        print(
            "Mean change in absolute correlation:",
            absolute_changes.mean()
        )

        print(
            "Pairs with reduced absolute correlation:",
            np.sum(absolute_changes < 0)
        )

    else:
        print("No valid paired comparisons.")

    return valid, changes


# ============================================================
# 6. PLOT CORRELATION DISTRIBUTIONS
# ============================================================

def plot_correlation_distributions(
    pairs_initial,
    pairs_learned,
    save_path,
    show=False
):
    """
    Plot initial and learned response-correlation
    distributions using identical histogram bins.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-1, 1, 61)

    ax.hist(
        pairs_initial[np.isfinite(pairs_initial)],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Initial"
    )

    ax.hist(
        pairs_learned[np.isfinite(pairs_learned)],
        bins=bins,
        alpha=0.6,
        density=True,
        label="Learned"
    )

    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Probability density")

    ax.set_title(
        "Pairwise Neuronal Response Correlations"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 7. VISUALIZE CORRELATION MATRICES
# ============================================================

def plot_correlation_matrices(
    C_initial,
    C_learned,
    save_path,
    show=False
):
    """
    Plot initial and learned correlation matrices.

    Both matrices use the same [-1, 1] color scale.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True
    )

    matrices = [
        C_initial,
        C_learned
    ]

    titles = [
        "Initial network",
        "Learned network"
    ]

    for ax, C, title in zip(
        axes,
        matrices,
        titles
    ):

        image = ax.imshow(
            C,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )

        ax.set_title(title)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Neuron")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.8,
        label="Pearson correlation"
    )

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 8. PAIRED CORRELATION SCATTER
# ============================================================

def plot_paired_correlations(
    pairs_initial,
    pairs_learned,
    save_path,
    show=False
):
    """
    Plot correlations of identical neuron pairs
    before and after learning.
    """

    valid = (
        np.isfinite(pairs_initial)
        & np.isfinite(pairs_learned)
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        pairs_initial[valid],
        pairs_learned[valid],
        s=5,
        alpha=0.2,
        rasterized=True
    )

    ax.plot(
        [-1, 1],
        [-1, 1],
        "--",
        color="gray"
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xlabel(
        "Initial response correlation"
    )

    ax.set_ylabel(
        "Learned response correlation"
    )

    ax.set_title(
        "Paired Neuronal Response Correlations"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. COMPLETE ANALYSIS
# ============================================================

def analyze_response_correlations(
    input_path=DEFAULT_INPUT,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run complete initial-versus-learned analysis.

    Saves the original correlation arrays and plots,
    with additional paired-comparison diagnostics.
    """

    input_path = Path(
        input_path
    ).expanduser()

    output_dir = Path(
        output_dir
    ).expanduser()

    initial_E, learned_E = (
        load_matched_responses(input_path)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Input:", input_path)

    print(
        "Initial responses:",
        initial_E.shape
    )

    print(
        "Learned responses:",
        learned_E.shape
    )

    # --------------------------------------------------------
    # A. Calculate correlations
    # --------------------------------------------------------

    C_initial, pairs_initial = (
        calculate_pairwise_correlations(initial_E)
    )

    C_learned, pairs_learned = (
        calculate_pairwise_correlations(learned_E)
    )

    # --------------------------------------------------------
    # B. Summary statistics
    # --------------------------------------------------------

    initial_stats = summarize_correlations(
        "Initial",
        pairs_initial
    )

    learned_stats = summarize_correlations(
        "Learned",
        pairs_learned
    )

    # --------------------------------------------------------
    # C. Paired comparison
    # --------------------------------------------------------

    valid_pairs, paired_changes = (
        compare_paired_correlations(
            pairs_initial,
            pairs_learned
        )
    )

    # --------------------------------------------------------
    # D. Plot correlation distributions
    # --------------------------------------------------------

    plot_correlation_distributions(
        pairs_initial,
        pairs_learned,
        output_dir / "neuronal_correlations.png",
        show=show
    )

    # --------------------------------------------------------
    # E. Plot correlation matrices
    # --------------------------------------------------------

    plot_correlation_matrices(
        C_initial,
        C_learned,
        output_dir / "neuronal_correlation_matrices.png",
        show=show
    )

    # --------------------------------------------------------
    # F. Plot paired correlations
    # --------------------------------------------------------

    plot_paired_correlations(
        pairs_initial,
        pairs_learned,
        output_dir / "neuronal_correlations_paired.png",
        show=show
    )

    # --------------------------------------------------------
    # G. Save numerical results
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "neuronal_correlations.npz",

        C_initial=C_initial,
        C_learned=C_learned,

        pairs_initial=pairs_initial,
        pairs_learned=pairs_learned,

        valid_pairs=valid_pairs,
        paired_changes=paired_changes,

        mean_initial=initial_stats["mean"],
        mean_learned=learned_stats["mean"],

        mean_abs_initial=initial_stats["mean_absolute"],
        mean_abs_learned=learned_stats["mean_absolute"]
    )

    print("\nCorrelation analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "C_initial": C_initial,
        "C_learned": C_learned,
        "pairs_initial": pairs_initial,
        "pairs_learned": pairs_learned,
        "valid_pairs": valid_pairs,
        "paired_changes": paired_changes
    }


# ============================================================
# 10. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare pairwise excitatory-neuron "
            "response correlations before and after learning."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to matched_responses.npz"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for figures and numerical results."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively."
    )

    args = parser.parse_args()

    analyze_response_correlations(
        input_path=args.input,
        output_dir=args.output_dir,
        show=args.show
    )
