
"""
Pairwise spatial receptive-field correlation analysis.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Purpose
-------
Compare the spatial similarity of excitatory V1 receptive
fields before and after synaptic learning.

Input
-----
Two checkpoints containing W_LGN_E:

    Initial untrained weights
    Final learned weights

Weight convention
-----------------
W_LGN_E[postsynaptic_neuron, presynaptic_neuron]

Each excitatory neuron receives:
    256 ON LGN inputs
    256 OFF LGN inputs

Signed receptive field:
    RF = ON - OFF

Correlation analysis
--------------------
1. Reconstruct signed RFs for all excitatory neurons.
2. Flatten each RF into a 256-dimensional vector.
3. Mean-center and L2-normalize each RF.
4. Calculate pairwise Pearson correlations.
5. Extract unique neuron pairs, excluding self-pairs.
6. Summarize signed, absolute, and RMS correlations.
7. Compare matching neuron pairs before and after learning.
8. Save numerical results and figures.

Constant receptive fields produce undefined correlations
and are represented by NaN.

This analysis measures spatial RF correlations, NOT
correlations between neuronal spike responses.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import N_E, N_LGN, PATCH_SIZE

from analysis.receptive_fields import (
    load_lgn_e_weights,
    reconstruct_rf,
)


# ============================================================
# 1. OUTPUT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "rf_correlations"
)


# ============================================================
# 2. RECONSTRUCT POPULATION RECEPTIVE FIELDS
# ============================================================

def reconstruct_all_rfs(W):
    """
    Reconstruct signed RFs for every excitatory neuron.

    Parameters
    ----------
    W : ndarray
        LGN-to-E weight matrix, shape (N_E, N_LGN).

    Returns
    -------
    ndarray
        Signed RFs with shape:
            (N_E, PATCH_SIZE, PATCH_SIZE)

    Notes
    -----
    Uses the same ON-OFF reconstruction as the
    validated receptive_fields.py module.
    """

    if W.shape != (N_E, N_LGN):
        raise ValueError(
            f"Expected {(N_E, N_LGN)}, got {W.shape}"
        )

    receptive_fields = np.stack([
        reconstruct_rf(W, neuron)[2]
        for neuron in range(N_E)
    ])

    return receptive_fields


# ============================================================
# 3. NORMALIZE RF VECTORS
# ============================================================

def normalize_rf_vectors(RF):
    """
    Mean-center and L2-normalize spatial RF vectors.

    Parameters
    ----------
    RF : ndarray
        Shape (n_neurons, n_pixels).

    Returns
    -------
    normalized : ndarray
        Centered, unit-norm vectors for valid RFs.

    valid : ndarray of bool
        True for finite RFs with nonzero spatial variance.

    Notes
    -----
    Invalid or constant receptive fields remain zero
    in the normalized array and are excluded later.
    """

    RF = np.asarray(RF, dtype=float)

    if RF.ndim != 2:
        raise ValueError(
            "RF input must have shape (neurons, pixels)."
        )

    if RF.shape[1] < 2:
        raise ValueError(
            "At least two spatial pixels are required."
        )

    finite = np.all(
        np.isfinite(RF),
        axis=1
    )

    centered = (
        RF - RF.mean(axis=1, keepdims=True)
    )

    norms = np.linalg.norm(
        centered,
        axis=1
    )

    valid = (
        finite
        & np.isfinite(norms)
        & (norms > 0)
    )

    normalized = np.zeros_like(
        RF,
        dtype=float
    )

    normalized[valid] = (
        centered[valid]
        / norms[valid, None]
    )

    return normalized, valid


# ============================================================
# 4. CALCULATE PAIRWISE RF CORRELATIONS
# ============================================================

def calculate_rf_correlations(RF):
    """
    Calculate Pearson correlations between spatial RFs.

    Parameters
    ----------
    RF : ndarray
        Shape:
            (n_neurons, height, width)
        or
            (n_neurons, n_pixels)

    Returns
    -------
    C : ndarray
        Full neuron-by-neuron correlation matrix.

    pairs : ndarray
        Upper-triangle correlations, excluding diagonal.

    Notes
    -----
    Upper-triangle ordering is preserved, including
    NaN placeholders for undefined correlations.

    This allows direct matching of neuron pairs between
    initial and learned networks.
    """

    RF = np.asarray(RF, dtype=float)

    if RF.ndim == 3:
        RF = RF.reshape(RF.shape[0], -1)

    if RF.ndim != 2:
        raise ValueError(
            "Expected 2D or 3D receptive-field array."
        )

    n_neurons = RF.shape[0]

    if n_neurons < 2:
        raise ValueError(
            "At least two neurons are required."
        )

    normalized, valid = normalize_rf_vectors(RF)

    # Dot product of centered, normalized RF vectors.
    C = normalized @ normalized.T

    C = np.clip(C, -1.0, 1.0)

    # Correlations involving constant or invalid RFs
    # are undefined.
    valid_matrix = (
        valid[:, None] & valid[None, :]
    )

    C[~valid_matrix] = np.nan

    np.fill_diagonal(
        C,
        np.where(valid, 1.0, np.nan)
    )

    upper_indices = np.triu_indices(
        n_neurons,
        k=1
    )

    pairs = C[upper_indices]

    return C, pairs


# ============================================================
# 5. RF CORRELATION SUMMARY STATISTICS
# ============================================================

def summarize_rf_correlations(name, pairs):
    """
    Summarize valid pairwise RF correlations.

    Returns
    -------
    dict
        Signed, absolute, RMS, and high-correlation
        statistics.
    """

    valid = pairs[np.isfinite(pairs)]

    print(f"\n{name.upper()}")
    print("-" * len(name))

    print("Total pairs:", pairs.size)
    print("Valid pairs:", valid.size)
    print("Undefined pairs:", pairs.size - valid.size)

    if valid.size == 0:
        return {
            "mean": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
            "median_abs": np.nan,
            "fraction_above_05": np.nan,
            "n_valid": 0
        }

    mean_corr = np.mean(valid)

    mean_abs = np.mean(
        np.abs(valid)
    )

    rms_corr = np.sqrt(
        np.mean(valid ** 2)
    )

    median_abs = np.median(
        np.abs(valid)
    )

    fraction_above_05 = np.mean(
        np.abs(valid) > 0.5
    )

    print("Mean signed correlation:", mean_corr)
    print("Mean absolute correlation:", mean_abs)
    print("RMS correlation:", rms_corr)
    print("Median absolute correlation:", median_abs)

    print(
        "Fraction |r| > 0.5:",
        fraction_above_05
    )

    return {
        "mean": float(mean_corr),
        "mean_abs": float(mean_abs),
        "rms": float(rms_corr),
        "median_abs": float(median_abs),
        "fraction_above_05": float(fraction_above_05),
        "n_valid": int(valid.size)
    }


# ============================================================
# 6. COMPARE MATCHED NEURON PAIRS
# ============================================================

def compare_rf_correlations(
    pairs_initial,
    pairs_learned
):
    """
    Compare the same RF pairs across learning stages.

    Only pairs with defined correlations in both
    stages are included in the paired comparison.
    """

    if pairs_initial.shape != pairs_learned.shape:
        raise ValueError(
            "Initial and learned pair arrays differ."
        )

    valid_pairs = (
        np.isfinite(pairs_initial)
        & np.isfinite(pairs_learned)
    )

    initial = pairs_initial[valid_pairs]
    learned = pairs_learned[valid_pairs]

    signed_change = learned - initial

    absolute_change = (
        np.abs(learned) - np.abs(initial)
    )

    print("\nPAIRED RF CORRELATION COMPARISON")

    print(
        "Valid paired comparisons:",
        valid_pairs.sum()
    )

    if signed_change.size:
        print(
            "Mean signed change:",
            signed_change.mean()
        )

        print(
            "Mean absolute-correlation change:",
            absolute_change.mean()
        )

        print(
            "Pairs with reduced |r|:",
            np.sum(absolute_change < 0)
        )

        print(
            "Pairs with increased |r|:",
            np.sum(absolute_change > 0)
        )

    else:
        print("No valid paired comparisons.")

    return valid_pairs, signed_change, absolute_change


# ============================================================
# 7. PLOT RF CORRELATION DISTRIBUTIONS
# ============================================================

def plot_rf_correlation_distributions(
    pairs_initial,
    pairs_learned,
    save_path,
    show=False
):
    """
    Compare RF correlation distributions.
    """

    initial = pairs_initial[
        np.isfinite(pairs_initial)
    ]

    learned = pairs_learned[
        np.isfinite(pairs_learned)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-1, 1, 81)

    if initial.size:
        ax.hist(
            initial,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Initial"
        )

    if learned.size:
        ax.hist(
            learned,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Learned"
        )

    ax.set_xlabel(
        "Pearson correlation between spatial RFs"
    )

    ax.set_ylabel("Probability density")

    ax.set_title(
        "Pairwise Receptive-Field Correlations"
    )

    if initial.size or learned.size:
        ax.legend()

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 8. PLOT CORRELATION MATRICES
# ============================================================

def plot_rf_correlation_matrices(
    C_initial,
    C_learned,
    save_path,
    show=False
):
    """
    Visualize initial and learned RF correlations.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True
    )

    for ax, C, title in zip(
        axes,
        (C_initial, C_learned),
        ("Initial RFs", "Learned RFs")
    ):

        image = ax.imshow(
            C,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )

        ax.set_title(title)
        ax.set_xlabel("Excitatory neuron")
        ax.set_ylabel("Excitatory neuron")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.8,
        label="Spatial RF Pearson correlation"
    )

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. PLOT RMS AND ABSOLUTE CORRELATIONS
# ============================================================

def plot_rf_summary(
    initial_stats,
    learned_stats,
    save_path,
    show=False
):
    """
    Plot two population RF correlation summaries.
    """

    labels = [
        "Mean |r|",
        "RMS r"
    ]

    initial_values = [
        initial_stats["mean_abs"],
        initial_stats["rms"]
    ]

    learned_values = [
        learned_stats["mean_abs"],
        learned_stats["rms"]
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        initial_values,
        width,
        label="Initial"
    )

    ax.bar(
        x + width / 2,
        learned_values,
        width,
        label="Learned"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RF correlation magnitude")

    ax.set_title(
        "Receptive-Field Similarity Before and After Learning"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 10. COMPLETE RF CORRELATION ANALYSIS
# ============================================================

def analyze_rf_correlations(
    initial_path,
    learned_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run population-wide RF correlation analysis.

    Reads initial and learned checkpoints, calculates
    pairwise spatial RF correlations, and saves results.
    """

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Load network weights
    # --------------------------------------------------------

    W_initial = load_lgn_e_weights(
        initial_path
    )

    W_learned = load_lgn_e_weights(
        learned_path
    )

    # --------------------------------------------------------
    # B. Reconstruct spatial receptive fields
    # --------------------------------------------------------

    RF_initial = reconstruct_all_rfs(
        W_initial
    )

    RF_learned = reconstruct_all_rfs(
        W_learned
    )

    print("Initial RF shape:", RF_initial.shape)
    print("Learned RF shape:", RF_learned.shape)

    # --------------------------------------------------------
    # C. Calculate correlations
    # --------------------------------------------------------

    C_initial, pairs_initial = (
        calculate_rf_correlations(RF_initial)
    )

    C_learned, pairs_learned = (
        calculate_rf_correlations(RF_learned)
    )

    # --------------------------------------------------------
    # D. Summary statistics
    # --------------------------------------------------------

    initial_stats = summarize_rf_correlations(
        "Initial RF correlations",
        pairs_initial
    )

    learned_stats = summarize_rf_correlations(
        "Learned RF correlations",
        pairs_learned
    )

    # --------------------------------------------------------
    # E. Compare identical neuron pairs
    # --------------------------------------------------------

    (
        valid_pairs,
        signed_change,
        absolute_change
    ) = compare_rf_correlations(
        pairs_initial,
        pairs_learned
    )

    # --------------------------------------------------------
    # F. Save figures
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_rf_correlation_distributions(
        pairs_initial,
        pairs_learned,
        output_dir / "rf_correlation_distributions.png",
        show=show
    )

    plot_rf_correlation_matrices(
        C_initial,
        C_learned,
        output_dir / "rf_correlation_matrices.png",
        show=show
    )

    plot_rf_summary(
        initial_stats,
        learned_stats,
        output_dir / "rf_correlation_summary.png",
        show=show
    )

    # --------------------------------------------------------
    # G. Save numerical results
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "rf_correlations.npz",

        C_initial=C_initial,
        C_learned=C_learned,

        pairs_initial=pairs_initial,
        pairs_learned=pairs_learned,

        valid_pairs=valid_pairs,
        signed_change=signed_change,
        absolute_change=absolute_change,

        mean_initial=initial_stats["mean"],
        mean_learned=learned_stats["mean"],

        mean_abs_initial=initial_stats["mean_abs"],
        mean_abs_learned=learned_stats["mean_abs"],

        rms_initial=initial_stats["rms"],
        rms_learned=learned_stats["rms"],

        fraction_above_05_initial=(
            initial_stats["fraction_above_05"]
        ),
        fraction_above_05_learned=(
            learned_stats["fraction_above_05"]
        )
    )

    print("\nRF correlation analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "C_initial": C_initial,
        "C_learned": C_learned,
        "pairs_initial": pairs_initial,
        "pairs_learned": pairs_learned,
        "initial_stats": initial_stats,
        "learned_stats": learned_stats
    }


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare pairwise spatial receptive-field "
            "correlations before and after V1 learning."
        )
    )

    parser.add_argument(
        "--initial",
        type=str,
        required=True,
        help="Path to initial weight checkpoint"
    )

    parser.add_argument(
        "--learned",
        type=str,
        required=True,
        help="Path to learned weight checkpoint"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for figures and results"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )

    args = parser.parse_args()

    analyze_rf_correlations(
        initial_path=args.initial,
        learned_path=args.learned,
        output_dir=args.output_dir,
        show=args.show
    )
