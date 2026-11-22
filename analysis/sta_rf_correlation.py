
"""
Population-level STA and weight-derived RF correlations.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Purpose
-------
Compare pairwise spatial receptive-field correlations
estimated using:

1. Learned LGN synaptic weights
2. Spike-triggered averaging (STA)

Populations
-----------
E-E : Excitatory neuron pairs
E-I : Excitatory-inhibitory neuron pairs
I-I : Inhibitory neuron pairs

For each category:
    - Mean Pearson correlation
    - Mean absolute correlation
    - RMS correlation
    - Median absolute correlation
    - Fraction with |r| > 0.5

Compare weight-derived and STA-derived correlations
using Pearson and Spearman agreement.

Input
-----
STA_2000_results.npz

Despite its name, the original file contains
10,000 white-noise stimulus presentations.

Outputs
-------
STA_RF_correlation_audit.npz
STA_vs_weight_RF_distributions.png
STA_vs_weight_RMS.png

Methodological safeguard
------------------------
Undefined RF correlations retain NaN placeholders.
Weight and STA agreement is calculated only for
matching valid neuron pairs.

The original analysis produces the same results
when all receptive fields are valid.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from config import N_E, N_I, PATCH_SIZE


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_STA_PATH = (
    PROJECT_DIR
    / "results"
    / "STA"
    / "STA_2000_results.npz"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "STA_RF"
)

N_PIXELS = PATCH_SIZE ** 2

EXPECTED_STIMULI = 10000

GROUP_NAMES = ("E-E", "E-I", "I-I")


# ============================================================
# 2. LOAD SAVED RECEPTIVE FIELDS
# ============================================================

def load_sta_receptive_fields(sta_path):
    """
    Load STA-derived and weight-derived RF vectors.

    Verify the number of stimulus presentations
    and expected matrix dimensions.
    """

    with np.load(
        sta_path,
        allow_pickle=False
    ) as data:

        STA_E = data["STA_E"].copy()
        STA_I = data["STA_I"].copy()

        weight_RF_E = data["weight_RF_E"].copy()
        weight_RF_I = data["weight_RF_I"].copy()

        n_stimuli = int(data["n_stimuli"])

    if n_stimuli != EXPECTED_STIMULI:
        raise ValueError(
            f"Expected {EXPECTED_STIMULI} stimuli, "
            f"found {n_stimuli}."
        )

    expected_E = (N_E, N_PIXELS)
    expected_I = (N_I, N_PIXELS)

    if STA_E.shape != expected_E:
        raise ValueError(
            f"Unexpected STA_E shape: {STA_E.shape}"
        )

    if STA_I.shape != expected_I:
        raise ValueError(
            f"Unexpected STA_I shape: {STA_I.shape}"
        )

    if weight_RF_E.shape != STA_E.shape:
        raise ValueError(
            "Weight and STA E dimensions differ."
        )

    if weight_RF_I.shape != STA_I.shape:
        raise ValueError(
            "Weight and STA I dimensions differ."
        )

    print("Number of STA stimuli:", n_stimuli)
    print("STA E shape:", STA_E.shape)
    print("STA I shape:", STA_I.shape)

    return {
        "STA_E": STA_E,
        "STA_I": STA_I,
        "weight_RF_E": weight_RF_E,
        "weight_RF_I": weight_RF_I,
        "n_stimuli": n_stimuli,
    }


# ============================================================
# 3. NORMALIZE RECEPTIVE FIELDS
# ============================================================

def normalize_receptive_fields(RF):
    """
    Mean-center and L2-normalize RF vectors.

    Parameters
    ----------
    RF : ndarray
        Shape (neurons, pixels).

    Returns
    -------
    normalized : ndarray
        Centered and normalized RF vectors.

    valid : ndarray
        Boolean mask for finite, nonconstant RFs.

    Invalid rows receive zero values internally.
    Their correlations are marked NaN later.
    """

    RF = np.asarray(RF, dtype=float)

    if RF.ndim != 2:
        raise ValueError(
            "RF must have shape (neurons, pixels)."
        )

    centered = (
        RF - RF.mean(axis=1, keepdims=True)
    )

    norms = np.linalg.norm(
        centered,
        axis=1,
        keepdims=True
    )

    valid = (
        np.all(np.isfinite(RF), axis=1)
        & np.isfinite(norms[:, 0])
        & (norms[:, 0] > 0)
    )

    normalized = np.zeros_like(
        RF,
        dtype=float
    )

    normalized[valid] = (
        centered[valid] / norms[valid]
    )

    return normalized, valid


# ============================================================
# 4. WITHIN-POPULATION CORRELATIONS
# ============================================================

def within_population_correlations(RF):
    """
    Calculate spatial Pearson correlations between
    unique neuron pairs within one population.

    Pair ordering follows np.triu_indices(..., k=1).

    Returns an array containing all possible pairs.
    Undefined correlations are represented by NaN.
    """

    normalized, valid = normalize_receptive_fields(
        RF
    )

    C = normalized @ normalized.T

    rows, cols = np.triu_indices(
        RF.shape[0],
        k=1
    )

    values = C[rows, cols].copy()

    pair_valid = (
        valid[rows] & valid[cols]
    )

    values[~pair_valid] = np.nan

    return values


# ============================================================
# 5. BETWEEN-POPULATION CORRELATIONS
# ============================================================

def between_population_correlations(RF_A, RF_B):
    """
    Calculate all A-versus-B spatial correlations.

    Returns a flattened array in row-major ordering.

    Each element retains its neuron-pair identity,
    including NaN for undefined correlations.
    """

    A, valid_A = normalize_receptive_fields(RF_A)
    B, valid_B = normalize_receptive_fields(RF_B)

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            "Populations must have matching pixel counts."
        )

    C = A @ B.T

    pair_valid = (
        valid_A[:, None]
        & valid_B[None, :]
    )

    C[~pair_valid] = np.nan

    return C.ravel()


# ============================================================
# 6. CALCULATE ALL SIX CORRELATION DISTRIBUTIONS
# ============================================================

def calculate_population_correlations(data):
    """
    Calculate E-E, E-I and I-I correlations for
    weight-derived and STA-derived receptive fields.
    """

    weight_EE = within_population_correlations(
        data["weight_RF_E"]
    )

    weight_EI = between_population_correlations(
        data["weight_RF_E"],
        data["weight_RF_I"]
    )

    weight_II = within_population_correlations(
        data["weight_RF_I"]
    )

    STA_EE = within_population_correlations(
        data["STA_E"]
    )

    STA_EI = between_population_correlations(
        data["STA_E"],
        data["STA_I"]
    )

    STA_II = within_population_correlations(
        data["STA_I"]
    )

    return {
        "weight_EE": weight_EE,
        "weight_EI": weight_EI,
        "weight_II": weight_II,
        "STA_EE": STA_EE,
        "STA_EI": STA_EI,
        "STA_II": STA_II,
    }


# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================

def summarize(name, values):
    """
    Calculate the original RF-correlation statistics.

    Undefined correlations are excluded.
    """

    values = np.asarray(values, dtype=float)

    valid = values[np.isfinite(values)]

    print(f"\n{name}")
    print("Number of pairs:", valid.size)
    print(
        "Undefined pairs:",
        values.size - valid.size
    )

    if valid.size == 0:
        raise ValueError(
            f"No valid correlations for {name}."
        )

    mean_corr = valid.mean()

    mean_abs = np.mean(
        np.abs(valid)
    )

    rms = np.sqrt(
        np.mean(valid ** 2)
    )

    median_abs = np.median(
        np.abs(valid)
    )

    fraction_above_05 = np.mean(
        np.abs(valid) > 0.5
    )

    print("Mean correlation:", mean_corr)
    print("Mean absolute correlation:", mean_abs)
    print("RMS correlation:", rms)
    print("Median absolute correlation:", median_abs)

    print(
        "Fraction |r| > 0.5:",
        fraction_above_05
    )

    return {
        "n_pairs": int(valid.size),
        "mean": float(mean_corr),
        "mean_abs": float(mean_abs),
        "rms": float(rms),
        "median_abs": float(median_abs),
        "fraction_above_05": float(fraction_above_05)
    }


# ============================================================
# 8. COMPARE MATCHED WEIGHT AND STA CORRELATIONS
# ============================================================

def compare_correlations(
    name,
    weight_values,
    sta_values
):
    """
    Calculate Pearson and Spearman agreement for
    matching valid neuron pairs.

    Ordering is identical in both arrays because
    undefined pairs retain NaN placeholders.
    """

    weight_values = np.asarray(
        weight_values,
        dtype=float
    )

    sta_values = np.asarray(
        sta_values,
        dtype=float
    )

    if weight_values.shape != sta_values.shape:
        raise ValueError(
            "Weight and STA pair dimensions differ."
        )

    valid = (
        np.isfinite(weight_values)
        & np.isfinite(sta_values)
    )

    weights = weight_values[valid]
    sta = sta_values[valid]

    if weights.size < 2:
        raise ValueError(
            f"Insufficient matched pairs for {name}."
        )

    pearson = np.corrcoef(
        weights,
        sta
    )[0, 1]

    spearman = spearmanr(
        weights,
        sta
    ).statistic

    print(f"\n{name}")
    print("Matched pairs:", weights.size)
    print("Weight vs STA Pearson:", pearson)
    print("Weight vs STA Spearman:", spearman)

    return pearson, spearman


# ============================================================
# 9. PLOT CORRELATION DISTRIBUTIONS
# ============================================================

def plot_distributions(
    correlations,
    save_path,
    show=False
):
    """
    Reproduce original three-panel comparison.
    """

    groups = [
        (
            "E-E",
            correlations["weight_EE"],
            correlations["STA_EE"]
        ),
        (
            "E-I",
            correlations["weight_EI"],
            correlations["STA_EI"]
        ),
        (
            "I-I",
            correlations["weight_II"],
            correlations["STA_II"]
        )
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4),
        constrained_layout=True
    )

    bins = np.linspace(-1, 1, 81)

    for ax, (name, weight_values, sta_values) in zip(
        axes,
        groups
    ):

        weight_valid = weight_values[
            np.isfinite(weight_values)
        ]

        sta_valid = sta_values[
            np.isfinite(sta_values)
        ]

        ax.hist(
            weight_valid,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Weights"
        )

        ax.hist(
            sta_valid,
            bins=bins,
            density=True,
            alpha=0.5,
            label="STA"
        )

        ax.set_title(
            f"{name} RF correlations"
        )

        ax.set_xlabel("Pearson correlation")
        ax.set_ylabel("Probability density")

        ax.legend()
        ax.grid(alpha=0.2)

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 10. RMS CORRELATION COMPARISON
# ============================================================

def calculate_rms(values):
    """
    Calculate RMS of defined spatial correlations.
    """

    valid = np.asarray(
        values,
        dtype=float
    )

    valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        return np.nan

    return np.sqrt(
        np.mean(valid ** 2)
    )


def plot_rms_comparison(
    weight_RMS,
    STA_RMS,
    save_path,
    show=False
):
    """
    Reproduce original grouped RMS bar chart.
    """

    labels = ["E-E", "E-I", "I-I"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        weight_RMS,
        width,
        label="Weight RF"
    )

    ax.bar(
        x + width / 2,
        STA_RMS,
        width,
        label="STA RF"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "RMS pairwise RF correlation"
    )

    ax.set_title(
        "Weight-Derived vs Functional RF Correlations"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 11. COMPLETE STA-RF CORRELATION ANALYSIS
# ============================================================

def analyze_sta_rf_correlations(
    sta_path=DEFAULT_STA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run population-level STA and weight RF analysis.

    Uses previously saved results and does not
    rerun white-noise V1 simulations.
    """

    sta_path = Path(
        sta_path
    ).expanduser()

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Load validated STA data
    # --------------------------------------------------------

    data = load_sta_receptive_fields(
        sta_path
    )

    # --------------------------------------------------------
    # B. Calculate pairwise RF correlations
    # --------------------------------------------------------

    correlations = calculate_population_correlations(
        data
    )

    # --------------------------------------------------------
    # C. Summarize population correlations
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("WEIGHT-DERIVED RF CORRELATIONS")
    print("=" * 55)

    weight_stats = {}

    for group in ("EE", "EI", "II"):

        weight_stats[group] = summarize(
            f"{group} WEIGHTS",
            correlations[f"weight_{group}"]
        )

    print("\n" + "=" * 55)
    print("STA-DERIVED RF CORRELATIONS")
    print("=" * 55)

    STA_stats = {}

    for group in ("EE", "EI", "II"):

        STA_stats[group] = summarize(
            f"{group} STA",
            correlations[f"STA_{group}"]
        )

    # --------------------------------------------------------
    # D. Weight-vs-STA pairwise agreement
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("WEIGHT-STA PAIRWISE AGREEMENT")
    print("=" * 55)

    agreement_EE = compare_correlations(
        "E-E",
        correlations["weight_EE"],
        correlations["STA_EE"]
    )

    agreement_EI = compare_correlations(
        "E-I",
        correlations["weight_EI"],
        correlations["STA_EI"]
    )

    agreement_II = compare_correlations(
        "I-I",
        correlations["weight_II"],
        correlations["STA_II"]
    )

    # --------------------------------------------------------
    # E. RMS comparison
    # --------------------------------------------------------

    weight_RMS = np.array([
        calculate_rms(correlations[f"weight_{group}"])
        for group in ("EE", "EI", "II")
    ])

    STA_RMS = np.array([
        calculate_rms(correlations[f"STA_{group}"])
        for group in ("EE", "EI", "II")
    ])

    # --------------------------------------------------------
    # F. Save results
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_dir / "STA_RF_correlation_audit.npz",

        weight_EE=correlations["weight_EE"],
        weight_EI=correlations["weight_EI"],
        weight_II=correlations["weight_II"],

        STA_EE=correlations["STA_EE"],
        STA_EI=correlations["STA_EI"],
        STA_II=correlations["STA_II"],

        weight_RMS=weight_RMS,
        STA_RMS=STA_RMS,

        agreement_EE=np.array(agreement_EE),
        agreement_EI=np.array(agreement_EI),
        agreement_II=np.array(agreement_II),

        n_stimuli=data["n_stimuli"]
    )

    # --------------------------------------------------------
    # G. Save figures
    # --------------------------------------------------------

    plot_distributions(
        correlations,
        output_dir / "STA_vs_weight_RF_distributions.png",
        show=show
    )

    plot_rms_comparison(
        weight_RMS,
        STA_RMS,
        output_dir / "STA_vs_weight_RMS.png",
        show=show
    )

    print("\nSTA RF correlation analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "correlations": correlations,
        "weight_RMS": weight_RMS,
        "STA_RMS": STA_RMS,
        "agreement_EE": agreement_EE,
        "agreement_EI": agreement_EI,
        "agreement_II": agreement_II,
    }


# ============================================================
# 12. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare population RF similarity measured "
            "from learned weights and STA estimates."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_STA_PATH),
        help="Saved 10,000-stimulus STA archive"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for correlation results"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively"
    )

    args = parser.parse_args()

    analyze_sta_rf_correlations(
        sta_path=args.input,
        output_dir=args.output_dir,
        show=args.show
    )
