
"""
Three-seed synaptic weight and RF correlation audit.

Reproduces the original:
    audit_three_seed_weights.py

LGN-V1 network replication.

Checks three trained networks at batch 1100:

1. All six weight matrices contain finite values.
2. All weights are nonnegative.
3. Postsynaptic row sums satisfy original L1 limits.
4. E-E and I-I self-connections are zero.
5. Compute signed ON-minus-OFF receptive fields.
6. Compute RMS pairwise spatial RF correlation.
7. Calculate mean excitatory/inhibitory thresholds.
8. Summarize RF RMS across three training runs.

Original project data and checkpoints are read-only.

The original audit printed results without saving an
archive. This implementation additionally saves the
twelve measurements for reproducibility.
"""

from pathlib import Path
import argparse

import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "audit"
    / "three_seed_weights"
)

NETWORK_RUNS = {
    "Reference": "training_test_25",
    "Seed 3001": "training_full_seed_3001",
    "Seed 3002": "training_full_seed_3002",
}

WEIGHT_NAMES = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E",
)

# Original per-postsynaptic-neuron L1 limits.
LIMITS = {
    "LGN_E": 100,
    "LGN_I": 80,
    "E_E": 10,
    "E_I": 240,
    "I_I": 120,
    "I_E": 120,
}

EXPECTED_SHAPES = {
    "LGN_E": (400, 512),
    "LGN_I": (100, 512),
    "E_E": (400, 400),
    "E_I": (100, 400),
    "I_I": (100, 100),
    "I_E": (400, 100),
}

EXPECTED_BATCH = 1100

L1_TOLERANCE = 1e-8

RESULT_METRICS = (
    "E_RF_RMS",
    "I_RF_RMS",
    "theta_E",
    "theta_I",
)


# ============================================================
# 2. ORIGINAL SIGNED RECEPTIVE-FIELD RMS
# ============================================================

def rf_rms(W):
    """
    Compute RMS spatial RF correlation.

    This function reproduces the exact original
    mathematical operations and their order.

    Parameters
    ----------
    W : ndarray
        LGN-to-cortical weight matrix.

        Rows: postsynaptic cortical neurons.

        Columns:
            First half: LGN ON weights
            Second half: LGN OFF weights

    Returns
    -------
    float
        RMS of unique off-diagonal pairwise
        receptive-field correlations.

    Steps
    -----
    1. Signed RF = ON - OFF.
    2. Subtract each RF's spatial mean.
    3. Normalize each RF to unit Euclidean norm.
    4. Calculate pairwise RF correlations.
    5. Select upper triangle without diagonal.
    6. Calculate sqrt(mean(correlation**2)).

    Zero-norm RFs are kept as zero vectors,
    matching the original implementation.
    """

    W = np.asarray(W)

    if W.ndim != 2:
        raise ValueError(
            "Weight matrix must be two-dimensional."
        )

    if W.shape[1] % 2 != 0:
        raise ValueError(
            "ON/OFF weight dimension must be even."
        )

    if W.shape[0] < 2:
        raise ValueError(
            "At least two neurons are needed."
        )

    # --------------------------------------------------------
    # Original ON-minus-OFF receptive fields
    # --------------------------------------------------------

    n_pixels = W.shape[1] // 2

    rf = (
        W[:, :n_pixels]
        - W[:, n_pixels:]
    )

    # --------------------------------------------------------
    # Original spatial centering
    # --------------------------------------------------------

    rf = (
        rf
        - rf.mean(
            axis=1,
            keepdims=True
        )
    )

    # --------------------------------------------------------
    # Original normalization
    # --------------------------------------------------------

    norm = np.linalg.norm(
        rf,
        axis=1,
        keepdims=True
    )

    rf = np.divide(
        rf,
        norm,
        out=np.zeros_like(rf),
        where=norm > 0
    )

    # --------------------------------------------------------
    # Original pairwise correlation
    # --------------------------------------------------------

    corr = rf @ rf.T

    upper = corr[
        np.triu_indices(
            len(rf),
            k=1
        )
    ]

    # --------------------------------------------------------
    # Original RMS statistic
    # --------------------------------------------------------

    return np.sqrt(
        np.mean(upper**2)
    )


# ============================================================
# 3. LOAD ORIGINAL TRAINED CHECKPOINT
# ============================================================

def load_checkpoint(path):
    """
    Load all six weight matrices and firing thresholds.

    Returns
    -------
    W : dict
        Six weight matrices.

    theta_E : ndarray
        Excitatory thresholds, shape (400, 1).

    theta_I : ndarray
        Inhibitory thresholds, shape (100, 1).

    batch : int
        Completed learning batch.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        W = {
            name: data[f"W_{name}"].copy()
            for name in WEIGHT_NAMES
        }

        theta_E = (
            data["threshold_E"].copy()
        )

        theta_I = (
            data["threshold_I"].copy()
        )

        batch = int(
            data["learning_batch"]
        )

    return W, theta_E, theta_I, batch


# ============================================================
# 4. VALIDATE SYNAPTIC WEIGHT CONSTRAINTS
# ============================================================

def validate_weights(W):
    """
    Check the original synaptic constraints.

    Preserves:
        np.all(np.isfinite(...))
        np.all(weights >= 0)
        row_sum <= L1_limit + 1e-8
        np.allclose(diagonal, 0)

    Additional dimension checks detect incorrect
    matrix ordering before scientific calculations.

    Raises
    ------
    ValueError
        If any matrix fails validation.

    Returns
    -------
    bool
        True if every matrix passes.
    """

    for name in WEIGHT_NAMES:

        matrix = W[name]

        if matrix.shape != EXPECTED_SHAPES[name]:
            raise ValueError(
                f"{name}: incorrect shape "
                f"{matrix.shape}"
            )

        if not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                f"{name}: non-finite weights"
            )

        if not np.all(
            matrix >= 0
        ):
            raise ValueError(
                f"{name}: negative weights"
            )

        row_sums = matrix.sum(axis=1)

        if not np.all(
            row_sums
            <= LIMITS[name] + L1_TOLERANCE
        ):
            raise ValueError(
                f"{name}: L1 row limit exceeded. "
                f"Maximum={row_sums.max()}, "
                f"limit={LIMITS[name]}"
            )

    if not np.allclose(
        np.diag(W["E_E"]),
        0
    ):
        raise ValueError(
            "E-E self-connections are nonzero."
        )

    if not np.allclose(
        np.diag(W["I_I"]),
        0
    ):
        raise ValueError(
            "I-I self-connections are nonzero."
        )

    return True


# ============================================================
# 5. ANALYZE ONE TRAINED CHECKPOINT
# ============================================================

def analyze_checkpoint(path, label):
    """
    Validate and analyze one network checkpoint.

    Returns
    -------
    dict
        E_RF_RMS
        I_RF_RMS
        theta_E
        theta_I
    """

    W, theta_E, theta_I, batch = (
        load_checkpoint(path)
    )

    if batch != EXPECTED_BATCH:
        raise ValueError(
            f"{label}: incomplete training "
            f"({batch} batches)"
        )

    validate_weights(W)

    if theta_E.shape != (400, 1):
        raise ValueError(
            f"{label}: incorrect E thresholds."
        )

    if theta_I.shape != (100, 1):
        raise ValueError(
            f"{label}: incorrect I thresholds."
        )

    if not (
        np.all(np.isfinite(theta_E))
        and np.all(np.isfinite(theta_I))
    ):
        raise ValueError(
            f"{label}: non-finite thresholds."
        )

    # Original calculations, without modifications.
    result = {
        "E_RF_RMS": rf_rms(W["LGN_E"]),
        "I_RF_RMS": rf_rms(W["LGN_I"]),
        "theta_E": theta_E.mean(),
        "theta_I": theta_I.mean(),
    }

    print(f"\n{'=' * 45}")
    print(label)
    print(f"{'=' * 45}")

    print("Completed batches:", batch)

    print(
        "E RF RMS:",
        result["E_RF_RMS"]
    )

    print(
        "I RF RMS:",
        result["I_RF_RMS"]
    )

    print(
        "Mean E threshold:",
        result["theta_E"]
    )

    print(
        "Mean I threshold:",
        result["theta_I"]
    )

    return result


# ============================================================
# 6. THREE-SEED POPULATION SUMMARY
# ============================================================

def summarize_three_seeds(results):
    """
    Print original three-seed RF RMS statistics.

    The original script summarizes:
        E_RF_RMS
        I_RF_RMS

    It uses sample standard deviation:
        np.std(values, ddof=1)
    """

    print("\n" + "=" * 45)
    print("THREE-SEED SUMMARY")
    print("=" * 45)

    summary = {}

    for metric in (
        "E_RF_RMS",
        "I_RF_RMS"
    ):

        values = np.array([
            results[name][metric]
            for name in NETWORK_RUNS
        ])

        mean = values.mean()
        sd = values.std(ddof=1)

        print(f"\n{metric}:")
        print("Values:", values)
        print("Mean:", mean)
        print("Sample SD:", sd)

        summary[metric] = {
            "values": values,
            "mean": mean,
            "sample_sd": sd,
        }

    return summary


# ============================================================
# 7. SAVE NEW REPRODUCIBILITY ARCHIVE
# ============================================================

def save_three_seed_weights(results, output_dir):
    """
    Save twelve measured values:

        3 networks x 4 metrics = 12 fields.

    Note
    ----
    This archive is a NEW addition to the cleaned
    repository. The original audit saved no NPZ file.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_data = {}

    for name, result in results.items():

        prefix = (
            name.lower()
            .replace(" ", "_")
        )

        for metric in RESULT_METRICS:

            save_data[
                f"{prefix}_{metric}"
            ] = result[metric]

    output_path = (
        output_dir
        / "three_seed_weights.npz"
    )

    np.savez_compressed(
        output_path,
        **save_data
    )

    return output_path


# ============================================================
# 8. COMPLETE ORIGINAL AUDIT
# ============================================================

def run_three_seed_weights(
    source_project,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Validate all three original trained checkpoints,
    compute RF RMS and threshold statistics,
    and save a new reproducibility archive.

    Original network data are never modified.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    results = {}

    for label, run_dir in NETWORK_RUNS.items():

        path = (
            source_project
            / "results"
            / "learning"
            / run_dir
            / "batch_1100.npz"
        )

        results[label] = analyze_checkpoint(
            path,
            label
        )

    summary = summarize_three_seeds(
        results
    )

    output_path = save_three_seed_weights(
        results,
        output_dir
    )

    print("\nALL THREE CHECKPOINTS VALIDATED")
    print("Saved results:", output_path)

    return results, summary


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce three-seed synaptic "
            "weight and RF correlation audit."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to original V1_network_learning"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    args = parser.parse_args()

    run_three_seed_weights(
        source_project=args.source_project,
        output_dir=args.output_dir
    )
