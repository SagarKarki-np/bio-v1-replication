
"""
Figure 10A — Track A functional STA provenance audit.

Reads the archived 100,000-stimulus STA file.

Checks:
- Saved metadata and array shapes
- Functional RF validity
- Pairwise RMS Pearson correlation
- Checkpoint progression
- Recorded split-half reliability
- Spike counts and saved diagnostics

Does NOT run simulations or modify existing files.

Important:
An NPZ file alone cannot prove stimulus generation,
preprocessing, or checkpoint provenance. Those require
a separate source-code audit.
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
)

STA_PATH = (
    BASE / "functional_sta"
    / "track_a"
    / "full_100000"
    / "STA_100000_results.npz"
)


def describe(name, value):
    """Print shape, dtype, and numerical summary."""

    a = np.asarray(value)

    print(f"\n{name}")
    print("  Shape:", a.shape)
    print("  Dtype:", a.dtype)

    if a.size == 0:
        print("  Empty array")
        return

    if a.dtype.kind not in "biufc":
        if a.ndim == 0:
            print("  Value:", str(a.item()))
        else:
            print("  Non-numeric metadata")
        return

    finite = np.isfinite(a)

    print("  Finite:", int(finite.sum()), "/", a.size)

    if not np.any(finite):
        return

    v = a[finite]

    print("  Min:", float(np.min(v)))
    print("  Mean:", float(np.mean(v)))
    print("  Max:", float(np.max(v)))


def pearson_rms(rf):
    """
    Pairwise RMS Pearson correlation of RF rows.
    Excludes diagonal and duplicate pairs.
    """

    rf = np.asarray(rf, dtype=np.float64)

    if rf.ndim != 2:
        raise ValueError("RF must be 2-dimensional")

    if not np.isfinite(rf).all():
        raise ValueError("RF contains nonfinite values")

    centered = rf - rf.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(centered, axis=1)

    valid = norms > 1e-12

    if valid.sum() < 2:
        raise ValueError("Insufficient valid RFs")

    standardized = (
        centered[valid] / norms[valid, None]
    )

    matrix = standardized @ standardized.T

    pairs = matrix[
        np.triu_indices(matrix.shape[0], k=1)
    ]

    return {
        "valid_rfs": int(valid.sum()),
        "pairs": int(pairs.size),
        "rms": float(np.sqrt(np.mean(pairs**2))),
        "mean": float(np.mean(pairs)),
        "mean_absolute": float(
            np.mean(np.abs(pairs))
        ),
    }


def checkpoint_audit(data):
    """Inspect stored convergence diagnostics."""

    print("\n" + "=" * 65)
    print("CHECKPOINT CONVERGENCE")
    print("=" * 65)

    keys = [
        "checkpoint_counts",
        "checkpoint_correlations_E",
        "checkpoint_correlations_I",
        "split_half_reliability",
    ]

    for key in keys:
        if key not in data.files:
            print(f"\n{key}: MISSING")
            continue

        try:
            values = data[key]
        except ValueError as error:
            print(
                f"\n{key}: Cannot read without pickle: "
                f"{error}"
            )
            continue

        describe(key, values)

        if values.dtype.kind not in "biufc":
            continue

        if values.ndim == 1 and values.size > 0:
            print(
                "  First:",
                values[:min(5, values.size)]
            )

            print(
                "  Last:",
                values[-min(5, values.size):]
            )

            if values.size >= 2:
                print(
                    "  Last-minus-previous:",
                    values[-1] - values[-2]
                )

    if "checkpoint_counts" in data.files:
        counts = data["checkpoint_counts"]

        if (
            counts.ndim == 1
            and counts.dtype.kind in "biuf"
            and counts.size >= 2
        ):
            monotonic = bool(
                np.all(np.diff(counts) > 0)
            )

            print("\nCheckpoint counts strictly increasing:",
                  monotonic)

            if "n_stimuli" in data.files:
                n = int(data["n_stimuli"])

                print(
                    "Final checkpoint equals n_stimuli:",
                    bool(counts[-1] == n)
                )


def main():

    print("=" * 65)
    print("FIGURE 10A — STA PROVENANCE AUDIT")
    print("=" * 65)

    if not STA_PATH.is_file():
        raise FileNotFoundError(STA_PATH)

    print("Source:", STA_PATH)

    with np.load(
        STA_PATH,
        allow_pickle=False
    ) as data:

        print("\nARCHIVE CONTENTS")
        print("-" * 65)

        for key in data.files:
            print(key)

        print("\n" + "=" * 65)
        print("EXPERIMENT METADATA")
        print("=" * 65)

        for key in [
            "n_stimuli",
            "batch_size",
            "seed",
            "mean_LGN_rate",
            "fraction_capped",
            "fraction_zero",
            "total_E_spikes",
            "total_I_spikes",
        ]:

            if key in data.files:
                try:
                    describe(key, data[key])
                except ValueError as error:
                    print(key, "unreadable:", error)
            else:
                print(key, "MISSING")

        print("\n" + "=" * 65)
        print("FUNCTIONAL RECEPTIVE FIELDS")
        print("=" * 65)

        for key, shape in [
            ("STA_E", (400, 256)),
            ("STA_I", (100, 256)),
        ]:

            if key not in data.files:
                raise KeyError(f"Missing {key}")

            rf = data[key]

            describe(key, rf)

            if rf.shape != shape:
                raise ValueError(
                    f"{key}: expected {shape}, "
                    f"found {rf.shape}"
                )

            result = pearson_rms(rf)

            print(f"\n{key} pairwise correlations")

            for result_key, value in result.items():
                print(" ", result_key, ":", value)

        print("\n" + "=" * 65)
        print("ADDITIONAL SAVED RF DIAGNOSTICS")
        print("=" * 65)

        for key in [
            "weight_RF_E",
            "weight_RF_I",
            "RF_correlations_E",
            "RF_correlations_I",
        ]:

            if key in data.files:
                describe(key, data[key])

        checkpoint_audit(data)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print("No network simulations performed.")
    print("No source NPZ or checkpoint modified.")

    print(
        "\nLIMITATION: Metadata alone does not "
        "prove the Gaussian-noise generator, "
        "preprocessing, or STA accumulation method."
    )


if __name__ == "__main__":
    main()
