
"""
Figure 10A-C: Read-only sparseness methodology audit.

Uses all five previously saved response evaluations.

Tests:
1. Uncorrected Treves-Rolls sparseness
2. Finite-sample-corrected sparseness
3. Undefined all-zero response handling
4. Agreement with stored sparseness values
5. Firing-rate and response-matrix consistency

No network simulation, training, or checkpoint changes.
"""

from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results" / "paper_protocol"
    / "seed_3002"
    / "figure10_firing_rate_sweep"
    / "paper"
)

OUT = BASE / "summary" / "sparseness_audit"

CONDITIONS = [
    ("E01_I02", 1, 2),
    ("E02_I04", 2, 4),
    ("E05_I10", 5, 10),
    ("E10_I20", 10, 20),
    ("E20_I40", 20, 40),
]


def treves_rolls(counts, axis, corrected=True):
    """
    Independently calculate Treves-Rolls sparseness.

    counts: shape (neurons, presentations)
    axis=1: temporal sparseness
    axis=0: population sparseness

    All-zero vectors are returned as NaN.
    """

    x = np.asarray(counts, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("Expected a 2D response matrix")

    if not np.isfinite(x).all():
        raise ValueError("Nonfinite responses")

    if np.any(x < 0):
        raise ValueError("Negative spike counts")

    n = x.shape[axis]

    if n <= 1:
        raise ValueError("At least two observations needed")

    mean = np.mean(x, axis=axis)
    mean_square = np.mean(x * x, axis=axis)

    result = np.full(
        mean.shape, np.nan, dtype=np.float64
    )

    valid = mean_square > 0

    result[valid] = (
        1.0
        - (mean[valid] ** 2) / mean_square[valid]
    )

    if corrected:
        result[valid] /= (1.0 - 1.0 / n)

    return result


def mean_with_policy(values, policy):
    """
    Policies:
    exclude: omit undefined responses
    zero: replace undefined responses with 0
    one: replace undefined responses with 1
    """

    values = np.asarray(
        values, dtype=np.float64
    )

    if policy == "exclude":
        return float(np.nanmean(values))

    if policy == "zero":
        return float(
            np.mean(np.nan_to_num(values, nan=0.0))
        )

    if policy == "one":
        return float(
            np.mean(np.nan_to_num(values, nan=1.0))
        )

    raise ValueError(policy)


def audit_population(name, counts, stored_temporal,
                     stored_population):

    if counts.ndim != 2:
        raise ValueError(f"{name}: invalid shape")

    temporal_raw = treves_rolls(
        counts, axis=1, corrected=False
    )

    temporal_corrected = treves_rolls(
        counts, axis=1, corrected=True
    )

    population_raw = treves_rolls(
        counts, axis=0, corrected=False
    )

    population_corrected = treves_rolls(
        counts, axis=0, corrected=True
    )

    if stored_temporal.shape != temporal_raw.shape:
        raise ValueError("Stored temporal shape mismatch")

    if stored_population.shape != population_raw.shape:
        raise ValueError("Stored population shape mismatch")

    raw_temporal_match = np.allclose(
        stored_temporal,
        temporal_raw,
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True
    )

    corrected_temporal_match = np.allclose(
        stored_temporal,
        temporal_corrected,
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True
    )

    raw_population_match = np.allclose(
        stored_population,
        population_raw,
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True
    )

    corrected_population_match = np.allclose(
        stored_population,
        population_corrected,
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True
    )

    silent_neurons = int(
        np.sum(counts.sum(axis=1) == 0)
    )

    silent_presentations = int(
        np.sum(counts.sum(axis=0) == 0)
    )

    print(f"\n{name}")
    print("-" * 60)

    print("Response matrix:", counts.shape)
    print("Silent neurons:", silent_neurons)
    print("Silent presentations:", silent_presentations)

    print("\nStored temporal mean:",
          float(np.nanmean(stored_temporal)))

    print("Uncorrected temporal:",
          mean_with_policy(temporal_raw, "exclude"))

    print("Corrected temporal:",
          mean_with_policy(temporal_corrected, "exclude"))

    print("\nStored population mean:",
          float(np.nanmean(stored_population)))

    print("Uncorrected population, exclude:",
          mean_with_policy(population_raw, "exclude"))

    print("Corrected population, exclude:",
          mean_with_policy(population_corrected, "exclude"))

    print("Corrected population, zero:",
          mean_with_policy(population_corrected, "zero"))

    print("Corrected population, one:",
          mean_with_policy(population_corrected, "one"))

    print("\nStored matches uncorrected:")
    print("Temporal:", raw_temporal_match)
    print("Population:", raw_population_match)

    print("Stored matches corrected:")
    print("Temporal:", corrected_temporal_match)
    print("Population:", corrected_population_match)

    return {
        "population": name,
        "silent_neurons": silent_neurons,
        "silent_presentations": silent_presentations,
        "valid_temporal": int(
            np.isfinite(temporal_corrected).sum()
        ),
        "valid_population": int(
            np.isfinite(population_corrected).sum()
        ),
        "stored_temporal": float(
            np.nanmean(stored_temporal)
        ),
        "stored_population": float(
            np.nanmean(stored_population)
        ),
        "temporal_raw": mean_with_policy(
            temporal_raw, "exclude"
        ),
        "temporal_corrected": mean_with_policy(
            temporal_corrected, "exclude"
        ),
        "population_raw": mean_with_policy(
            population_raw, "exclude"
        ),
        "population_corrected_exclude": mean_with_policy(
            population_corrected, "exclude"
        ),
        "population_corrected_zero": mean_with_policy(
            population_corrected, "zero"
        ),
        "population_corrected_one": mean_with_policy(
            population_corrected, "one"
        ),
        "stored_matches_raw_temporal": raw_temporal_match,
        "stored_matches_raw_population": raw_population_match,
        "stored_matches_corrected_temporal":
            corrected_temporal_match,
        "stored_matches_corrected_population":
            corrected_population_match,
    }


def main():

    print("=" * 65)
    print("FIGURE 10 — SPARSENESS METHODOLOGY AUDIT")
    print("=" * 65)

    rows = []

    for condition, target_e, target_i in CONDITIONS:

        path = (
            BASE / condition / "evaluation"
            / "paper_1000_source"
            / "response_evaluation.npz"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:

            if str(data["condition"]) != condition:
                raise ValueError("Condition mismatch")

            if str(data["protocol"]) != "paper_1000_source":
                raise ValueError("Protocol mismatch")

            duration = float(data["duration"])

            for label, prefix, target in (
                ("E", "E", target_e),
                ("I", "I", target_i),
            ):

                counts = data[f"{prefix}_counts"]

                expected_shape = (
                    400 if prefix == "E" else 100,
                    100
                )

                if counts.shape != expected_shape:
                    raise ValueError(
                        f"Incorrect response shape: {counts.shape}"
                    )

                observed = float(
                    counts.mean() / duration
                )

                stored_rate = float(
                    data[f"{prefix}_rate"]
                )

                if not np.isclose(
                    observed, stored_rate,
                    rtol=1e-10
                ):
                    raise ValueError(
                        "Stored firing rate mismatch"
                    )

                print("\n" + "=" * 65)
                print(
                    condition,
                    label,
                    f"Target {target} Hz",
                    f"Observed {observed:.6f} Hz"
                )

                result = audit_population(
                    label,
                    counts,
                    data[f"{prefix}_temporal"],
                    data[f"{prefix}_population"]
                )

                result["condition"] = condition
                result["target_hz"] = target
                result["observed_hz"] = observed

                rows.append(result)

    OUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUT / "sparseness_method_comparison.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETE")
    print("=" * 65)

    print("Conditions:", len(CONDITIONS))
    print("Population comparisons:", len(rows))
    print("Saved:", csv_path)

    print(
        "\nNo checkpoint or response evaluation "
        "file was modified."
    )


if __name__ == "__main__":
    main()
