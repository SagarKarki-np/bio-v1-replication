
"""
Figure 10: Gabor morphology comparison.

Compare E02/I04 and E05/I10 using previously saved
100,000-stimulus STAs and Gabor fits.

Measurements:
1. Boundary energy of each receptive field
2. Gabor parameters near optimizer bounds
3. Relationship to fit success and failure

No training, STA simulation, or refitting.
Original results are read-only.
"""

from pathlib import Path
import numpy as np

from gabor_fitting import (
    as_map,
    LOWER,
    UPPER,
    PARAMETER_NAMES,
    check_spatial_order,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
)

CONDITIONS = ("E02_I04", "E05_I10")
ERROR_THRESHOLD = 0.10

check_spatial_order()


def boundary_energy_fraction(rf, width=2):
    """Fraction of squared RF energy within the outer edge."""

    energy = np.asarray(rf, dtype=float) ** 2

    mask = np.zeros((16, 16), dtype=bool)
    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True

    total = np.sum(energy)

    if total <= 0 or not np.isfinite(total):
        return np.nan

    return float(np.sum(energy[mask]) / total)


def parameter_bound_flags(params):
    """Flag parameters within 1% of optimization bounds."""

    params = np.asarray(params, dtype=float)

    if not np.isfinite(params).all():
        return np.zeros(len(LOWER), dtype=bool)

    tolerance = 0.01 * (UPPER - LOWER)

    return (
        (np.abs(params - LOWER) <= tolerance)
        | (np.abs(params - UPPER) <= tolerance)
    )


def load_condition(condition):
    directory = RESULT_ROOT / condition

    sta_path = (
        directory / "sta_100k" / "STA_100000_results.npz"
    )

    fit_path = (
        directory / "gabor_100k" / "gabor_all_400_fits.npz"
    )

    with np.load(sta_path, allow_pickle=False) as data:
        sta = data["STA_E"].copy()
        n_stimuli = int(data["n_stimuli"])

    with np.load(fit_path, allow_pickle=False) as data:
        indices = data["neuron_indices_zero_based"].copy()
        parameters = data["parameters"].copy()
        errors = data["fit_errors"].copy()
        passed = data["passes_10_percent"].copy()
        n_starts = int(data["n_starts"])
        seed = int(data["seed"])

    assert sta.shape == (400, 256)
    assert parameters.shape == (400, 8)
    assert errors.shape == (400,)
    assert np.array_equal(indices, np.arange(400))
    assert n_stimuli == 100000
    assert n_starts == 12
    assert seed == 2705
    assert np.isfinite(errors).all()
    assert np.array_equal(passed, errors < ERROR_THRESHOLD)

    boundary = np.array([
        boundary_energy_fraction(as_map(rf))
        for rf in sta
    ])

    bound_matrix = np.array([
        parameter_bound_flags(params)
        for params in parameters
    ])

    return {
        "sta": sta,
        "errors": errors,
        "passed": passed,
        "boundary": boundary,
        "bound_matrix": bound_matrix,
        "any_bound": bound_matrix.any(axis=1),
    }


def summarize_group(label, data, selection):
    errors = data["errors"][selection]
    boundary = data["boundary"][selection]
    bounds = data["any_bound"][selection]

    print(f"\n{label}")
    print("-" * 50)
    print("Number of neurons:", len(errors))

    if len(errors) == 0:
        return

    print("Median Gabor error:", np.median(errors))
    print("Median boundary energy:", np.nanmedian(boundary))
    print("Mean boundary energy:", np.nanmean(boundary))
    print("Fraction with boundary energy > 0.50:",
          np.mean(boundary > 0.50))
    print("Fraction touching parameter bounds:",
          np.mean(bounds))


def main():
    baseline = load_condition("E02_I04")
    higher = load_condition("E05_I10")

    print("\nFIGURE 10: GABOR MORPHOLOGY COMPARISON")
    print("=" * 60)

    for condition, data in [
        ("E02_I04", baseline),
        ("E05_I10", higher),
    ]:
        print(f"\n{'=' * 60}")
        print(condition)

        passed = data["passed"]

        summarize_group(
            "All neurons",
            data,
            np.ones(400, dtype=bool)
        )

        summarize_group(
            "Successful Gabor fits",
            data,
            passed
        )

        summarize_group(
            "Failed Gabor fits",
            data,
            ~passed
        )

        print("\nParameters near fitting bounds:")
        for name, flags in zip(
            PARAMETER_NAMES,
            data["bound_matrix"].T
        ):
            print(f"  {name}: {int(flags.sum())}/400")

    print("\n" + "=" * 60)
    print("PAIRED CONDITION COMPARISON")
    print("=" * 60)

    a = baseline["passed"]
    b = higher["passed"]

    groups = {
        "Passed both": a & b,
        "Failed both": ~a & ~b,
        "Baseline only": a & ~b,
        "Higher-rate only": ~a & b,
    }

    for label, selection in groups.items():
        print(f"\n{label}: {selection.sum()} neurons")

        for name, data in [
            ("E02/I04", baseline),
            ("E05/I10", higher),
        ]:
            boundary = data["boundary"][selection]
            bounds = data["any_bound"][selection]

            print(
                f"  {name}: "
                f"median boundary={np.nanmedian(boundary):.4f}, "
                f"bound fraction={np.mean(bounds):.3f}"
            )

    print("\nBOUNDARY ENERGY CHANGES")

    delta = higher["boundary"] - baseline["boundary"]

    print("Median change:", np.nanmedian(delta))
    print("Mean change:", np.nanmean(delta))
    print("Increased boundary energy:", np.sum(delta > 0))
    print("Decreased boundary energy:", np.sum(delta < 0))

    print("\nAUDIT COMPLETE")
    print("No existing result files were modified.")


if __name__ == "__main__":
    main()
