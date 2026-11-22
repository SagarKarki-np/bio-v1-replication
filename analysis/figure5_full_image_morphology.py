
"""Figure 5: full-population RF morphology audit.

Read-only analysis of saved full-image STA and Gabor fits.
No network simulations or Gabor optimization.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import gabor_fitting as gf
from analysis import figure5_interpolation_refit as refit


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results" / "paper_protocol" / "seed_3002"
    / "functional_sta"
)

OUTPUT = ROOT / "figure5_full_image_morphology"
THRESHOLD = 0.10


def boundary_energy_fraction(rf, width=2):
    """Fraction of squared RF energy in the outer pixel border."""
    rf = np.asarray(rf, dtype=float)

    if rf.shape != (16, 16):
        raise ValueError(f"Unexpected RF shape: {rf.shape}")

    energy = rf ** 2
    mask = np.zeros((16, 16), dtype=bool)

    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True

    total = float(np.sum(energy))

    if total <= 0 or not np.isfinite(total):
        return np.nan

    return float(np.sum(energy[mask]) / total)


def near_parameter_bounds(params, scale):
    """Check 1%-of-range proximity in optimizer units."""
    p = np.asarray(params, dtype=float).copy()

    if scale <= 0 or not np.isfinite(scale):
        return np.zeros(len(gf.LOWER), dtype=bool)

    # Refitted amplitude is stored in original STA units.
    # The optimizer used amplitude divided by the STA scale.
    p[0] /= scale

    tolerance = 0.01 * (gf.UPPER - gf.LOWER)

    return (
        (np.abs(p - gf.LOWER) <= tolerance)
        | (np.abs(p - gf.UPPER) <= tolerance)
    )


def load_track(track):
    base = ROOT / f"track_{track}"

    sta_path = (
        base / "full_image_100000"
        / "STA_100000_results.npz"
    )

    fit_path = (
        base / "full_image_gabor_160"
        / "gabor_160_comparison_400.npz"
    )

    with np.load(sta_path, allow_pickle=False) as data:
        sta = data["STA_E"].copy()

    with np.load(fit_path, allow_pickle=False) as data:
        params = data["parameters_160"].copy()
        errors = data["refitted_errors_160"].copy()
        indices = data["indices"].copy()

    if sta.shape != (400, 256):
        raise ValueError(f"Unexpected STA shape: {sta.shape}")

    if params.shape != (400, 8):
        raise ValueError(f"Unexpected parameter shape: {params.shape}")

    if errors.shape != (400,):
        raise ValueError(f"Unexpected error shape: {errors.shape}")

    if not np.array_equal(indices, np.arange(400)):
        raise ValueError("Neuron ordering mismatch.")

    boundary = np.full(400, np.nan)
    bounds = np.zeros((400, 8), dtype=bool)
    computed_errors = np.full(400, np.nan)

    for neuron in range(400):
        rf16 = gf.as_map(sta[neuron])
        rf160 = refit.interpolate_160(rf16)

        predicted160 = refit.gabor_on_grid(
            params[neuron],
            refit.X160,
            refit.Y160
        ).reshape(160, 160)

        energy = float(np.sum(rf160 ** 2))

        if energy > 0 and np.isfinite(energy):
            residual = rf160 - predicted160
            computed_errors[neuron] = (
                float(np.sum(residual ** 2)) / energy
            )

        boundary[neuron] = boundary_energy_fraction(rf16)

        scale = float(np.max(np.abs(rf16)))
        bounds[neuron] = near_parameter_bounds(
            params[neuron], scale
        )

    if not np.allclose(
        computed_errors, errors,
        rtol=1e-4, atol=1e-6,
        equal_nan=True
    ):
        difference = np.nanmax(
            np.abs(computed_errors - errors)
        )
        raise ValueError(
            f"Saved error validation failed for "
            f"Track {track}: max difference={difference}"
        )

    return {
        "errors": errors,
        "boundary": boundary,
        "bounds": bounds,
        "any_bound": bounds.any(axis=1),
        "passed": np.isfinite(errors) & (errors < THRESHOLD),
    }


def summarize(track, data):
    passed = data["passed"]
    failed = ~passed

    print("\n" + "=" * 65)
    print(f"TRACK {track.upper()} ? MORPHOLOGY SUMMARY")
    print("=" * 65)

    print("Passed:", int(passed.sum()))
    print("Failed:", int(failed.sum()))

    for label, mask in (
        ("Successful", passed),
        ("Failed", failed)
    ):
        boundary = data["boundary"][mask]
        bound_flags = data["any_bound"][mask]

        print(f"\n{label} neurons: {int(mask.sum())}")
        print(
            "Median boundary energy:",
            float(np.nanmedian(boundary))
        )
        print(
            "Boundary energy > 50%:",
            int(np.sum(boundary > 0.5))
        )
        print(
            "Near any parameter bound:",
            int(bound_flags.sum())
        )
        print(
            "Median Gabor error:",
            float(np.nanmedian(data["errors"][mask]))
        )

    print("\nParameters near bounds, all 400:")
    for name, count in zip(
        gf.PARAMETER_NAMES,
        data["bounds"].sum(axis=0)
    ):
        print(f"  {name}: {int(count)}")


def make_figure(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = {"a": "tab:blue", "b": "tab:orange"}

    # A. Boundary energy distributions
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 31)

    for track in ("a", "b"):
        values = results[track]["boundary"]
        ax.hist(
            values[np.isfinite(values)],
            bins=bins,
            alpha=0.5,
            label=f"Track {track.upper()}",
            color=colors[track]
        )

    ax.set_xlabel("Boundary energy fraction")
    ax.set_ylabel("Number of neurons")
    ax.set_title("A. RF energy near patch boundaries")
    ax.legend()

    # B. Boundary energy versus Gabor error
    ax = axes[0, 1]

    for track in ("a", "b"):
        data = results[track]
        ax.scatter(
            data["boundary"],
            data["errors"],
            alpha=0.4,
            s=12,
            color=colors[track],
            label=f"Track {track.upper()}"
        )

    ax.axhline(THRESHOLD, color="black", linestyle="--")
    ax.set_xlabel("Boundary energy fraction")
    ax.set_ylabel("160x160 normalized Gabor error")
    ax.set_title("B. Boundary energy versus fitting error")
    ax.legend()

    # C. Group medians
    ax = axes[1, 0]

    positions = [0, 1, 3, 4]
    values = []
    labels = []

    for track in ("a", "b"):
        data = results[track]

        for group_name, mask in (
            ("Pass", data["passed"]),
            ("Fail", ~data["passed"])
        ):
            values.append(
                float(np.nanmedian(data["boundary"][mask]))
            )
            labels.append(
                f"{track.upper()}\n{group_name}"
            )

    ax.bar(
        positions,
        values,
        color=[
            colors["a"], colors["a"],
            colors["b"], colors["b"]
        ],
        alpha=0.75
    )

    ax.set_xticks(positions, labels)
    ax.set_ylabel("Median boundary energy fraction")
    ax.set_title("C. Boundary energy by fitting outcome")
    ax.set_ylim(0, 1)

    # D. Percentage near parameter bounds
    ax = axes[1, 1]

    percentages = []

    for track in ("a", "b"):
        data = results[track]

        for mask in (data["passed"], ~data["passed"]):
            percentage = (
                100 * np.mean(data["any_bound"][mask])
                if mask.any() else np.nan
            )
            percentages.append(percentage)

    ax.bar(
        positions,
        percentages,
        color=[
            colors["a"], colors["a"],
            colors["b"], colors["b"]
        ],
        alpha=0.75
    )

    ax.set_xticks(positions, labels)
    ax.set_ylabel("Neurons near parameter bounds (%)")
    ax.set_ylim(0, 100)
    ax.set_title("D. Parameter-bound proximity")

    fig.suptitle(
        "Figure 5 ? Receptive-field morphology audit",
        fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=250)
    plt.close(fig)


def main():
    gf.check_spatial_order()

    results = {
        "a": load_track("a"),
        "b": load_track("b"),
    }

    for track, data in results.items():
        summarize(track, data)

    figure_path = OUTPUT / "morphology_comparison.png"
    statistics_path = OUTPUT / "morphology_statistics.npz"
    csv_path = OUTPUT / "morphology_per_neuron.csv"

    for path in (figure_path, statistics_path, csv_path):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    make_figure(results, figure_path)

    np.savez_compressed(
        statistics_path,
        threshold=THRESHOLD,
        parameter_names=np.asarray(gf.PARAMETER_NAMES),
        track_a_errors=results["a"]["errors"],
        track_b_errors=results["b"]["errors"],
        track_a_boundary=results["a"]["boundary"],
        track_b_boundary=results["b"]["boundary"],
        track_a_bounds=results["a"]["bounds"],
        track_b_bounds=results["b"]["bounds"],
        track_a_passed=results["a"]["passed"],
        track_b_passed=results["b"]["passed"]
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track", "neuron_id", "gabor_error_160",
            "passed", "boundary_energy",
            "near_any_parameter_bound"
        ])

        for track, data in results.items():
            for i in range(400):
                writer.writerow([
                    track.upper(),
                    i + 1,
                    data["errors"][i],
                    bool(data["passed"][i]),
                    data["boundary"][i],
                    bool(data["any_bound"][i])
                ])

    print("\nSaved:", figure_path)
    print("Saved:", statistics_path)
    print("Saved:", csv_path)
    print("\nMorphology audit complete.")


if __name__ == "__main__":
    main()
