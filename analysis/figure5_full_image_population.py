
"""Figure 5: population Gabor fitting analysis for both tracks."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results" / "paper_protocol" / "seed_3002"
    / "functional_sta"
)

OUTPUT = ROOT / "figure5_full_image_population"
THRESHOLD = 0.10


def load_track(track):
    base = ROOT / f"track_{track}"

    fit16 = (
        base / "full_image_gabor_16"
        / "gabor_all_400_fits.npz"
    )

    fit160 = (
        base / "full_image_gabor_160"
        / "gabor_160_comparison_400.npz"
    )

    with np.load(fit16, allow_pickle=False) as data:
        errors16 = data["fit_errors"].copy()

    with np.load(fit160, allow_pickle=False) as data:
        errors160 = data["refitted_errors_160"].copy()
        original16 = data["original_errors_16"].copy()
        indices = data["indices"].copy()

    if errors16.shape != (400,) or errors160.shape != (400,):
        raise ValueError(f"Unexpected shapes for Track {track}")

    if not np.array_equal(indices, np.arange(400)):
        raise ValueError(f"Unexpected neuron ordering: Track {track}")

    if not np.allclose(
        errors16,
        original16,
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True
    ):
        raise ValueError(
            f"Initial Gabor errors disagree: Track {track}"
        )

    return errors16, errors160


def empirical_cdf(errors):
    values = np.sort(errors[np.isfinite(errors)])
    fractions = np.arange(1, len(values) + 1) / len(values)
    return values, fractions


def main():
    a16, a160 = load_track("a")
    b16, b160 = load_track("b")

    colors = {
        "A": "tab:blue",
        "B": "tab:orange",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Error distributions
    ax = axes[0, 0]

    bins = np.linspace(0, 0.5, 45)

    for track, errors in (("A", a160), ("B", b160)):
        ax.hist(
            errors[np.isfinite(errors)],
            bins=bins,
            alpha=0.5,
            label=f"Track {track}",
            color=colors[track]
        )

    ax.axvline(
        THRESHOLD, color="black", linestyle="--",
        label="10% error threshold"
    )
    ax.set_xlabel("Normalized squared fitting error")
    ax.set_ylabel("Number of neurons")
    ax.set_title("A. Gabor fitting-error distributions")
    ax.legend()

    # Panel B: Empirical CDF
    ax = axes[0, 1]

    for track, errors in (("A", a160), ("B", b160)):
        x, y = empirical_cdf(errors)
        ax.step(
            x, y, where="post",
            label=f"Track {track}",
            color=colors[track]
        )

    ax.axvline(
        THRESHOLD, color="black", linestyle="--"
    )
    ax.set_xlabel("Normalized squared fitting error")
    ax.set_ylabel("Cumulative fraction of neurons")
    ax.set_title("B. Cumulative error distributions")
    ax.legend()

    # Panel C: Pass/fail proportions
    ax = axes[1, 0]

    passed = [
        np.sum(a160 < THRESHOLD),
        np.sum(b160 < THRESHOLD)
    ]

    failed = [400 - value for value in passed]

    labels = ["Track A", "Track B"]

    ax.bar(labels, passed, label="Passed")
    ax.bar(
        labels, failed, bottom=passed,
        label="Failed", alpha=0.4
    )

    for i, count in enumerate(passed):
        ax.text(
            i, count / 2,
            f"{count}/400\n({count / 4:.2f}%)",
            ha="center", va="center",
            fontweight="bold"
        )

    ax.set_ylim(0, 420)
    ax.set_ylabel("Excitatory neurons")
    ax.set_title("C. Gabor fitting success")
    ax.legend()

    # Panel D: Paired original and interpolated errors
    ax = axes[1, 1]

    for track, e16, e160 in (
        ("A", a16, a160),
        ("B", b16, b160)
    ):
        valid = np.isfinite(e16) & np.isfinite(e160)

        ax.scatter(
            e16[valid],
            e160[valid],
            s=13,
            alpha=0.45,
            label=f"Track {track}",
            color=colors[track]
        )

    ax.plot(
        [0, 0.5], [0, 0.5],
        linestyle="--", color="black",
        linewidth=1
    )

    ax.axhline(
        THRESHOLD, color="gray", linestyle=":"
    )
    ax.axvline(
        THRESHOLD, color="gray", linestyle=":"
    )

    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Initial 16x16 fitting error")
    ax.set_ylabel("Reoptimized 160x160 fitting error")
    ax.set_title("D. Paired fitting errors")
    ax.legend()

    fig.suptitle(
        "Figure 5 ? Standardized full-image STA: "
        "population Gabor analysis",
        fontsize=14
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT / "gabor_population_comparison.png"

    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_file}"
        )

    fig.savefig(output_file, dpi=250)
    plt.close(fig)

    for track, e16, e160 in (
        ("A", a16, a160),
        ("B", b16, b160)
    ):
        print(f"\nTRACK {track}")
        print("16x16 passes:", int(np.sum(e16 < THRESHOLD)))
        print("160x160 passes:", int(np.sum(e160 < THRESHOLD)))
        print("Median initial error:", np.nanmedian(e16))
        print("Median refitted error:", np.nanmedian(e160))
        print(
            "Numerically lower error:",
            int(np.sum(e160 < e16))
        )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
