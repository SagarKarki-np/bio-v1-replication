
"""Figure 5: structural versus functional RF comparison."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import gabor_fitting as gf

ROOT = (
    Path(__file__).resolve().parents[1]
    / "results/paper_protocol/seed_3002/functional_sta"
)
OUTPUT = ROOT / "figure5_structural_functional"
THRESHOLD = 0.10


def load_track(track):
    base = ROOT / f"track_{track}"

    sta_file = (
        base / "full_image_100000"
        / "STA_100000_results.npz"
    )
    gabor_file = (
        base / "full_image_gabor_160"
        / "gabor_160_comparison_400.npz"
    )

    with np.load(sta_file, allow_pickle=False) as data:
        sta = data["STA_E"].copy()
        weights = data["weight_RF_E"].copy()
        correlations = data["RF_correlations_E"].copy()
        reliability = data["split_half_reliability"].copy()

    with np.load(gabor_file, allow_pickle=False) as data:
        errors = data["refitted_errors_160"].copy()
        indices = data["indices"].copy()

    assert sta.shape == weights.shape == (400, 256)
    assert correlations.shape == reliability.shape == errors.shape == (400,)
    assert np.array_equal(indices, np.arange(400))

    # Independently verify saved correlations.
    recalculated = np.array([
        np.corrcoef(sta[i], weights[i])[0, 1]
        if np.std(sta[i]) > 0 and np.std(weights[i]) > 0
        else np.nan
        for i in range(400)
    ])

    if not np.allclose(
        correlations, recalculated,
        atol=1e-8, rtol=1e-8, equal_nan=True
    ):
        raise ValueError(f"Correlation mismatch for Track {track}")

    return dict(
        sta=sta,
        weights=weights,
        corr=correlations,
        reliability=reliability,
        errors=errors,
        passed=np.isfinite(errors) & (errors < THRESHOLD)
    )


def summary(track, d):
    print("\n" + "=" * 60)
    print(f"TRACK {track.upper()}")
    print("=" * 60)

    for name, mask in (
        ("Passed", d["passed"]),
        ("Failed", ~d["passed"])
    ):
        values = d["corr"][mask]
        print(
            name,
            "n =", int(mask.sum()),
            "| median STA-weight correlation =",
            float(np.nanmedian(values)),
            "| median STA reliability =",
            float(np.nanmedian(d["reliability"][mask]))
        )

    print(
        "Neurons with correlation > 0.8:",
        int(np.sum(d["corr"] > 0.8))
    )


def make_figure(results, filename):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {"a": "tab:blue", "b": "tab:orange"}

    # A: Correlation distributions
    ax = axes[0, 0]
    bins = np.linspace(-1, 1, 41)

    for track in ("a", "b"):
        c = results[track]["corr"]
        ax.hist(
            c[np.isfinite(c)],
            bins=bins, alpha=0.5,
            label=f"Track {track.upper()}",
            color=colors[track]
        )

    ax.set_xlabel("STA?weight Pearson correlation")
    ax.set_ylabel("Neurons")
    ax.set_title("A. Structural?functional agreement")
    ax.legend()

    # B: Correlation versus Gabor error
    ax = axes[0, 1]

    for track in ("a", "b"):
        d = results[track]
        ax.scatter(
            d["corr"], d["errors"],
            s=13, alpha=0.45,
            color=colors[track],
            label=f"Track {track.upper()}"
        )

    ax.axhline(THRESHOLD, linestyle="--", color="black")
    ax.set_xlabel("STA?weight Pearson correlation")
    ax.set_ylabel("160?160 Gabor fitting error")
    ax.set_title("B. Agreement versus Gabor error")
    ax.legend()

    # C: Pass versus failure correlation
    ax = axes[1, 0]
    groups = []
    labels = []

    for track in ("a", "b"):
        d = results[track]
        for name, mask in (
            ("Pass", d["passed"]),
            ("Fail", ~d["passed"])
        ):
            groups.append(d["corr"][mask & np.isfinite(d["corr"])])
            labels.append(f"{track.upper()} {name}")

    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("STA?weight Pearson correlation")
    ax.set_title("C. Agreement by fitting outcome")

    # D: Reliability versus agreement
    ax = axes[1, 1]

    for track in ("a", "b"):
        d = results[track]
        ax.scatter(
            d["reliability"], d["corr"],
            s=13, alpha=0.45,
            color=colors[track],
            label=f"Track {track.upper()}"
        )

    ax.set_xlabel("STA split-half reliability")
    ax.set_ylabel("STA?weight Pearson correlation")
    ax.set_title("D. Reliability versus agreement")
    ax.legend()

    fig.suptitle(
        "Figure 5 ? Structural and functional RF comparison",
        fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(filename, dpi=250)
    plt.close(fig)


def main():
    gf.check_spatial_order()

    results = {
        "a": load_track("a"),
        "b": load_track("b")
    }

    for track, data in results.items():
        summary(track, data)

    figure = OUTPUT / "structural_functional_comparison.png"
    csv_file = OUTPUT / "per_neuron_comparison.csv"

    for path in (figure, csv_file):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    make_figure(results, figure)

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track", "neuron", "sta_weight_correlation",
            "sta_split_half_reliability",
            "gabor_error_160", "gabor_passed"
        ])

        for track, d in results.items():
            for i in range(400):
                writer.writerow([
                    track.upper(), i + 1,
                    d["corr"][i],
                    d["reliability"][i],
                    d["errors"][i],
                    bool(d["passed"][i])
                ])

    print("\nSaved:", figure)
    print("Saved:", csv_file)


if __name__ == "__main__":
    main()
