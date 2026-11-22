
"""
Figure 7: Shared-range RF connectivity audit.

Adapted from figure7_shared_range_audit.py.

Restricts initial and learned RF-weight relationships
to their overlapping central 98% correlation ranges.

Uses saved Track A/B learning-audit arrays.
No simulation or training.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
    / "figure7_learning_audit"
)

CONNECTIONS = (
    "I_to_E",
    "E_to_I",
    "I_to_I",
    "E_to_E",
)

MIN_COUNT = 30
N_BINS = 12


def pearson(x, y):

    if (
        len(x) < 3
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def bin_statistics(x, y, edges):

    means = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for i in range(len(means)):

        if i == len(means) - 1:
            mask = (
                (x >= edges[i])
                & (x <= edges[i + 1])
            )
        else:
            mask = (
                (x >= edges[i])
                & (x < edges[i + 1])
            )

        counts[i] = np.sum(mask)

        if counts[i] >= MIN_COUNT:
            means[i] = np.mean(y[mask])

    return means, counts


def main(track):

    input_path = (
        BASE
        / f"track_{track}"
        / "figure7_learning_audit.npz"
    )

    output = (
        BASE
        / f"track_{track}"
    )

    figure_path = output / "figure7_shared_range.png"

    stats_path = (
        output / "figure7_shared_range_statistics.npz"
    )

    if figure_path.exists() or stats_path.exists():
        raise FileExistsError(
            "Shared-range results already exist."
        )

    print("=" * 65)
    print(f"FIGURE 7 — SHARED-RANGE AUDIT — TRACK {track.upper()}")
    print("=" * 65)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 10),
        constrained_layout=True
    )

    saved = {}

    with np.load(input_path, allow_pickle=False) as data:

        for ax, name in zip(axes.flat, CONNECTIONS):

            xi = data[f"{name}_initial_RF_corr"].copy()
            yi = data[f"{name}_initial_weights"].copy()

            xl = data[f"{name}_learned_RF_corr"].copy()
            yl = data[f"{name}_learned_weights"].copy()

            valid_i = np.isfinite(xi) & np.isfinite(yi)
            valid_l = np.isfinite(xl) & np.isfinite(yl)

            xi, yi = xi[valid_i], yi[valid_i]
            xl, yl = xl[valid_l], yl[valid_l]

            # Overlap between central 98% of both
            # RF-correlation distributions.
            lower = max(
                np.quantile(xi, 0.01),
                np.quantile(xl, 0.01)
            )

            upper = min(
                np.quantile(xi, 0.99),
                np.quantile(xl, 0.99)
            )

            if lower >= upper:
                raise ValueError(
                    f"No shared range for {name}"
                )

            mask_i = (xi >= lower) & (xi <= upper)
            mask_l = (xl >= lower) & (xl <= upper)

            xi, yi = xi[mask_i], yi[mask_i]
            xl, yl = xl[mask_l], yl[mask_l]

            ri = pearson(xi, yi)
            rl = pearson(xl, yl)

            edges = np.linspace(
                lower,
                upper,
                N_BINS + 1
            )

            centers = (edges[:-1] + edges[1:]) / 2

            mean_i, count_i = bin_statistics(
                xi, yi, edges
            )

            mean_l, count_l = bin_statistics(
                xl, yl, edges
            )

            print(f"\n{name}")
            print("-" * 45)
            print("Shared range:", lower, upper)
            print("Initial pairs:", len(xi))
            print("Learned pairs:", len(xl))
            print("Initial Pearson r:", ri)
            print("Learned Pearson r:", rl)
            print("Change:", rl - ri)

            ax.plot(
                centers,
                mean_i,
                "o-",
                label="Initial",
                markersize=4
            )

            ax.plot(
                centers,
                mean_l,
                "o-",
                label="Learned",
                markersize=4
            )

            ax.set_title(
                f"{name}: Initial r={ri:.3f}, "
                f"Learned r={rl:.3f}"
            )

            ax.set_xlabel("RF correlation (shared range)")
            ax.set_ylabel("Mean synaptic weight")
            ax.set_xlim(lower, upper)
            ax.legend()
            ax.grid(alpha=0.2)

            saved[f"{name}_lower"] = lower
            saved[f"{name}_upper"] = upper
            saved[f"{name}_r_initial"] = ri
            saved[f"{name}_r_learned"] = rl
            saved[f"{name}_count_initial"] = count_i
            saved[f"{name}_count_learned"] = count_l
            saved[f"{name}_mean_initial"] = mean_i
            saved[f"{name}_mean_learned"] = mean_l
            saved[f"{name}_bin_centers"] = centers

    fig.suptitle(
        f"Figure 7 — Track {track.upper()}: "
        "Shared-Range Connectivity Audit",
        fontsize=15
    )

    fig.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    np.savez_compressed(
        stats_path,
        **saved
    )

    print("\nAudit completed.")
    print("Figure:", figure_path)
    print("Statistics:", stats_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=("a", "b"),
        required=True
    )

    args = parser.parse_args()

    main(args.track)
