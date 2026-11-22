
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT / "results"
    / "figure7_learning_audit"
    / "figure7_learning_audit.npz"
)

OUTPUT = (
    PROJECT / "results"
    / "figure7_learning_audit"
)

CONNECTIONS = [
    "I_to_E",
    "E_to_I",
    "I_to_I",
    "E_to_E"
]

MIN_COUNT = 30
N_BINS = 12


def pearson(x, y):
    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def bin_statistics(x, y, edges):
    means = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for i in range(len(edges) - 1):

        if i == len(edges) - 2:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])

        counts[i] = mask.sum()

        if counts[i] >= MIN_COUNT:
            means[i] = np.mean(y[mask])

    return means, counts


fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 10),
    constrained_layout=True
)

saved_results = {}

with np.load(INPUT, allow_pickle=False) as data:

    for ax, name in zip(axes.flat, CONNECTIONS):

        xi = data[f"{name}_initial_RF_corr"].copy()
        yi = data[f"{name}_initial_weights"].copy()

        xl = data[f"{name}_learned_RF_corr"].copy()
        yl = data[f"{name}_learned_weights"].copy()

        valid_i = np.isfinite(xi) & np.isfinite(yi)
        valid_l = np.isfinite(xl) & np.isfinite(yl)

        xi, yi = xi[valid_i], yi[valid_i]
        xl, yl = xl[valid_l], yl[valid_l]

        # Robust common range: overlap between the central
        # 98% of initial and learned RF correlations.
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
                f"No shared percentile range for {name}"
            )

        mi = (xi >= lower) & (xi <= upper)
        ml = (xl >= lower) & (xl <= upper)

        xi_shared = xi[mi]
        yi_shared = yi[mi]

        xl_shared = xl[ml]
        yl_shared = yl[ml]

        ri = pearson(xi_shared, yi_shared)
        rl = pearson(xl_shared, yl_shared)

        edges = np.linspace(lower, upper, N_BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        mean_i, count_i = bin_statistics(
            xi_shared, yi_shared, edges
        )

        mean_l, count_l = bin_statistics(
            xl_shared, yl_shared, edges
        )

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        print(
            f"Shared range: [{lower:.4f}, {upper:.4f}]"
        )

        print("Initial pairs in range:", len(xi_shared))
        print("Learned pairs in range:", len(xl_shared))

        print("Initial shared-range r:", ri)
        print("Learned shared-range r:", rl)
        print("Change:", rl - ri)

        print("\nBin counts:")
        print("Bin center | Initial | Learned")

        for center, ci, cl in zip(
            centers, count_i, count_l
        ):
            print(
                f"{center:+.3f}     {ci:7d}   {cl:7d}"
            )

        ax.plot(
            centers, mean_i,
            "o-",
            label="Initial",
            markersize=4
        )

        ax.plot(
            centers, mean_l,
            "o-",
            label="Learned",
            markersize=4
        )

        # Display how many pairs are in each bin
        ax2 = ax.twinx()

        ax2.bar(
            centers,
            count_i,
            width=(edges[1] - edges[0]) * 0.32,
            alpha=0.12,
            align="edge"
        )

        ax2.bar(
            centers,
            count_l,
            width=-(edges[1] - edges[0]) * 0.32,
            alpha=0.12,
            align="edge"
        )

        ax2.set_ylabel(
            "Number of pairs",
            color="gray"
        )

        ax.set_title(
            f"{name.replace('_', ' ')}\n"
            f"Initial r={ri:.3f}, Learned r={rl:.3f}"
        )

        ax.set_xlabel("RF correlation (shared range)")
        ax.set_ylabel("Mean synaptic weight")
        ax.set_xlim(lower, upper)

        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)

        saved_results[f"{name}_lower"] = lower
        saved_results[f"{name}_upper"] = upper

        saved_results[f"{name}_r_initial"] = ri
        saved_results[f"{name}_r_learned"] = rl

        saved_results[f"{name}_count_initial"] = count_i
        saved_results[f"{name}_count_learned"] = count_l

        saved_results[f"{name}_mean_initial"] = mean_i
        saved_results[f"{name}_mean_learned"] = mean_l

        saved_results[f"{name}_bin_centers"] = centers


fig.suptitle(
    "Figure 7: Shared-range RF connectivity audit",
    fontsize=14
)

fig.savefig(
    OUTPUT / "figure7_shared_range.png",
    dpi=250
)

plt.close(fig)

np.savez_compressed(
    OUTPUT / "figure7_shared_range_statistics.npz",
    **saved_results
)

print("\nShared-range audit completed.")
print("Results saved to:", OUTPUT)
