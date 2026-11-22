
"""
Figure 7: Learning-emergence audit for Track A and Track B.

Adapted from analysis/figure7_learning_audit.py.

Compare initial and trained RF-weight relationships
using ON-minus-OFF feedforward weight-derived RFs.

No training or simulation.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_protocol" / "seed_3002"

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)

CONNECTIONS = (
    "I_to_E", "E_to_I", "I_to_I", "E_to_E"
)


def load_weights(path):

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {
            name: data[name].copy()
            for name in WEIGHT_NAMES
        }


def reconstruct_rf(W_LGN):

    assert W_LGN.shape[1] == 512

    return W_LGN[:, :256] - W_LGN[:, 256:]


def normalize_rf(rf):

    rf = np.asarray(rf, dtype=float)

    centered = rf - rf.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)

    valid = (
        np.all(np.isfinite(rf), axis=1)
        & np.isfinite(norms)
        & (norms > 0)
    )

    normalized = np.zeros_like(centered)
    normalized[valid] = (
        centered[valid] / norms[valid, None]
    )

    return normalized, valid


def get_connection_pairs(weights):

    E_RF = reconstruct_rf(weights["W_LGN_E"])
    I_RF = reconstruct_rf(weights["W_LGN_I"])

    E, valid_E = normalize_rf(E_RF)
    I, valid_I = normalize_rf(I_RF)

    C_EI = np.clip(E @ I.T, -1, 1)
    C_EE = np.clip(E @ E.T, -1, 1)
    C_II = np.clip(I @ I.T, -1, 1)

    C_EI[
        ~(valid_E[:, None] & valid_I[None, :])
    ] = np.nan

    C_EE[
        ~(valid_E[:, None] & valid_E[None, :])
    ] = np.nan

    C_II[
        ~(valid_I[:, None] & valid_I[None, :])
    ] = np.nan

    connections = {
        "I_to_E": (C_EI, weights["W_I_E"]),
        "E_to_I": (C_EI.T, weights["W_E_I"]),
        "I_to_I": (C_II, weights["W_I_I"]),
        "E_to_E": (C_EE, weights["W_E_E"])
    }

    results = {}

    for name, (C, W) in connections.items():

        assert C.shape == W.shape

        mask = np.isfinite(C) & np.isfinite(W)

        if name in ("I_to_I", "E_to_E"):
            np.fill_diagonal(mask, False)

        results[name] = (C[mask], W[mask])

    return results


def pearson(x, y):

    if (
        len(x) < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def binned_mean(x, y, bins, min_count=10):

    means = np.full(len(bins) - 1, np.nan)
    counts = np.zeros(len(bins) - 1, dtype=int)

    for i in range(len(means)):

        if i == len(means) - 1:
            mask = (x >= bins[i]) & (x <= bins[i + 1])
        else:
            mask = (x >= bins[i]) & (x < bins[i + 1])

        counts[i] = mask.sum()

        if counts[i] >= min_count:
            means[i] = y[mask].mean()

    return means, counts


def main(track):

    initial_path = (
        BASE / "initialization" / "weights.npz"
    )

    learned_path = (
        BASE / "synaptic_training"
        / f"track_{track}"
        / "checkpoints"
        / "batch_1100.npz"
    )

    output = (
        BASE / "figure7_learning_audit"
        / f"track_{track}"
    )

    figure_path = output / "figure7_learning_comparison.png"
    statistics_path = output / "figure7_learning_audit.npz"

    for path in (figure_path, statistics_path):
        if path.exists():
            raise FileExistsError(path)

    initial = load_weights(initial_path)
    learned = load_weights(learned_path)

    print("=" * 65)
    print(f"FIGURE 7 — LEARNING AUDIT — TRACK {track.upper()}")
    print("=" * 65)

    print("Initial:", initial_path)
    print("Learned:", learned_path)

    initial_pairs = get_connection_pairs(initial)
    learned_pairs = get_connection_pairs(learned)

    bins = np.linspace(-1, 1, 31)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 10),
        constrained_layout=True
    )

    saved = {
        "initial_checkpoint": str(initial_path),
        "learned_checkpoint": str(learned_path),
        "track": track
    }

    for ax, name in zip(axes.flat, CONNECTIONS):

        xi, yi = initial_pairs[name]
        xl, yl = learned_pairs[name]

        ri = pearson(xi, yi)
        rl = pearson(xl, yl)

        print(f"\n{name}")
        print("-" * 45)
        print("Initial Pearson r:", ri)
        print("Learned Pearson r:", rl)
        print("Change:", rl - ri)
        print("Initial mean |RF correlation|:", np.mean(np.abs(xi)))
        print("Learned mean |RF correlation|:", np.mean(np.abs(xl)))
        print("Initial mean weight:", np.mean(yi))
        print("Learned mean weight:", np.mean(yl))

        mi, ni = binned_mean(xi, yi, bins)
        ml, nl = binned_mean(xl, yl, bins)

        ax.plot(
            centers, mi,
            "o-", markersize=3,
            label="Initial"
        )

        ax.plot(
            centers, ml,
            "o-", markersize=3,
            label="Learned"
        )

        ax.set_title(
            f"{name}: Initial r={ri:.3f}, Learned r={rl:.3f}"
        )
        ax.set_xlabel("Weight-derived RF correlation")
        ax.set_ylabel("Mean synaptic weight")
        ax.set_xlim(-1, 1)
        ax.legend()
        ax.grid(alpha=0.2)

        saved[f"{name}_initial_RF_corr"] = xi
        saved[f"{name}_initial_weights"] = yi
        saved[f"{name}_learned_RF_corr"] = xl
        saved[f"{name}_learned_weights"] = yl
        saved[f"{name}_initial_r"] = ri
        saved[f"{name}_learned_r"] = rl
        saved[f"{name}_initial_bin_counts"] = ni
        saved[f"{name}_learned_bin_counts"] = nl

    fig.suptitle(
        f"Figure 7 — Track {track.upper()}: "
        "RF–Weight Relationships Before and After Training"
    )

    output.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight"
    )
    plt.close(fig)

    np.savez_compressed(statistics_path, **saved)

    print("\nAnalysis complete.")
    print("Figure:", figure_path)
    print("Statistics:", statistics_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=("a", "b"),
        required=True
    )

    args = parser.parse_args()

    main(args.track)
