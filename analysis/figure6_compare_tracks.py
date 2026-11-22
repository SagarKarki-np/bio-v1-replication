
"""
Figure 6: Activity correlations for Track A and Track B.

Uses saved, matched 100-image spike-count matrices.

No simulation or training.
Does not modify existing response files.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.response_correlations import calculate_pairwise_correlations


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
    / "figure6_correlations"
)

INPUT_DIR = BASE / "matched_responses"
OUTPUT_DIR = BASE / "correlation_analysis"

TRACKS = ("a", "b")

POPULATIONS = {
    "LGN": ("LGN_counts", 512),
    "E": ("E_counts", 400),
    "I": ("I_counts", 100),
}


def load_responses(track):

    path = INPUT_DIR / f"track_{track}_responses.npz"

    with np.load(path, allow_pickle=False) as data:

        responses = {}

        for name, (key, n_neurons) in POPULATIONS.items():

            array = data[key].copy()

            assert array.shape == (n_neurons, 100)
            assert np.isfinite(array).all()

            responses[name] = array

        metadata = {
            "duration": float(data["duration"]),
            "seed": int(data["seed"]),
            "noise_std": float(data["noise_std"]),
            "stimulus_path": str(data["stimulus_path"]),
            "checkpoint_path": str(data["checkpoint_path"]),
        }

    return responses, metadata


def summarize_correlations(counts):

    # Each row is a neuron; each column is an image.
    correlation_matrix, pairs = (
        calculate_pairwise_correlations(counts)
    )

    valid = pairs[np.isfinite(pairs)]

    constant_neurons = int(
        np.count_nonzero(
            np.ptp(counts, axis=1) == 0
        )
    )

    if len(valid) == 0:
        raise ValueError(
            "No valid neuron pairs for correlation analysis."
        )

    summary = {
        "total_pairs": len(pairs),
        "valid_pairs": len(valid),
        "constant_neurons": constant_neurons,
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "mean_absolute": float(np.mean(np.abs(valid))),
        "std": float(np.std(valid)),
        "fraction_positive": float(np.mean(valid > 0)),
    }

    return correlation_matrix, pairs, valid, summary


def main():

    print("=" * 65)
    print("FIGURE 6 — PAIRED ACTIVITY CORRELATIONS")
    print("=" * 65)

    datasets = {}
    metadata = {}

    for track in TRACKS:
        datasets[track], metadata[track] = (
            load_responses(track)
        )

    # Verify identical LGN spike counts for both tracks.
    np.testing.assert_array_equal(
        datasets["a"]["LGN"],
        datasets["b"]["LGN"]
    )

    assert metadata["a"]["duration"] == metadata["b"]["duration"]
    assert metadata["a"]["seed"] == metadata["b"]["seed"]
    assert metadata["a"]["noise_std"] == metadata["b"]["noise_std"]
    assert metadata["a"]["stimulus_path"] == metadata["b"]["stimulus_path"]

    print("\nMatched LGN inputs: VERIFIED")
    print("Stimulus seed:", metadata["a"]["seed"])
    print("Duration:", metadata["a"]["duration"])

    results = {}

    for track in TRACKS:

        results[track] = {}

        print(f"\nTRACK {track.upper()}")
        print("-" * 65)

        for population, counts in datasets[track].items():

            C, pairs, valid, summary = (
                summarize_correlations(counts)
            )

            results[track][population] = {
                "matrix": C,
                "pairs": pairs,
                "valid": valid,
                "summary": summary,
            }

            print(f"\n{population} population")
            print("Neurons:", counts.shape[0])
            print("Total pairs:", summary["total_pairs"])
            print("Valid pairs:", summary["valid_pairs"])
            print(
                "Constant-response neurons:",
                summary["constant_neurons"]
            )
            print("Mean correlation:", summary["mean"])
            print("Median correlation:", summary["median"])
            print(
                "Mean absolute correlation:",
                summary["mean_absolute"]
            )
            print(
                "Fraction positive:",
                summary["fraction_positive"]
            )

    # Plot population distributions.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    bins = np.linspace(-1, 1, 51)

    for row, track in enumerate(TRACKS):

        for col, population in enumerate(POPULATIONS):

            ax = axes[row, col]
            result = results[track][population]

            values = result["valid"]
            mean = result["summary"]["mean"]

            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.8,
                color="steelblue",
                edgecolor="none"
            )

            ax.axvline(
                mean,
                color="black",
                linestyle="--",
                label=f"Mean = {mean:.3f}"
            )

            ax.axvline(
                0,
                color="gray",
                linestyle=":",
                linewidth=1
            )

            ax.set_title(
                f"Track {track.upper()} — {population}"
            )
            ax.set_xlabel("Pearson correlation")
            ax.set_xlim(-1, 1)
            ax.legend()
            ax.grid(alpha=0.2)

        axes[row, 0].set_ylabel("Probability density")

    fig.suptitle(
        "Figure 6 — Pairwise Activity Correlations",
        fontsize=16
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_path = (
        OUTPUT_DIR / "figure6_track_comparison.png"
    )

    if figure_path.exists():
        raise FileExistsError(figure_path)

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Save full numeric results with provenance.
    output_file = (
        OUTPUT_DIR / "figure6_correlation_statistics.npz"
    )

    if output_file.exists():
        raise FileExistsError(output_file)

    save_data = {}

    for track in TRACKS:

        for population in POPULATIONS:

            prefix = f"track_{track}_{population}"
            result = results[track][population]

            save_data[f"{prefix}_matrix"] = result["matrix"]
            save_data[f"{prefix}_pairs"] = result["pairs"]

            for key, value in result["summary"].items():

                save_data[f"{prefix}_{key}"] = np.array(
                    value
                )

        save_data[f"track_{track}_checkpoint"] = np.array(
            metadata[track]["checkpoint_path"]
        )

    save_data["stimulus_path"] = np.array(
        metadata["a"]["stimulus_path"]
    )
    save_data["stimulus_seed"] = np.array(
        metadata["a"]["seed"]
    )
    save_data["duration"] = np.array(
        metadata["a"]["duration"]
    )

    np.savez_compressed(
        output_file,
        **save_data
    )

    print("\n" + "=" * 65)
    print("FIGURE 6 ANALYSIS COMPLETE")
    print("=" * 65)
    print("Figure:", figure_path)
    print("Statistics:", output_file)


if __name__ == "__main__":
    main()
