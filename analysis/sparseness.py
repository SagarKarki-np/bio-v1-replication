
"""
Sparseness analysis for the LGN-V1 replication.

Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Two analysis workflows:

1. Figure 3:
   ON, OFF, excitatory and inhibitory populations.
   Uses the published sparseness equations.

2. Legacy:
   Initial-versus-learned excitatory sparseness.
   Preserves the existing finite-size-corrected analysis.

No network simulation or training is performed.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent

import os as _os

# Optional: set V1_ORIGINAL_PROJECT to a folder holding earlier archived data.
ORIGINAL_PROJECT = Path(_os.environ.get("V1_ORIGINAL_PROJECT", str(PROJECT_DIR)))

DEFAULT_INPUT = (
    ORIGINAL_PROJECT
    / "results"
    / "sparsity"
    / "matched_responses.npz"
)

FIGURE3_OUTPUT = (
    PROJECT_DIR / "results" / "figure3_seed3001_audit"
)

LEGACY_OUTPUT = (
    PROJECT_DIR / "results" / "sparsity"
)

PRESENTATION_DURATION = 0.4

PUBLISHED_TEMPORAL = {
    "ON": 0.666,
    "OFF": 0.688,
    "E": 0.889,
    "I": 0.851,
}

PUBLISHED_POPULATION = {
    "ON": 0.656,
    "OFF": 0.678,
    "E": 0.891,
    "I": 0.836,
}


# ============================================================
# 1. LOAD ARCHIVED RESPONSES
# ============================================================

def load_responses(path):
    """Load all archived LGN and cortical spike counts."""

    with np.load(path, allow_pickle=False) as data:
        required = (
            "LGN_counts",
            "initial_E",
            "initial_I",
            "learned_E",
            "learned_I",
        )

        missing = [
            name for name in required
            if name not in data.files
        ]

        if missing:
            raise KeyError(
                f"Missing response fields: {missing}"
            )

        arrays = {
            name: data[name].copy()
            for name in required
        }

    expected = {
        "LGN_counts": (512, 100),
        "initial_E": (400, 100),
        "initial_I": (100, 100),
        "learned_E": (400, 100),
        "learned_I": (100, 100),
    }

    for name, shape in expected.items():
        R = arrays[name]

        if R.shape != shape:
            raise ValueError(
                f"{name}: expected {shape}, got {R.shape}"
            )

        if not np.isfinite(R).all() or np.any(R < 0):
            raise ValueError(
                f"{name} contains invalid responses."
            )

    return arrays


# ============================================================
# 2. SPARSENESS CALCULATIONS
# ============================================================

def sparseness(R, axis, finite_correction=False):
    """
    Calculate the Treves-Rolls sparseness metric.

    Published form:
        S = 1 - mean(r)^2 / mean(r^2)

    axis=0: across neurons (population sparseness)
    axis=1: across images (temporal sparseness)

    Silent groups are assigned NaN.

    finite_correction=True preserves the older analysis.
    """

    R = np.asarray(R, dtype=float)

    mean_r = np.mean(R, axis=axis)
    mean_r2 = np.mean(R ** 2, axis=axis)

    S = np.divide(
        mean_r ** 2,
        mean_r2,
        out=np.full_like(mean_r, np.nan),
        where=mean_r2 > 0,
    )

    S = 1.0 - S

    if finite_correction:
        n = R.shape[axis]
        S = S / (1.0 - 1.0 / n)

    return S


def summarize_population(R):
    """Compute paper-form sparseness statistics."""

    temporal = sparseness(R, axis=1)
    population = sparseness(R, axis=0)

    return {
        "temporal_values": temporal,
        "population_values": population,
        "temporal_mean": float(
            np.nanmean(temporal)
        ),
        "population_mean": float(
            np.nanmean(population)
        ),
        "silent_neurons": int(
            np.isnan(temporal).sum()
        ),
        "silent_presentations": int(
            np.isnan(population).sum()
        ),
        "mean_rate_hz": float(R.mean()),
    }


# ============================================================
# 3. SORTED RESPONSE CURVES
# ============================================================

def temporal_curve(R):
    """
    Figure 3B.

    Sort each neuron's responses across images,
    then average the sorted rows.
    """
    return np.sort(R, axis=1).mean(axis=0)


def population_curve(R):
    """
    Figure 3C.

    Sort neuronal responses independently for each
    image, then average the sorted columns.
    """
    return np.sort(R, axis=0).mean(axis=1)


def normalized_rank(n):
    """Rank coordinates from 0 to 1."""
    return np.linspace(0.0, 1.0, n)


# ============================================================
# 4. FIGURE 3A: RESPONSE MATRICES
# ============================================================

def plot_response_matrices(populations, path):
    """
    Show 30 neurons responding to 30 presentations.

    The subsets are the first 30 neurons and images.
    """

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9)
    )

    for ax, (name, R) in zip(
        axes.flat, populations.items()
    ):
        subset = R[:30, :30]

        im = ax.imshow(
            subset,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
        )

        ax.set_title(f"{name} population")
        ax.set_xlabel("Image presentation")
        ax.set_ylabel("Neuron index")

        fig.colorbar(
            im, ax=ax, label="Firing rate (Hz)"
        )

    fig.suptitle(
        "Figure 3A: Neural responses across natural images"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================
# 5. FIGURE 3B AND 3C
# ============================================================

def plot_sparseness_curves(
    populations, statistics, path
):
    """Plot all four population response curves."""

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5)
    )

    for name, R in populations.items():
        stats = statistics[name]

        curve_T = temporal_curve(R)
        curve_P = population_curve(R)

        axes[0].plot(
            normalized_rank(curve_T.size),
            curve_T,
            label=(
                f"{name} "
                f"(S={stats['temporal_mean']:.3f})"
            ),
            linewidth=1.8,
        )

        axes[1].plot(
            normalized_rank(curve_P.size),
            curve_P,
            label=(
                f"{name} "
                f"(S={stats['population_mean']:.3f})"
            ),
            linewidth=1.8,
        )

    axes[0].set_title(
        "B. Temporal sparseness"
    )
    axes[0].set_xlabel(
        "Images ranked least to most preferred"
    )
    axes[0].set_ylabel("Mean firing rate (Hz)")

    axes[1].set_title(
        "C. Population sparseness"
    )
    axes[1].set_xlabel(
        "Neurons ranked least to most responsive"
    )
    axes[1].set_ylabel("Mean firing rate (Hz)")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)

    fig.suptitle(
        "Figure 3: Sparseness in ON, OFF, E and I populations"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================
# 6. FIGURE 3 COMPLETE ANALYSIS
# ============================================================

def run_figure3(input_path, output_dir):

    arrays = load_responses(input_path)

    # Spike counts converted to firing rates in Hz.
    populations = {
        "ON": arrays["LGN_counts"][:256]
              / PRESENTATION_DURATION,

        "OFF": arrays["LGN_counts"][256:]
               / PRESENTATION_DURATION,

        "E": arrays["learned_E"]
             / PRESENTATION_DURATION,

        "I": arrays["learned_I"]
             / PRESENTATION_DURATION,
    }

    statistics = {
        name: summarize_population(R)
        for name, R in populations.items()
    }

    print("\nFIGURE 3: PUBLISHED SPARSENESS METRICS")
    print("=" * 75)

    print(
        f"{'Population':<12}"
        f"{'Temporal':>11}"
        f"{'Paper T':>11}"
        f"{'Population':>13}"
        f"{'Paper P':>11}"
    )

    for name, stats in statistics.items():
        print(
            f"{name:<12}"
            f"{stats['temporal_mean']:>11.4f}"
            f"{PUBLISHED_TEMPORAL[name]:>11.4f}"
            f"{stats['population_mean']:>13.4f}"
            f"{PUBLISHED_POPULATION[name]:>11.4f}"
        )

    print("\nACTIVITY AND VALIDITY")

    for name, stats in statistics.items():
        print(
            f"{name}: "
            f"mean rate={stats['mean_rate_hz']:.3f} Hz, "
            f"silent neurons={stats['silent_neurons']}, "
            f"silent presentations="
            f"{stats['silent_presentations']}"
        )

    output_dir = Path(output_dir)

    outputs = (
        output_dir / "figure3_response_matrices.png",
        output_dir / "figure3_sparseness_curves.png",
        output_dir / "figure3_statistics.npz",
    )

    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Figure 3 output already exists. "
            "Use another output directory."
        )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    plot_response_matrices(
        populations, outputs[0]
    )

    plot_sparseness_curves(
        populations, statistics, outputs[1]
    )

    archive = {
        "presentation_duration": PRESENTATION_DURATION,
        "input_path": str(input_path),
    }

    for name, R in populations.items():
        archive[f"{name}_rates"] = R
        archive[f"{name}_temporal_curve"] = (
            temporal_curve(R)
        )
        archive[f"{name}_population_curve"] = (
            population_curve(R)
        )

        for key, value in statistics[name].items():
            archive[f"{name}_{key}"] = value

    np.savez_compressed(
        outputs[2], **archive
    )

    print("\nFigure 3 outputs saved:")
    for path in outputs:
        print(path)


# ============================================================
# 7. PRESERVE ORIGINAL E-ONLY SPARSENESS ANALYSIS
# ============================================================

def save_distribution(initial, learned, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 31)

    ax.hist(
        initial[np.isfinite(initial)],
        bins=bins,
        alpha=0.6,
        label="Initial",
    )
    ax.hist(
        learned[np.isfinite(learned)],
        bins=bins,
        alpha=0.6,
        label="Learned",
    )
    ax.set_title(title)
    ax.set_xlabel("Sparseness")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_paired(initial, learned, title, path):
    valid = (
        np.isfinite(initial)
        & np.isfinite(learned)
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(initial[valid], learned[valid], alpha=0.65)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Initial sparseness")
    ax.set_ylabel("Learned sparseness")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_legacy(input_path, output_dir):
    """
    Preserve original finite-size-corrected E analysis.

    Saves the same four plots and two NPZ archives
    as the previous sparseness.py workflow.
    """
    arrays = load_responses(input_path)

    initial = arrays["initial_E"]
    learned = arrays["learned_E"]

    P0 = sparseness(initial, axis=0, finite_correction=True)
    P1 = sparseness(learned, axis=0, finite_correction=True)

    T0 = sparseness(initial, axis=1, finite_correction=True)
    T1 = sparseness(learned, axis=1, finite_correction=True)

    F0 = (initial > 0).mean(axis=0)
    F1 = (learned > 0).mean(axis=0)

    output_dir = Path(output_dir)

    names = (
        "population_sparseness.npz",
        "lifetime_sparseness.npz",
        "population_sparseness.png",
        "population_sparseness_paired.png",
        "lifetime_sparseness.png",
        "lifetime_sparseness_paired.png",
    )

    if any((output_dir / name).exists() for name in names):
        raise FileExistsError(
            "Legacy outputs already exist. "
            "They will not be overwritten."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / names[0],
        S_initial=P0,
        S_learned=P1,
        fraction_initial=F0,
        fraction_learned=F1,
    )

    np.savez_compressed(
        output_dir / names[1],
        S_initial=T0,
        S_learned=T1,
    )

    save_distribution(
        P0, P1,
        "Population Sparseness Before and After Learning",
        output_dir / names[2],
    )

    save_paired(
        P0, P1,
        "Paired Population Sparseness",
        output_dir / names[3],
    )

    save_distribution(
        T0, T1,
        "Lifetime Sparseness Before and After Learning",
        output_dir / names[4],
    )

    save_paired(
        T0, T1,
        "Paired Lifetime Sparseness",
        output_dir / names[5],
    )

    print("Legacy sparseness analysis completed.")
    print("Results saved to:", output_dir)


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LGN-V1 sparseness analysis"
    )

    parser.add_argument(
        "--mode",
        choices=("figure3", "legacy"),
        default="figure3",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.mode == "figure3":
        output = (
            args.output_dir
            if args.output_dir is not None
            else FIGURE3_OUTPUT
        )

        run_figure3(args.input, output)

    else:
        output = (
            args.output_dir
            if args.output_dir is not None
            else LEGACY_OUTPUT
        )

        run_legacy(args.input, output)
