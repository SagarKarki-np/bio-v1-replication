
"""
Compare final learned LGN-to-V1 receptive-field weight maps.

Track A: original 100-batch calibration.
Track B: extended 680-batch calibration.

This is a feedforward weight-map analysis, NOT
a functional spike-triggered-average (STA).

No training checkpoints are modified.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import figure10_config
from analysis import figure10_train as tr


PROJECT = Path(__file__).resolve().parents[1]

NETWORK_SEED = 3002
TRAINING_SEED = 303002
BATCH_SIZE = 100
FINAL_BATCH = 1100
PATCH_SIZE = 16
N_PIXELS = PATCH_SIZE ** 2

ROOT = (
    PROJECT / "results" / "paper_protocol"
    / f"seed_{NETWORK_SEED}"
)

OUTPUT = (
    ROOT / "receptive_field_comparison"
)

condition = figure10_config.make_condition(
    2.0,
    parameterization="paper"
)


def load_track(track):
    folder = (
        ROOT / "synaptic_training"
        / f"track_{track.lower()}"
    )

    (
        completed,
        weights,
        theta_E,
        theta_I,
        history_E,
        history_I
    ) = tr.load_latest_checkpoint(
        folder,
        condition,
        BATCH_SIZE,
        TRAINING_SEED
    )

    if completed != FINAL_BATCH:
        raise ValueError(
            f"Track {track}: expected batch "
            f"{FINAL_BATCH}, got {completed}"
        )

    return weights


def make_weight_fields(matrix):
    """
    Each row is one V1 neuron's LGN input weights.

    ON:  columns 0:256
    OFF: columns 256:512

    Returns ON, OFF, and ON-minus-OFF maps.
    """

    if matrix.ndim != 2:
        raise ValueError("Expected 2D weight matrix")

    if matrix.shape[1] != 2 * N_PIXELS:
        raise ValueError(
            f"Unexpected LGN dimension: {matrix.shape}"
        )

    n_neurons = matrix.shape[0]

    on = matrix[:, :N_PIXELS].reshape(
        n_neurons, PATCH_SIZE, PATCH_SIZE,
        order="C"
    )

    off = matrix[:, N_PIXELS:].reshape(
        n_neurons, PATCH_SIZE, PATCH_SIZE,
        order="C"
    )

    return on, off, on - off


def compare_population(name, key, weights_A, weights_B):

    on_A, off_A, rf_A = make_weight_fields(
        weights_A[key]
    )

    on_B, off_B, rf_B = make_weight_fields(
        weights_B[key]
    )

    n = rf_A.shape[0]

    # Spatial contrast strength:
    # standard deviation across 16x16 positions.
    contrast_A = rf_A.reshape(n, -1).std(axis=1)
    contrast_B = rf_B.reshape(n, -1).std(axis=1)

    # Select neuron identities by their combined
    # spatial variability, without choosing
    # different neurons for the two tracks.
    combined = (contrast_A + contrast_B) / 2

    indices = np.argsort(combined)[-12:][::-1]

    # Compare spatial patterns of the same neuron
    # across tracks using Pearson correlation.
    correlations = []

    for neuron in range(n):
        a = rf_A[neuron].ravel()
        b = rf_B[neuron].ravel()

        if a.std() > 0 and b.std() > 0:
            corr = np.corrcoef(a, b)[0, 1]
        else:
            corr = np.nan

        correlations.append(corr)

    correlations = np.asarray(correlations)

    print(f"\n{name} NEURONS")
    print("-" * 55)

    print("Number of neurons:", n)

    print(
        "Mean spatial contrast — Track A:",
        float(contrast_A.mean())
    )

    print(
        "Mean spatial contrast — Track B:",
        float(contrast_B.mean())
    )

    print(
        "Mean absolute A-B RF difference:",
        float(np.abs(rf_B - rf_A).mean())
    )

    print(
        "Mean same-neuron spatial correlation:",
        float(np.nanmean(correlations))
    )

    print("Displayed neuron indices:", indices.tolist())

    # Save full numeric maps.
    np.savez_compressed(
        OUTPUT / f"{name.lower()}_weight_fields.npz",
        RF_A=rf_A,
        RF_B=rf_B,
        ON_A=on_A,
        OFF_A=off_A,
        ON_B=on_B,
        OFF_B=off_B,
        contrast_A=contrast_A,
        contrast_B=contrast_B,
        spatial_correlations=correlations,
        displayed_indices=indices
    )

    # Shared color scale across both tracks.
    displayed = np.concatenate(
        [rf_A[indices], rf_B[indices]],
        axis=0
    )

    limit = float(np.max(np.abs(displayed)))

    if limit == 0:
        limit = 1.0

    fig, axes = plt.subplots(
        2, len(indices),
        figsize=(18, 4.5),
        squeeze=False
    )

    for column, neuron in enumerate(indices):

        for row, fields in enumerate((rf_A, rf_B)):

            ax = axes[row, column]

            ax.imshow(
                fields[neuron],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(
                f"Neuron {neuron}",
                fontsize=8
            )

            if column == 0:
                ax.set_ylabel(
                    "Track A" if row == 0
                    else "Track B",
                    fontsize=11
                )

    fig.suptitle(
        f"{name} — LGN ON minus OFF weight maps",
        fontsize=14
    )

    fig.tight_layout()

    figure_path = (
        OUTPUT
        / f"{name.lower()}_track_comparison.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Figure saved:", figure_path)


def main():

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("FINAL NETWORK RECEPTIVE-FIELD COMPARISON")
    print("=" * 65)

    print("Loading Track A...")
    weights_A = load_track("A")

    print("Loading Track B...")
    weights_B = load_track("B")

    compare_population(
        "Excitatory",
        "LGN_E",
        weights_A,
        weights_B
    )

    compare_population(
        "Inhibitory",
        "LGN_I",
        weights_A,
        weights_B
    )

    print("\n" + "=" * 65)
    print("COMPARISON COMPLETE")
    print("=" * 65)

    print("Results saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
