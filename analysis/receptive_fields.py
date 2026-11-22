
"""
Receptive-field development analysis for the LGN-V1 network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This module reconstructs feedforward receptive fields from
LGN-to-excitatory synaptic weights.

Weight convention
-----------------
W_LGN_E.shape = (400, 512)

Each excitatory neuron receives:
    256 ON LGN inputs
    256 OFF LGN inputs

Receptive field:
    RF = ON - OFF

Spatial maps are reconstructed using column-major ordering,
consistent with the original MATLAB implementation and our
LGN preprocessing.

Analysis
--------
1. Reconstruct ON, OFF and signed RF maps.
2. Visualize a single neuron's ON/OFF maps.
3. Compare selected neurons across training stages.
4. Calculate mean absolute RF change from initialization.
5. Save figures and numerical results.

The original checkpoints are read without modification.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import PATCH_SIZE, N_E, N_LGN


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "receptive_fields"
)

DEFAULT_NEURONS = [0, 25, 100, 250]


# ============================================================
# 2. LOAD LGN -> E WEIGHTS
# ============================================================

def load_lgn_e_weights(checkpoint_path):
    """
    Load LGN-to-excitatory weights from an NPZ checkpoint.

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path
        Checkpoint containing W_LGN_E.

    Returns
    -------
    numpy.ndarray
        Weight matrix with shape (N_E, N_LGN).
    """

    checkpoint_path = Path(
        checkpoint_path
    ).expanduser()

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as data:

        W = data["W_LGN_E"].copy()

    expected_shape = (N_E, N_LGN)

    if W.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, got {W.shape}."
        )

    if not np.isfinite(W).all():
        raise ValueError(
            "Weight matrix contains non-finite values."
        )

    return W


# ============================================================
# 3. RECONSTRUCT RECEPTIVE FIELD
# ============================================================

def reconstruct_rf(W, neuron):
    """
    Reconstruct ON, OFF and signed receptive-field maps.

    Parameters
    ----------
    W : numpy.ndarray
        LGN-to-E weights, shape (N_E, N_LGN).

    neuron : int
        Excitatory neuron index.

    Returns
    -------
    ON : numpy.ndarray
        ON-channel weights, shape (PATCH_SIZE, PATCH_SIZE).

    OFF : numpy.ndarray
        OFF-channel weights, shape (PATCH_SIZE, PATCH_SIZE).

    RF : numpy.ndarray
        Signed receptive field, ON - OFF.

    Notes
    -----
    MATLAB column-major ordering is preserved using
    reshape(..., order="F").
    """

    if not 0 <= neuron < W.shape[0]:
        raise IndexError(
            f"Neuron index {neuron} is out of range."
        )

    n_pixels = PATCH_SIZE ** 2

    weights = W[neuron]

    ON = weights[:n_pixels].reshape(
        PATCH_SIZE,
        PATCH_SIZE,
        order="F"
    )

    OFF = weights[n_pixels:].reshape(
        PATCH_SIZE,
        PATCH_SIZE,
        order="F"
    )

    RF = ON - OFF

    return ON, OFF, RF


# ============================================================
# 4. MEAN ABSOLUTE RECEPTIVE-FIELD CHANGE
# ============================================================

def mean_absolute_rf_change(
    RF_initial,
    RF_learned
):
    """
    Calculate mean absolute change between two RF maps.

    This preserves the metric in the original script:

        mean(abs(RF_learned - RF_initial))
    """

    if RF_initial.shape != RF_learned.shape:
        raise ValueError(
            "Receptive fields must have matching shapes."
        )

    return np.mean(
        np.abs(RF_learned - RF_initial)
    )


# ============================================================
# 5. VISUALIZE ON AND OFF CHANNELS
# ============================================================

def plot_on_off_maps(
    W,
    neuron,
    save_path,
    stage_label="Selected checkpoint",
    show=False
):
    """
    Plot ON weights, OFF weights and signed RF
    for one excitatory neuron.
    """

    ON, OFF, RF = reconstruct_rf(
        W,
        neuron
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    max_weight = max(
        ON.max(),
        OFF.max()
    )

    max_rf = np.abs(RF).max()

    # Avoid identical zero color limits.
    if max_weight == 0:
        max_weight = 1.0

    if max_rf == 0:
        max_rf = 1.0

    axes[0].imshow(
        ON,
        cmap="Reds",
        vmin=0,
        vmax=max_weight,
        interpolation="nearest"
    )

    axes[1].imshow(
        OFF,
        cmap="Blues",
        vmin=0,
        vmax=max_weight,
        interpolation="nearest"
    )

    axes[2].imshow(
        RF,
        cmap="bwr",
        vmin=-max_rf,
        vmax=max_rf,
        interpolation="nearest"
    )

    titles = [
        "ON weights",
        "OFF weights",
        "ON - OFF"
    ]

    for ax, title in zip(axes, titles):

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Excitatory neuron {neuron} - {stage_label}"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)

    return ON, OFF, RF


# ============================================================
# 6. RECEPTIVE-FIELD DEVELOPMENT GRID
# ============================================================

def plot_rf_development(
    receptive_fields,
    neurons,
    stage_names,
    save_path,
    show=False
):
    """
    Plot signed receptive fields across training stages.

    Rows correspond to selected neurons.
    Columns correspond to training checkpoints.

    A common symmetric color scale is used for all maps.
    """

    n_neurons = len(neurons)
    n_stages = len(stage_names)

    fig, axes = plt.subplots(
        n_neurons,
        n_stages,
        figsize=(3.3 * n_stages, 3 * n_neurons),
        squeeze=False,
        constrained_layout=True
    )

    color_limit = np.max(
        np.abs(receptive_fields)
    )

    if color_limit == 0:
        color_limit = 1.0

    for row, neuron in enumerate(neurons):

        for col, stage in enumerate(stage_names):

            ax = axes[row, col]

            image = ax.imshow(
                receptive_fields[row, col],
                cmap="seismic",
                vmin=-color_limit,
                vmax=color_limit,
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(
                    stage,
                    fontsize=13
                )

            if col == 0:
                ax.set_ylabel(
                    f"Neuron {neuron}",
                    fontsize=12
                )

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.65,
        label="ON - OFF synaptic weight"
    )

    fig.suptitle(
        "Receptive-field development during V1 learning",
        fontsize=16
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 7. COMPLETE RECEPTIVE-FIELD ANALYSIS
# ============================================================

def analyze_receptive_fields(
    initial_path,
    batch_100_path,
    batch_1100_path,
    neurons=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Compare receptive fields before and during learning.

    Parameters
    ----------
    initial_path : str or Path
        Initial untrained weights.

    batch_100_path : str or Path
        Checkpoint after 100 learning batches.

    batch_1100_path : str or Path
        Checkpoint after 1100 learning batches.

    neurons : list of int
        Selected excitatory neuron indices.

    output_dir : str or Path
        Directory for analysis outputs.

    show : bool
        Whether to display figures interactively.
    """

    if neurons is None:
        neurons = DEFAULT_NEURONS

    output_dir = Path(
        output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # A. Load three training stages
    # --------------------------------------------------------

    W_initial = load_lgn_e_weights(
        initial_path
    )

    W_100 = load_lgn_e_weights(
        batch_100_path
    )

    W_1100 = load_lgn_e_weights(
        batch_1100_path
    )

    weight_sets = [
        W_initial,
        W_100,
        W_1100
    ]

    stage_names = [
        "Batch 0",
        "Batch 100",
        "Batch 1100"
    ]

    # --------------------------------------------------------
    # B. Reconstruct selected receptive fields
    # --------------------------------------------------------

    receptive_fields = np.array([
        [
            reconstruct_rf(W, neuron)[2]
            for W in weight_sets
        ]
        for neuron in neurons
    ])

    print(
        "Receptive fields shape:",
        receptive_fields.shape
    )

    print(
        "Shared color limit:",
        np.max(np.abs(receptive_fields))
    )

    # --------------------------------------------------------
    # C. Visualize RF development
    # --------------------------------------------------------

    plot_rf_development(
        receptive_fields,
        neurons,
        stage_names,
        output_dir / "RF_batch_0_100_1100.png",
        show=show
    )

    # --------------------------------------------------------
    # D. Quantify mean absolute RF change
    # --------------------------------------------------------

    changes_100 = []
    changes_1100 = []

    print(
        "\nMEAN ABSOLUTE RF CHANGE FROM BATCH 0"
    )

    for row, neuron in enumerate(neurons):

        initial_rf = receptive_fields[row, 0]

        change_100 = mean_absolute_rf_change(
            initial_rf,
            receptive_fields[row, 1]
        )

        change_1100 = mean_absolute_rf_change(
            initial_rf,
            receptive_fields[row, 2]
        )

        changes_100.append(change_100)
        changes_1100.append(change_1100)

        print(f"\nNeuron {neuron}")
        print("Batch 0 -> 100:", change_100)
        print("Batch 0 -> 1100:", change_1100)

    # --------------------------------------------------------
    # E. Save numerical results
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "rf_development.npz",
        neurons=np.asarray(neurons),
        receptive_fields=receptive_fields,
        change_100=np.asarray(changes_100),
        change_1100=np.asarray(changes_1100)
    )

    # --------------------------------------------------------
    # F. Plot ON/OFF maps for the first selected neuron
    # --------------------------------------------------------

    first_neuron = neurons[0]

    plot_on_off_maps(
        W_1100,
        first_neuron,
        output_dir / f"on_off_neuron_{first_neuron}.png",
        stage_label="Batch 1100",
        show=show
    )

    print("\nReceptive-field analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "neurons": np.asarray(neurons),
        "receptive_fields": receptive_fields,
        "change_100": np.asarray(changes_100),
        "change_1100": np.asarray(changes_1100)
    }


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Visualize LGN-to-V1 receptive-field "
            "development across training."
        )
    )

    parser.add_argument(
        "--initial",
        type=str,
        required=True,
        help="Path to initial weights.npz"
    )

    parser.add_argument(
        "--batch100",
        type=str,
        required=True,
        help="Path to batch 100 checkpoint"
    )

    parser.add_argument(
        "--batch1100",
        type=str,
        required=True,
        help="Path to batch 1100 checkpoint"
    )

    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=DEFAULT_NEURONS,
        help="Excitatory neuron indices"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )

    args = parser.parse_args()

    analyze_receptive_fields(
        initial_path=args.initial,
        batch_100_path=args.batch100,
        batch_1100_path=args.batch1100,
        neurons=args.neurons,
        output_dir=args.output_dir,
        show=args.show
    )
