
"""
Figure 7: Receptive-field correlations and synaptic connectivity.

Adapted from the original Figure 7 MATLAB-demo analysis.

Works with Track A or Track B, using:
- 100,000-stimulus STA receptive fields
- Final batch_1100 trained weights

No training or simulation.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT
    / "results"
    / "paper_protocol"
    / "seed_3002"
)


def load_data(track):

    sta_path = (
        BASE
        / "functional_sta"
        / f"track_{track}"
        / "full_image_100000"
        / "STA_100000_results.npz"
    )

    checkpoint_path = (
        BASE
        / "synaptic_training"
        / f"track_{track}"
        / "checkpoints"
        / "batch_1100.npz"
    )

    print("STA:", sta_path)
    print("Checkpoint:", checkpoint_path)

    with np.load(sta_path, allow_pickle=False) as data:

        E_RF = data["STA_E"].copy()
        I_RF = data["STA_I"].copy()

        assert int(data["n_stimuli"]) == 100000

    with np.load(checkpoint_path, allow_pickle=False) as data:

        W_I_E = data["W_I_E"].copy()
        W_E_I = data["W_E_I"].copy()
        W_I_I = data["W_I_I"].copy()
        W_E_E = data["W_E_E"].copy()

        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"

    assert E_RF.shape == (400, 256)
    assert I_RF.shape == (100, 256)

    assert W_I_E.shape == (400, 100)
    assert W_E_I.shape == (100, 400)
    assert W_I_I.shape == (100, 100)
    assert W_E_E.shape == (400, 400)

    weights = {
        "I_to_E": W_I_E,
        "E_to_I": W_E_I,
        "I_to_I": W_I_I,
        "E_to_E": W_E_E,
    }

    return E_RF, I_RF, weights, sta_path, checkpoint_path


def normalize_rfs(rf):

    rf = np.asarray(rf, dtype=float)

    centered = rf - rf.mean(
        axis=1,
        keepdims=True
    )

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


def calculate_rf_correlations(E_RF, I_RF):

    E_norm, E_valid = normalize_rfs(E_RF)
    I_norm, I_valid = normalize_rfs(I_RF)

    print("Valid E RFs:", E_valid.sum())
    print("Valid I RFs:", I_valid.sum())

    C_EI = np.clip(
        E_norm @ I_norm.T,
        -1, 1
    )

    C_EE = np.clip(
        E_norm @ E_norm.T,
        -1, 1
    )

    C_II = np.clip(
        I_norm @ I_norm.T,
        -1, 1
    )

    C_EI[
        ~(E_valid[:, None] & I_valid[None, :])
    ] = np.nan

    C_EE[
        ~(E_valid[:, None] & E_valid[None, :])
    ] = np.nan

    C_II[
        ~(I_valid[:, None] & I_valid[None, :])
    ] = np.nan

    return C_EI, C_EE, C_II


def extract_pairs(correlation, weights, recurrent=False):

    assert correlation.shape == weights.shape

    if recurrent:

        # Remove self-connections.
        # Retain both directed pairs: i->j and j->i.
        mask = ~np.eye(
            weights.shape[0],
            dtype=bool
        )

    else:
        mask = np.ones(
            weights.shape,
            dtype=bool
        )

    mask &= np.isfinite(correlation)
    mask &= np.isfinite(weights)

    return correlation[mask], weights[mask]


def main(track):

    print("=" * 65)
    print(f"FIGURE 7 — TRACK {track.upper()}")
    print("=" * 65)

    output = (
        BASE
        / "figure7_rf_connectivity_full_image"
        / f"track_{track}"
    )

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output}"
        )

    E_RF, I_RF, weights, sta_path, checkpoint_path = (
        load_data(track)
    )

    C_EI, C_EE, C_II = calculate_rf_correlations(
        E_RF,
        I_RF
    )

    # Weight convention: W[postsynaptic, presynaptic].
    connections = {
        "I_to_E": (C_EI, weights["I_to_E"]),
        "E_to_I": (C_EI.T, weights["E_to_I"]),
        "I_to_I": (C_II, weights["I_to_I"]),
        "E_to_E": (C_EE, weights["E_to_E"]),
    }

    results = {}

    print("\nRF SIMILARITY AND SYNAPTIC WEIGHTS")

    for name, (corr, W) in connections.items():

        recurrent = name in ("I_to_I", "E_to_E")

        x, y = extract_pairs(
            corr,
            W,
            recurrent=recurrent
        )

        results[name] = (x, y)

        print(f"\n{name}")
        print("Connections:", len(x))
        print("Mean RF correlation:", np.mean(x))
        print(
            "Mean absolute RF correlation:",
            np.mean(np.abs(x))
        )
        print("Mean synaptic weight:", np.mean(y))

        if len(x) > 1 and np.std(y) > 0:
            print(
                "Pearson RF-weight correlation:",
                np.corrcoef(x, y)[0, 1]
            )

    # --------------------------------------
    # Figure 7: Six-panel visualization
    # --------------------------------------

    EXAMPLE_E = 0

    example_corr = C_EI[EXAMPLE_E]
    example_weights = weights["I_to_E"][EXAMPLE_E]

    bins = np.arange(-1, 1.021, 0.02)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
        constrained_layout=True
    )

    # Panel A
    ax = axes[0, 0]

    ax.hist(
        example_corr[np.isfinite(example_corr)],
        bins=bins,
        color="steelblue"
    )

    ax.set_title(
        f"A. E neuron {EXAMPLE_E}: RF correlations with I"
    )
    ax.set_xlabel("RF correlation")
    ax.set_ylabel("Number of inhibitory neurons")
    ax.set_xlim(-1, 1)

    # Panel B
    ax = axes[0, 1]

    ax.scatter(
        example_corr,
        example_weights,
        s=20,
        alpha=0.7
    )

    ax.set_title(
        f"B. Inhibitory input to E neuron {EXAMPLE_E}"
    )
    ax.set_xlabel("RF correlation")
    ax.set_ylabel("I -> E weight")
    ax.set_xlim(-1, 1)

    # Panels C-F
    panel_specs = [
        (axes[0, 2], "I_to_E", "C. I -> E"),
        (axes[1, 0], "E_to_I", "D. E -> I"),
        (axes[1, 1], "I_to_I", "E. I -> I"),
        (axes[1, 2], "E_to_E", "F. E -> E"),
    ]

    for ax, name, title in panel_specs:

        x, y = results[name]

        ax.scatter(
            x,
            y,
            s=2,
            alpha=0.08
        )

        ax.set_title(title)
        ax.set_xlabel("Pairwise RF correlation")
        ax.set_ylabel("Synaptic weight")
        ax.set_xlim(-1, 1)

        inset = ax.inset_axes([
            0.62, 0.62, 0.34, 0.32
        ])

        inset.hist(
            x,
            bins=bins,
            color="gray"
        )

        inset.set_xlim(-1, 1)
        inset.tick_params(labelsize=7)

    fig.suptitle(
        f"Figure 7 — Track {track.upper()}: "
        "RF Similarity and Synaptic Connectivity",
        fontsize=16
    )

    output.mkdir(parents=True, exist_ok=True)

    figure_path = (
        output / "figure7_sta_connectivity.png"
    )

    fig.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------
    # Save numerical results
    # --------------------------------------

    saved = {
        "E_RF": E_RF,
        "I_RF": I_RF,
        "C_EI": C_EI,
        "C_EE": C_EE,
        "C_II": C_II,
        "example_E": EXAMPLE_E,
        "example_correlations": example_corr,
        "example_weights": example_weights,
        "sta_source": str(sta_path),
        "checkpoint_source": str(checkpoint_path),
    }

    for name, (x, y) in results.items():

        saved[f"{name}_correlations"] = x
        saved[f"{name}_weights"] = y

    statistics_path = (
        output / "figure7_statistics.npz"
    )

    np.savez_compressed(
        statistics_path,
        **saved
    )

    print("\n" + "=" * 65)
    print("FIGURE 7 ANALYSIS COMPLETE")
    print("=" * 65)
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
