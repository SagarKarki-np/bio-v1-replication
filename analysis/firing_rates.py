
"""
Figure 2: V1 membrane voltage, synaptic inputs and spike raster.

Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Evaluates the fully trained seed-3001 V1 network.

Uses:
    - Learned synaptic weights and thresholds
    - One natural-image presentation (400 ms)
    - Evaluation seed 2029
    - Excitatory neuron E-132

Outputs:
    figure2_activity.png
    figure2_activity.npz

No network training or synaptic plasticity occurs.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND IMPORTS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

import os as _os

# Optional: set V1_ORIGINAL_PROJECT to a folder holding earlier archived data.
ORIGINAL_PROJECT = Path(_os.environ.get("V1_ORIGINAL_PROJECT", str(PROJECT_DIR)))

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import (
    N_E,
    N_I,
    DT,
    N_STEPS,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
)

from lgn_spikes import generate_lgn_batch
from simulation import update_v1_timestep


CHECKPOINT_PATH = (
    PROJECT_DIR
    / "results"
    / "learning"
    / "training_test_3_seed_3001"
    / "batch_1100.npz"
)

IMAGES_PATH = (
    ORIGINAL_PROJECT
    / "data"
    / "van_hateren_dog.npy"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "figure2_seed3001_E132"
)

EVALUATION_SEED = 2029
SELECTED_NEURON = 132  # One-based neuron numbering


WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)


# ============================================================
# 2. LOAD TRAINED NETWORK
# ============================================================

def load_network(checkpoint_path):

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        threshold_E = (
            data["threshold_E"]
            .copy()
            .reshape(N_E, 1)
        )

        threshold_I = (
            data["threshold_I"]
            .copy()
            .reshape(N_I, 1)
        )

        learning_batch = int(
            data["learning_batch"]
        )

    return (
        weights,
        threshold_E,
        threshold_I,
        learning_batch
    )


# ============================================================
# 3. POPULATION SPARSENESS
# ============================================================

def population_sparseness(spike_counts):
    """
    Paper Equation 17:

    S = 1 - mean(r)^2 / mean(r^2)

    For identical presentation durations, spike counts
    and firing rates yield the same sparseness.
    """

    rates = np.asarray(
        spike_counts,
        dtype=float
    )

    mean_squared_rate = np.mean(rates ** 2)

    if mean_squared_rate == 0:
        return np.nan

    return (
        1.0
        - np.mean(rates) ** 2 / mean_squared_rate
    )


# ============================================================
# 4. SIMULATE ONE PRESENTATION
# ============================================================

def evaluate_network(
    images,
    weights,
    threshold_E,
    threshold_I,
    neuron_number=SELECTED_NEURON,
    seed=EVALUATION_SEED
):

    if not 1 <= neuron_number <= N_E:
        raise ValueError("Invalid excitatory neuron number.")

    neuron_index = neuron_number - 1

    np.random.seed(seed)

    LGN_spikes = generate_lgn_batch(
        images,
        batch_size=1,
        n_steps=N_STEPS,
        dt=DT
    )

    V_E = np.zeros((N_E, 1))
    V_I = np.zeros((N_I, 1))

    E_spikes = np.zeros(
        (N_E, N_STEPS),
        dtype=bool
    )

    I_spikes = np.zeros(
        (N_I, N_STEPS),
        dtype=bool
    )

    voltage = np.zeros(N_STEPS)
    excitation = np.zeros(N_STEPS)
    inhibition = np.zeros(N_STEPS)

    # --------------------------------------------------------
    # Run the V1 network
    # --------------------------------------------------------

    for t in range(N_STEPS):

        # Save membrane voltage before spike detection.
        voltage[t] = V_E[neuron_index, 0]

        (
            V_E,
            V_I,
            S_E,
            S_I,
            inputs
        ) = update_v1_timestep(
            V_E,
            V_I,
            threshold_E,
            threshold_I,
            LGN_spikes[:, :, t],
            weights,
            dt=DT,
            tau_m=TAU_M,
            noise_std=NOISE_STD,
            v_min=V_MIN,
            v_reset=V_RESET,
            return_inputs=True
        )

        E_spikes[:, t] = S_E[:, 0]
        I_spikes[:, t] = S_I[:, 0]

        excitation[t] = (
            inputs["LGN_E"][neuron_index, 0]
            + inputs["E_E"][neuron_index, 0]
        )

        inhibition[t] = (
            inputs["I_E"][neuron_index, 0]
        )

    # --------------------------------------------------------
    # Calculate activity measurements
    # --------------------------------------------------------

    E_counts = E_spikes.sum(axis=1)
    I_counts = I_spikes.sum(axis=1)

    duration = N_STEPS * DT

    mean_E_rate = E_counts.mean() / duration
    mean_I_rate = I_counts.mean() / duration

    sparseness_E = population_sparseness(E_counts)
    sparseness_I = population_sparseness(I_counts)

    results = {
        "time_ms": np.arange(N_STEPS) * DT * 1000,
        "voltage": voltage,
        "threshold": float(
            threshold_E[neuron_index, 0]
        ),
        "excitation": excitation,
        "inhibition": inhibition,
        "E_spikes": E_spikes,
        "I_spikes": I_spikes,
        "LGN_spikes": LGN_spikes[:, 0, :],
        "E_counts": E_counts,
        "I_counts": I_counts,
        "mean_E_rate": mean_E_rate,
        "mean_I_rate": mean_I_rate,
        "sparseness_E": sparseness_E,
        "sparseness_I": sparseness_I,
        "neuron_number": neuron_number,
        "neuron_spike_count": int(
            E_counts[neuron_index]
        ),
        "seed": seed
    }

    return results


# ============================================================
# 5. PLOT FIGURE 2
# ============================================================

def plot_figure2(results, save_path):

    time_ms = results["time_ms"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.3, 1.0, 1.6]
        }
    )

    # --------------------------------------------------------
    # A. Membrane voltage and learned threshold
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        time_ms,
        results["voltage"],
        color="tab:orange",
        linewidth=1,
        label="Membrane voltage"
    )

    ax.axhline(
        results["threshold"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Learned threshold"
    )

    spike_times = time_ms[
        results["E_spikes"][
            results["neuron_number"] - 1
        ]
    ]

    ax.scatter(
        spike_times,
        np.full(
            spike_times.shape,
            results["threshold"]
        ),
        s=16,
        color="red",
        marker="v",
        label="Spike events",
        zorder=3
    )

    ax.set_title(
        f"A. Membrane voltage of E-"
        f"{results['neuron_number']:03d}"
    )

    ax.set_ylabel("Membrane potential")
    ax.legend(loc="upper right")

    # --------------------------------------------------------
    # B. Instantaneous synaptic inputs
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        time_ms,
        results["excitation"],
        color="tab:green",
        linewidth=0.9,
        label="Excitation"
    )

    ax.plot(
        time_ms,
        -results["inhibition"],
        color="tab:purple",
        linewidth=0.9,
        label="Inhibition"
    )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.6
    )

    ax.set_title(
        "B. Instantaneous excitatory and inhibitory inputs"
    )

    ax.set_ylabel("Synaptic input")
    ax.legend(loc="upper right")

    # --------------------------------------------------------
    # C. Population spike raster
    # --------------------------------------------------------

    ax = axes[2]

    E_ids, E_t = np.where(
        results["E_spikes"]
    )

    I_ids, I_t = np.where(
        results["I_spikes"]
    )

    ax.scatter(
        E_t * DT * 1000,
        E_ids + 1,
        s=3,
        color="tab:green",
        marker=".",
        label="Excitatory",
        rasterized=True
    )

    ax.scatter(
        I_t * DT * 1000,
        N_E + I_ids + 1,
        s=4,
        color="tab:purple",
        marker=".",
        label="Inhibitory",
        rasterized=True
    )

    ax.axhline(
        N_E + 0.5,
        color="black",
        linewidth=0.8
    )

    ax.set_title(
        "C. V1 population raster "
        f"(E={results['sparseness_E']:.3f}, "
        f"I={results['sparseness_I']:.3f})"
    )

    ax.set_ylabel("Neuron index")
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0, N_STEPS * DT * 1000)

    ax.legend(loc="upper right")

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 6. MAIN WORKFLOW
# ============================================================

def main():

    for path in (CHECKPOINT_PATH, IMAGES_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    (
        weights,
        threshold_E,
        threshold_I,
        learning_batch
    ) = load_network(CHECKPOINT_PATH)

    images = np.load(
        IMAGES_PATH,
        mmap_mode="r"
    )

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Learning batch:", learning_batch)
    print("Images:", images.shape)

    results = evaluate_network(
        images,
        weights,
        threshold_E,
        threshold_I
    )

    print("\nFIGURE 2: E-132 EVALUATION")
    print("=" * 45)

    print(
        "Selected neuron:",
        f"E-{results['neuron_number']:03d}"
    )

    print(
        "Selected neuron spikes:",
        results["neuron_spike_count"]
    )

    print(
        "Selected neuron threshold:",
        results["threshold"]
    )

    print(
        "Total E spikes:",
        results["E_counts"].sum()
    )

    print(
        "Total I spikes:",
        results["I_counts"].sum()
    )

    print(
        "Mean E firing rate:",
        results["mean_E_rate"],
        "Hz"
    )

    print(
        "Mean I firing rate:",
        results["mean_I_rate"],
        "Hz"
    )

    print(
        "Population sparseness E:",
        results["sparseness_E"]
    )

    print(
        "Population sparseness I:",
        results["sparseness_I"]
    )

    # --------------------------------------------------------
    # Save without overwriting the original evaluation
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    figure_path = (
        OUTPUT_DIR / "figure2_activity.png"
    )

    data_path = (
        OUTPUT_DIR / "figure2_activity.npz"
    )

    if figure_path.exists() or data_path.exists():
        raise FileExistsError(
            "Figure 2 output already exists. "
            "Existing results were not overwritten."
        )

    plot_figure2(
        results,
        figure_path
    )

    np.savez_compressed(
        data_path,
        **results
    )

    print("\nFigure saved:", figure_path)
    print("Data saved:", data_path)


if __name__ == "__main__":
    main()
