
"""
Compare structural and functional orientation preferences.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.

Reproduces the original:
    compare_structural_functional_orientation.py

This analysis compares, for each learned excitatory neuron:

1. Structural preferred orientation from receptive-field
   grating projections at 2 cycles per patch.
2. Functional preferred orientation from stimulus-evoked
   spike-based tuning curves.
3. Circular difference between those preferences.
4. Alignment against 1,000 shuffled-neuron controls.

Original settings
-----------------
Network checkpoint: batch 1100
Neurons:            400 excitatory neurons
Spatial frequency:  2 cycles per patch
Orientations:       0 to 165 degrees, step 15
Shuffles:           1000
RNG seed:           2031

The five original saved arrays are preserved exactly.

Inputs
------
results/final_analysis/final_grating_analysis.npz
results/functional_orientation/
    functional_orientation_analysis.npz
results/functional_orientation/
    functional_orientation_responses.npz

Outputs
-------
results/functional_orientation/
    structural_functional_comparison.npz
    structural_functional_difference.png

No new simulations are performed.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND ORIGINAL PARAMETERS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

STRUCTURAL_PATH = (
    PROJECT_DIR
    / "results"
    / "final_analysis"
    / "final_grating_analysis.npz"
)

FUNCTIONAL_ANALYSIS_PATH = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
    / "functional_orientation_analysis.npz"
)

FUNCTIONAL_RESPONSES_PATH = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
    / "functional_orientation_responses.npz"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
)

TARGET_FREQUENCY = 2.0

N_SHUFFLES = 1000
SHUFFLE_SEED = 2031

EXAMPLE_NEURONS = (0, 25, 100, 250)


# ============================================================
# 2. LOAD VALIDATED STRUCTURAL AND FUNCTIONAL DATA
# ============================================================

def load_orientation_data(
    structural_path=STRUCTURAL_PATH,
    functional_analysis_path=FUNCTIONAL_ANALYSIS_PATH,
    functional_responses_path=FUNCTIONAL_RESPONSES_PATH
):
    """
    Load the three previously validated archives.

    Returns
    -------
    dict
        angles
        frequencies
        structural
        functional
        functional_OSI
        spike_counts
        total_spikes

    Structural response shape:
        (neurons, frequencies, orientations)

    Functional tuning shape:
        (neurons, orientations)

    Spike-count shape:
        (neurons, presentations)
    """

    with np.load(
        structural_path,
        allow_pickle=False
    ) as data:

        angles = data["angles"].copy()
        frequencies = data["frequencies"].copy()
        structural = data["responses_1100"].copy()

    with np.load(
        functional_analysis_path,
        allow_pickle=False
    ) as data:

        functional = data["learned_tuning"].copy()
        functional_OSI = data["OSI_learned"].copy()

        # The functional analysis stores its own angles.
        functional_angles = data["angles"].copy()

    with np.load(
        functional_responses_path,
        allow_pickle=False
    ) as data:

        spike_counts = data["learned_E"].copy()

    total_spikes = spike_counts.sum(axis=1)

    # Confirm that structural and functional angle grids
    # refer to the same orientations and ordering.
    if not np.array_equal(
        angles,
        functional_angles
    ):
        raise ValueError(
            "Structural and functional angle grids differ."
        )

    if structural.ndim != 3:
        raise ValueError(
            "Structural responses must be 3D."
        )

    n_neurons = structural.shape[0]

    expected_structural_shape = (
        n_neurons,
        len(frequencies),
        len(angles)
    )

    if structural.shape != expected_structural_shape:
        raise ValueError(
            "Unexpected structural response dimensions."
        )

    expected_functional_shape = (
        n_neurons,
        len(angles)
    )

    if functional.shape != expected_functional_shape:
        raise ValueError(
            "Unexpected functional tuning dimensions."
        )

    if functional_OSI.shape != (n_neurons,):
        raise ValueError(
            "Unexpected functional OSI dimensions."
        )

    if (
        spike_counts.ndim != 2
        or spike_counts.shape[0] != n_neurons
    ):
        raise ValueError(
            "Spike-count neuron dimension mismatch."
        )

    print("STRUCTURAL AND FUNCTIONAL DATA LOADED")
    print("Structural:", structural.shape)
    print("Functional:", functional.shape)
    print("Spike counts:", spike_counts.shape)
    print("Neurons:", n_neurons)

    return {
        "angles": angles,
        "frequencies": frequencies,
        "structural": structural,
        "functional": functional,
        "functional_OSI": functional_OSI,
        "spike_counts": spike_counts,
        "total_spikes": total_spikes,
    }


# ============================================================
# 3. SELECT STRUCTURAL RESPONSES AT 2 CYCLES/PATCH
# ============================================================

def select_structural_frequency(
    structural,
    frequencies,
    target_frequency=TARGET_FREQUENCY
):
    """
    Select structural grating responses at exactly
    the spatial frequency used for functional testing.

    The index selection reproduces the original:
        np.where(np.isclose(frequencies, 2.0))[0][0]

    Returns
    -------
    structural_at_frequency : ndarray
        Shape (neurons, orientations).

    freq_idx : int
        Index of selected frequency.
    """

    matches = np.where(
        np.isclose(
            frequencies,
            target_frequency
        )
    )[0]

    if matches.size == 0:
        raise ValueError(
            f"Frequency {target_frequency} not found."
        )

    freq_idx = matches[0]

    structural_at_frequency = (
        structural[:, freq_idx, :]
    )

    return structural_at_frequency, freq_idx


# ============================================================
# 4. PREFERRED ORIENTATIONS
# ============================================================

def preferred_orientations(responses, angles):
    """
    Determine each neuron's preferred orientation.

    Uses np.argmax exactly as in the original script.

    Tied maxima select the first matching orientation.

    Parameters
    ----------
    responses : ndarray
        Shape (neurons, orientations).

    angles : ndarray
        Orientation values in degrees.

    Returns
    -------
    preferred : ndarray
        Preferred orientation for each neuron.
    """

    responses = np.asarray(responses)
    angles = np.asarray(angles)

    if responses.ndim != 2:
        raise ValueError(
            "Responses must be two-dimensional."
        )

    if responses.shape[1] != len(angles):
        raise ValueError(
            "Orientation dimension mismatch."
        )

    return angles[
        np.argmax(responses, axis=1)
    ]


# ============================================================
# 5. CIRCULAR ORIENTATION DIFFERENCE
# ============================================================

def orientation_difference(a, b):
    """
    Calculate minimal angular separation for orientations.

    Orientations have 180-degree periodicity.

    Examples
    --------
    0 vs 165 degrees -> 15 degrees
    15 vs 105 degrees -> 90 degrees
    60 vs 60 degrees -> 0 degrees

    Returns
    -------
    ndarray or scalar
        Orientation difference in degrees, 0 through 90.
    """

    return np.abs(
        (a - b + 90) % 180 - 90
    )


# ============================================================
# 6. ORIGINAL SHUFFLED-NEURON CONTROL
# ============================================================

def shuffled_neuron_control(
    structural_pref,
    functional_pref,
    n_shuffles=N_SHUFFLES,
    seed=SHUFFLE_SEED
):
    """
    Break structural-functional neuron correspondence
    while preserving each preference distribution.

    Original implementation:
        rng = np.random.default_rng(2031)
        rng.permutation(functional_pref)

    Returns
    -------
    shuffled_means : ndarray
        Mean orientation difference for each permutation.
    """

    structural_pref = np.asarray(structural_pref)
    functional_pref = np.asarray(functional_pref)

    if structural_pref.shape != functional_pref.shape:
        raise ValueError(
            "Structural and functional preference "
            "arrays must have matching shapes."
        )

    rng = np.random.default_rng(seed)

    shuffled_means = np.zeros(n_shuffles)

    for i in range(n_shuffles):

        shuffled_pref = rng.permutation(
            functional_pref
        )

        shuffled_means[i] = orientation_difference(
            structural_pref,
            shuffled_pref
        ).mean()

    return shuffled_means


# ============================================================
# 7. PERMUTATION P-VALUE
# ============================================================

def calculate_permutation_p_value(
    observed_mean,
    shuffled_means
):
    """
    One-sided permutation p-value.

    Smaller observed angular differences represent
    stronger structural-functional alignment.

    The +1 correction exactly matches the original.

    Returns
    -------
    float
    """

    n_shuffles = len(shuffled_means)

    return (
        1 + np.sum(shuffled_means <= observed_mean)
    ) / (n_shuffles + 1)


# ============================================================
# 8. PLOT ORIENTATION DIFFERENCES
# ============================================================

def plot_orientation_differences(
    differences,
    output_path,
    show=False
):
    """
    Reproduce the original orientation-difference
    histogram, with bins every 15 degrees.
    """

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        differences,
        bins=np.arange(-7.5, 105, 15),
        edgecolor="black"
    )

    ax.set_xlabel(
        "Orientation difference (degrees)"
    )

    ax.set_ylabel(
        "Number of neurons"
    )

    ax.set_title(
        "Structural vs Functional Preferred Orientation"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. COMPLETE STRUCTURAL-FUNCTIONAL COMPARISON
# ============================================================

def analyze_structural_functional_orientation(
    structural_path=STRUCTURAL_PATH,
    functional_analysis_path=FUNCTIONAL_ANALYSIS_PATH,
    functional_responses_path=FUNCTIONAL_RESPONSES_PATH,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Reproduce the original structural-functional
    orientation comparison.

    This function loads existing numerical results,
    calculates preferred orientations and their
    differences, runs the original 1000 permutations,
    and saves the original five arrays.

    No trained checkpoints or simulations are modified.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # A. Load existing analysis results
    # --------------------------------------------------------

    data = load_orientation_data(
        structural_path,
        functional_analysis_path,
        functional_responses_path
    )

    angles = data["angles"]
    frequencies = data["frequencies"]

    structural = data["structural"]
    functional = data["functional"]

    functional_OSI = data["functional_OSI"]
    total_spikes = data["total_spikes"]

    # --------------------------------------------------------
    # B. Select matching spatial frequency
    # --------------------------------------------------------

    structural_at_2, freq_idx = (
        select_structural_frequency(
            structural,
            frequencies,
            TARGET_FREQUENCY
        )
    )

    print(
        "\nSelected spatial frequency:",
        frequencies[freq_idx]
    )

    print(
        "Structural response shape:",
        structural_at_2.shape
    )

    # --------------------------------------------------------
    # C. Compute preferred orientations
    # --------------------------------------------------------

    structural_pref = preferred_orientations(
        structural_at_2,
        angles
    )

    functional_pref = preferred_orientations(
        functional,
        angles
    )

    # --------------------------------------------------------
    # D. Calculate circular differences
    # --------------------------------------------------------

    differences = orientation_difference(
        structural_pref,
        functional_pref
    )

    print("\nSTRUCTURAL VS FUNCTIONAL ORIENTATION")

    print(
        "Mean difference:",
        differences.mean()
    )

    print(
        "Median difference:",
        np.median(differences)
    )

    print(
        "Within 15 degrees:",
        np.sum(differences <= 15)
    )

    print(
        "Within 30 degrees:",
        np.sum(differences <= 30)
    )

    # --------------------------------------------------------
    # E. Spike-count diagnostics
    # --------------------------------------------------------

    print("\nSPIKE COUNT CHECK")

    print(
        "Minimum total spikes:",
        total_spikes.min()
    )

    print(
        "Median total spikes:",
        np.median(total_spikes)
    )

    print(
        "Maximum total spikes:",
        total_spikes.max()
    )

    # --------------------------------------------------------
    # F. Example neurons
    # --------------------------------------------------------

    print("\nEXAMPLE NEURONS")

    for neuron in EXAMPLE_NEURONS:

        if neuron >= len(structural_pref):
            continue

        print(f"\nNeuron {neuron}")

        print(
            "Structural:",
            structural_pref[neuron]
        )

        print(
            "Functional:",
            functional_pref[neuron]
        )

        print(
            "Difference:",
            differences[neuron]
        )

        print(
            "Functional OSI:",
            functional_OSI[neuron]
        )

        print(
            "Total spikes:",
            total_spikes[neuron]
        )

    # --------------------------------------------------------
    # G. Original shuffled-neuron control
    # --------------------------------------------------------

    shuffled_means = shuffled_neuron_control(
        structural_pref,
        functional_pref,
        n_shuffles=N_SHUFFLES,
        seed=SHUFFLE_SEED
    )

    observed_mean = differences.mean()

    p_value = calculate_permutation_p_value(
        observed_mean,
        shuffled_means
    )

    print("\nSHUFFLED CONTROL")

    print(
        "Observed mean difference:",
        observed_mean
    )

    print(
        "Shuffled mean difference:",
        shuffled_means.mean()
    )

    print(
        "Permutation p-value:",
        p_value
    )

    # --------------------------------------------------------
    # H. Save figure
    # --------------------------------------------------------

    plot_orientation_differences(
        differences,
        output_dir / "structural_functional_difference.png"
    )

    # --------------------------------------------------------
    # I. Save original five numerical arrays
    # --------------------------------------------------------

    output_path = (
        output_dir
        / "structural_functional_comparison.npz"
    )

    np.savez_compressed(
        output_path,

        structural_pref=structural_pref,
        functional_pref=functional_pref,
        differences=differences,
        total_spikes=total_spikes,
        shuffled_means=shuffled_means
    )

    print("\nComparison completed!")
    print("Output:", output_path)

    return {
        "structural_pref": structural_pref,
        "functional_pref": functional_pref,
        "differences": differences,
        "total_spikes": total_spikes,
        "shuffled_means": shuffled_means,
        "p_value": p_value
    }


# ============================================================
# 10. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare learned structural and functional "
            "orientation preferences."
        )
    )

    parser.add_argument(
        "--structural",
        type=str,
        default=str(STRUCTURAL_PATH)
    )

    parser.add_argument(
        "--functional-analysis",
        type=str,
        default=str(FUNCTIONAL_ANALYSIS_PATH)
    )

    parser.add_argument(
        "--functional-responses",
        type=str,
        default=str(FUNCTIONAL_RESPONSES_PATH)
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    args = parser.parse_args()

    analyze_structural_functional_orientation(
        structural_path=args.structural,
        functional_analysis_path=args.functional_analysis,
        functional_responses_path=args.functional_responses,
        output_dir=args.output_dir
    )
