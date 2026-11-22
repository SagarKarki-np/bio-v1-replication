
"""
Weight-derived orientation selectivity analysis.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This script analyzes orientation and spatial-frequency
selectivity from LGN-to-V1 excitatory synaptic weights.

IMPORTANT
---------
This is a structural RF-based grating projection analysis.

The grating responses are NOT simulated V1 spike rates.

Original analysis protocol
--------------------------
1. Load the LGN-to-E weight matrix.
2. Reconstruct ON and OFF receptive-field maps.
3. Compute signed receptive fields: ON - OFF.
4. Spatially center and L2-normalize each RF.
5. Generate sine/cosine grating templates.
6. Orthonormalize grating templates using QR.
7. Calculate RF projection magnitudes.
8. Determine preferred orientation and frequency.
9. Calculate vector-based orientation selectivity.

Outputs
-------
population_orientation_analysis.npz
population_OSI.npz
population_orientation_histogram.png
population_frequency_histogram.png
population_OSI_histogram.png
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import PATCH_SIZE, N_E, N_LGN


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "orientation_selectivity"
)

ANGLES = np.arange(0, 180, 15)

FREQUENCIES = np.arange(0.5, 4.5, 0.5)

EXAMPLE_NEURONS = (0, 25, 100, 250)


# ============================================================
# 2. LOAD LGN -> E WEIGHTS
# ============================================================

def load_lgn_e_weights(checkpoint_path):
    """
    Load excitatory feedforward weights from a checkpoint.

    Expected shape:
        (N_E, N_LGN)
    """

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as data:
        W = data["W_LGN_E"].copy()

    if W.shape != (N_E, N_LGN):
        raise ValueError(
            f"Unexpected LGN-to-E shape: {W.shape}"
        )

    if not np.isfinite(W).all():
        raise ValueError(
            "Synaptic weights must be finite."
        )

    return W


# ============================================================
# 3. RECONSTRUCT POPULATION RECEPTIVE FIELDS
# ============================================================

def reconstruct_population_rfs(W):
    """
    Reconstruct all excitatory receptive fields.

    Returns
    -------
    ndarray
        Signed receptive fields with shape:
            (n_neurons, PATCH_SIZE, PATCH_SIZE)

    The two LGN channels are reconstructed with
    MATLAB column-major ordering.
    """

    n_pixels = PATCH_SIZE ** 2

    if W.ndim != 2 or W.shape[1] != 2 * n_pixels:
        raise ValueError(
            "Unexpected LGN input dimensions."
        )

    n_neurons = W.shape[0]

    ON = np.stack([
        W[i, :n_pixels].reshape(
            PATCH_SIZE, PATCH_SIZE, order="F"
        )
        for i in range(n_neurons)
    ])

    OFF = np.stack([
        W[i, n_pixels:].reshape(
            PATCH_SIZE, PATCH_SIZE, order="F"
        )
        for i in range(n_neurons)
    ])

    return ON - OFF


# ============================================================
# 4. GENERATE ORIENTED GRATING
# ============================================================

def create_grating(angle, phase, frequency):
    """
    Generate a two-dimensional cosine grating.

    Parameters
    ----------
    angle : float
        Orientation in degrees.
    phase : float
        Grating phase in radians.
    frequency : float
        Spatial frequency in cycles per patch.

    Returns
    -------
    ndarray
        Grating with shape (PATCH_SIZE, PATCH_SIZE).

    Notes
    -----
    The spatial coordinate convention matches
    the original population orientation script.
    """

    y, x = np.meshgrid(
        np.arange(PATCH_SIZE),
        np.arange(PATCH_SIZE),
        indexing="ij"
    )

    center = (PATCH_SIZE - 1) / 2

    x = x - center
    y = y - center

    theta = np.deg2rad(angle)

    coordinate = (
        -x * np.sin(theta)
        + y * np.cos(theta)
    )

    return np.cos(
        2 * np.pi * frequency
        * coordinate / PATCH_SIZE
        + phase
    )


# ============================================================
# 5. NORMALIZE RECEPTIVE FIELDS
# ============================================================

def normalize_receptive_fields(RF):
    """
    Spatially center and L2-normalize each RF.

    Input shape:
        (n_neurons, PATCH_SIZE, PATCH_SIZE)

    Returns
    -------
    ndarray
        Normalized RF vectors:
            (n_neurons, PATCH_SIZE**2)

    Zero-norm RFs remain zero.
    """

    n_neurons = RF.shape[0]

    RF_flat = RF.reshape(
        n_neurons,
        PATCH_SIZE ** 2
    )

    RF_centered = (
        RF_flat
        - RF_flat.mean(axis=1, keepdims=True)
    )

    RF_norms = np.linalg.norm(
        RF_centered,
        axis=1,
        keepdims=True
    )

    RF_normalized = np.divide(
        RF_centered,
        RF_norms,
        out=np.zeros_like(RF_centered),
        where=RF_norms > 0
    )

    return RF_normalized


# ============================================================
# 6. CALCULATE POPULATION GRATING RESPONSES
# ============================================================

def calculate_grating_responses(
    RF_normalized,
    angles=ANGLES,
    frequencies=FREQUENCIES
):
    """
    Calculate RF projection magnitudes onto grating bases.

    Returns
    -------
    ndarray
        Shape:
            (n_neurons, n_frequencies, n_orientations)

    Each grating basis contains:
        - Cosine phase (0 radians)
        - Sine phase (pi/2 radians)

    The QR decomposition constructs an orthonormal
    basis spanning these two templates.

    The response is the norm of the RF projection
    onto the resulting two-dimensional subspace.
    """

    RF_normalized = np.asarray(
        RF_normalized,
        dtype=float
    )

    n_neurons = RF_normalized.shape[0]

    responses = np.zeros((
        n_neurons,
        len(frequencies),
        len(angles)
    ))

    for i, frequency in enumerate(frequencies):

        for j, angle in enumerate(angles):

            # Original two-phase grating construction.
            G_cos = create_grating(
                angle, 0, frequency
            ).flatten()

            G_sin = create_grating(
                angle, np.pi / 2, frequency
            ).flatten()

            # Remove the spatial means.
            G_cos = G_cos - G_cos.mean()
            G_sin = G_sin - G_sin.mean()

            templates = np.column_stack(
                (G_cos, G_sin)
            )

            # Orthonormalize phase templates.
            Q, R = np.linalg.qr(
                templates,
                mode="reduced"
            )

            # Preserve the original degeneracy check.
            if np.min(np.abs(np.diag(R))) < 1e-10:
                continue

            # Project normalized RFs onto the basis.
            projections = RF_normalized @ Q

            responses[:, i, j] = np.linalg.norm(
                projections,
                axis=1
            )

    return responses


# ============================================================
# 7. FIND PREFERRED ORIENTATION AND FREQUENCY
# ============================================================

def calculate_preferences(
    responses,
    angles=ANGLES,
    frequencies=FREQUENCIES
):
    """
    Find the strongest frequency-orientation response.

    Original selection rule:
        Find the single largest response across all
        frequency and orientation combinations.

    Returns
    -------
    preferred_angles : ndarray
    preferred_frequencies : ndarray
    max_responses : ndarray
    frequency_indices : ndarray
    """

    n_neurons = responses.shape[0]

    expected_shape = (
        n_neurons,
        len(frequencies),
        len(angles)
    )

    if responses.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, "
            f"got {responses.shape}."
        )

    best_flat = responses.reshape(
        n_neurons, -1
    ).argmax(axis=1)

    best_indices = np.unravel_index(
        best_flat,
        (len(frequencies), len(angles))
    )

    frequency_indices = best_indices[0]

    preferred_frequencies = frequencies[
        frequency_indices
    ]

    preferred_angles = angles[
        best_indices[1]
    ]

    max_responses = responses.reshape(
        n_neurons, -1
    ).max(axis=1)

    return (
        preferred_angles,
        preferred_frequencies,
        max_responses,
        frequency_indices
    )


# ============================================================
# 8. CALCULATE VECTOR-BASED OSI
# ============================================================

def calculate_vector_osi(
    orientation_responses,
    angles=ANGLES
):
    """
    Calculate vector-based orientation selectivity.

    OSI = abs(sum(R(theta) * exp(2j * theta)))
          / sum(R(theta))

    Angles are converted from degrees to radians.

    Neurons with zero total orientation response
    receive OSI = 0, matching the original script.
    """

    orientation_responses = np.asarray(
        orientation_responses,
        dtype=float
    )

    if orientation_responses.ndim != 2:
        raise ValueError(
            "Orientation responses must be 2D."
        )

    if orientation_responses.shape[1] != len(angles):
        raise ValueError(
            "Response and angle dimensions differ."
        )

    theta = np.deg2rad(angles)

    orientation_vectors = np.exp(
        2j * theta
    )

    numerator = np.abs(
        orientation_responses @ orientation_vectors
    )

    denominator = orientation_responses.sum(
        axis=1
    )

    OSI = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0
    )

    return OSI


# ============================================================
# 9. PLOT DISTRIBUTIONS
# ============================================================

def plot_preference_histograms(
    preferred_angles,
    preferred_frequencies,
    angles,
    frequencies,
    output_dir,
    show=False
):
    """
    Plot distributions of preferred orientations
    and preferred spatial frequencies.
    """

    # Preferred orientations.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        preferred_angles,
        bins=np.arange(-7.5, 187.5, 15),
        edgecolor="black"
    )

    ax.set_xlabel(
        "Preferred orientation (degrees)"
    )

    ax.set_ylabel("Number of neurons")

    ax.set_title(
        "Preferred orientations of V1 neurons"
    )

    ax.set_xticks(angles)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "population_orientation_histogram.png",
        dpi=200
    )

    if show:
        plt.show()

    plt.close(fig)

    # Preferred spatial frequencies.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        preferred_frequencies,
        bins=np.arange(0.25, 4.75, 0.5),
        edgecolor="black"
    )

    ax.set_xlabel(
        "Preferred spatial frequency (cycles/patch)"
    )

    ax.set_ylabel("Number of neurons")

    ax.set_title(
        "Preferred spatial frequencies of V1 neurons"
    )

    ax.set_xticks(frequencies)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "population_frequency_histogram.png",
        dpi=200
    )

    if show:
        plt.show()

    plt.close(fig)


def plot_osi_histogram(
    OSI,
    output_dir,
    show=False
):
    """
    Plot population distribution of orientation selectivity.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        OSI,
        bins=np.linspace(0, 1, 31),
        edgecolor="black"
    )

    ax.set_xlabel(
        "Vector-based orientation selectivity index"
    )

    ax.set_ylabel("Number of neurons")

    ax.set_title(
        "Weight-derived V1 Orientation Selectivity"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "population_OSI_histogram.png",
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 10. COMPLETE ORIENTATION ANALYSIS
# ============================================================

def analyze_orientation_selectivity(
    checkpoint_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run weight-derived population orientation analysis.

    Loads one checkpoint and saves:
        - Grating projection responses
        - Preferred angles and frequencies
        - OSI values
        - Population histograms
    """

    checkpoint_path = Path(
        checkpoint_path
    ).expanduser()

    output_dir = Path(
        output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # A. Load network and reconstruct RFs
    # --------------------------------------------------------

    W = load_lgn_e_weights(
        checkpoint_path
    )

    RF = reconstruct_population_rfs(W)

    print("Checkpoint:", checkpoint_path)
    print("Receptive fields:", RF.shape)

    # --------------------------------------------------------
    # B. Normalize the RFs
    # --------------------------------------------------------

    RF_normalized = normalize_receptive_fields(
        RF
    )

    # --------------------------------------------------------
    # C. Calculate projection responses
    # --------------------------------------------------------

    responses = calculate_grating_responses(
        RF_normalized,
        ANGLES,
        FREQUENCIES
    )

    print("Response shape:", responses.shape)

    # --------------------------------------------------------
    # D. Find preferred frequency and orientation
    # --------------------------------------------------------

    (
        preferred_angles,
        preferred_frequencies,
        max_responses,
        frequency_indices
    ) = calculate_preferences(
        responses,
        ANGLES,
        FREQUENCIES
    )

    # --------------------------------------------------------
    # E. Select each neuron's preferred frequency
    # --------------------------------------------------------

    orientation_responses = responses[
        np.arange(responses.shape[0]),
        frequency_indices,
        :
    ]

    print(
        "Orientation response shape:",
        orientation_responses.shape
    )

    # --------------------------------------------------------
    # F. Calculate OSI
    # --------------------------------------------------------

    OSI = calculate_vector_osi(
        orientation_responses,
        ANGLES
    )

    # --------------------------------------------------------
    # G. Print example neurons
    # --------------------------------------------------------

    for neuron in EXAMPLE_NEURONS:

        if neuron >= W.shape[0]:
            continue

        print(f"\nNeuron {neuron}")

        print(
            "Preferred orientation:",
            preferred_angles[neuron]
        )

        print(
            "Preferred spatial frequency:",
            preferred_frequencies[neuron]
        )

        print(
            "Maximum projection response:",
            max_responses[neuron]
        )

        print("OSI:", OSI[neuron])

    # --------------------------------------------------------
    # H. Print population statistics
    # --------------------------------------------------------

    print("\nPOPULATION ORIENTATION ANALYSIS")
    print("Mean maximum response:", max_responses.mean())
    print("Median maximum response:", np.median(max_responses))

    print("\nPOPULATION OSI")
    print("Mean:", OSI.mean())
    print("Median:", np.median(OSI))
    print("Minimum:", OSI.min())
    print("Maximum:", OSI.max())

    # --------------------------------------------------------
    # I. Save original numerical output fields
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "population_orientation_analysis.npz",
        angles=ANGLES,
        frequencies=FREQUENCIES,
        responses=responses,
        preferred_angles=preferred_angles,
        preferred_frequencies=preferred_frequencies,
        max_responses=max_responses
    )

    np.savez_compressed(
        output_dir / "population_OSI.npz",
        OSI=OSI,
        orientation_responses=orientation_responses
    )

    # --------------------------------------------------------
    # J. Save figures
    # --------------------------------------------------------

    plot_preference_histograms(
        preferred_angles,
        preferred_frequencies,
        ANGLES,
        FREQUENCIES,
        output_dir,
        show=show
    )

    plot_osi_histogram(
        OSI,
        output_dir,
        show=show
    )

    print("\nOrientation analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "responses": responses,
        "angles": ANGLES,
        "frequencies": FREQUENCIES,
        "preferred_angles": preferred_angles,
        "preferred_frequencies": preferred_frequencies,
        "orientation_responses": orientation_responses,
        "OSI": OSI
    }


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate weight-derived V1 orientation "
            "selectivity from a trained checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to a checkpoint containing W_LGN_E"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for analysis outputs"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively"
    )

    args = parser.parse_args()

    analyze_orientation_selectivity(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        show=args.show
    )
