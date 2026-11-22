
"""
Sinusoidal grating stimulus generation for V1.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.

This module reproduces the original
generate_grating_stimuli.py implementation.

Original protocol
-----------------
Patch size:       16 x 16 pixels
Orientations:     0 to 165 degrees, step 15
Default frequency: 2 cycles per patch
Default phase:     0 radians

The original stimulus formula and coordinate
convention are preserved.

The generator does not perform LGN preprocessing
or generate Poisson spikes.

Running this module directly saves the original
example-grating figure and prints diagnostics.

Importing it does not execute the demonstration.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "functional_orientation"
)

PATCH_SIZE = 16

ANGLES = np.arange(0, 180, 15)

FREQUENCY = 2.0
PHASE = 0.0

EXAMPLE_ANGLES = (0, 45, 90, 135)


# ============================================================
# 2. GENERATE SINUSOIDAL GRATING
# ============================================================

def generate_grating(
    angle,
    frequency=FREQUENCY,
    phase=PHASE,
    patch_size=PATCH_SIZE
):
    """
    Generate a sinusoidal grating.

    Parameters
    ----------
    angle : float
        Orientation of the stripes in degrees.

    frequency : float
        Number of cycles across the patch.

    phase : float
        Phase offset in radians.

    patch_size : int
        Spatial width and height.

    Returns
    -------
    grating : ndarray
        Shape (patch_size, patch_size).
        Values between -1 and +1.

    Notes
    -----
    The original centered coordinate grid and
    orientation convention are preserved.

    This function does not generate LGN firing
    rates or spike trains.
    """

    y, x = np.meshgrid(
        np.arange(patch_size),
        np.arange(patch_size),
        indexing="ij"
    )

    x = x - (patch_size - 1) / 2
    y = y - (patch_size - 1) / 2

    theta = np.deg2rad(angle)

    coordinate = (
        -x * np.sin(theta)
        + y * np.cos(theta)
    )

    grating = np.cos(
        2 * np.pi
        * frequency
        * coordinate / patch_size
        + phase
    )

    return grating


# ============================================================
# 3. SAVE EXAMPLE GRATING FIGURE
# ============================================================

def plot_example_gratings(
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Reproduce the original four-orientation figure.

    Original output filename:
        example_gratings.png
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(11, 3)
    )

    for ax, angle in zip(
        axes,
        EXAMPLE_ANGLES
    ):

        grating = generate_grating(angle)

        ax.imshow(
            grating,
            cmap="gray",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )

        ax.set_title(f"{angle}°")

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Sinusoidal grating stimuli: 2 cycles/patch"
    )

    fig.tight_layout()

    save_path = (
        output_dir / "example_gratings.png"
    )

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)

    return save_path


# ============================================================
# 4. PRINT STIMULUS DIAGNOSTICS
# ============================================================

def verify_stimuli():
    """
    Print original stimulus statistics for
    0, 45, 90 and 135 degrees.

    Returns
    -------
    dict
        Orientation-specific diagnostic values.
    """

    print("STIMULUS VERIFICATION")

    results = {}

    for angle in EXAMPLE_ANGLES:

        grating = generate_grating(angle)

        stats = {
            "shape": grating.shape,
            "min": grating.min(),
            "max": grating.max(),
            "mean": grating.mean(),
            "std": grating.std()
        }

        results[angle] = stats

        print(f"\nOrientation: {angle}°")
        print("Shape:", stats["shape"])
        print("Minimum:", stats["min"])
        print("Maximum:", stats["max"])
        print("Mean:", stats["mean"])
        print("Standard deviation:", stats["std"])

    return results


# ============================================================
# 5. MAIN
# ============================================================

def main():
    """
    Reproduce the original stimulus demonstration.
    """

    save_path = plot_example_gratings()

    verify_stimuli()

    print("\nStimulus generation completed!")
    print("Figure saved to:", save_path)


if __name__ == "__main__":
    main()
