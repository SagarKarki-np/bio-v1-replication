
"""
Figure 5: Visual comparison of Gabor receptive fields.

For Track A and Track B, display:
1. Original 16 x 16 STA
2. Bilinearly interpolated 160 x 160 STA
3. Original Gabor fit, evaluated at 160 x 160
4. Reoptimized Gabor fit at 160 x 160

Neuron categories are selected using ORIGINAL 16 x 16
fitting errors to avoid selecting based on the new results.

No simulations or optimization are performed.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis import gabor_fitting as gf

from analysis.figure5_interpolation_refit import (
    interpolate_160,
    gabor_on_grid,
    X160,
    Y160,
)


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT
    / "results"
    / "paper_protocol"
    / "seed_3002"
)

THRESHOLD = 0.10


def get_paths(track):

    track = track.lower()

    return {
        "sta": (
            BASE / "functional_sta"
            / f"track_{track}"
            / "full_100000"
            / "STA_100000_results.npz"
        ),

        "original": (
            BASE / "gabor_fitting"
            / f"track_{track}_100000"
            / "gabor_all_400_fits.npz"
        ),

        "refitted": (
            BASE / "gabor_fitting"
            / f"interpolation_sensitivity_track_{track}"
            / "gabor_160_refit_progress.npz"
        ),
    }


def load_track(track):

    paths = get_paths(track)

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} file:\n{path}"
            )

    with np.load(paths["sta"]) as data:
        sta = data["STA_E"].copy()

    with np.load(paths["original"]) as data:
        old_params = data["parameters"].copy()
        old_errors = data["fit_errors"].copy()

    with np.load(paths["refitted"]) as data:
        new_params = data["parameters_160"].copy()
        new_errors = data["errors_160"].copy()
        completed = data["completed"].copy()

    assert sta.shape == (400, 256)
    assert old_params.shape == (400, 8)
    assert new_params.shape == (400, 8)
    assert np.all(completed), (
        f"Track {track}: incomplete 160 x 160 fitting"
    )

    return sta, old_params, old_errors, new_params, new_errors


def select_representatives(errors):

    errors = np.asarray(errors)

    successful = np.flatnonzero(
        np.isfinite(errors) & (errors < THRESHOLD)
    )

    unsuccessful = np.flatnonzero(
        np.isfinite(errors) & (errors >= THRESHOLD)
    )

    if not len(successful) or not len(unsuccessful):
        raise ValueError(
            "Need both successful and failed fits"
        )

    # A typical successful neuron, not the best case.
    successful_median = np.median(errors[successful])

    typical = successful[
        np.argmin(
            np.abs(
                errors[successful] - successful_median
            )
        )
    ]

    # Nearest failure above the threshold.
    borderline = unsuccessful[
        np.argmin(errors[unsuccessful])
    ]

    # Largest original fitting error.
    severe = unsuccessful[
        np.argmax(errors[unsuccessful])
    ]

    return [
        ("Typical success", int(typical)),
        ("Borderline failure", int(borderline)),
        ("High-error failure", int(severe)),
    ]


def plot_track(track, output):

    (
        sta,
        old_params,
        old_errors,
        new_params,
        new_errors
    ) = load_track(track)

    selections = select_representatives(old_errors)

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15, 11),
        constrained_layout=True
    )

    headings = [
        "Original STA\n16 x 16",
        "Interpolated STA\n160 x 160",
        "Original Gabor\n160 x 160",
        "Reoptimized Gabor\n160 x 160",
    ]

    for col, heading in enumerate(headings):
        axes[0, col].set_title(
            heading,
            fontsize=12,
            fontweight="bold"
        )

    print(f"\nTRACK {track.upper()}")
    print("-" * 55)

    for row, (category, neuron) in enumerate(selections):

        rf16 = gf.as_map(sta[neuron])
        rf160 = interpolate_160(rf16)

        predicted_old = gabor_on_grid(
            old_params[neuron],
            X160,
            Y160
        ).reshape(160, 160)

        predicted_new = gabor_on_grid(
            new_params[neuron],
            X160,
            Y160
        ).reshape(160, 160)

        images = [
            rf16,
            rf160,
            predicted_old,
            predicted_new
        ]

        # Same color scale for all images of this neuron.
        vmax = max(
            float(np.max(np.abs(image)))
            for image in images
        )

        vmax = max(vmax, 1e-12)

        for col, image in enumerate(images):

            ax = axes[row, col]

            ax.imshow(
                image,
                cmap="RdBu_r",
                origin="lower",
                interpolation=(
                    "nearest" if col == 0 else "none"
                ),
                vmin=-vmax,
                vmax=vmax,
                extent=[0, 15, 0, 15]
            )

            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel(
            f"{category}\n"
            f"Neuron E-{neuron + 1:03d}\n"
            f"Original error: {old_errors[neuron]:.4f}",
            fontsize=10
        )

        axes[row, 3].set_xlabel(
            f"Refitted 160 error: "
            f"{new_errors[neuron]:.4f}",
            fontsize=10
        )

        print(
            f"{category}: E-{neuron + 1:03d} | "
            f"original={old_errors[neuron]:.5f} | "
            f"refitted160={new_errors[neuron]:.5f}"
        )

    fig.suptitle(
        f"Track {track.upper()}: "
        "Gabor Receptive-Field Analysis",
        fontsize=16,
        fontweight="bold"
    )

    filename = output / (
        f"figure5_gabor_examples_track_{track}.png"
    )

    fig.savefig(
        filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    print("Saved:", filename)

    return selections


def main():

    gf.check_spatial_order()

    output = (
        BASE
        / "gabor_fitting"
        / "figure5_visual_comparison"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = {}

    for track in ("a", "b"):

        selected[track] = plot_track(
            track,
            output
        )

    with open(
        output / "selected_neurons.txt",
        "w",
        encoding="utf-8"
    ) as file:

        for track, selections in selected.items():

            file.write(
                f"\nTrack {track.upper()}\n"
            )

            for category, neuron in selections:
                file.write(
                    f"{category}: E-{neuron + 1:03d}\n"
                )

    print("\nFigure 5 visualization completed.")
    print("Output directory:", output)


if __name__ == "__main__":
    main()
