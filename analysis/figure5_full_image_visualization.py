
"""Figure 5: standardized full-image STA and 160x160 Gabor panels."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import gabor_fitting as gf
from analysis import figure5_interpolation_refit as refit


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results" / "paper_protocol" / "seed_3002"
    / "functional_sta"
)


def select_representatives(errors):
    valid = np.isfinite(errors)
    success = np.flatnonzero(valid & (errors < 0.10))
    failure = np.flatnonzero(valid & (errors >= 0.10))

    if not len(success) or not len(failure):
        raise ValueError("Need both successful and failed fits.")

    typical = success[
        np.argmin(
            np.abs(errors[success] - np.median(errors[success]))
        )
    ]
    borderline = failure[np.argmin(errors[failure])]
    severe = failure[np.argmax(errors[failure])]

    return [
        ("Typical success", int(typical)),
        ("Borderline failure", int(borderline)),
        ("High-error failure", int(severe)),
    ]


def plot_track(track):
    base = ROOT / f"track_{track.lower()}"

    sta_path = (
        base / "full_image_100000"
        / "STA_100000_results.npz"
    )
    fit16_path = (
        base / "full_image_gabor_16"
        / "gabor_all_400_fits.npz"
    )
    fit160_path = (
        base / "full_image_gabor_160"
        / "gabor_160_comparison_400.npz"
    )

    output = base / "figure5_full_image_visualizations"
    output_file = output / "sta_gabor_residual_examples.png"

    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing figure: {output_file}"
        )

    for path in (sta_path, fit16_path, fit160_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(sta_path, allow_pickle=False) as data:
        sta = data["STA_E"].copy()

    with np.load(fit16_path, allow_pickle=False) as data:
        errors16 = data["fit_errors"].copy()

    with np.load(fit160_path, allow_pickle=False) as data:
        parameters = data["parameters_160"].copy()
        errors160 = data["refitted_errors_160"].copy()
        reference_errors16 = data["original_errors_16"].copy()
        indices = data["indices"].copy()

    if sta.shape != (400, 256):
        raise ValueError(f"Unexpected STA shape: {sta.shape}")

    if parameters.shape != (400, 8):
        raise ValueError(
            f"Unexpected Gabor parameter shape: {parameters.shape}"
        )

    if errors160.shape != (400,) or errors16.shape != (400,):
        raise ValueError("Unexpected Gabor error shape.")

    if not np.array_equal(indices, np.arange(400)):
        raise ValueError("Gabor neuron indices do not match.")

    if not np.allclose(
        errors16, reference_errors16,
        rtol=1e-8, atol=1e-10, equal_nan=True
    ):
        raise ValueError(
            "16x16 Gabor errors disagree between saved files."
        )

    representatives = select_representatives(errors160)

    fig, axes = plt.subplots(
        3, 4, figsize=(13, 9), squeeze=False
    )

    headers = [
        "Original STA (16x16)",
        "Interpolated STA (160x160)",
        "Optimized Gabor (160x160)",
        "Residual (STA - Gabor)",
    ]

    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=11)

    for row, (category, neuron) in enumerate(representatives):
        rf16 = gf.as_map(sta[neuron])
        rf160 = refit.interpolate_160(rf16)

        gabor160 = refit.gabor_on_grid(
            parameters[neuron],
            refit.X160,
            refit.Y160
        ).reshape(refit.FINE_SIDE, refit.FINE_SIDE)

        residual = rf160 - gabor160

        scale = max(
            float(np.max(np.abs(rf160))),
            float(np.max(np.abs(gabor160))),
            1e-12
        )
        residual_scale = max(
            float(np.max(np.abs(residual))), 1e-12
        )

        images = [rf16, rf160, gabor160, residual]

        for col, image in enumerate(images):
            limit = residual_scale if col == 3 else scale

            axes[row, col].imshow(
                image,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                origin="upper"
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(
            f"{category}\nE-{neuron + 1:03d}",
            fontsize=10
        )

        axes[row, 2].set_xlabel(
            f"Normalized error = {errors160[neuron]:.4f}",
            fontsize=10
        )

        print(
            f"  {category}: E-{neuron + 1:03d}, "
            f"error={errors160[neuron]:.6f}"
        )

    fig.suptitle(
        f"Track {track.upper()} ? Full-image STA and "
        "160x160 Gabor fitting",
        fontsize=14
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("16x16 passes:", int(np.sum(errors16 < 0.10)))
    print("160x160 passes:", int(np.sum(errors160 < 0.10)))
    print("Saved:", output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track", choices=("A", "B"), required=True
    )
    args = parser.parse_args()

    gf.check_spatial_order()
    plot_track(args.track)


if __name__ == "__main__":
    main()
