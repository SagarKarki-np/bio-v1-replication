
"""
Alternative full-image STA preprocessing pilot.

Uses the frozen MATLAB-demo E02_I04 network.

Only the white-noise stimulus generator differs.
The original STA analysis functions are reused.

This pilot does not modify existing results.
"""

from pathlib import Path
import argparse
import numpy as np

from analysis import spike_triggered_average as sta
from preprocessing import (
    divisive_dog,
    normalize_standard_deviation,
)


PROJECT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
    / "E02_I04"
)

CHECKPOINT = (
    ROOT
    / "training"
    / "checkpoints"
    / "batch_1100.npz"
)

OUTPUT = (
    ROOT
    / "sta_full_image_normalization_test"
    / "pilot_20"
)

IMAGE_SIZE = 128
BUFFER = 20


def generate_full_image_sta_batch(
    center_kernel,
    surround_kernel,
    divisive_kernel,
    diagnostics
):
    """
    Generate full-image-preprocessed white-noise stimuli.

    Original signed noise supplies the STA stimulus.
    The corresponding filtered patch supplies LGN rates.
    """

    stimuli = np.zeros(
        (sta.STA_BATCH_SIZE, sta.N_PIXELS)
    )

    LGN_spikes = np.zeros(
        (
            sta.N_LGN,
            sta.STA_BATCH_SIZE,
            sta.N_STEPS
        ),
        dtype=bool
    )

    for k in range(sta.STA_BATCH_SIZE):

        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(IMAGE_SIZE, IMAGE_SIZE)
        )

        grayscale = np.clip(
            50.0 * noise + 100.0,
            0.0,
            255.0
        )

        filtered = divisive_dog(
            grayscale,
            center_kernel,
            surround_kernel,
            divisive_kernel
        )

        normalized = normalize_standard_deviation(
            filtered
        )

        row = np.random.randint(
            BUFFER,
            IMAGE_SIZE - sta.PATCH_SIZE - BUFFER + 1
        )

        col = np.random.randint(
            BUFFER,
            IMAGE_SIZE - sta.PATCH_SIZE - BUFFER + 1
        )

        row_end = row + sta.PATCH_SIZE
        col_end = col + sta.PATCH_SIZE

        # Corresponding locations in the original noise
        # and the fully preprocessed image.
        stimulus_patch = noise[
            row:row_end,
            col:col_end
        ]

        filtered_patch = normalized[
            row:row_end,
            col:col_end
        ]

        assert stimulus_patch.shape == (
            sta.PATCH_SIZE, sta.PATCH_SIZE
        )

        assert filtered_patch.shape == (
            sta.PATCH_SIZE, sta.PATCH_SIZE
        )

        stimuli[k] = stimulus_patch.flatten(
            order="F"
        )

        signed = (
            filtered_patch.flatten(order="F")
            * sta.LGN_STA_SCALE
        )

        rates = np.concatenate((
            np.maximum(signed, 0.0),
            np.maximum(-signed, 0.0)
        ))

        rates = np.clip(
            rates,
            0.0,
            sta.LGN_MAX_RATE
        )

        assert rates.shape == (sta.N_LGN,)
        assert np.isfinite(rates).all()

        # Preserve the original LGN diagnostics.
        diagnostics["total_rate_values"] += rates.size

        diagnostics["total_capped"] += np.count_nonzero(
            rates >= sta.LGN_MAX_RATE
        )

        diagnostics["total_zero"] += np.count_nonzero(
            rates == 0
        )

        diagnostics["total_rate_sum"] += rates.sum()

        diagnostics["total_rate_squared_sum"] += np.sum(
            rates ** 2
        )

        diagnostics["max_rate_observed"] = max(
            diagnostics["max_rate_observed"],
            rates.max()
        )

        probabilities = rates[:, None] * sta.DT

        LGN_spikes[:, k, :] = (
            np.random.random(
                (sta.N_LGN, sta.N_STEPS)
            ) < probabilities
        )

    return stimuli, LGN_spikes


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batches",
        type=int,
        default=1
    )

    args = parser.parse_args()

    if args.batches != 1:
        parser.error(
            "This script currently supports only "
            "the 1-batch technical pilot."
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to reuse pilot output: {OUTPUT}"
        )

    # Confirm the original STA loader accepts
    # the frozen MATLAB-demo checkpoint.
    weights, theta_E, theta_I = (
        sta.load_trained_network(CHECKPOINT)
    )

    assert len(weights) == 6
    assert theta_E.shape == (sta.N_E, 1)
    assert theta_I.shape == (sta.N_I, 1)

    print("Frozen MATLAB-demo checkpoint verified.")
    print("Alternative preprocessing: full-image.")
    print("Image size:", IMAGE_SIZE)
    print("Stimuli:", sta.STA_BATCH_SIZE)
    print("Output:", OUTPUT)

    # Temporary module-level substitution, used only
    # inside this standalone pilot process.
    # No source files are modified.
    original_generator = sta.generate_sta_batch

    try:
        sta.generate_sta_batch = (
            generate_full_image_sta_batch
        )

        sta.analyze_spike_triggered_average(
            source_project=PROJECT,
            n_batches=1,
            output_dir=OUTPUT,
            show=False,
            checkpoint_override=CHECKPOINT
        )

    finally:
        sta.generate_sta_batch = original_generator

    print("\nAlternative STA pilot completed.")
    print("Existing baseline outputs unchanged.")


if __name__ == "__main__":
    main()
