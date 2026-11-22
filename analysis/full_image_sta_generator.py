"""Shared full-image STA generator.

Validated against MATLAB-demo STA implementation.
Parent image: 128 x 128 Gaussian white noise.
Extracted patch: 16 x 16.
Original signed noise supplies STA stimuli.

The 128 x 128 parent size is an implementation
assumption, not a verified paper parameter.
"""

import numpy as np

from analysis import spike_triggered_average as sta
from preprocessing import (
    divisive_dog,
    normalize_standard_deviation,
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

