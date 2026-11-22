
"""
LGN spike generation from natural-image patches.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This module converts preprocessed Van Hateren image patches
into stochastic LGN spike trains.

Processing pipeline
-------------------
1. Randomly select a preprocessed natural image.
2. Extract and rotate a random 16 x 16 patch.
3. Flatten the patch using MATLAB column-major ordering.
4. Scale the signed DoG responses.
5. Separate positive and negative responses into ON/OFF channels.
6. Cap LGN firing rates at the specified maximum.
7. Generate discrete-time Poisson-like spike trains.

The input images must already have undergone the required
DoG filtering, divisive normalization, and standardization.

Spike generation uses independent Bernoulli draws with:

    P(spike in timestep) = firing_rate * dt

The original random-number-generation sequence is preserved
for reproducibility with legacy NumPy RNG checkpoints.
"""

import numpy as np

from config import (
    PATCH_SIZE,
    LGN_RATE_SCALE,
    LGN_MAX_RATE,
    N_LGN,
)

from preprocessing import extract_random_patch
from simulation import generate_lgn_population_spikes


# ============================================================
# 1. SINGLE-NEURON SPIKE GENERATION
# ============================================================

def generate_single_neuron_spikes(rate, dt, n_steps):
    """
    Generate a discrete-time Poisson-like spike train
    for one LGN neuron.

    Parameters
    ----------
    rate : float
        Neuron firing rate in Hz.
    dt : float
        Simulation timestep in seconds.
    n_steps : int
        Number of simulation timesteps.

    Returns
    -------
    numpy.ndarray
        Boolean spike array with shape (n_steps,).

    Notes
    -----
    Each timestep is an independent Bernoulli trial.

    This function is retained as a convenient utility.
    The population spike generator is used for training.
    """
    if not np.isfinite(rate):
        raise ValueError("Firing rate must be finite.")

    if rate < 0 or rate * dt > 1:
        raise ValueError(
            "Invalid firing rate for spike generation."
        )

    p = rate * dt

    random_number = np.random.rand(n_steps)

    lgn_spikes = random_number < p

    return lgn_spikes


# ============================================================
# 2. BATCH LGN SPIKE GENERATION
# ============================================================

def generate_lgn_batch(images, batch_size, n_steps, dt):
    """
    Generate LGN spike trains from random natural-image patches.

    Parameters
    ----------
    images : numpy.ndarray
        Preprocessed image array with shape:
            (height, width, n_images)

    batch_size : int
        Number of independent image presentations.

    n_steps : int
        Number of timesteps per presentation.

    dt : float
        Simulation timestep in seconds.

    Returns
    -------
    numpy.ndarray
        Boolean LGN spike array with shape:
            (N_LGN, batch_size, n_steps)

    Notes
    -----
    Each presentation independently selects an image
    and extracts a randomly rotated patch.

    The order of random sampling operations is preserved:
        1. Image index
        2. Patch coordinates
        3. Patch rotation
        4. LGN spike draws
    """

    # Allocate output for the entire batch.
    LGN_spikes_batch = np.zeros(
        (N_LGN, batch_size, n_steps),
        dtype=bool
    )

    for i in range(batch_size):

        # ----------------------------------------------------
        # 1. Randomly select a preprocessed image
        # ----------------------------------------------------

        image_index = np.random.randint(
            0,
            images.shape[2]
        )

        image = images[:, :, image_index]

        # ----------------------------------------------------
        # 2. Extract a random 16 x 16 patch
        # ----------------------------------------------------

        patch = extract_random_patch(
            image,
            PATCH_SIZE,
            20
        )

        # ----------------------------------------------------
        # 3. Flatten using MATLAB column-major ordering
        # ----------------------------------------------------

        patch_vector = patch.flatten(
            order="F"
        )

        # ----------------------------------------------------
        # 4. Scale the signed DoG responses
        # ----------------------------------------------------

        scaled_patch = (
            patch_vector * LGN_RATE_SCALE
        )

        # ----------------------------------------------------
        # 5. Separate ON and OFF responses
        # ----------------------------------------------------

        ON_rates = np.maximum(
            scaled_patch, 0
        )

        OFF_rates = np.maximum(
            -scaled_patch, 0
        )

        LGN_rates = np.concatenate(
            (ON_rates, OFF_rates)
        )

        # ----------------------------------------------------
        # 6. Cap LGN firing rates
        # ----------------------------------------------------

        LGN_rates = np.clip(
            LGN_rates,
            0,
            LGN_MAX_RATE
        )

        # ----------------------------------------------------
        # 7. Generate stochastic LGN spikes
        # ----------------------------------------------------

        single_LGN_spikes = generate_lgn_population_spikes(
            LGN_rates,
            dt,
            n_steps
        )

        # ----------------------------------------------------
        # 8. Store this presentation
        # ----------------------------------------------------

        LGN_spikes_batch[:, i, :] = (
            single_LGN_spikes
        )

    return LGN_spikes_batch
