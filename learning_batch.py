
"""
Batch-level synaptic learning for the LGN-V1 spiking network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Weight convention
-----------------
W[postsynaptic_neuron, presynaptic_neuron]

This module connects the plasticity functions to the training
pipeline.

For each batch:
    1. Initialize STDP traces.
    2. Initialize accumulated weight-change matrices.
    3. Calculate trace decay factors.
    4. Accumulate STDP across all timesteps.
    5. Divide by batch size and apply learning rates.
    6. Update and normalize synaptic weights.

The input spike trains are generated before calling this
function. Synaptic weights remain unchanged during the
presentation and STDP accumulation.

Learning is applied after the complete batch.
"""

import numpy as np

from config import (
    TRIPLET_POTENTIATION,
    TRIPLET_DEPRESSION,
    SYMMETRIC_STDP_COEFFICIENT,
)

from plasticity import (
    initialize_stdp_traces,
    update_stdp_timestep,
    scale_stdp_updates,
    apply_all_weight_updates,
)


def learn_from_spikes(
    LGN_spikes,
    E_spikes,
    I_spikes,
    weights,
    learning_rates,
    L1_limits,
    dt=0.001,
    tau_short=0.020,
    tau_long=0.050
):
    """
    Perform one batch of synaptic learning.

    Parameters
    ----------
    LGN_spikes : numpy.ndarray
        Boolean LGN spikes with shape:
        (n_lgn, batch_size, n_steps).

    E_spikes : numpy.ndarray
        Excitatory V1 spikes with shape:
        (n_e, batch_size, n_steps).

    I_spikes : numpy.ndarray
        Inhibitory V1 spikes with shape:
        (n_i, batch_size, n_steps).

    weights : dict
        Six current synaptic weight matrices.

    learning_rates : dict
        Learning rates for the six synaptic pathways.

    L1_limits : dict
        Maximum incoming weight sums for each pathway.

    dt : float
        Simulation timestep in seconds.

    tau_short : float
        Short STDP trace time constant in seconds.

    tau_long : float
        Long STDP trace time constant in seconds.

    Returns
    -------
    updated_weights : dict
        Synaptic matrices after STDP and normalization.

    scaled_dW : dict
        Accumulated STDP updates after batch averaging
        and learning-rate scaling, but before weight
        clipping and normalization.

    Notes
    -----
    No synaptic weights are changed inside the timestep
    loop.

    The supplied weight dictionary is not modified in place.
    """

    # --------------------------------------------------------
    # 1. Read population and batch dimensions
    # --------------------------------------------------------

    n_lgn, batch_size, n_steps = LGN_spikes.shape

    n_e = E_spikes.shape[0]
    n_i = I_spikes.shape[0]

    if E_spikes.shape != (n_e, batch_size, n_steps):
        raise ValueError(
            "Excitatory spike-array dimensions do not match."
        )

    if I_spikes.shape != (n_i, batch_size, n_steps):
        raise ValueError(
            "Inhibitory spike-array dimensions do not match."
        )

    if batch_size < 1 or n_steps < 1:
        raise ValueError(
            "Batch size and number of timesteps must be positive."
        )

    # --------------------------------------------------------
    # 2. Initialize STDP traces
    # --------------------------------------------------------

    traces = initialize_stdp_traces(
        n_lgn,
        n_e,
        n_i,
        batch_size
    )

    # --------------------------------------------------------
    # 3. Initialize accumulated weight changes
    # --------------------------------------------------------

    shapes = {
        "LGN_E": (n_e, n_lgn),
        "E_E": (n_e, n_e),
        "LGN_I": (n_i, n_lgn),
        "E_I": (n_i, n_e),
        "I_E": (n_e, n_i),
        "I_I": (n_i, n_i)
    }

    dW = {
        name: np.zeros(shape)
        for name, shape in shapes.items()
    }

    # --------------------------------------------------------
    # 4. Calculate exponential trace decay factors
    # --------------------------------------------------------

    decay_short = np.exp(
        -dt / tau_short
    )

    decay_long = np.exp(
        -dt / tau_long
    )

    # STDP coefficients from config.py.
    A_p = TRIPLET_POTENTIATION
    A_d = TRIPLET_DEPRESSION
    A_i = SYMMETRIC_STDP_COEFFICIENT

    # --------------------------------------------------------
    # 5. Accumulate STDP across all timesteps
    # --------------------------------------------------------

    for t in range(n_steps):

        S_LGN = LGN_spikes[:, :, t]
        S_E = E_spikes[:, :, t]
        S_I = I_spikes[:, :, t]

        traces, dW = update_stdp_timestep(
            S_LGN,
            S_E,
            S_I,
            traces,
            dW,
            decay_short,
            decay_long,
            tau_short,
            tau_long,
            A_p,
            A_d,
            A_i
        )

    # --------------------------------------------------------
    # 6. Average and scale accumulated STDP contributions
    # --------------------------------------------------------

    scaled_dW = scale_stdp_updates(
        dW,
        batch_size,
        learning_rates
    )

    # --------------------------------------------------------
    # 7. Apply weight updates and normalize
    # --------------------------------------------------------

    updated_weights = apply_all_weight_updates(
        weights,
        scaled_dW,
        L1_limits
    )

    return updated_weights, scaled_dW
