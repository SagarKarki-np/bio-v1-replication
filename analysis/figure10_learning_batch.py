
"""
Figure 10: Parameterized synaptic learning.

Ruslim et al. (2025) LGN-V1 network replication.

This module implements one batch of synaptic learning
with explicitly configurable STDP coefficients.

It reuses the validated plasticity functions from
the main project.

Differences from learning_batch.py
----------------------------------
1. Triplet potentiation A_p is an argument.
2. Triplet depression A_d is an argument.
3. Symmetric STDP coefficient A_i is an argument.

The STDP equations, trace initialization, learning-rate
scaling and weight normalization remain unchanged.

The original learning_batch.py is not modified.
"""

import numpy as np

from plasticity import (
    initialize_stdp_traces,
    update_stdp_timestep,
    scale_stdp_updates,
    apply_all_weight_updates
)


# ============================================================
# 1. PARAMETERIZED BATCH LEARNING
# ============================================================

def learn_from_spikes_figure10(
    LGN_spikes,
    E_spikes,
    I_spikes,
    weights,
    learning_rates,
    L1_limits,
    A_p,
    A_d,
    A_i,
    dt=0.001,
    tau_short=0.020,
    tau_long=0.050
):
    """
    Perform one batch of condition-specific synaptic learning.

    Parameters
    ----------
    LGN_spikes : ndarray
        Shape (n_lgn, batch_size, n_steps).

    E_spikes : ndarray
        Shape (n_e, batch_size, n_steps).

    I_spikes : ndarray
        Shape (n_i, batch_size, n_steps).

    weights : dict
        Six synaptic weight matrices.

    learning_rates : dict
        Learning rates for all six pathways.

    L1_limits : dict
        Incoming L1 weight limits.

    A_p : float
        Triplet STDP potentiation coefficient.

    A_d : float
        Triplet STDP depression coefficient.

    A_i : float
        Symmetric inhibitory STDP coefficient.

    dt : float
        Simulation timestep in seconds.

    tau_short : float
        Short STDP time constant.

    tau_long : float
        Long STDP time constant.

    Returns
    -------
    updated_weights : dict
        Updated and normalized weight matrices.

    scaled_dW : dict
        Scaled STDP changes before weight application.
    """

    # --------------------------------------------------------
    # 1. Validate spike-array dimensions
    # --------------------------------------------------------

    n_lgn, batch_size, n_steps = LGN_spikes.shape

    n_e = E_spikes.shape[0]
    n_i = I_spikes.shape[0]

    if E_spikes.shape != (
        n_e, batch_size, n_steps
    ):
        raise ValueError(
            "Invalid excitatory spike dimensions."
        )

    if I_spikes.shape != (
        n_i, batch_size, n_steps
    ):
        raise ValueError(
            "Invalid inhibitory spike dimensions."
        )

    if batch_size < 1 or n_steps < 1:
        raise ValueError(
            "Batch size and timesteps must be positive."
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
    # 4. Calculate STDP trace decay factors
    # --------------------------------------------------------

    decay_short = np.exp(
        -dt / tau_short
    )

    decay_long = np.exp(
        -dt / tau_long
    )

    # --------------------------------------------------------
    # 5. Accumulate STDP across timesteps
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
    # 6. Average and scale weight changes
    # --------------------------------------------------------

    scaled_dW = scale_stdp_updates(
        dW,
        batch_size,
        learning_rates
    )

    # --------------------------------------------------------
    # 7. Apply updates and normalize
    # --------------------------------------------------------

    updated_weights = apply_all_weight_updates(
        weights,
        scaled_dW,
        L1_limits
    )

    return updated_weights, scaled_dW
