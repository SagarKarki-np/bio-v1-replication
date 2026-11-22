
"""
Figure 10: Condition-specific homeostatic adaptation.

Preserves the original threshold-update equation
while allowing different E and I firing-rate targets.

Does not modify the original train_v1.py.
"""

import numpy as np


def update_figure10_thresholds(
    E_spikes,
    I_spikes,
    threshold_E,
    threshold_I,
    target_E_Hz,
    target_I_Hz,
    eta_theta_E,
    eta_theta_I,
    dt=0.001
):
    """
    Update E and I firing thresholds for one learning batch.

    Spike array shapes:
        E_spikes: (n_e, batch_size, n_steps)
        I_spikes: (n_i, batch_size, n_steps)

    Threshold shapes:
        threshold_E: (n_e, 1)
        threshold_I: (n_i, 1)
    """

    if E_spikes.ndim != 3 or I_spikes.ndim != 3:
        raise ValueError("Spike arrays must be three-dimensional.")

    if E_spikes.shape[1:] != I_spikes.shape[1:]:
        raise ValueError("E and I batch dimensions must match.")

    n_steps = E_spikes.shape[2]
    duration = n_steps * dt

    if duration <= 0:
        raise ValueError("Presentation duration must be positive.")

    if target_E_Hz <= 0 or target_I_Hz <= 0:
        raise ValueError("Target firing rates must be positive.")

    threshold_E = np.asarray(threshold_E)
    threshold_I = np.asarray(threshold_I)

    if threshold_E.shape != (E_spikes.shape[0], 1):
        raise ValueError("Invalid E threshold shape.")

    if threshold_I.shape != (I_spikes.shape[0], 1):
        raise ValueError("Invalid I threshold shape.")

    # Expected spike count per presentation.
    target_E = target_E_Hz * duration
    target_I = target_I_Hz * duration

    # Observed mean spike count for each neuron.
    E_counts = E_spikes.sum(axis=2).mean(
        axis=1,
        keepdims=True
    )

    I_counts = I_spikes.sum(axis=2).mean(
        axis=1,
        keepdims=True
    )

    # Original homeostatic update equation.
    delta_E = np.minimum(
        eta_theta_E * (E_counts - target_E),
        eta_theta_E * target_E
    )

    delta_I = np.minimum(
        eta_theta_I * (I_counts - target_I),
        eta_theta_I * target_I
    )

    new_threshold_E = threshold_E + delta_E
    new_threshold_I = threshold_I + delta_I

    return new_threshold_E, new_threshold_I
