
"""
Synaptic plasticity and homeostatic learning for the LGN-V1 network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Weight convention
-----------------
W[postsynaptic_neuron, presynaptic_neuron]

This module implements:
    - Homeostatic threshold adaptation
    - Triplet spike-timing-dependent plasticity
    - Symmetric spike-timing-dependent plasticity
    - Short and long spike traces
    - STDP accumulation across timesteps
    - Learning-rate and batch-size scaling
    - Subtractive and multiplicative weight normalization

IMPORTANT
---------
The order of STDP operations is preserved:

1. Decay existing traces.
2. Calculate STDP contributions.
3. Accumulate weight changes.
4. Add current spikes to traces.

Actual synaptic weights are updated separately from trace
accumulation.

All pathway matrices follow the postsynaptic-by-presynaptic
convention.
"""

import numpy as np


# ============================================================
# 1. HOMEOSTATIC THRESHOLD ADAPTATION
# ============================================================

def update_thresholds(
    thresholds,
    observed_counts,
    target,
    eta
):
    """
    Update neuronal firing thresholds using homeostasis.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Current neuronal firing thresholds.
    observed_counts : numpy.ndarray
        Average observed spike counts per presentation
        for each neuron.
    target : float
        Target spike count per presentation.
    eta : float
        Homeostatic learning coefficient.

    Returns
    -------
    numpy.ndarray
        Updated neuronal thresholds.

    Notes
    -----
    The positive threshold adjustment is capped at
    target * eta.

    Negative adjustments are not capped by this rule.

    The caller must ensure that observed_counts and
    target use the same units.
    """
    delta_theta = (
        observed_counts - target
    ) * eta

    delta_theta = np.minimum(
        delta_theta,
        target * eta
    )

    new_thresholds = thresholds + delta_theta

    return new_thresholds


# ============================================================
# 2. TRIPLET STDP
# ============================================================

def triplet_stdp_matrix(
    S_pre,
    S_post,
    U_pre_short,
    U_post_short,
    U_post_long,
    A_p,
    A_d
):
    """
    Calculate one timestep of triplet-STDP weight changes.

    Parameters
    ----------
    S_pre : numpy.ndarray
        Presynaptic spikes, shape (n_pre, batch_size).
    S_post : numpy.ndarray
        Postsynaptic spikes, shape (n_post, batch_size).
    U_pre_short : numpy.ndarray
        Short presynaptic spike traces.
    U_post_short : numpy.ndarray
        Short postsynaptic spike traces.
    U_post_long : numpy.ndarray
        Long postsynaptic spike traces.
    A_p : float
        Potentiation coefficient.
    A_d : float
        Depression coefficient.

    Returns
    -------
    numpy.ndarray
        Weight-change matrix with shape (n_post, n_pre).

    Notes
    -----
    Contributions are summed over the batch.

    Batch-size division and learning-rate scaling
    are performed separately.
    """
    potentiation = A_p * (
        (S_post * U_post_long) @ U_pre_short.T
    )

    depression = A_d * (
        U_post_short @ S_pre.T
    )

    dW = potentiation - depression

    return dW


# ============================================================
# 3. SYMMETRIC STDP
# ============================================================

def symmetric_stdp_matrix(
    S_pre,
    S_post,
    U_pre,
    U_post,
    A_i
):
    """
    Calculate one timestep of symmetric STDP.

    Includes:
        - Presynaptic-before-postsynaptic contribution
        - Postsynaptic-before-presynaptic contribution
        - Simultaneous spike contribution

    Parameters
    ----------
    S_pre, S_post : numpy.ndarray
        Presynaptic and postsynaptic spike matrices.
    U_pre, U_post : numpy.ndarray
        Corresponding short spike traces.
    A_i : float
        Symmetric STDP coefficient.

    Returns
    -------
    numpy.ndarray
        Weight-change matrix with shape (n_post, n_pre).
    """
    pre_before_post = A_i * (
        S_post @ U_pre.T
    )

    post_before_pre = A_i * (
        U_post @ S_pre.T
    )

    simultaneous = A_i * (
        S_post @ S_pre.T
    )

    dW = (
        pre_before_post
        + post_before_pre
        + simultaneous
    )

    return dW


# ============================================================
# 4. STDP TRACE INITIALIZATION
# ============================================================

def initialize_stdp_traces(
    n_lgn,
    n_e,
    n_i,
    batch_size
):
    """
    Initialize short and long spike traces.

    All trace arrays have shape:
        (number_of_neurons, batch_size)

    Returns
    -------
    dict
        Dictionary containing five trace matrices.
    """
    traces = {
        "LGN_short": np.zeros((n_lgn, batch_size)),
        "E_short": np.zeros((n_e, batch_size)),
        "E_long": np.zeros((n_e, batch_size)),
        "I_short": np.zeros((n_i, batch_size)),
        "I_long": np.zeros((n_i, batch_size))
    }

    return traces


# ============================================================
# 5. STDP FOR ALL SIX PATHWAYS
# ============================================================

def calculate_all_stdp(
    S_LGN,
    S_E,
    S_I,
    traces,
    A_p,
    A_d,
    A_i
):
    """
    Calculate one timestep's STDP contributions.

    The supplied traces must already be decayed.

    Triplet STDP:
        LGN -> E
        E   -> E

    Symmetric STDP:
        LGN -> I
        E   -> I
        I   -> E
        I   -> I

    Returns
    -------
    dict
        Unscaled weight changes for all six pathways.
    """

    # LGN -> E: Triplet STDP
    dW_LGN_E = triplet_stdp_matrix(
        S_LGN,
        S_E,
        traces["LGN_short"],
        traces["E_short"],
        traces["E_long"],
        A_p,
        A_d
    )

    # E -> E: Triplet STDP
    dW_E_E = triplet_stdp_matrix(
        S_E,
        S_E,
        traces["E_short"],
        traces["E_short"],
        traces["E_long"],
        A_p,
        A_d
    )

    # LGN -> I: Symmetric STDP
    dW_LGN_I = symmetric_stdp_matrix(
        S_LGN,
        S_I,
        traces["LGN_short"],
        traces["I_short"],
        A_i
    )

    # E -> I: Symmetric STDP
    dW_E_I = symmetric_stdp_matrix(
        S_E,
        S_I,
        traces["E_short"],
        traces["I_short"],
        A_i
    )

    # I -> E: Symmetric STDP
    dW_I_E = symmetric_stdp_matrix(
        S_I,
        S_E,
        traces["I_short"],
        traces["E_short"],
        A_i
    )

    # I -> I: Symmetric STDP
    dW_I_I = symmetric_stdp_matrix(
        S_I,
        S_I,
        traces["I_short"],
        traces["I_short"],
        A_i
    )

    return {
        "LGN_E": dW_LGN_E,
        "E_E": dW_E_E,
        "LGN_I": dW_LGN_I,
        "E_I": dW_E_I,
        "I_E": dW_I_E,
        "I_I": dW_I_I
    }


# ============================================================
# 6. ONE TIMESTEP OF STDP
# ============================================================

def update_stdp_timestep(
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
):
    """
    Update STDP traces and accumulate weight changes.

    The trace and dW dictionaries are modified in place.

    Actual synaptic weights are not changed here.

    Operation order:
        1. Decay old traces
        2. Calculate STDP contributions
        3. Accumulate dW
        4. Add current spikes to traces

    Returns
    -------
    tuple
        Updated (traces, dW) dictionaries.
    """

    # 1. Decay existing traces.
    traces["LGN_short"] *= decay_short
    traces["E_short"] *= decay_short
    traces["E_long"] *= decay_long
    traces["I_short"] *= decay_short
    traces["I_long"] *= decay_long

    # 2. Calculate weight-change contributions.
    changes = calculate_all_stdp(
        S_LGN,
        S_E,
        S_I,
        traces,
        A_p,
        A_d,
        A_i
    )

    # 3. Accumulate changes across timesteps.
    for name in dW:
        dW[name] += changes[name]

    # 4. Add current spikes to traces.
    traces["LGN_short"] += S_LGN / tau_short
    traces["E_short"] += S_E / tau_short
    traces["E_long"] += S_E / tau_long
    traces["I_short"] += S_I / tau_short
    traces["I_long"] += S_I / tau_long

    return traces, dW


# ============================================================
# 7. SCALE ACCUMULATED STDP UPDATES
# ============================================================

def scale_stdp_updates(
    dW,
    batch_size,
    learning_rates
):
    """
    Scale accumulated weight updates by batch size and
    pathway-specific learning rates.

    scaled_dW = dW / batch_size * learning_rate

    Does not modify actual synaptic weights.
    """
    scaled_dW = {}

    for name, matrix in dW.items():

        eta = learning_rates[name]

        scaled_dW[name] = (
            matrix / batch_size
        ) * eta

    return scaled_dW


# ============================================================
# 8. LEARNED WEIGHT NORMALIZATION
# ============================================================

def normalize_learned_weights(
    W,
    dW,
    L1_limit,
    recurrent=False
):
    """
    Apply STDP updates and weight normalization.

    Processing order:
        1. Apply STDP update
        2. Clip negative weights
        3. Remove recurrent self-connections
        4. Calculate incoming row sums
        5. Apply subtractive normalization
        6. Clip negative weights again
        7. Apply multiplicative normalization

    Parameters
    ----------
    W : numpy.ndarray
        Current weight matrix, shape (n_post, n_pre).
    dW : numpy.ndarray
        Scaled STDP weight changes.
    L1_limit : float
        Maximum total incoming weight per neuron.
    recurrent : bool
        Remove diagonal self-connections when True.

    Returns
    -------
    numpy.ndarray
        Updated nonnegative weight matrix.

    Notes
    -----
    The original normalization denominator W.shape[1]
    is preserved, including recurrent matrices.

    Rows whose updated sum is zero remain zero.
    """

    # 1. Apply synaptic changes.
    W_new = W + dW

    # 2. Prevent negative synaptic weights.
    W_new = np.maximum(W_new, 0.0)

    # 3. Remove recurrent self-connections.
    if recurrent:
        np.fill_diagonal(W_new, 0.0)

    # 4. Calculate incoming weight totals.
    row_sums = np.sum(
        W_new,
        axis=1,
        keepdims=True
    )

    # 5. Subtractive normalization.
    correction = (
        -row_sums + np.minimum(row_sums, L1_limit)
    ) / W_new.shape[1]

    W_new = W_new + correction

    # 6. Clip negative weights again.
    W_new = np.maximum(W_new, 0.0)

    # 7. Multiplicative normalization.
    row_sums = np.sum(
        W_new,
        axis=1,
        keepdims=True
    )

    targets = np.minimum(
        row_sums,
        L1_limit
    )

    W_new = np.divide(
        W_new * targets,
        row_sums,
        out=np.zeros_like(W_new),
        where=row_sums > 0
    )

    return W_new


# ============================================================
# 9. APPLY UPDATES TO ALL PATHWAYS
# ============================================================

def apply_all_weight_updates(
    weights,
    scaled_dW,
    L1_limits
):
    """
    Apply synaptic updates to all six pathways.

    Parameters
    ----------
    weights : dict
        Current synaptic weight matrices.
    scaled_dW : dict
        Learning-rate-scaled STDP updates.
    L1_limits : dict
        Incoming L1 limits for each pathway.

    Returns
    -------
    dict
        Updated normalized weight matrices.

    Recurrent self-connections are removed for
    E -> E and I -> I pathways.
    """
    updated_weights = {}

    for name, W in weights.items():

        recurrent = name in ("E_E", "I_I")

        updated_weights[name] = normalize_learned_weights(
            W,
            scaled_dW[name],
            L1_limits[name],
            recurrent=recurrent
        )

    return updated_weights
