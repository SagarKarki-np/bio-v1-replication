
"""
Network initialization and connectivity for the LGN-V1 spiking network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Weight convention
-----------------
All weight matrices use:

    W[postsynaptic_neuron, presynaptic_neuron]

Matrix naming convention
------------------------
SOURCE_TO_TARGET

Examples:
    W_LGN_E : LGN -> excitatory V1
    W_I_E   : inhibitory V1 -> excitatory V1

This module provides:
    - Synaptic weight initialization
    - Incoming L1 weight normalization
    - Construction of six connectivity matrices
    - Initialization of V1 membrane potentials and thresholds

Membrane dynamics and spike generation are handled separately
in simulation.py.
"""

import numpy as np

from config import (
    N_LGN,
    N_E,
    N_I,
    L1_LGN_TO_E,
    L1_LGN_TO_I,
    L1_E_TO_E,
    L1_E_TO_I,
    L1_I_TO_E,
    L1_I_TO_I,
    INITIAL_WEIGHT_MEAN,
    INITIAL_WEIGHT_STD,
    STRONG_CONNECTION_PROBABILITY,
    WEAK_CONNECTION_DIVISOR,
    INITIAL_THRESHOLD_E,
    INITIAL_THRESHOLD_I,
)


def initialize_weights(
    n_post,
    n_pre,
    mean,
    std,
    strong_probability,
    weak_divisor
):
    """
    Initialize a nonnegative synaptic weight matrix.

    Each weight is first sampled from a Gaussian distribution.

    A fraction of connections, determined by strong_probability,
    retain their sampled values. The remaining connections are
    weakened by dividing their values by weak_divisor.

    Negative sampled weights are clipped to zero.

    Parameters
    ----------
    n_post : int
        Number of postsynaptic neurons.
    n_pre : int
        Number of presynaptic neurons.
    mean : float
        Mean of the Gaussian weight distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    strong_probability : float
        Probability that a connection retains its sampled value.
    weak_divisor : float
        Factor used to weaken the remaining connections.

    Returns
    -------
    numpy.ndarray
        Nonnegative weight matrix with shape (n_post, n_pre).
    """
    if not 0 <= strong_probability <= 1:
        raise ValueError(
            "strong_probability must be between 0 and 1."
        )

    if weak_divisor <= 0:
        raise ValueError(
            "weak_divisor must be positive."
        )

    weights = np.random.normal(
        loc=mean,
        scale=std,
        size=(n_post, n_pre)
    )

    random_number = np.random.rand(n_post, n_pre)

    weights = np.where(
        random_number > strong_probability,
        weights / weak_divisor,
        weights
    )

    weights = np.maximum(weights, 0)

    return weights


def normalize_initial_weights(weights, L1_limit):
    """
    Normalize each postsynaptic neuron's incoming weights.

    All weight values are assumed nonnegative.

    For each row, the sum of incoming synaptic weights
    is scaled to the specified L1_limit.

    Parameters
    ----------
    weights : numpy.ndarray
        Synaptic weight matrix.
    L1_limit : float
        Desired sum of incoming weights per postsynaptic neuron.

    Returns
    -------
    numpy.ndarray
        Row-normalized synaptic weight matrix.

    Raises
    ------
    ValueError
        If a row has zero incoming weight.
    """
    sum_row = np.sum(
        weights,
        axis=1,
        keepdims=True
    )

    if np.any(sum_row == 0):
        raise ValueError(
            "Sum of incoming weights cannot be zero."
        )

    normalized_weights = (
        weights / sum_row
    ) * L1_limit

    return normalized_weights


def create_network_matrices():
    """
    Initialize and normalize six LGN-V1 connectivity matrices.

    Weight convention
    -----------------
    Rows: postsynaptic neurons
    Columns: presynaptic neurons

    Connections
    -----------
    LGN -> E : (N_E, N_LGN)
    LGN -> I : (N_I, N_LGN)
    E   -> E : (N_E, N_E)
    E   -> I : (N_I, N_E)
    I   -> I : (N_I, N_I)
    I   -> E : (N_E, N_I)

    Recurrent E -> E and I -> I self-connections
    are removed before normalization.

    Returns
    -------
    tuple of numpy.ndarray
        Six normalized weight matrices in the original
        simulator's expected ordering.
    """

    # --------------------------------------------------------
    # 1. Initialize synaptic weight matrices
    # --------------------------------------------------------

    W_LGN_E = initialize_weights(
        N_E,
        N_LGN,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    W_LGN_I = initialize_weights(
        N_I,
        N_LGN,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    W_E_E = initialize_weights(
        N_E,
        N_E,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    W_E_I = initialize_weights(
        N_I,
        N_E,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    W_I_I = initialize_weights(
        N_I,
        N_I,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    W_I_E = initialize_weights(
        N_E,
        N_I,
        INITIAL_WEIGHT_MEAN,
        INITIAL_WEIGHT_STD,
        STRONG_CONNECTION_PROBABILITY,
        WEAK_CONNECTION_DIVISOR
    )

    # --------------------------------------------------------
    # 2. Remove recurrent self-connections
    # --------------------------------------------------------

    np.fill_diagonal(W_E_E, 0)
    np.fill_diagonal(W_I_I, 0)

    # --------------------------------------------------------
    # 3. Normalize incoming synaptic weights
    # --------------------------------------------------------

    W_LGN_E = normalize_initial_weights(
        W_LGN_E, L1_LGN_TO_E
    )

    W_LGN_I = normalize_initial_weights(
        W_LGN_I, L1_LGN_TO_I
    )

    W_E_E = normalize_initial_weights(
        W_E_E, L1_E_TO_E
    )

    W_E_I = normalize_initial_weights(
        W_E_I, L1_E_TO_I
    )

    W_I_E = normalize_initial_weights(
        W_I_E, L1_I_TO_E
    )

    W_I_I = normalize_initial_weights(
        W_I_I, L1_I_TO_I
    )

    return (
        W_LGN_E,
        W_LGN_I,
        W_E_E,
        W_E_I,
        W_I_I,
        W_I_E
    )


def initialize_neuron_states(batch_size=1):
    """
    Initialize V1 membrane potentials and firing thresholds.

    Parameters
    ----------
    batch_size : int
        Number of independent stimuli being simulated.

    Returns
    -------
    V_E : numpy.ndarray
        Excitatory voltages with shape (N_E, batch_size).
    V_I : numpy.ndarray
        Inhibitory voltages with shape (N_I, batch_size).
    threshold_E : numpy.ndarray
        Excitatory thresholds with shape (N_E, 1).
    threshold_I : numpy.ndarray
        Inhibitory thresholds with shape (N_I, 1).

    Notes
    -----
    Threshold arrays have one column and can be broadcast
    across multiple stimuli in the same simulation batch.
    """
    V_E = np.zeros((N_E, batch_size))
    V_I = np.zeros((N_I, batch_size))

    threshold_E = np.full(
        (N_E, 1),
        INITIAL_THRESHOLD_E
    )

    threshold_I = np.full(
        (N_I, 1),
        INITIAL_THRESHOLD_I
    )

    return V_E, V_I, threshold_E, threshold_I
