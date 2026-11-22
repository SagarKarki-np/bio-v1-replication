
"""
Spiking dynamics for the LGN-V1 neural network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Weight convention
-----------------
W[postsynaptic_neuron, presynaptic_neuron]

Weight tuple ordering:
    W_LGN_E, W_LGN_I, W_E_E,
    W_E_I, W_I_I, W_I_E

This module provides:
    - LGN Poisson-like spike generation
    - Membrane potential decay
    - Synaptic input calculation
    - Excitatory and inhibitory input integration
    - Gaussian membrane noise
    - Spike detection and voltage reset
    - Single-stimulus and batch simulation
    - Optional synaptic-input diagnostics

The voltage update order follows the original project:

    1. Detect spikes from current membrane voltages.
    2. Reset firing neurons.
    3. Calculate feedforward and recurrent synaptic inputs.
    4. Apply membrane decay, synaptic input and noise.
    5. Apply the minimum voltage limit.

Consequently, spikes are recorded before the new voltage
for that timestep is calculated.
"""

import numpy as np

from config import (
    N_LGN,
    N_E,
    N_I,
    DT,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
)

from network import initialize_neuron_states


# ============================================================
# 1. LGN SPIKE GENERATION
# ============================================================

def generate_lgn_population_spikes(LGN_rates, dt, n_steps):
    """
    Generate discrete-time Poisson-like LGN spike trains.

    Each neuron performs one Bernoulli trial per timestep.

    Parameters
    ----------
    LGN_rates : numpy.ndarray
        Firing rates in Hz, shape (N_LGN,).
    dt : float
        Simulation timestep in seconds.
    n_steps : int
        Number of simulation timesteps.

    Returns
    -------
    numpy.ndarray
        Boolean spike matrix of shape (N_LGN, n_steps).
    """
    LGN_rates = np.asarray(LGN_rates)

    if LGN_rates.ndim != 1:
        raise ValueError("LGN_rates must be one-dimensional.")

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    probabilities = LGN_rates * dt

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("LGN rates must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(
            "LGN spike probabilities must lie between 0 and 1."
        )

    random_numbers = np.random.rand(
        LGN_rates.shape[0],
        n_steps
    )

    return random_numbers < probabilities[:, np.newaxis]


# ============================================================
# 2. MEMBRANE POTENTIAL DECAY
# ============================================================

def membrane_decay(V, dt, tau_m):
    """
    Apply exponential decay to membrane potentials.

    V_decayed = V * exp(-dt / tau_m)
    """
    decay = np.exp(-dt / tau_m)

    return V * decay


# ============================================================
# 3. SYNAPTIC INPUT CALCULATIONS
# ============================================================

def calculate_synaptic_input(weights, presynaptic_spikes):
    """
    Calculate weighted synaptic input.

    Parameters
    ----------
    weights : numpy.ndarray
        Weight matrix with shape (n_post, n_pre).
    presynaptic_spikes : numpy.ndarray
        Spike array with shape (n_pre, batch_size).

    Returns
    -------
    numpy.ndarray
        Input array with shape (n_post, batch_size).
    """
    return weights @ presynaptic_spikes


def calculate_all_synaptic_inputs(
    W_LGN_E,
    W_LGN_I,
    W_E_E,
    W_E_I,
    W_I_E,
    W_I_I,
    S_LGN,
    S_E,
    S_I
):
    """
    Calculate inputs from all six synaptic pathways.

    Returns
    -------
    tuple
        LGN_E, LGN_I, E_E, E_I, I_E, I_I
    """
    input_LGN_E = calculate_synaptic_input(
        W_LGN_E, S_LGN
    )

    input_LGN_I = calculate_synaptic_input(
        W_LGN_I, S_LGN
    )

    input_E_E = calculate_synaptic_input(
        W_E_E, S_E
    )

    input_E_I = calculate_synaptic_input(
        W_E_I, S_E
    )

    input_I_E = calculate_synaptic_input(
        W_I_E, S_I
    )

    input_I_I = calculate_synaptic_input(
        W_I_I, S_I
    )

    return (
        input_LGN_E,
        input_LGN_I,
        input_E_E,
        input_E_I,
        input_I_E,
        input_I_I
    )


def combine_synaptic_inputs(
    input_LGN_E,
    input_LGN_I,
    input_E_E,
    input_E_I,
    input_I_E,
    input_I_I
):
    """
    Combine excitatory and inhibitory synaptic contributions.

    Inhibitory inputs are stored as nonnegative magnitudes
    and subtracted from the total membrane drive.
    """
    total_input_E = (
        input_LGN_E
        + input_E_E
        - input_I_E
    )

    total_input_I = (
        input_LGN_I
        + input_E_I
        - input_I_I
    )

    return total_input_E, total_input_I


# ============================================================
# 4. MEMBRANE POTENTIAL UPDATE
# ============================================================

def update_membrane_potentials(
    V_E,
    V_I,
    total_input_E,
    total_input_I,
    dt,
    tau_m,
    noise_std
):
    """
    Update E and I membrane potentials.

    The update retains the original project's
    exponential decay, independent Gaussian noise,
    and additive synaptic-input convention.
    """
    V_E_decayed = membrane_decay(
        V_E, dt, tau_m
    )

    V_I_decayed = membrane_decay(
        V_I, dt, tau_m
    )

    noise_E = np.random.normal(
        0,
        noise_std,
        size=V_E.shape
    )

    noise_I = np.random.normal(
        0,
        noise_std,
        size=V_I.shape
    )

    V_E_next = (
        V_E_decayed
        + noise_E
        + total_input_E
    )

    V_I_next = (
        V_I_decayed
        + noise_I
        + total_input_I
    )

    return V_E_next, V_I_next


def apply_voltage_floor(V, v_min):
    """
    Enforce a minimum membrane potential.
    """
    return np.maximum(V, v_min)


# ============================================================
# 5. SPIKE DETECTION AND RESET
# ============================================================

def detect_spikes_and_reset(V, threshold, v_reset):
    """
    Detect neurons reaching threshold and reset voltage.

    Spike condition:
        V >= threshold

    Returns
    -------
    spikes : numpy.ndarray
        Boolean spike array.
    V_reset : numpy.ndarray
        Membrane voltages after resetting firing neurons.
    """
    spikes = V >= threshold

    V_after_reset = np.where(
        spikes,
        v_reset,
        V
    )

    return spikes, V_after_reset


# ============================================================
# 6. ONE V1 SIMULATION TIMESTEP
# ============================================================

def update_v1_timestep(
    V_E,
    V_I,
    threshold_E,
    threshold_I,
    S_LGN,
    weights,
    dt,
    tau_m,
    noise_std,
    v_min,
    v_reset,
    return_inputs=False
):
    """
    Advance the E and I neuron populations by one timestep.

    All spike and voltage arrays use shape:
        (number_of_neurons, batch_size)

    The returned S_E and S_I are threshold events
    detected from the incoming membrane voltages.
    """

    # 1. Unpack weights in the established tuple order.
    (
        W_LGN_E,
        W_LGN_I,
        W_E_E,
        W_E_I,
        W_I_I,
        W_I_E
    ) = weights

    # 2. Detect spikes and reset firing neurons.
    S_E, V_E_reset = detect_spikes_and_reset(
        V_E,
        threshold_E,
        v_reset
    )

    S_I, V_I_reset = detect_spikes_and_reset(
        V_I,
        threshold_I,
        v_reset
    )

    # 3. Calculate six synaptic pathways.
    (
        input_LGN_E,
        input_LGN_I,
        input_E_E,
        input_E_I,
        input_I_E,
        input_I_I
    ) = calculate_all_synaptic_inputs(
        W_LGN_E,
        W_LGN_I,
        W_E_E,
        W_E_I,
        W_I_E,
        W_I_I,
        S_LGN,
        S_E,
        S_I
    )

    # 4. Combine excitation and inhibition.
    total_input_E, total_input_I = combine_synaptic_inputs(
        input_LGN_E,
        input_LGN_I,
        input_E_E,
        input_E_I,
        input_I_E,
        input_I_I
    )

    # 5. Update membrane voltages.
    V_E_next, V_I_next = update_membrane_potentials(
        V_E_reset,
        V_I_reset,
        total_input_E,
        total_input_I,
        dt,
        tau_m,
        noise_std
    )

    # 6. Apply minimum voltage.
    V_E_next = apply_voltage_floor(
        V_E_next, v_min
    )

    V_I_next = apply_voltage_floor(
        V_I_next, v_min
    )

    # 7. Optionally return synaptic-input diagnostics.
    
# 7. Optionally return synaptic-input diagnostics.
    if return_inputs:
       diagnostics = {
        # Inputs received by excitatory neurons
        "LGN_E": input_LGN_E,
        "E_E": input_E_E,
        "I_E": input_I_E,
        "total_E": total_input_E,

        # Inputs received by inhibitory neurons
        "LGN_I": input_LGN_I,
        "E_I": input_E_I,
        "I_I": input_I_I,
        "total_I": total_input_I
       }
       return (
        V_E_next,
        V_I_next,
        S_E,
        S_I,
        diagnostics
    )

    return V_E_next, V_I_next, S_E, S_I


# ============================================================
# 7. BATCH V1 SIMULATION
# ============================================================

def simulate_v1_batch(
    LGN_spikes_batch,
    weights,
    dt=DT,
    tau_m=TAU_M,
    noise_std=NOISE_STD,
    v_min=V_MIN,
    v_reset=V_RESET,
    threshold_E=None,
    threshold_I=None,
    record_inputs=False
):
    """
    Simulate a batch of independent LGN-driven presentations.

    Parameters
    ----------
    LGN_spikes_batch : numpy.ndarray
        Boolean LGN spikes with shape:
        (N_LGN, batch_size, n_steps).
    weights : tuple
        Six weight matrices in the established order.
    threshold_E : numpy.ndarray or None
        Excitatory threshold array, shape (N_E, 1).
        Uses initial thresholds when None.
    threshold_I : numpy.ndarray or None
        Inhibitory threshold array, shape (N_I, 1).
        Uses initial thresholds when None.
    record_inputs : bool
        Whether to record synaptic-input diagnostics.

    Returns
    -------
    E_spikes_batch : numpy.ndarray
        Shape (N_E, batch_size, n_steps).
    I_spikes_batch : numpy.ndarray
        Shape (N_I, batch_size, n_steps).

    When record_inputs=True, also returns a dictionary
    with LGN_E, E_E, I_E and total_E input traces.
    """
    LGN_spikes_batch = np.asarray(LGN_spikes_batch)

    if LGN_spikes_batch.ndim != 3:
        raise ValueError(
            "LGN_spikes_batch must have three dimensions."
        )

    if LGN_spikes_batch.shape[0] != N_LGN:
        raise ValueError(
            f"Expected {N_LGN} LGN neurons."
        )

    batch_size = LGN_spikes_batch.shape[1]
    n_steps = LGN_spikes_batch.shape[2]

    if batch_size < 1 or n_steps < 1:
        raise ValueError(
            "Batch size and number of timesteps must be positive."
        )

    (
        V_E,
        V_I,
        initial_threshold_E,
        initial_threshold_I
    ) = initialize_neuron_states(
        batch_size=batch_size
    )

    if threshold_E is None:
        threshold_E = initial_threshold_E

    if threshold_I is None:
        threshold_I = initial_threshold_I

    threshold_E = np.asarray(threshold_E)
    threshold_I = np.asarray(threshold_I)

    if threshold_E.shape not in ((N_E, 1), (N_E, batch_size)):
        raise ValueError(
            "Unexpected excitatory threshold shape."
        )

    if threshold_I.shape not in ((N_I, 1), (N_I, batch_size)):
        raise ValueError(
            "Unexpected inhibitory threshold shape."
        )

    E_spikes_batch = np.zeros(
        (N_E, batch_size, n_steps),
        dtype=bool
    )

    I_spikes_batch = np.zeros(
        (N_I, batch_size, n_steps),
        dtype=bool
    )

    # Allocate diagnostic arrays only if requested.
    
    if record_inputs:
      input_traces = {}

      # Record inputs received by excitatory neurons
      for name in ("LGN_E", "E_E", "I_E", "total_E"):
        input_traces[name] = np.zeros(
            (N_E, batch_size, n_steps)
        )

      # Record inputs received by inhibitory neurons
      for name in ("LGN_I", "E_I", "I_I", "total_I"):
        input_traces[name] = np.zeros(
            (N_I, batch_size, n_steps)
        )

    # Simulation loop.
    for t in range(n_steps):

        S_LGN = LGN_spikes_batch[:, :, t]

        if record_inputs:
            (
                V_E,
                V_I,
                S_E,
                S_I,
                diagnostics
            ) = update_v1_timestep(
                V_E,
                V_I,
                threshold_E,
                threshold_I,
                S_LGN,
                weights,
                dt,
                tau_m,
                noise_std,
                v_min,
                v_reset,
                return_inputs=True
            )

            for name in input_traces:
                input_traces[name][:, :, t] = (
                    diagnostics[name]
                )

        else:
            (
                V_E,
                V_I,
                S_E,
                S_I
            ) = update_v1_timestep(
                V_E,
                V_I,
                threshold_E,
                threshold_I,
                S_LGN,
                weights,
                dt,
                tau_m,
                noise_std,
                v_min,
                v_reset
            )

        E_spikes_batch[:, :, t] = S_E
        I_spikes_batch[:, :, t] = S_I

    if record_inputs:
        return (
            E_spikes_batch,
            I_spikes_batch,
            input_traces
        )

    return E_spikes_batch, I_spikes_batch


# ============================================================
# 8. SINGLE-STIMULUS V1 SIMULATION
# ============================================================

def simulate_v1(
    LGN_spikes,
    weights,
    dt=DT,
    tau_m=TAU_M,
    noise_std=NOISE_STD,
    v_min=V_MIN,
    v_reset=V_RESET
):
    """
    Simulate one LGN-driven visual presentation.

    Parameters
    ----------
    LGN_spikes : numpy.ndarray
        Shape (N_LGN, n_steps).

    Returns
    -------
    E_spikes : numpy.ndarray
        Shape (N_E, n_steps).
    I_spikes : numpy.ndarray
        Shape (N_I, n_steps).

    This wrapper reuses the batch simulation engine to
    ensure consistent membrane and spike dynamics.
    """
    LGN_spikes = np.asarray(LGN_spikes)

    if LGN_spikes.ndim != 2:
        raise ValueError(
            "LGN_spikes must have shape (N_LGN, n_steps)."
        )

    LGN_spikes_batch = LGN_spikes[:, np.newaxis, :]

    E_batch, I_batch = simulate_v1_batch(
        LGN_spikes_batch,
        weights,
        dt=dt,
        tau_m=tau_m,
        noise_std=noise_std,
        v_min=v_min,
        v_reset=v_reset
    )

    return E_batch[:, 0, :], I_batch[:, 0, :]
