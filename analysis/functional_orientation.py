
"""
Functional orientation responses in the LGN-V1 network.

Reproduces evaluate_functional_orientation.py from the
original V1_network_learning project.

Experiment
----------
12 orientations x 4 phases x 3 repetitions
= 144 stimulus presentations.

Each sinusoidal grating is converted directly into
LGN ON/OFF firing rates, without DoG filtering.

The same LGN spike trains and matched NumPy simulation
noise are used to evaluate:

1. The initial network with calibrated thresholds.
2. The learned network at batch 1100.

The original parameter values, stimulus ordering,
random seed, simulation calls and saved array names
are preserved.

Output
------
results/functional_orientation/
    functional_orientation_responses.npz
"""

from pathlib import Path
import argparse

import numpy as np

from config import (
    DT,
    N_STEPS,
    TAU_M,
    V_MIN,
    V_RESET,
    NOISE_STD,
    LGN_RATE_SCALE,
    LGN_MAX_RATE,
    N_E,
    N_I,
    N_LGN,
)

from analysis.grating_stimuli import generate_grating


from simulation import simulate_v1_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
)

ANGLES = np.arange(0, 180, 15)

PHASES = np.linspace(
    0,
    2 * np.pi,
    4,
    endpoint=False
)

N_REPEATS = 3

FREQUENCY = 2.0

SEED = 2030

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E"
)


# ============================================================
# 2. ORIGINAL GRATING-TO-LGN CONVERSION
# ============================================================

def grating_to_lgn_rates(grating):
    """
    Convert a signed grating into LGN firing rates.

    Important:
        This is the original direct conversion.

        No DoG filtering.
        No divisive normalization.
        No standard-deviation normalization.

    The grating is flattened in MATLAB/Fortran order
    before ON/OFF splitting.

    Returns
    -------
    rates : ndarray, shape (N_LGN,)
    """

    patch_vector = grating.flatten(order="F")

    scaled = patch_vector * LGN_RATE_SCALE

    ON = np.maximum(scaled, 0.0)

    OFF = np.maximum(-scaled, 0.0)

    rates = np.concatenate((ON, OFF))

    return np.clip(
        rates,
        0.0,
        LGN_MAX_RATE
    )


def generate_lgn_population_spikes(rates, dt, n_steps):
    """
    Generate independent Poisson-approximation LGN spikes
    for one stimulus presentation.

    Uses the legacy NumPy RNG to preserve the original
    functional-orientation experiment's random sequence.

    Parameters
    ----------
    rates : ndarray, shape (N_LGN,)
        LGN firing rates in Hz.
    dt : float
        Simulation time step in seconds.
    n_steps : int
        Number of simulation time steps.

    Returns
    -------
    spikes : ndarray, shape (N_LGN, n_steps)
        Boolean LGN spike matrix.
    """

    rates = np.asarray(rates)

    if rates.shape != (N_LGN,):
        raise ValueError(
            f"Expected {(N_LGN,)}, got {rates.shape}"
        )

    probabilities = rates[:, None] * dt

    spikes = (
        np.random.random((N_LGN, n_steps))
        < probabilities
    )

    return spikes


# ============================================================
# 3. GENERATE MATCHED LGN SPIKE TRAINS
# ============================================================

def generate_orientation_lgn_inputs():
    """
    Generate all original orientation presentations.

    Random seed 2030 is set before generating
    Poisson LGN spikes.

    Original loop ordering:
        angle
            phase
                repeat

    Returns
    -------
    LGN_spikes : ndarray
        Shape (N_LGN, 144, N_STEPS).

    presentation_angles : ndarray
    presentation_phases : ndarray
    presentation_repeats : ndarray

    expected_spikes : ndarray
        Expected total LGN spike count per presentation.

    noise_state : tuple
        NumPy random state after LGN generation.
        Reused for both network simulations.
    """

    np.random.seed(SEED)

    n_presentations = (
        len(ANGLES)
        * len(PHASES)
        * N_REPEATS
    )

    LGN_spikes = np.zeros(
        (N_LGN, n_presentations, N_STEPS),
        dtype=bool
    )

    presentation_angles = []
    presentation_phases = []
    presentation_repeats = []
    expected_spikes = []

    index = 0

    for angle in ANGLES:

        for phase in PHASES:

            grating = generate_grating(
                angle,
                frequency=FREQUENCY,
                phase=phase
            )

            rates = grating_to_lgn_rates(
                grating
            )

            if rates.shape != (N_LGN,):
                raise ValueError(
                    f"Unexpected LGN rate shape: "
                    f"{rates.shape}"
                )

            for repeat in range(N_REPEATS):

                spikes = generate_lgn_population_spikes(
                    rates,
                    DT,
                    N_STEPS
                )

                LGN_spikes[:, index, :] = spikes

                presentation_angles.append(angle)
                presentation_phases.append(phase)
                presentation_repeats.append(repeat)

                expected_spikes.append(
                    rates.sum() * DT * N_STEPS
                )

                index += 1

    if index != n_presentations:
        raise RuntimeError(
            "Incorrect number of presentations."
        )

    print("LGN shape:", LGN_spikes.shape)
    print("Presentations:", index)

    print(
        "Mean expected LGN spikes:",
        np.mean(expected_spikes)
    )

    print(
        "Mean observed LGN spikes:",
        LGN_spikes.sum(axis=(0, 2)).mean()
    )

    noise_state = np.random.get_state()

    return {
        "LGN_spikes": LGN_spikes,

        "presentation_angles": np.array(
            presentation_angles
        ),

        "presentation_phases": np.array(
            presentation_phases
        ),

        "presentation_repeats": np.array(
            presentation_repeats
        ),

        "expected_spikes": np.array(
            expected_spikes
        ),

        "noise_state": noise_state,
    }


# ============================================================
# 4. LOAD NETWORK WEIGHTS
# ============================================================

def load_weights(path):
    """
    Load the original six weight matrices.

    Original ordering:
        LGN -> E
        LGN -> I
        E -> E
        E -> I
        I -> I
        I -> E
    """

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

    expected_shapes = (
        (N_E, N_LGN),
        (N_I, N_LGN),
        (N_E, N_E),
        (N_I, N_E),
        (N_I, N_I),
        (N_E, N_I),
    )

    for name, matrix, shape in zip(
        WEIGHT_NAMES,
        weights,
        expected_shapes
    ):

        if matrix.shape != shape:
            raise ValueError(
                f"{name}: expected {shape}, "
                f"got {matrix.shape}"
            )

    return weights


# ============================================================
# 5. LOAD INITIAL AND LEARNED NETWORKS
# ============================================================

def load_network_states(
    source_project,
    learned_run="training_test_25"
):
    """
    Load both original network states.

    Initial weights:
        results/untrained/batch_001/weights.npz

    Initial thresholds:
        results/calibration/full_100_batches/
            calibrated_thresholds.npz

    Learned weights and thresholds:
        results/learning/<learned_run>/
            batch_1100.npz

    Threshold shapes are preserved as (N, 1).
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    initial_weights_path = (
        source_project
        / "results"
        / "untrained"
        / "batch_001"
        / "weights.npz"
    )

    learned_path = (
        source_project
        / "results"
        / "learning"
        / learned_run
        / "batch_1100.npz"
    )

    calibrated_path = (
        source_project
        / "results"
        / "calibration"
        / "full_100_batches"
        / "calibrated_thresholds.npz"
    )

    initial_weights = load_weights(
        initial_weights_path
    )

    learned_weights = load_weights(
        learned_path
    )

    with np.load(
        calibrated_path,
        allow_pickle=False
    ) as data:

        initial_threshold_E = (
            data["threshold_E"].copy()
        )

        initial_threshold_I = (
            data["threshold_I"].copy()
        )

    with np.load(
        learned_path,
        allow_pickle=False
    ) as data:

        learned_threshold_E = (
            data["threshold_E"].copy()
        )

        learned_threshold_I = (
            data["threshold_I"].copy()
        )

    thresholds = (
        ("initial E", initial_threshold_E, (N_E, 1)),
        ("initial I", initial_threshold_I, (N_I, 1)),
        ("learned E", learned_threshold_E, (N_E, 1)),
        ("learned I", learned_threshold_I, (N_I, 1)),
    )

    for name, values, expected_shape in thresholds:

        if values.shape != expected_shape:
            raise ValueError(
                f"{name} threshold shape: "
                f"expected {expected_shape}, "
                f"got {values.shape}"
            )

    print("\nLoaded network states.")

    print("Initial weights:", initial_weights_path)
    print("Learned weights:", learned_path)

    print(
        "Initial threshold shapes:",
        initial_threshold_E.shape,
        initial_threshold_I.shape
    )

    print(
        "Learned threshold shapes:",
        learned_threshold_E.shape,
        learned_threshold_I.shape
    )

    return {
        "initial_weights": initial_weights,
        "learned_weights": learned_weights,

        "initial_threshold_E": initial_threshold_E,
        "initial_threshold_I": initial_threshold_I,

        "learned_threshold_E": learned_threshold_E,
        "learned_threshold_I": learned_threshold_I,
    }


# ============================================================
# 6. EVALUATE ONE FROZEN NETWORK
# ============================================================

def evaluate_network(
    LGN_spikes,
    weights,
    threshold_E,
    threshold_I,
    noise_state
):
    """
    Evaluate one frozen V1 network.

    The same LGN spike trains and NumPy random state
    can be supplied to two network evaluations.

    Returns
    -------
    E_counts : ndarray, shape (N_E, 144)
    I_counts : ndarray, shape (N_I, 144)
    """

    # Restore matched simulation noise.
    np.random.set_state(noise_state)

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    E_counts = E_spikes.sum(axis=2)
    I_counts = I_spikes.sum(axis=2)

    return E_counts, I_counts


# ============================================================
# 7. COMPLETE FUNCTIONAL ORIENTATION EXPERIMENT
# ============================================================

def analyze_functional_orientation(
    source_project,
    learned_run="training_test_25",
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Reproduce initial and learned V1 responses
    to oriented grating stimuli.

    The original project is read-only.
    """

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Generate the original 144 LGN presentations
    # --------------------------------------------------------

    inputs = generate_orientation_lgn_inputs()

    LGN_spikes = inputs["LGN_spikes"]

    noise_state = inputs["noise_state"]

    # --------------------------------------------------------
    # B. Load initial and learned network states
    # --------------------------------------------------------

    states = load_network_states(
        source_project,
        learned_run
    )

    # --------------------------------------------------------
    # C. Simulate initial network
    # --------------------------------------------------------

    print("\nEvaluating initial network...")

    initial_E, initial_I = evaluate_network(
        LGN_spikes,
        states["initial_weights"],
        states["initial_threshold_E"],
        states["initial_threshold_I"],
        noise_state
    )

    # --------------------------------------------------------
    # D. Simulate learned network
    # --------------------------------------------------------

    print("Evaluating learned network...")

    learned_E, learned_I = evaluate_network(
        LGN_spikes,
        states["learned_weights"],
        states["learned_threshold_E"],
        states["learned_threshold_I"],
        noise_state
    )

    # --------------------------------------------------------
    # E. Check response dimensions
    # --------------------------------------------------------

    n_presentations = (
        len(ANGLES) * len(PHASES) * N_REPEATS
    )

    expected_E = (N_E, n_presentations)
    expected_I = (N_I, n_presentations)

    for name, array, expected in (
        ("initial_E", initial_E, expected_E),
        ("learned_E", learned_E, expected_E),
        ("initial_I", initial_I, expected_I),
        ("learned_I", learned_I, expected_I),
    ):

        if array.shape != expected:
            raise ValueError(
                f"{name}: expected {expected}, "
                f"got {array.shape}"
            )

    print("\nRESPONSE SHAPES")
    print("Initial E:", initial_E.shape)
    print("Learned E:", learned_E.shape)
    print("Initial I:", initial_I.shape)
    print("Learned I:", learned_I.shape)

    # --------------------------------------------------------
    # F. Print firing rates
    # --------------------------------------------------------

    duration = N_STEPS * DT

    print("\nPOPULATION FIRING RATES")

    print(
        "Initial E:",
        initial_E.mean() / duration,
        "Hz"
    )

    print(
        "Learned E:",
        learned_E.mean() / duration,
        "Hz"
    )

    print(
        "Initial I:",
        initial_I.mean() / duration,
        "Hz"
    )

    print(
        "Learned I:",
        learned_I.mean() / duration,
        "Hz"
    )

    # --------------------------------------------------------
    # G. Save all original output fields
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "functional_orientation_responses.npz"
    )

    np.savez_compressed(
        output_path,

        angles=ANGLES,
        phases=PHASES,
        frequency=FREQUENCY,
        n_repeats=N_REPEATS,

        presentation_angles=inputs[
            "presentation_angles"
        ],

        presentation_phases=inputs[
            "presentation_phases"
        ],

        presentation_repeats=inputs[
            "presentation_repeats"
        ],

        initial_E=initial_E,
        learned_E=learned_E,

        initial_I=initial_I,
        learned_I=learned_I,

        LGN_spike_counts=LGN_spikes.sum(axis=2),

        duration=duration
    )

    print("\nFunctional responses saved!")
    print("Output:", output_path)

    return {
        "initial_E": initial_E,
        "learned_E": learned_E,
        "initial_I": initial_I,
        "learned_I": learned_I,
        "LGN_spike_counts": LGN_spikes.sum(axis=2),
        "duration": duration,
    }


# ============================================================
# 8. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate functional V1 orientation responses "
            "in initial and learned networks."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to original V1 project"
    )

    parser.add_argument(
        "--learned-run",
        type=str,
        default="training_test_25"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    args = parser.parse_args()

    analyze_functional_orientation(
        source_project=args.source_project,
        learned_run=args.learned_run,
        output_dir=args.output_dir
    )
