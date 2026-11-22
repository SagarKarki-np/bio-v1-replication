
"""
Excitatory and inhibitory synaptic-input analysis.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This module combines two original analyses:

1. Population-average excitation and inhibition.
2. Lagged E/I synaptic-input correlations.

Protocol
--------
- Compare initial and trained V1 networks.
- Present identical LGN spike trains to both networks.
- Restore the same NumPy RNG state before each simulation.
- Freeze synaptic weights during evaluation.
- Record full E and I synaptic-input traces.
- Calculate population-average input magnitudes.
- Calculate temporal correlations for neuron-trial pairs.

Input definitions
-----------------
Excitation = LGN_E + E_E
Inhibition = I_E

Inhibition is stored as a positive magnitude.

Positive correlation lag means inhibition follows
excitation under the applied time-shift convention.

Outputs
-------
EI_full_traces.npz
EI_population_traces.npz
EI_lag_analysis.npz

EI_population_traces.png
EI_lag_cross_correlation.png
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import (
    DT,
    N_STEPS,
    TAU_M,
    V_MIN,
    V_RESET,
    BATCH_SIZE,
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "EI_balance"
)

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)

EVALUATION_SEED = 2029
EVALUATION_BATCH_SIZE = 10
EVALUATION_NOISE_STD = 1.0
MAX_LAG_MS = 20


# ============================================================
# 2. LOAD NETWORK PARAMETERS
# ============================================================

def load_weights(path):
    """
    Load the six synaptic weight matrices.

    Matrix ordering matches simulation.py.
    """

    with np.load(path, allow_pickle=False) as data:
        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

    return weights


def load_thresholds(path):
    """
    Load excitatory and inhibitory firing thresholds.
    """

    with np.load(path, allow_pickle=False) as data:
        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

    return threshold_E, threshold_I


# ============================================================
# 3. EVALUATE A NETWORK WITH FROZEN WEIGHTS
# ============================================================

def evaluate_network(
    LGN_spikes,
    weights,
    threshold_E,
    threshold_I,
    noise_state
):
    """
    Simulate a frozen network and record E/I inputs.

    The RNG state is restored before simulation so
    initial and trained networks receive the same
    underlying random-number sequence.

    Returns
    -------
    excitation : ndarray
        Shape (n_e, batch_size, n_steps).

    inhibition : ndarray
        Shape (n_e, batch_size, n_steps).
    """

    np.random.set_state(noise_state)

    E_spikes, I_spikes, traces = simulate_v1_batch(
        LGN_spikes,
        weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=EVALUATION_NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I,
        record_inputs=True
    )

    excitation = (
        traces["LGN_E"] + traces["E_E"]
    )

    inhibition = traces["I_E"]

    return excitation, inhibition


# ============================================================
# 4. SUMMARIZE SYNAPTIC INPUT MAGNITUDES
# ============================================================

def summarize_inputs(name, excitation, inhibition):
    """
    Calculate mean synaptic-input magnitudes.

    Time is preserved in the population-average traces.
    """

    if excitation.shape != inhibition.shape:
        raise ValueError(
            "Excitation and inhibition shapes differ."
        )

    if excitation.ndim != 3:
        raise ValueError(
            "Expected (neurons, presentations, timesteps)."
        )

    mean_exc = excitation.mean()
    mean_inh = inhibition.mean()

    mean_exc_time = excitation.mean(axis=(0, 1))
    mean_inh_time = inhibition.mean(axis=(0, 1))

    ratio = (
        mean_exc / mean_inh
        if mean_inh > 0
        else np.nan
    )

    print(f"\n{name.upper()}")
    print("Mean excitation:", mean_exc)
    print("Mean inhibition:", mean_inh)
    print("Excitation / inhibition ratio:", ratio)

    return mean_exc_time, mean_inh_time


# ============================================================
# 5. CORRELATION AT ONE TEMPORAL LAG
# ============================================================

def correlation_at_lag(E, I, lag):
    """
    Calculate E/I Pearson correlation at one lag.

    Parameters
    ----------
    E, I : ndarray
        Shape (neuron_trial_pairs, timesteps).

    lag : int
        Temporal displacement in timesteps.

        Positive:
            Inhibition follows excitation.

        Negative:
            Inhibition precedes excitation.

    Returns
    -------
    ndarray
        One Pearson correlation per neuron-trial pair.

    A constant trace gives an undefined (NaN)
    correlation.
    """

    E = np.asarray(E, dtype=float)
    I = np.asarray(I, dtype=float)

    if E.ndim != 2 or E.shape != I.shape:
        raise ValueError(
            "E and I must be matching 2D arrays."
        )

    if abs(lag) >= E.shape[1]:
        raise ValueError(
            "Lag must be shorter than the time series."
        )

    if lag > 0:
        X = E[:, :-lag]
        Y = I[:, lag:]

    elif lag < 0:
        X = E[:, -lag:]
        Y = I[:, :lag]

    else:
        X = E
        Y = I

    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)

    numerator = np.sum(
        X * Y,
        axis=1
    )

    denominator = np.sqrt(
        np.sum(X ** 2, axis=1)
        * np.sum(Y ** 2, axis=1)
    )

    correlations = np.divide(
        numerator,
        denominator,
        out=np.full(E.shape[0], np.nan),
        where=denominator > 0
    )

    return correlations


# ============================================================
# 6. CALCULATE LAG-CORRELATION CURVES
# ============================================================

def calculate_lag_correlations(
    initial_exc,
    initial_inh,
    learned_exc,
    learned_inh,
    dt=DT,
    max_lag_ms=MAX_LAG_MS
):
    """
    Calculate temporally shifted E/I correlations.

    Uses the same matched-valid-pair policy as
    the original E/I lag analysis.

    Each neuron-presentation pair must have a valid
    correlation at every lag in both networks.
    """

    shapes = {
        initial_exc.shape,
        initial_inh.shape,
        learned_exc.shape,
        learned_inh.shape
    }

    if len(shapes) != 1:
        raise ValueError(
            "All four full-trace arrays must match."
        )

    if initial_exc.ndim != 3:
        raise ValueError(
            "Expected three-dimensional full traces."
        )

    n_steps = initial_exc.shape[2]

    max_lag_steps = int(
        round(max_lag_ms / (dt * 1000))
    )

    if max_lag_steps >= n_steps:
        raise ValueError(
            "Maximum lag exceeds the available trace length."
        )

    lags = np.arange(
        -max_lag_steps,
        max_lag_steps + 1
    )

    lag_ms = lags * dt * 1000

    # One row per neuron-presentation pair.
    initial_E = initial_exc.reshape(-1, n_steps).astype(float)
    initial_I = initial_inh.reshape(-1, n_steps).astype(float)

    learned_E = learned_exc.reshape(-1, n_steps).astype(float)
    learned_I = learned_inh.reshape(-1, n_steps).astype(float)

    initial_correlations = np.array([
        correlation_at_lag(initial_E, initial_I, lag)
        for lag in lags
    ])

    learned_correlations = np.array([
        correlation_at_lag(learned_E, learned_I, lag)
        for lag in lags
    ])

    # Require validity at every lag in both networks.
    valid_pairs = (
        np.isfinite(initial_correlations).all(axis=0)
        & np.isfinite(learned_correlations).all(axis=0)
    )

    print("\nMATCHED VALIDITY")
    print("Total neuron-trial pairs:", valid_pairs.size)
    print("Matched valid pairs:", valid_pairs.sum())
    print("Excluded pairs:", (~valid_pairs).sum())

    if not valid_pairs.any():
        raise RuntimeError(
            "No pairs with valid correlations at every lag."
        )

    initial_matched = initial_correlations[:, valid_pairs]
    learned_matched = learned_correlations[:, valid_pairs]

    mean_initial = initial_matched.mean(axis=1)
    mean_learned = learned_matched.mean(axis=1)

    return {
        "lag_ms": lag_ms,
        "lags": lags,
        "initial_correlations": initial_matched,
        "learned_correlations": learned_matched,
        "mean_initial": mean_initial,
        "mean_learned": mean_learned,
        "valid_pairs": valid_pairs
    }


# ============================================================
# 7. REPORT LAG-CORRELATION RESULTS
# ============================================================

def summarize_lag_results(results):
    """
    Print peak correlation lags and zero-lag values.
    """

    lag_ms = results["lag_ms"]
    lags = results["lags"]

    mean_initial = results["mean_initial"]
    mean_learned = results["mean_learned"]

    initial_peak_idx = np.argmax(mean_initial)
    learned_peak_idx = np.argmax(mean_learned)

    zero_idx = np.where(lags == 0)[0][0]

    print("\nINITIAL NETWORK")
    print("Peak lag (ms):", lag_ms[initial_peak_idx])
    print(
        "Peak mean correlation:",
        mean_initial[initial_peak_idx]
    )

    print("\nLEARNED NETWORK")
    print("Peak lag (ms):", lag_ms[learned_peak_idx])
    print(
        "Peak mean correlation:",
        mean_learned[learned_peak_idx]
    )

    print("\nZERO-LAG COMPARISON")
    print("Initial:", mean_initial[zero_idx])
    print("Learned:", mean_learned[zero_idx])


# ============================================================
# 8. PLOT POPULATION-AVERAGE INPUT TRACES
# ============================================================

def plot_population_traces(
    time_ms,
    initial_exc,
    initial_inh,
    learned_exc,
    learned_inh,
    save_path,
    show=False
):
    """
    Plot average synaptic inputs over presentation time.
    """

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 7),
        sharex=True
    )

    stages = [
        ("Initial network", initial_exc, initial_inh),
        ("Learned network - batch 1100", learned_exc, learned_inh)
    ]

    for ax, (title, excitation, inhibition) in zip(
        axes, stages
    ):
        ax.plot(time_ms, excitation, label="Excitation")
        ax.plot(time_ms, inhibition, label="Inhibition")

        ax.set_title(title)
        ax.set_ylabel("Mean synaptic input")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[1].set_xlabel("Time (ms)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. PLOT TEMPORAL CROSS-CORRELATIONS
# ============================================================

def plot_lag_correlations(results, save_path, show=False):
    """
    Plot mean lagged E/I correlations for both networks.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        results["lag_ms"],
        results["mean_initial"],
        marker="o",
        markersize=3,
        label="Initial"
    )

    ax.plot(
        results["lag_ms"],
        results["mean_learned"],
        marker="o",
        markersize=3,
        label="Learned"
    )

    ax.axvline(
        0,
        linestyle="--",
        color="gray"
    )

    ax.set_xlabel(
        "Lag (ms): positive = inhibition follows excitation"
    )

    ax.set_ylabel("Mean Pearson cross-correlation")
    ax.set_title("Temporal Relationship Between E and I Inputs")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 10. COMPLETE E/I ANALYSIS
# ============================================================

def analyze_excitation_inhibition(
    source_project,
    learned_run="training_test_25",
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run both original E/I analyses in one workflow.

    The original project is used only for loading
    images, checkpoints and threshold arrays.

    New numerical results and figures are saved
    to output_dir in the new repository.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Locate input files
    # --------------------------------------------------------

    initial_path = (
        source_project / "results"
        / "untrained" / "batch_001" / "weights.npz"
    )

    calibration_path = (
        source_project / "results"
        / "calibration" / "full_100_batches"
        / "calibrated_thresholds.npz"
    )

    learned_path = (
        source_project / "results"
        / "learning" / learned_run / "batch_1100.npz"
    )

    images_path = (
        source_project / "data" / "van_hateren_dog.npy"
    )

    # --------------------------------------------------------
    # B. Load weights and thresholds
    # --------------------------------------------------------

    initial_weights = load_weights(initial_path)
    learned_weights = load_weights(learned_path)

    initial_threshold_E, initial_threshold_I = (
        load_thresholds(calibration_path)
    )

    learned_threshold_E, learned_threshold_I = (
        load_thresholds(learned_path)
    )

    # --------------------------------------------------------
    # C. Generate matched LGN input
    # --------------------------------------------------------

    images = np.load(
        images_path,
        mmap_mode="r"
    )

    np.random.seed(EVALUATION_SEED)

    LGN_spikes = generate_lgn_batch(
        images,
        batch_size=EVALUATION_BATCH_SIZE,
        n_steps=N_STEPS,
        dt=DT
    )

    # Capture RNG state after the LGN inputs are generated.
    noise_state = np.random.get_state()

    print("LGN spike shape:", LGN_spikes.shape)

    # --------------------------------------------------------
    # D. Simulate initial network
    # --------------------------------------------------------

    print("\nEvaluating initial network...")

    initial_exc, initial_inh = evaluate_network(
        LGN_spikes,
        initial_weights,
        initial_threshold_E,
        initial_threshold_I,
        noise_state
    )

    initial_exc_time, initial_inh_time = (
        summarize_inputs(
            "Initial network",
            initial_exc,
            initial_inh
        )
    )

    # --------------------------------------------------------
    # E. Simulate trained network
    # --------------------------------------------------------

    print("\nEvaluating learned network...")

    learned_exc, learned_inh = evaluate_network(
        LGN_spikes,
        learned_weights,
        learned_threshold_E,
        learned_threshold_I,
        noise_state
    )

    learned_exc_time, learned_inh_time = (
        summarize_inputs(
            "Learned network",
            learned_exc,
            learned_inh
        )
    )

    # --------------------------------------------------------
    # F. Save full and population-average traces
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    time_ms = np.arange(N_STEPS) * DT * 1000

    np.savez_compressed(
        output_dir / "EI_full_traces.npz",
        initial_exc=initial_exc,
        initial_inh=initial_inh,
        learned_exc=learned_exc,
        learned_inh=learned_inh,
        dt=DT
    )

    np.savez_compressed(
        output_dir / "EI_population_traces.npz",
        time_ms=time_ms,
        initial_exc=initial_exc_time,
        initial_inh=initial_inh_time,
        learned_exc=learned_exc_time,
        learned_inh=learned_inh_time
    )

    plot_population_traces(
        time_ms,
        initial_exc_time,
        initial_inh_time,
        learned_exc_time,
        learned_inh_time,
        output_dir / "EI_population_traces.png",
        show=show
    )

    # --------------------------------------------------------
    # G. Calculate temporal E/I correlations
    # --------------------------------------------------------

    results = calculate_lag_correlations(
        initial_exc,
        initial_inh,
        learned_exc,
        learned_inh,
        dt=DT,
        max_lag_ms=MAX_LAG_MS
    )

    summarize_lag_results(results)

    # --------------------------------------------------------
    # H. Save lag results and figure
    # --------------------------------------------------------

    np.savez_compressed(
        output_dir / "EI_lag_analysis.npz",
        lag_ms=results["lag_ms"],
        initial_correlations=results["initial_correlations"],
        learned_correlations=results["learned_correlations"],
        mean_initial=results["mean_initial"],
        mean_learned=results["mean_learned"],
        valid_pairs=results["valid_pairs"]
    )

    plot_lag_correlations(
        results,
        output_dir / "EI_lag_cross_correlation.png",
        show=show
    )

    print("\nE/I analysis completed!")
    print("Results saved to:", output_dir)

    return results


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare excitatory and inhibitory synaptic "
            "inputs before and after V1 learning."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to archived original V1 project"
    )

    parser.add_argument(
        "--learned-run",
        type=str,
        default="training_test_25",
        help="Original training run containing batch_1100.npz"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated figures"
    )

    args = parser.parse_args()

    analyze_excitation_inhibition(
        source_project=args.source_project,
        learned_run=args.learned_run,
        output_dir=args.output_dir,
        show=args.show
    )
