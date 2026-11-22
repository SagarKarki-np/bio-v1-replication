
"""
Response-weighted spike-triggered averaging for V1.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

This reproduces the original 10,000-stimulus STA analysis.

Protocol
--------
- Trained network: batch_1100.npz
- Gaussian 16x16 white-noise stimuli
- Divisive DoG preprocessing
- Per-stimulus standard-deviation normalization
- ON/OFF LGN rate generation and clipping
- Recurrent V1 spiking simulation
- Response-weighted stimulus accumulation
- E and I STA-to-weight RF correlations
- Alternating-presentation split-half reliability
- Convergence checkpoints and LGN diagnostics

STA here is a stimulus average weighted by each
neuron's total spike count during a presentation.

The original RNG sequence and calculations are kept.

Default:
    20 presentations/batch x 500 batches = 10,000

A --batches option permits shorter smoke tests.

The original archive uses the filename
STA_2000_results.npz despite containing 10,000
stimulus presentations. This filename is retained
for exact archive comparisons at the default setting.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import (
    DT,
    N_STEPS,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
    N_E,
    N_I,
    N_LGN,
    PATCH_SIZE,
)

from preprocessing import (
    gaussian_kernel,
    divisive_dog,
    normalize_standard_deviation,
)

from simulation import simulate_v1_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "STA"
)

WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)

N_PIXELS = PATCH_SIZE ** 2

STA_BATCH_SIZE = 20
DEFAULT_N_BATCHES = 500

LGN_STA_SCALE = 140.0
LGN_MAX_RATE = 100.0

SEED = 2041

# This is the final, effective checkpoint list
# from the original script.
CONVERGENCE_CHECKPOINTS = (
    200,
    500,
    1000,
    2000,
)

EXAMPLE_NEURONS = (0, 25, 100, 250)


# ============================================================
# 2. LOAD TRAINED NETWORK
# ============================================================

def load_trained_network(checkpoint_path):
    """
    Load trained synaptic weights and thresholds.

    Preserve threshold shapes:
        E: (N_E, 1)
        I: (N_I, 1)
    """

    with np.load(
        checkpoint_path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

    if theta_E.shape != (N_E, 1):
        raise ValueError(
            f"Unexpected E threshold shape: {theta_E.shape}"
        )

    if theta_I.shape != (N_I, 1):
        raise ValueError(
            f"Unexpected I threshold shape: {theta_I.shape}"
        )

    if weights[0].shape != (N_E, N_LGN):
        raise ValueError("Unexpected W_LGN_E shape.")

    if weights[1].shape != (N_I, N_LGN):
        raise ValueError("Unexpected W_LGN_I shape.")

    return weights, theta_E, theta_I


# ============================================================
# 3. WEIGHT-DERIVED RECEPTIVE FIELDS
# ============================================================

def calculate_weight_rfs(weights):
    """
    Construct flattened signed ON-OFF receptive fields.

    Returns
    -------
    weight_RF_E : (N_E, N_PIXELS)
    weight_RF_I : (N_I, N_PIXELS)

    Flattening is already encoded in each weight row.
    """

    W_LGN_E = weights[0]
    W_LGN_I = weights[1]

    weight_RF_E = (
        W_LGN_E[:, :N_PIXELS]
        - W_LGN_E[:, N_PIXELS:]
    )

    weight_RF_I = (
        W_LGN_I[:, :N_PIXELS]
        - W_LGN_I[:, N_PIXELS:]
    )

    return weight_RF_E, weight_RF_I


# ============================================================
# 4. PEARSON RF CORRELATION
# ============================================================

def pearson_rf(a, b):
    """
    Calculate Pearson correlation between RF vectors.

    Undefined RF correlations return NaN.
    """

    if not (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
    ):
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def calculate_rf_correlations(STA, weight_RF):
    """
    Calculate STA-to-weight correlation per neuron.
    """

    if STA.shape != weight_RF.shape:
        raise ValueError(
            "STA and weight RF shapes must match."
        )

    return np.array([
        pearson_rf(STA[i], weight_RF[i])
        for i in range(STA.shape[0])
    ])


# ============================================================
# 5. CALCULATE STA FROM ACCUMULATED RESPONSES
# ============================================================

def calculate_sta(sta_sum, spike_counts):
    """
    Calculate response-weighted receptive-field estimates.

    Silent neurons are assigned NaN, as in the original.
    """

    if (
        sta_sum.ndim != 2
        or spike_counts.shape != (sta_sum.shape[0],)
    ):
        raise ValueError(
            "Incompatible STA accumulator dimensions."
        )

    return np.divide(
        sta_sum,
        spike_counts[:, None],
        out=np.full_like(sta_sum, np.nan),
        where=spike_counts[:, None] > 0
    )


# ============================================================
# 6. PREPARE ORIGINAL GAUSSIAN FILTERS
# ============================================================

def create_sta_kernels():
    """
    Construct original center, surround, and
    divisive normalization kernels.
    """

    center_kernel = gaussian_kernel(
        sigma=1.0,
        radius=4
    )

    surround_kernel = gaussian_kernel(
        sigma=1.5,
        radius=6
    )

    divisive_kernel = gaussian_kernel(
        sigma=1.5,
        radius=6
    )

    return (
        center_kernel,
        surround_kernel,
        divisive_kernel
    )


# ============================================================
# 7. GENERATE ORIGINAL WHITE-NOISE STIMULUS
# ============================================================

def generate_white_noise():
    """
    Generate Gaussian noise and its grayscale version.

    The signed original noise is used for STA.
    Grayscale is used only for LGN preprocessing.
    """

    noise = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(PATCH_SIZE, PATCH_SIZE)
    )

    grayscale = np.clip(
        50.0 * noise + 100.0,
        0.0,
        255.0
    )

    return noise, grayscale


# ============================================================
# 8. CONVERT WHITE NOISE TO LGN FIRING RATES
# ============================================================

def white_noise_to_lgn_rates(
    grayscale,
    center_kernel,
    surround_kernel,
    divisive_kernel
):
    """
    Convert one grayscale stimulus to ON/OFF LGN rates.

    Processing matches the original STA script:
        divisive DoG
        standard deviation normalization
        Fortran-order flattening
        multiplication by 140
        ON/OFF splitting
        clipping at 100 Hz

    Per-stimulus SD normalization is a methodological
    assumption inherited from the original experiment.
    """

    filtered = divisive_dog(
        grayscale,
        center_kernel,
        surround_kernel,
        divisive_kernel
    )

    filtered = normalize_standard_deviation(
        filtered
    )

    signed = (
        filtered.flatten(order="F")
        * LGN_STA_SCALE
    )

    ON = np.maximum(signed, 0.0)
    OFF = np.maximum(-signed, 0.0)

    rates = np.concatenate((ON, OFF))

    return np.clip(
        rates,
        0.0,
        LGN_MAX_RATE
    )


# ============================================================
# 9. INITIALIZE ACCUMULATORS
# ============================================================

def initialize_accumulators():
    """
    Create full and split-half STA accumulators.
    """

    return {
        "STA_sum_E": np.zeros((N_E, N_PIXELS)),
        "STA_sum_I": np.zeros((N_I, N_PIXELS)),

        "total_E_spikes": np.zeros(N_E),
        "total_I_spikes": np.zeros(N_I),

        "half_sum_E": [
            np.zeros((N_E, N_PIXELS)),
            np.zeros((N_E, N_PIXELS)),
        ],

        "half_spikes_E": [
            np.zeros(N_E),
            np.zeros(N_E),
        ],
    }


def initialize_diagnostics():
    """
    Create original LGN firing-rate diagnostic totals.
    """

    return {
        "total_rate_values": 0,
        "total_capped": 0,
        "total_zero": 0,
        "total_rate_sum": 0.0,
        "total_rate_squared_sum": 0.0,
        "max_rate_observed": 0.0,
    }


# ============================================================
# 10. GENERATE ONE BATCH OF LGN SPIKES
# ============================================================

def generate_sta_batch(
    center_kernel,
    surround_kernel,
    divisive_kernel,
    diagnostics
):
    """
    Generate one batch of noise stimuli and LGN spikes.

    The original NumPy random calls and their ordering
    are preserved.

    Returns
    -------
    stimuli : (STA_BATCH_SIZE, N_PIXELS)
    LGN_spikes : (N_LGN, STA_BATCH_SIZE, N_STEPS)
    """

    stimuli = np.zeros(
        (STA_BATCH_SIZE, N_PIXELS)
    )

    LGN_spikes = np.zeros(
        (N_LGN, STA_BATCH_SIZE, N_STEPS),
        dtype=bool
    )

    for k in range(STA_BATCH_SIZE):

        noise, grayscale = generate_white_noise()

        stimuli[k] = noise.flatten(order="F")

        rates = white_noise_to_lgn_rates(
            grayscale,
            center_kernel,
            surround_kernel,
            divisive_kernel
        )

        if rates.shape != (N_LGN,):
            raise ValueError(
                f"Unexpected LGN rate shape: {rates.shape}"
            )

        # Original diagnostic accumulation order.
        diagnostics["total_rate_values"] += rates.size

        diagnostics["total_capped"] += np.count_nonzero(
            rates >= LGN_MAX_RATE
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

        probabilities = rates[:, None] * DT

        LGN_spikes[:, k, :] = (
            np.random.random(
                (N_LGN, N_STEPS)
            ) < probabilities
        )

    return stimuli, LGN_spikes


# ============================================================
# 11. ACCUMULATE SPIKE-WEIGHTED STIMULI
# ============================================================

def accumulate_sta(
    accumulators,
    stimuli,
    E_counts,
    I_counts
):
    """
    Accumulate response-weighted stimuli.

    Preserves original dot-product ordering and
    alternating-presentation split-half allocation.
    """

    accumulators["STA_sum_E"] += (
        E_counts @ stimuli
    )

    accumulators["STA_sum_I"] += (
        I_counts @ stimuli
    )

    accumulators["total_E_spikes"] += (
        E_counts.sum(axis=1)
    )

    accumulators["total_I_spikes"] += (
        I_counts.sum(axis=1)
    )

    for half in (0, 1):

        indices = np.arange(
            half,
            STA_BATCH_SIZE,
            2
        )

        accumulators["half_sum_E"][half] += (
            E_counts[:, indices]
            @ stimuli[indices]
        )

        accumulators["half_spikes_E"][half] += (
            E_counts[:, indices].sum(axis=1)
        )


# ============================================================
# 12. CALCULATE CONVERGENCE CHECKPOINT
# ============================================================

def calculate_convergence(
    accumulators,
    weight_RF_E,
    weight_RF_I
):
    """
    Calculate current median STA-to-weight correlation.
    """

    current_STA_E = calculate_sta(
        accumulators["STA_sum_E"],
        accumulators["total_E_spikes"]
    )

    current_STA_I = calculate_sta(
        accumulators["STA_sum_I"],
        accumulators["total_I_spikes"]
    )

    correlations_E = calculate_rf_correlations(
        current_STA_E,
        weight_RF_E
    )

    correlations_I = calculate_rf_correlations(
        current_STA_I,
        weight_RF_I
    )

    return (
        np.nanmedian(correlations_E),
        np.nanmedian(correlations_I)
    )


# ============================================================
# 13. CALCULATE FINAL RESULTS
# ============================================================

def finalize_sta(
    accumulators,
    weight_RF_E,
    weight_RF_I
):
    """
    Calculate final STAs, RF correlations, and
    excitatory split-half reliability.
    """

    STA_E = calculate_sta(
        accumulators["STA_sum_E"],
        accumulators["total_E_spikes"]
    )

    STA_I = calculate_sta(
        accumulators["STA_sum_I"],
        accumulators["total_I_spikes"]
    )

    RF_correlations_E = calculate_rf_correlations(
        STA_E,
        weight_RF_E
    )

    RF_correlations_I = calculate_rf_correlations(
        STA_I,
        weight_RF_I
    )

    half_STA_E = []

    for half in (0, 1):

        estimate = calculate_sta(
            accumulators["half_sum_E"][half],
            accumulators["half_spikes_E"][half]
        )

        half_STA_E.append(estimate)

    split_half_reliability = np.array([
        pearson_rf(
            half_STA_E[0][i],
            half_STA_E[1][i]
        )
        for i in range(N_E)
    ])

    return {
        "STA_E": STA_E,
        "STA_I": STA_I,
        "RF_correlations_E": RF_correlations_E,
        "RF_correlations_I": RF_correlations_I,
        "split_half_reliability": split_half_reliability,
    }


# ============================================================
# 14. CALCULATE LGN DIAGNOSTICS
# ============================================================

def summarize_lgn_diagnostics(diagnostics):
    """
    Calculate LGN rate statistics as in the original.
    """

    n = diagnostics["total_rate_values"]

    if n == 0:
        raise ValueError("No LGN rates were recorded.")

    mean_rate = (
        diagnostics["total_rate_sum"] / n
    )

    std_rate = np.sqrt(
        max(
            0.0,
            diagnostics["total_rate_squared_sum"] / n
            - mean_rate ** 2
        )
    )

    fraction_capped = (
        diagnostics["total_capped"] / n
    )

    fraction_zero = (
        diagnostics["total_zero"] / n
    )

    print("\nWHITE-NOISE LGN DIAGNOSTICS")
    print("Mean LGN rate:", mean_rate)
    print("LGN rate SD:", std_rate)

    print(
        "Maximum LGN rate:",
        diagnostics["max_rate_observed"]
    )

    print(
        "Fraction capped at 100 Hz:",
        fraction_capped
    )

    print("Fraction at zero:", fraction_zero)

    return mean_rate, std_rate, fraction_capped, fraction_zero


# ============================================================
# 15. PLOT EXAMPLE RECEPTIVE FIELDS
# ============================================================

def plot_sta_examples(
    STA_E,
    weight_RF_E,
    save_path,
    show=False
):
    """
    Plot original weight RFs beside estimated STAs.
    """

    neurons = [
        n for n in EXAMPLE_NEURONS
        if n < STA_E.shape[0]
    ]

    fig, axes = plt.subplots(
        len(neurons),
        2,
        figsize=(7, 3 * len(neurons)),
        constrained_layout=True,
        squeeze=False
    )

    for row, neuron in enumerate(neurons):

        weight_map = weight_RF_E[neuron].reshape(
            PATCH_SIZE,
            PATCH_SIZE,
            order="F"
        )

        sta_map = STA_E[neuron].reshape(
            PATCH_SIZE,
            PATCH_SIZE,
            order="F"
        )

        for col, (image, label) in enumerate([
            (weight_map, "Weights"),
            (sta_map, "STA")
        ]):

            ax = axes[row, col]

            if np.all(np.isfinite(image)):

                limit = np.max(np.abs(image))

                if limit == 0:
                    limit = 1.0

                ax.imshow(
                    image,
                    cmap="coolwarm",
                    vmin=-limit,
                    vmax=limit
                )

            ax.set_title(
                f"Neuron {neuron}: {label}"
            )

            ax.axis("off")

    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 16. PLOT CONVERGENCE
# ============================================================

def plot_convergence(
    counts,
    E_values,
    I_values,
    save_path,
    show=False
):
    """
    Plot median STA-to-weight correlation over stimuli.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        counts,
        E_values,
        marker="o",
        label="Excitatory"
    )

    ax.plot(
        counts,
        I_values,
        marker="o",
        label="Inhibitory"
    )

    ax.set_xlabel("Number of white-noise stimuli")

    ax.set_ylabel(
        "Median STA-weight Pearson correlation"
    )

    ax.set_title(
        "Convergence of V1 Receptive-Field Estimates"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 17. PLOT SPLIT-HALF RELIABILITY
# ============================================================

def plot_split_half(
    split_half_reliability,
    save_path,
    show=False
):
    """
    Plot excitatory STA split-half correlations.
    """

    valid = split_half_reliability[
        np.isfinite(split_half_reliability)
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        valid,
        bins=np.linspace(-1, 1, 41),
        edgecolor="black"
    )

    ax.set_xlabel(
        "Split-half STA Pearson correlation"
    )

    ax.set_ylabel(
        "Number of excitatory neurons"
    )

    ax.set_title("STA Split-Half Reliability")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 18. COMPLETE STA EXPERIMENT
# ============================================================

def analyze_spike_triggered_average(
    source_project,
    learned_run="training_test_25",
    n_batches=DEFAULT_N_BATCHES,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False,
    checkpoint_override=None
):
    """
    Run the original white-noise STA experiment.

    The original checkpoint is read-only.

    Setting n_batches=500 reproduces the full
    10,000-stimulus protocol.

    Smaller values permit deterministic smoke tests.
    """

    if n_batches < 1:
        raise ValueError(
            "Number of batches must be positive."
        )

    source_project = Path(
        source_project
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser()

    checkpoint_path = (
    source_project
    / "results"
    / "learning"
    / learned_run
    / "batch_1100.npz"
  )
    if checkpoint_override is not None:
        checkpoint_path = (
        Path(checkpoint_override)
        .expanduser()
        .resolve()
      )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # A. Initialize original global RNG
    # --------------------------------------------------------

    np.random.seed(SEED)

    # --------------------------------------------------------
    # B. Load trained network
    # --------------------------------------------------------

    weights, theta_E, theta_I = load_trained_network(
        checkpoint_path
    )

    print("Trained network loaded.")
    print("Checkpoint:", checkpoint_path)

    weight_RF_E, weight_RF_I = calculate_weight_rfs(
        weights
    )

    # --------------------------------------------------------
    # C. Prepare original DoG kernels
    # --------------------------------------------------------

    (
        center_kernel,
        surround_kernel,
        divisive_kernel
    ) = create_sta_kernels()

    # --------------------------------------------------------
    # D. Initialize accumulators and diagnostics
    # --------------------------------------------------------

    accumulators = initialize_accumulators()
    diagnostics = initialize_diagnostics()

    N_STIMULI = STA_BATCH_SIZE * n_batches

    checkpoints = [
        n for n in CONVERGENCE_CHECKPOINTS
        if n <= N_STIMULI
    ]

    convergence_E = {}
    convergence_I = {}

    print("\nStarting STA simulation...")
    print("Number of stimuli:", N_STIMULI)
    print("Batch size:", STA_BATCH_SIZE)
    print("Number of batches:", n_batches)

    # --------------------------------------------------------
    # E. Simulate all white-noise presentations
    # --------------------------------------------------------

    for batch in range(n_batches):

        stimuli, LGN_spikes = generate_sta_batch(
            center_kernel,
            surround_kernel,
            divisive_kernel,
            diagnostics
        )

        E_spikes, I_spikes = simulate_v1_batch(
            LGN_spikes,
            weights,
            dt=DT,
            tau_m=TAU_M,
            noise_std=NOISE_STD,
            v_min=V_MIN,
            v_reset=V_RESET,
            threshold_E=theta_E,
            threshold_I=theta_I
        )

        E_counts = E_spikes.sum(axis=2)
        I_counts = I_spikes.sum(axis=2)

        accumulate_sta(
            accumulators,
            stimuli,
            E_counts,
            I_counts
        )

        completed = (batch + 1) * STA_BATCH_SIZE

        if completed in checkpoints:

            corr_E, corr_I = calculate_convergence(
                accumulators,
                weight_RF_E,
                weight_RF_I
            )

            convergence_E[completed] = corr_E
            convergence_I[completed] = corr_I

            print(
                f"\nSTA checkpoint: {completed} stimuli"
            )

            print(
                "Median E STA-weight correlation:",
                corr_E
            )

            print(
                "Median I STA-weight correlation:",
                corr_I
            )

        if (batch + 1) % 10 == 0:

            print(
                f"Batch {batch + 1}/{n_batches} | "
                f"E spikes: {E_counts.sum()} | "
                f"I spikes: {I_counts.sum()}"
            )

    # --------------------------------------------------------
    # F. Calculate final STA and reliability
    # --------------------------------------------------------

    final = finalize_sta(
        accumulators,
        weight_RF_E,
        weight_RF_I
    )

    STA_E = final["STA_E"]
    STA_I = final["STA_I"]

    RF_correlations_E = final["RF_correlations_E"]
    RF_correlations_I = final["RF_correlations_I"]

    split_half_reliability = (
        final["split_half_reliability"]
    )

    # --------------------------------------------------------
    # G. Print final results
    # --------------------------------------------------------

    duration = N_STIMULI * N_STEPS * DT

    total_E_spikes = accumulators["total_E_spikes"]
    total_I_spikes = accumulators["total_I_spikes"]

    mean_E_rate = total_E_spikes.mean() / duration
    mean_I_rate = total_I_spikes.mean() / duration

    print("\n" + "=" * 55)
    print("STA RESULTS")
    print("=" * 55)

    print("Total stimuli:", N_STIMULI)

    print(
        "Mean E firing rate:",
        mean_E_rate,
        "Hz"
    )

    print(
        "Mean I firing rate:",
        mean_I_rate,
        "Hz"
    )

    print(
        "E neurons with zero spikes:",
        np.sum(total_E_spikes == 0)
    )

    print(
        "I neurons with zero spikes:",
        np.sum(total_I_spikes == 0)
    )

    print(
        "Median spikes per E neuron:",
        np.median(total_E_spikes)
    )

    print(
        "Median spikes per I neuron:",
        np.median(total_I_spikes)
    )

    print("\nFINAL STA-WEIGHT CORRELATIONS")

    print(
        "Median E correlation:",
        np.nanmedian(RF_correlations_E)
    )

    print(
        "Median I correlation:",
        np.nanmedian(RF_correlations_I)
    )

    print("\nSPLIT-HALF RELIABILITY")

    print(
        "Median E split-half correlation:",
        np.nanmedian(split_half_reliability)
    )

    print(
        "Neurons with valid split-half estimates:",
        np.isfinite(split_half_reliability).sum()
    )

    print("\nSTA CONVERGENCE")

    for count in checkpoints:

        print(
            f"{count} stimuli | "
            f"E correlation: {convergence_E[count]:.4f} | "
            f"I correlation: {convergence_I[count]:.4f}"
        )

    # --------------------------------------------------------
    # H. Calculate LGN diagnostics
    # --------------------------------------------------------

    (
        mean_LGN_rate,
        std_LGN_rate,
        fraction_capped,
        fraction_zero
    ) = summarize_lgn_diagnostics(
        diagnostics
    )

    # --------------------------------------------------------
    # I. Prepare original checkpoint arrays
    # --------------------------------------------------------

    counts = np.array(checkpoints)

    E_values = np.array([
        convergence_E[n]
        for n in checkpoints
    ])

    I_values = np.array([
        convergence_I[n]
        for n in checkpoints
    ])

    # --------------------------------------------------------
    # J. Save results
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve original filename only for the
    # original full 10,000-stimulus experiment.
    if n_batches == DEFAULT_N_BATCHES:
        prefix = "STA_2000"
    else:
        prefix = f"STA_{N_STIMULI}"

    np.savez_compressed(
        output_dir / f"{prefix}_results.npz",

        STA_E=STA_E,
        STA_I=STA_I,

        weight_RF_E=weight_RF_E,
        weight_RF_I=weight_RF_I,

        RF_correlations_E=RF_correlations_E,
        RF_correlations_I=RF_correlations_I,

        total_E_spikes=total_E_spikes,
        total_I_spikes=total_I_spikes,

        checkpoint_counts=counts,
        checkpoint_correlations_E=E_values,
        checkpoint_correlations_I=I_values,

        split_half_reliability=split_half_reliability,

        mean_LGN_rate=mean_LGN_rate,
        fraction_capped=fraction_capped,
        fraction_zero=fraction_zero,

        n_stimuli=N_STIMULI,
        batch_size=STA_BATCH_SIZE,
        seed=SEED
    )

    # --------------------------------------------------------
    # K. Save figures
    # --------------------------------------------------------

    plot_sta_examples(
        STA_E,
        weight_RF_E,
        output_dir / f"{prefix}_examples.png",
        show=show
    )

    if len(checkpoints) > 0:
        plot_convergence(
            counts,
            E_values,
            I_values,
            output_dir / f"{prefix}_convergence.png",
            show=show
        )

    plot_split_half(
        split_half_reliability,
        output_dir / f"{prefix}_split_half_reliability.png",
        show=show
    )

    print("\nSTA analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "STA_E": STA_E,
        "STA_I": STA_I,
        "RF_correlations_E": RF_correlations_E,
        "RF_correlations_I": RF_correlations_I,
        "split_half_reliability": split_half_reliability,
        "n_stimuli": N_STIMULI,
    }


# ============================================================
# 19. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct V1 receptive fields from "
            "spike-count-weighted white-noise stimuli."
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
        "--batches",
        type=int,
        default=DEFAULT_N_BATCHES,
        help="Number of batches (default: 500)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )
    parser.add_argument(
    "--checkpoint-path",
    type=str,
    default=None,
    help="Optional trained checkpoint overriding --learned-run"
    )

    args = parser.parse_args()

    analyze_spike_triggered_average(
    source_project=args.source_project,
    learned_run=args.learned_run,
    n_batches=args.batches,
    output_dir=args.output_dir,
    show=args.show,
    checkpoint_override=args.checkpoint_path
   )