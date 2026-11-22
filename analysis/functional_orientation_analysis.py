
"""
Functional orientation tuning and spike-based OSI.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Reproduces the original analyze_functional_orientation.py.

Input
-----
results/functional_orientation/
    functional_orientation_responses.npz

The input contains spike counts from both the
initial and learned V1 networks, evaluated using
144 matched grating presentations.

Protocol
--------
12 orientations
4 spatial phases
3 repetitions
400 excitatory neurons

For each neuron:
1. Reshape spike counts into angle/phase/repeat axes.
2. Average across phases and repetitions.
3. Divide by stimulus duration to obtain firing rates.
4. Calculate circular-vector orientation selectivity.
5. Compare initial and learned OSI.

Zero-response neurons receive NaN OSI.

Original output
---------------
functional_orientation_analysis.npz
functional_tuning_examples.png
functional_OSI_distribution.png

All original saved numerical field names and
calculations are preserved for exact comparison.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import N_E


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
    / "functional_orientation_responses.npz"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "functional_orientation"
)

EXAMPLE_NEURONS = (0, 25, 100, 250)


# ============================================================
# 2. LOAD FUNCTIONAL SPIKE RESPONSES
# ============================================================

def load_functional_responses(path):
    """
    Load the original functional-response archive.

    Returns
    -------
    dict
        Angles, phases, initial and learned E counts,
        stimulus duration, and repeat count.

    Counts have shape:
        (N_E, angles * phases * repeats)
    """

    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Functional response archive not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        angles = data["angles"].copy()
        phases = data["phases"].copy()

        initial_E = data["initial_E"].copy()
        learned_E = data["learned_E"].copy()

        duration = float(data["duration"])
        n_repeats = int(data["n_repeats"])

    n_angles = len(angles)
    n_phases = len(phases)

    expected_presentations = (
        n_angles * n_phases * n_repeats
    )

    expected_shape = (
        N_E,
        expected_presentations
    )

    if initial_E.shape != expected_shape:
        raise ValueError(
            f"Expected initial_E shape {expected_shape}, "
            f"got {initial_E.shape}"
        )

    if learned_E.shape != initial_E.shape:
        raise ValueError(
            "Initial and learned response shapes differ."
        )

    if duration <= 0 or not np.isfinite(duration):
        raise ValueError(
            "Presentation duration must be positive."
        )

    if n_repeats < 1:
        raise ValueError(
            "Number of repetitions must be positive."
        )

    if not (
        np.all(np.isfinite(initial_E))
        and np.all(np.isfinite(learned_E))
    ):
        raise ValueError(
            "Spike counts contain non-finite values."
        )

    if (
        np.any(initial_E < 0)
        or np.any(learned_E < 0)
    ):
        raise ValueError(
            "Spike counts cannot be negative."
        )

    print("Functional response archive:", path)
    print("Initial E shape:", initial_E.shape)
    print("Learned E shape:", learned_E.shape)
    print("Number of orientations:", n_angles)
    print("Number of phases:", n_phases)
    print("Repetitions:", n_repeats)
    print("Duration:", duration, "s")

    return {
        "angles": angles,
        "phases": phases,
        "initial_E": initial_E,
        "learned_E": learned_E,
        "duration": duration,
        "n_repeats": n_repeats,
    }


# ============================================================
# 3. CONSTRUCT ORIENTATION TUNING CURVES
# ============================================================

def make_tuning_curves(
    spike_counts,
    n_angles,
    n_phases,
    n_repeats,
    duration
):
    """
    Convert spike counts into orientation tuning curves.

    Parameters
    ----------
    spike_counts : ndarray
        Shape (neurons, presentations).

    n_angles : int
    n_phases : int
    n_repeats : int
    duration : float
        Seconds per presentation.

    Returns
    -------
    firing_rates : ndarray
        Shape (neurons, n_angles).

    Original presentation ordering:
        angle
            phase
                repeat

    Original averaging:
        mean(axis=(2, 3))
    """

    spike_counts = np.asarray(spike_counts)

    if spike_counts.ndim != 2:
        raise ValueError(
            "Spike counts must be two-dimensional."
        )

    n_neurons = spike_counts.shape[0]

    expected_presentations = (
        n_angles * n_phases * n_repeats
    )

    if spike_counts.shape[1] != expected_presentations:
        raise ValueError(
            "Incorrect number of stimulus presentations."
        )

    if duration <= 0:
        raise ValueError(
            "Duration must be positive."
        )

    responses = spike_counts.reshape(
        n_neurons,
        n_angles,
        n_phases,
        n_repeats
    )

    # Original averaging across phases and repetitions.
    mean_spikes = responses.mean(axis=(2, 3))

    # Convert spike counts to firing rates (Hz).
    firing_rates = mean_spikes / duration

    return firing_rates


# ============================================================
# 4. CALCULATE SPIKE-BASED OSI
# ============================================================

def calculate_functional_osi(tuning, angles):
    """
    Calculate circular-vector orientation selectivity.

    Parameters
    ----------
    tuning : ndarray
        Shape (neurons, orientations).
        Nonnegative firing rates in Hz.

    angles : ndarray
        Orientation angles in degrees.

    Returns
    -------
    osi : ndarray
        Shape (neurons,).

    Notes
    -----
    Unlike structural OSI, silent neurons receive
    NaN rather than zero.

    Preserves the original np.divide calculation.
    """

    tuning = np.asarray(tuning)
    angles = np.asarray(angles)

    if tuning.ndim != 2:
        raise ValueError(
            "Tuning curves must be two-dimensional."
        )

    if tuning.shape[1] != len(angles):
        raise ValueError(
            "Angle count does not match tuning shape."
        )

    if not np.all(np.isfinite(tuning)):
        raise ValueError(
            "Tuning curves contain non-finite values."
        )

    if np.any(tuning < 0):
        raise ValueError(
            "Firing rates cannot be negative."
        )

    theta = np.deg2rad(angles)

    vectors = np.exp(2j * theta)

    numerator = np.abs(tuning @ vectors)
    denominator = tuning.sum(axis=1)

    osi = np.divide(
        numerator,
        denominator,
        out=np.full(
            len(denominator),
            np.nan
        ),
        where=denominator > 0
    )

    return osi


# ============================================================
# 5. SUMMARIZE POPULATION OSI
# ============================================================

def summarize_osi(name, osi):
    """
    Print original descriptive OSI statistics.

    Undefined neurons are excluded from mean,
    median, minimum and maximum calculations.
    """

    valid = osi[np.isfinite(osi)]

    print(f"\n{name}")
    print("Defined OSI neurons:", len(valid))

    print(
        "Undefined OSI neurons:",
        np.isnan(osi).sum()
    )

    if valid.size == 0:
        print("No defined orientation selectivity values.")

        return {
            "n_valid": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    print("Mean OSI:", valid.mean())
    print("Median OSI:", np.median(valid))
    print("Minimum:", valid.min())
    print("Maximum:", valid.max())

    return {
        "n_valid": int(valid.size),
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


# ============================================================
# 6. PAIRED INITIAL-VS-LEARNED COMPARISON
# ============================================================

def compare_functional_osi(
    OSI_initial,
    OSI_learned
):
    """
    Compare matched excitatory neurons with defined
    OSI in both network states.

    Returns
    -------
    valid : ndarray
        Boolean mask of matched valid neurons.

    change : ndarray
        Learned minus initial OSI for valid neurons.
    """

    if OSI_initial.shape != OSI_learned.shape:
        raise ValueError(
            "Initial and learned OSI shapes differ."
        )

    valid = (
        np.isfinite(OSI_initial)
        & np.isfinite(OSI_learned)
    )

    change = (
        OSI_learned[valid]
        - OSI_initial[valid]
    )

    print("\nPAIRED FUNCTIONAL OSI")
    print("Valid neurons:", valid.sum())

    if change.size == 0:
        print("No valid paired neurons.")

    else:
        print("Mean change:", change.mean())

        print(
            "Increased:",
            np.sum(change > 0)
        )

        print(
            "Decreased:",
            np.sum(change < 0)
        )

    return valid, change


# ============================================================
# 7. PLOT EXAMPLE ORIENTATION TUNING CURVES
# ============================================================

def plot_example_tuning(
    angles,
    initial_tuning,
    learned_tuning,
    save_path,
    show=False
):
    """
    Reproduce original four-neuron tuning figure.
    """

    neurons = [
        n for n in EXAMPLE_NEURONS
        if n < initial_tuning.shape[0]
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        constrained_layout=True
    )

    for ax, neuron in zip(
        axes.flat,
        neurons
    ):

        ax.plot(
            angles,
            initial_tuning[neuron],
            marker="o",
            label="Initial"
        )

        ax.plot(
            angles,
            learned_tuning[neuron],
            marker="o",
            label="Learned"
        )

        ax.set_title(f"Neuron {neuron}")

        ax.set_xlabel("Orientation (degrees)")
        ax.set_ylabel("Mean firing rate (Hz)")

        ax.set_xticks(angles)

        ax.tick_params(
            axis="x",
            labelrotation=45
        )

        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 8. PLOT POPULATION OSI DISTRIBUTIONS
# ============================================================

def plot_functional_osi(
    OSI_initial,
    OSI_learned,
    save_path,
    show=False
):
    """
    Reproduce the original functional OSI histogram.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 31)

    ax.hist(
        OSI_initial[np.isfinite(OSI_initial)],
        bins=bins,
        alpha=0.6,
        label="Initial"
    )

    ax.hist(
        OSI_learned[np.isfinite(OSI_learned)],
        bins=bins,
        alpha=0.6,
        label="Learned"
    )

    ax.set_xlabel(
        "Spike-based orientation selectivity index"
    )

    ax.set_ylabel("Number of neurons")

    ax.set_title(
        "Functional Orientation Selectivity"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. COMPLETE ANALYSIS
# ============================================================

def analyze_functional_orientation(
    input_path=DEFAULT_INPUT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Calculate orientation tuning and functional OSI
    from the validated initial and learned spike counts.

    This module does not regenerate LGN spikes or
    rerun V1 simulations.
    """

    input_path = Path(
        input_path
    ).expanduser()

    output_dir = Path(
        output_dir
    ).expanduser()

    # --------------------------------------------------------
    # A. Load validated functional responses
    # --------------------------------------------------------

    data = load_functional_responses(
        input_path
    )

    angles = data["angles"]
    phases = data["phases"]

    initial_E = data["initial_E"]
    learned_E = data["learned_E"]

    duration = data["duration"]
    n_repeats = data["n_repeats"]

    n_angles = len(angles)
    n_phases = len(phases)

    # --------------------------------------------------------
    # B. Calculate orientation tuning curves
    # --------------------------------------------------------

    initial_tuning = make_tuning_curves(
        initial_E,
        n_angles,
        n_phases,
        n_repeats,
        duration
    )

    learned_tuning = make_tuning_curves(
        learned_E,
        n_angles,
        n_phases,
        n_repeats,
        duration
    )

    print("\nTuning curve shape:", learned_tuning.shape)

    # --------------------------------------------------------
    # C. Calculate spike-based OSI
    # --------------------------------------------------------

    OSI_initial = calculate_functional_osi(
        initial_tuning,
        angles
    )

    OSI_learned = calculate_functional_osi(
        learned_tuning,
        angles
    )

    # --------------------------------------------------------
    # D. Summarize functional orientation selectivity
    # --------------------------------------------------------

    print("\nFUNCTIONAL ORIENTATION SELECTIVITY")

    initial_stats = summarize_osi(
        "Initial",
        OSI_initial
    )

    learned_stats = summarize_osi(
        "Learned",
        OSI_learned
    )

    # --------------------------------------------------------
    # E. Compare matched neurons
    # --------------------------------------------------------

    valid, change = compare_functional_osi(
        OSI_initial,
        OSI_learned
    )

    # --------------------------------------------------------
    # F. Save original result archive
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_path = (
        output_dir
        / "functional_orientation_analysis.npz"
    )

    np.savez_compressed(
        result_path,

        angles=angles,

        initial_tuning=initial_tuning,
        learned_tuning=learned_tuning,

        OSI_initial=OSI_initial,
        OSI_learned=OSI_learned
    )

    # --------------------------------------------------------
    # G. Save original figures
    # --------------------------------------------------------

    plot_example_tuning(
        angles,
        initial_tuning,
        learned_tuning,
        output_dir / "functional_tuning_examples.png",
        show=show
    )

    plot_functional_osi(
        OSI_initial,
        OSI_learned,
        output_dir / "functional_OSI_distribution.png",
        show=show
    )

    print("\nFunctional orientation analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "angles": angles,
        "initial_tuning": initial_tuning,
        "learned_tuning": learned_tuning,
        "OSI_initial": OSI_initial,
        "OSI_learned": OSI_learned,
        "valid": valid,
        "change": change,
        "initial_stats": initial_stats,
        "learned_stats": learned_stats,
    }


# ============================================================
# 10. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analyze spike-based orientation tuning "
            "in initial and learned V1 networks."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Functional orientation response archive"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for results and figures"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )

    args = parser.parse_args()

    analyze_functional_orientation(
        input_path=args.input,
        output_dir=args.output_dir,
        show=args.show
    )
