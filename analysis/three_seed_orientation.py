
"""
Three-seed functional orientation reproducibility audit.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.

Reproduces the original:
    audit_three_seed_orientation.py

Purpose
-------
Compare spike-based orientation selectivity across:
    Initial
    Reference
    Seed 3001
    Seed 3002

All four networks receive the same 144 grating-driven
LGN spike presentations and matched membrane-noise
random states.

IMPORTANT
---------
This experiment has a DIFFERENT grating coordinate
convention from analysis/grating_stimuli.py.

It also uses two separate NumPy RNG streams:
    default_rng(9123) for LGN spikes
    np.random.seed(9124) for simulation noise

Both details are preserved exactly.

Original output:
    results/audit/three_seed_orientation/
        three_seed_orientation.npz
        three_seed_orientation.png

The original data and checkpoints are read-only.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import (
    DT,
    N_STEPS,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
)

from simulation import simulate_v1_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "audit"
    / "three_seed_orientation"
)

NETWORK_RUNS = {
    "Reference": "training_test_25",
    "Seed 3001": "training_full_seed_3001",
    "Seed 3002": "training_full_seed_3002",
}

WEIGHT_ORDER = (
    "LGN_E",
    "LGN_I",
    "E_E",
    "E_I",
    "I_I",
    "I_E",
)

PATCH_SIZE = 16

SPATIAL_FREQUENCY = 2

ORIENTATIONS = np.arange(0, 180, 15)

PHASES = np.deg2rad([
    0, 90, 180, 270
])

N_REPEATS = 3

LGN_RATE_SCALE = 100.0
LGN_RATE_CAP = 100.0

EVAL_SEED = 9123

DURATION = DT * N_STEPS

EXPECTED_BATCH = 1100

N_PRESENTATIONS = (
    len(ORIENTATIONS)
    * len(PHASES)
    * N_REPEATS
)

NETWORK_NAMES = (
    "Initial",
    "Reference",
    "Seed 3001",
    "Seed 3002",
)


# ============================================================
# 2. LOAD SIX WEIGHT MATRICES
# ============================================================

def load_weights(path):
    """
    Load the six weight matrices in original order.

    Original array names:
        W_LGN_E
        W_LGN_I
        W_E_E
        W_E_I
        W_I_I
        W_I_E
    """

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        weights = tuple(
            data[f"W_{name}"].copy()
            for name in WEIGHT_ORDER
        )

    return weights


# ============================================================
# 3. LOAD ALL FOUR NETWORK STATES
# ============================================================

def load_networks(source_project):
    """
    Load the initial network and three trained networks.

    The initial network uses untrained weights and
    calibrated thresholds.

    Each trained network uses its own weights and
    thresholds from batch 1100.

    Dictionary insertion order is preserved:
        Initial
        Reference
        Seed 3001
        Seed 3002
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

    initial_thresholds_path = (
        source_project
        / "results"
        / "calibration"
        / "full_100_batches"
        / "calibrated_thresholds.npz"
    )

    networks = {}

    with np.load(
        initial_thresholds_path,
        allow_pickle=False
    ) as data:

        initial_theta_E = data[
            "threshold_E"
        ].copy()

        initial_theta_I = data[
            "threshold_I"
        ].copy()

    networks["Initial"] = {
        "weights": load_weights(
            initial_weights_path
        ),
        "theta_E": initial_theta_E,
        "theta_I": initial_theta_I,
    }

    for name, run_dir in NETWORK_RUNS.items():

        path = (
            source_project
            / "results"
            / "learning"
            / run_dir
            / "batch_1100.npz"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(
            path,
            allow_pickle=False
        ) as data:

            if int(data["learning_batch"]) != EXPECTED_BATCH:
                raise ValueError(
                    f"{name}: incomplete training run"
                )

            networks[name] = {
                "weights": tuple(
                    data[f"W_{key}"].copy()
                    for key in WEIGHT_ORDER
                ),
                "theta_E": data[
                    "threshold_E"
                ].copy(),
                "theta_I": data[
                    "threshold_I"
                ].copy(),
            }

    for name, network in networks.items():

        if network["theta_E"].shape != (400, 1):
            raise ValueError(
                f"{name}: incorrect E threshold shape"
            )

        if network["theta_I"].shape != (100, 1):
            raise ValueError(
                f"{name}: incorrect I threshold shape"
            )

        print(f"Loaded network: {name}")

    return networks


# ============================================================
# 4. ORIGINAL GRATING-TO-LGN RATE GENERATOR
# ============================================================

# Construct coordinate grid once.
# This matches the original script.

y, x = np.meshgrid(
    np.arange(PATCH_SIZE),
    np.arange(PATCH_SIZE),
    indexing="ij"
)

x = x - (PATCH_SIZE - 1) / 2
y = y - (PATCH_SIZE - 1) / 2


def generate_grating_rates(orientation, phase):
    """
    Generate LGN firing rates for one sinusoidal grating.

    This preserves the original coordinate convention:
        x*cos(theta) + y*sin(theta)

    The signed grating is flattened in Fortran order
    and split into ON/OFF channels.

    No DoG or divisive preprocessing is applied.

    Returns
    -------
    rates : ndarray, shape (512,)
    """

    theta = np.deg2rad(orientation)

    spatial_coordinate = (
        x * np.cos(theta)
        + y * np.sin(theta)
    )

    grating = np.cos(
        2 * np.pi
        * SPATIAL_FREQUENCY
        * spatial_coordinate / PATCH_SIZE
        + phase
    )

    signed = grating.flatten(order="F")

    ON = np.maximum(signed, 0)
    OFF = np.maximum(-signed, 0)

    rates = LGN_RATE_SCALE * np.concatenate(
        (ON, OFF)
    )

    return np.clip(
        rates,
        0,
        LGN_RATE_CAP
    )


# ============================================================
# 5. CONSTRUCT ORIGINAL STIMULUS TABLE
# ============================================================

def generate_stimulus_table():
    """
    Construct all 144 grating presentations.

    Original loop ordering:
        orientation
            phase
                repeat

    Returns
    -------
    stimulus_rates : ndarray, shape (144, 512)

    stimulus_labels : ndarray, shape (144, 3)
        Columns:
            orientation in degrees
            phase in radians
            repetition index
    """

    stimulus_labels = []
    stimulus_rates = []

    for orientation in ORIENTATIONS:

        for phase in PHASES:

            for repeat in range(N_REPEATS):

                rates = generate_grating_rates(
                    orientation,
                    phase
                )

                stimulus_rates.append(rates)

                stimulus_labels.append(
                    (orientation, phase, repeat)
                )

    stimulus_rates = np.array(
        stimulus_rates
    )

    stimulus_labels = np.array(
        stimulus_labels
    )

    if len(stimulus_rates) != N_PRESENTATIONS:
        raise RuntimeError(
            "Incorrect number of presentations."
        )

    return stimulus_rates, stimulus_labels


# ============================================================
# 6. GENERATE MATCHED LGN SPIKE TRAINS
# ============================================================

def generate_matched_lgn_spikes(stimulus_rates):
    """
    Generate the original LGN Poisson spike matrix.

    IMPORTANT:
    Uses np.random.default_rng(9123), not
    the legacy np.random.random() function.

    The full random array is generated at once with
    shape (512, 144, N_STEPS).

    After LGN generation, the legacy RNG is seeded
    separately for the membrane-noise state.

    Returns
    -------
    LGN_spikes : ndarray
        Shape (512, 144, N_STEPS)

    noise_state : tuple
        Legacy NumPy RNG state seeded with 9124.
    """

    rng = np.random.default_rng(EVAL_SEED)

    LGN_spikes = (
        rng.random(
            (512, N_PRESENTATIONS, N_STEPS)
        )
        < stimulus_rates.T[:, :, None] * DT
    )

    np.random.seed(EVAL_SEED + 1)

    noise_state = np.random.get_state()

    print(
        "Number of presentations:",
        N_PRESENTATIONS
    )

    print(
        "LGN spike shape:",
        LGN_spikes.shape
    )

    print(
        "Mean LGN input rate:",
        stimulus_rates.mean()
    )

    return LGN_spikes, noise_state


# ============================================================
# 7. ORIGINAL SPIKE-BASED OSI CALCULATION
# ============================================================

def calculate_osi(E_counts, stimulus_labels):
    """
    Calculate functional orientation selectivity.

    Parameters
    ----------
    E_counts : ndarray
        Shape (neurons, presentations).

    stimulus_labels : ndarray
        Shape (presentations, 3).

    Returns
    -------
    OSI : ndarray, shape (neurons,)
    tuning : ndarray, shape (neurons, 12)
    preferred_orientation : ndarray, shape (neurons,)

    The original complex-vector OSI, sum ordering,
    preferred-angle calculation and NaN handling
    are preserved.
    """

    n_neurons = E_counts.shape[0]

    tuning = np.zeros(
        (n_neurons, len(ORIENTATIONS))
    )

    for index, orientation in enumerate(ORIENTATIONS):

        mask = (
            stimulus_labels[:, 0] == orientation
        )

        tuning[:, index] = (
            E_counts[:, mask].mean(axis=1)
            / DURATION
        )

    angles = np.deg2rad(
        ORIENTATIONS
    )

    vector = np.sum(
        tuning
        * np.exp(2j * angles)[None, :],
        axis=1
    )

    denominator = tuning.sum(axis=1)

    OSI = np.divide(
        np.abs(vector),
        denominator,
        out=np.full(n_neurons, np.nan),
        where=denominator > 0
    )

    preferred_orientation = (
        np.rad2deg(np.angle(vector)) / 2
    ) % 180

    preferred_orientation[
        denominator == 0
    ] = np.nan

    return OSI, tuning, preferred_orientation


# ============================================================
# 8. SIMULATE ONE NETWORK
# ============================================================

def evaluate_network(
    LGN_spikes,
    network,
    noise_state,
    stimulus_labels
):
    """
    Evaluate one frozen network.

    Uses the same LGN spikes and resets the same
    membrane-noise state for every network.
    """

    np.random.set_state(noise_state)

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        network["weights"],
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=network["theta_E"],
        threshold_I=network["theta_I"]
    )

    E_counts = E_spikes.sum(axis=2)

    OSI, tuning, preferred = calculate_osi(
        E_counts,
        stimulus_labels
    )

    result = {
        "OSI": OSI,
        "tuning": tuning,
        "preferred": preferred,
        "mean_rate": E_counts.mean() / DURATION,
        "valid": np.isfinite(OSI).sum(),
    }

    return result


# ============================================================
# 9. PRINT REPRODUCIBILITY STATISTICS
# ============================================================

def print_orientation_results(results):
    """
    Print the original per-network and cross-run
    orientation-selectivity summaries.
    """

    trained_names = [
        "Reference",
        "Seed 3001",
        "Seed 3002"
    ]

    learned_means = np.array([
        np.nanmean(results[name]["OSI"])
        for name in trained_names
    ])

    initial_mean = np.nanmean(
        results["Initial"]["OSI"]
    )

    print("\n" + "=" * 55)

    print(
        "THREE-SEED ORIENTATION REPRODUCIBILITY"
    )

    print("=" * 55)

    print("Initial mean OSI:", initial_mean)
    print("Learned mean OSIs:", learned_means)

    print(
        "Across-run mean:",
        learned_means.mean()
    )

    print(
        "Across-run sample SD:",
        learned_means.std(ddof=1)
    )

    for name in trained_names:

        initial_osi = results["Initial"]["OSI"]
        learned_osi = results[name]["OSI"]

        valid = (
            np.isfinite(initial_osi)
            & np.isfinite(learned_osi)
        )

        fraction_increased = (
            np.mean(
                learned_osi[valid]
                > initial_osi[valid]
            )
            if valid.any()
            else np.nan
        )

        print(
            name,
            "fraction of comparable neurons "
            "with increased OSI:",
            fraction_increased
        )


# ============================================================
# 10. ORIGINAL COMPARISON FIGURE
# ============================================================

def plot_orientation_results(
    results,
    output_dir,
    show=False
):
    """
    Reproduce the original two-panel figure.
    """

    names = list(results.keys())

    means = [
        np.nanmean(results[name]["OSI"])
        for name in names
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].bar(
        names,
        means
    )

    axes[0].set_ylabel(
        "Mean functional OSI"
    )

    axes[0].set_title(
        "Orientation Selectivity Across Networks"
    )

    axes[0].set_ylim(0, 1)

    axes[0].tick_params(
        axis="x",
        rotation=20
    )

    for name in names:

        values = results[name]["OSI"]

        axes[1].hist(
            values[np.isfinite(values)],
            bins=np.linspace(0, 1, 26),
            density=True,
            histtype="step",
            linewidth=2,
            label=name
        )

    axes[1].set_xlabel("Functional OSI")

    axes[1].set_ylabel(
        "Probability density"
    )

    axes[1].set_title(
        "Distribution of Neuronal OSI"
    )

    axes[1].legend()

    fig.tight_layout()

    output_path = (
        output_dir
        / "three_seed_orientation.png"
    )

    fig.savefig(
        output_path,
        dpi=300
    )

    if show:
        plt.show()

    plt.close(fig)

    return output_path


# ============================================================
# 11. SAVE ORIGINAL 15-FIELD ARCHIVE
# ============================================================

def save_orientation_results(
    results,
    output_dir
):
    """
    Save the exact original result fields.

    Three metadata fields plus:
        OSI
        tuning
        preferred

    for each of four networks.
    """

    save_data = {
        "orientations": ORIENTATIONS,
        "n_presentations": N_PRESENTATIONS,
        "eval_seed": EVAL_SEED
    }

    for name in results:

        prefix = (
            name.lower()
            .replace(" ", "_")
        )

        save_data[f"{prefix}_OSI"] = (
            results[name]["OSI"]
        )

        save_data[f"{prefix}_tuning"] = (
            results[name]["tuning"]
        )

        save_data[f"{prefix}_preferred"] = (
            results[name]["preferred"]
        )

    output_path = (
        output_dir
        / "three_seed_orientation.npz"
    )

    np.savez_compressed(
        output_path,
        **save_data
    )

    return output_path


# ============================================================
# 12. COMPLETE FOUR-NETWORK ORIENTATION AUDIT
# ============================================================

def run_three_seed_orientation(
    source_project,
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Reproduce the original orientation audit.

    Original execution order:
        load networks
        generate stimulus rates and labels
        generate LGN spikes using default_rng(9123)
        seed membrane noise with 9124
        evaluate all four networks
        calculate OSI and tuning
        print reproducibility summary
        plot
        save original result fields
    """

    output_dir = Path(
        output_dir
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # A. Load all four network states.
    networks = load_networks(
        source_project
    )

    # B. Generate the 144 stimulus presentations.
    stimulus_rates, stimulus_labels = (
        generate_stimulus_table()
    )

    # C. Generate matched LGN spikes and noise state.
    LGN_spikes, noise_state = (
        generate_matched_lgn_spikes(
            stimulus_rates
        )
    )

    # D. Evaluate all four networks.
    results = {}

    for name, network in networks.items():

        print("\nEvaluating:", name)

        result = evaluate_network(
            LGN_spikes,
            network,
            noise_state,
            stimulus_labels
        )

        results[name] = result

        print(
            "Mean OSI:",
            np.nanmean(result["OSI"])
        )

        print(
            "Median OSI:",
            np.nanmedian(result["OSI"])
        )

        print(
            "Mean E firing rate:",
            result["mean_rate"]
        )

        print(
            "Valid neurons:",
            result["valid"]
        )

    # E. Print cross-network results.
    print_orientation_results(
        results
    )

    # F. Generate original figure.
    figure_path = plot_orientation_results(
        results,
        output_dir,
        show=show
    )

    # G. Save original result archive.
    archive_path = save_orientation_results(
        results,
        output_dir
    )

    print(
        "\nORIENTATION REPRODUCIBILITY AUDIT COMPLETE"
    )

    print("Numerical archive:", archive_path)
    print("Figure:", figure_path)

    return results


# ============================================================
# 13. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce four-network orientation "
            "selectivity audit."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to original V1_network_learning"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true"
    )

    args = parser.parse_args()

    run_three_seed_orientation(
        source_project=args.source_project,
        output_dir=args.output_dir,
        show=args.show
    )
