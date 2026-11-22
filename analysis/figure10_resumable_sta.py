
"""
Resumable 100,000-stimulus functional STA.

Reuses the original spike_triggered_average functions.
Does not modify trained weights or thresholds.

Important:
Checkpoint .pkl files must only be loaded from this
trusted local project directory.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from analysis import spike_triggered_average as sta


PROJECT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT
    / "results"
    / "paper_protocol"
    / "seed_3002"
)

TOTAL_STIMULI = 100_000
BATCH_SIZE = sta.STA_BATCH_SIZE
TOTAL_BATCHES = TOTAL_STIMULI // BATCH_SIZE

CHECKPOINT_STIMULI = (
    200, 500, 1000, 2000,
    10000, 20000, 50000, 100000
)


def paths_for_track(track):
    """
    Resolve the STA adapter and output directory
    for one Figure 10 firing-rate condition.
    """
    condition = track.upper()

    allowed = (
        "E01_I02",
        "E05_I10",
        "E10_I20",
        "E20_I40",
    )

    if condition not in allowed:
        raise ValueError(
            f"Unsupported condition: {condition}"
        )

    base = (
        ROOT
        / "figure10_firing_rate_sweep"
        / "paper"
        / condition
        / "functional_sta"
    )

    adapter = (
        base / "sta_compatible_checkpoint.npz"
    )

    output = base / "full_100000"

    return adapter, output

def save_progress(path, state):
    """Atomically save accumulators and RNG state."""

    temp = path.with_suffix(".tmp")

    with open(temp, "wb") as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    os.replace(temp, path)


def load_progress(path, track):
    """Restore an exact simulation checkpoint."""

    with open(path, "rb") as f:
        state = pickle.load(f)

    if state["track"] != track:
        raise ValueError("Track mismatch")

    if state["batch_size"] != BATCH_SIZE:
        raise ValueError("Batch size mismatch")

    if state["seed"] != sta.SEED:
        raise ValueError("STA seed mismatch")

    if not 0 <= state["completed"] <= TOTAL_BATCHES:
        raise ValueError("Invalid completed batch count")

    np.random.set_state(
        state["rng_state"]
    )

    return state


def run(track, checkpoint_every):

    track = track.upper()

    adapter, output = paths_for_track(track)

    if not adapter.is_file():
        raise FileNotFoundError(
            f"Missing STA adapter: {adapter}\n"
            "Run analysis.final_tracks_sta first "
            "to create the adapter."
        )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    progress_path = (
        output / "sta_progress.pkl"
    )

    results_path = (
        output / "STA_100000_results.npz"
    )

    if results_path.exists():
        print(
            "Completed results already exist:",
            results_path
        )
        return

    print("=" * 65)
    print(f"FULL 100,000-STIMULUS STA — TRACK {track}")
    print("=" * 65)

    weights, theta_E, theta_I = (
        sta.load_trained_network(adapter)
    )

    weight_RF_E, weight_RF_I = (
        sta.calculate_weight_rfs(weights)
    )

    (
        center_kernel,
        surround_kernel,
        divisive_kernel
    ) = sta.create_sta_kernels()

    if progress_path.exists():

        state = load_progress(
            progress_path,
            track
        )

        completed = state["completed"]
        accumulators = state["accumulators"]
        diagnostics = state["diagnostics"]

        convergence_E = state["convergence_E"]
        convergence_I = state["convergence_I"]

        print(
            "Resuming from batch:",
            completed
        )

    else:

        # Same seed as the original STA implementation.
        np.random.seed(sta.SEED)

        completed = 0

        accumulators = (
            sta.initialize_accumulators()
        )

        diagnostics = (
            sta.initialize_diagnostics()
        )

        convergence_E = {}
        convergence_I = {}

        print("Starting new 100,000-stimulus run.")

    print("Batch size:", BATCH_SIZE)
    print("Total batches:", TOTAL_BATCHES)
    print("Already completed:", completed)
    print("Output directory:", output)

    # --------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------

    for batch in range(completed, TOTAL_BATCHES):

        stimuli, LGN_spikes = (
            sta.generate_sta_batch(
                center_kernel,
                surround_kernel,
                divisive_kernel,
                diagnostics
            )
        )

        E_spikes, I_spikes = (
            sta.simulate_v1_batch(
                LGN_spikes,
                weights,
                dt=sta.DT,
                tau_m=sta.TAU_M,
                noise_std=sta.NOISE_STD,
                v_min=sta.V_MIN,
                v_reset=sta.V_RESET,
                threshold_E=theta_E,
                threshold_I=theta_I
            )
        )

        E_counts = E_spikes.sum(axis=2)
        I_counts = I_spikes.sum(axis=2)

        sta.accumulate_sta(
            accumulators,
            stimuli,
            E_counts,
            I_counts
        )

        completed = batch + 1

        n_stimuli = completed * BATCH_SIZE

        if n_stimuli in CHECKPOINT_STIMULI:

            corr_E, corr_I = (
                sta.calculate_convergence(
                    accumulators,
                    weight_RF_E,
                    weight_RF_I
                )
            )

            convergence_E[n_stimuli] = corr_E
            convergence_I[n_stimuli] = corr_I

            print(
                f"\nSTA checkpoint: {n_stimuli}"
                f" | E: {corr_E:.4f}"
                f" | I: {corr_I:.4f}",
                flush=True
            )

        if completed % 10 == 0:

            print(
                f"Batch {completed}/{TOTAL_BATCHES}"
                f" | E spikes: {E_counts.sum()}"
                f" | I spikes: {I_counts.sum()}",
                flush=True
            )

        if (
            completed % checkpoint_every == 0
            or completed == TOTAL_BATCHES
        ):

            state = {
                "track": track,
                "seed": sta.SEED,
                "batch_size": BATCH_SIZE,
                "completed": completed,
                "accumulators": accumulators,
                "diagnostics": diagnostics,
                "convergence_E": convergence_E,
                "convergence_I": convergence_I,
                "rng_state": np.random.get_state()
            }

            save_progress(
                progress_path,
                state
            )

            print(
                f"Saved resumable checkpoint "
                f"at {n_stimuli} stimuli",
                flush=True
            )

    # --------------------------------------------------
    # Finalize using the original STA functions
    # --------------------------------------------------

    final = sta.finalize_sta(
        accumulators,
        weight_RF_E,
        weight_RF_I
    )

    (
        mean_LGN_rate,
        std_LGN_rate,
        fraction_capped,
        fraction_zero
    ) = sta.summarize_lgn_diagnostics(
        diagnostics
    )

    total_E = accumulators["total_E_spikes"]
    total_I = accumulators["total_I_spikes"]

    duration = (
        TOTAL_STIMULI
        * sta.N_STEPS
        * sta.DT
    )

    E_rate = total_E.mean() / duration
    I_rate = total_I.mean() / duration

    counts = np.array(
        sorted(convergence_E.keys())
    )

    E_values = np.array([
        convergence_E[n]
        for n in counts
    ])

    I_values = np.array([
        convergence_I[n]
        for n in counts
    ])

    np.savez_compressed(
        results_path,

        STA_E=final["STA_E"],
        STA_I=final["STA_I"],

        weight_RF_E=weight_RF_E,
        weight_RF_I=weight_RF_I,

        RF_correlations_E=(
            final["RF_correlations_E"]
        ),

        RF_correlations_I=(
            final["RF_correlations_I"]
        ),

        total_E_spikes=total_E,
        total_I_spikes=total_I,

        checkpoint_counts=counts,
        checkpoint_correlations_E=E_values,
        checkpoint_correlations_I=I_values,

        split_half_reliability=(
            final["split_half_reliability"]
        ),

        mean_LGN_rate=mean_LGN_rate,
        fraction_capped=fraction_capped,
        fraction_zero=fraction_zero,

        n_stimuli=TOTAL_STIMULI,
        batch_size=BATCH_SIZE,
        seed=sta.SEED
    )

    # --------------------------------------------------
    # Save the original STA diagnostic figures
    # --------------------------------------------------

    sta.plot_sta_examples(
        final["STA_E"],
        weight_RF_E,
        output / "STA_100000_examples.png",
        show=False
    )

    sta.plot_convergence(
        counts,
        E_values,
        I_values,
        output / "STA_100000_convergence.png",
        show=False
    )

    sta.plot_split_half(
        final["split_half_reliability"],
        output / "STA_100000_split_half.png",
        show=False
    )

    print("\n" + "=" * 65)
    print(f"TRACK {track}: 100,000-STIMULUS STA COMPLETE")
    print("=" * 65)

    print("Mean E rate:", E_rate, "Hz")
    print("Mean I rate:", I_rate, "Hz")

    print(
        "Median E STA-weight correlation:",
        np.nanmedian(
            final["RF_correlations_E"]
        )
    )

    print(
        "Median I STA-weight correlation:",
        np.nanmedian(
            final["RF_correlations_I"]
        )
    )

    print(
        "Median E split-half reliability:",
        np.nanmedian(
            final["split_half_reliability"]
        )
    )

    print("Results saved:", results_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=(
            "E01_I02",
            "E05_I10",
            "E10_I20",
            "E20_I40",
        ),
        required=True
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25
    )

    args = parser.parse_args()

    if args.checkpoint_every < 1:
        parser.error(
            "--checkpoint-every must be positive"
        )

    if TOTAL_STIMULI % BATCH_SIZE != 0:
        raise ValueError(
            "Stimulus count must divide by batch size"
        )

    run(
        args.condition,
        args.checkpoint_every
    )


if __name__ == "__main__":
    main()
