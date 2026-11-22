
"""
Convergence-controlled E10_I20 functional STA.

Reuses the existing Figure 10 STA simulation unchanged.
Loads the trained checkpoint directly.
Saves results in an independent output directory.
"""

import argparse
from pathlib import Path

import numpy as np

from analysis import figure10_resumable_sta as runner
from analysis import spike_triggered_average as sta


PROJECT = Path(__file__).resolve().parents[1]

BASE = (
    PROJECT / "results" / "paper_protocol"
    / "seed_3002" / "convergence_experiments"
    / "E10_I20"
)

CHECKPOINT = (
    BASE / "training" / "checkpoints"
    / "batch_1100.npz"
)

OUTPUT = BASE / "functional_sta" / "full_100000"


def paths_for_condition(condition):
    if condition != "E10_I20":
        raise ValueError("Unexpected condition")

    return CHECKPOINT, OUTPUT


def preflight():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    with np.load(CHECKPOINT, allow_pickle=False) as data:
        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"
        assert np.isclose(float(data["target_E_Hz"]), 10.0)
        assert np.isclose(float(data["target_I_Hz"]), 20.0)

    weights, theta_E, theta_I = (
        sta.load_trained_network(CHECKPOINT)
    )

    assert len(weights) == 6
    assert theta_E.shape == (400, 1)
    assert theta_I.shape == (100, 1)

    original_output = (
        runner.ROOT / "figure10_firing_rate_sweep"
        / "paper" / "E10_I20"
        / "functional_sta" / "full_100000"
    )

    assert OUTPUT.resolve() != original_output.resolve()

    if OUTPUT.exists():
        existing = [
            p.name for p in OUTPUT.iterdir()
            if p.name != "sta_progress.pkl"
        ]
        if existing:
            raise FileExistsError(
                "Output contains existing files: "
                + ", ".join(existing)
            )

    print("=" * 60)
    print("E10/I20 CONVERGENCE-CONTROLLED STA PREFLIGHT")
    print("=" * 60)
    print("Checkpoint:", CHECKPOINT)
    print("Output:", OUTPUT)
    print("Weight shapes:", [w.shape for w in weights])
    print("E thresholds:", theta_E.shape)
    print("I thresholds:", theta_I.shape)
    print("STA seed:", sta.SEED)
    print("STA batch size:", runner.BATCH_SIZE)
    print("Total stimuli:", runner.TOTAL_STIMULI)
    print("Original runner: unchanged")
    print("Preflight passed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight",
        action="store_true"
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25
    )

    args = parser.parse_args()

    if args.checkpoint_every < 1:
        parser.error("Checkpoint interval must be positive")

    preflight()

    if args.preflight:
        return

    # Redirect checkpoint and output selection only.
    # Scientific simulation functions are unchanged.
    runner.paths_for_track = paths_for_condition

    runner.run(
        "E10_I20",
        args.checkpoint_every
    )


if __name__ == "__main__":
    main()
