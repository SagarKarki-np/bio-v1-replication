
"""
Functional STA analysis for completed Track A and B.

Reuses analysis.spike_triggered_average.

- 10,000 white-noise presentations per track
- Same stimulus and noise seed
- Frozen network weights and thresholds
- Original training checkpoints remain unchanged
- Separate analysis outputs for both tracks
"""

import argparse
import json
from pathlib import Path

import numpy as np

import figure10_config

from analysis import figure10_train as tr
from analysis import spike_triggered_average as sta


PROJECT = Path(__file__).resolve().parents[1]

NETWORK_SEED = 3002
TRAINING_SEED = 303002

FINAL_BATCH = 1100
TRAINING_BATCH_SIZE = 100

STA_BATCHES = 500
EXPECTED_STIMULI = 10000

ROOT = (
    PROJECT
    / "results"
    / "paper_protocol"
    / f"seed_{NETWORK_SEED}"
)

condition = figure10_config.make_condition(
    2.0,
    parameterization="paper"
)


def run_track(track):

    track = track.upper()

    training_dir = (
        ROOT
        / "synaptic_training"
        / f"track_{track.lower()}"
    )

    output_dir = (
        ROOT
        / "functional_sta"
        / f"track_{track.lower()}"
    )

    checkpoint = (
        training_dir
        / "checkpoints"
        / "batch_1100.npz"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    print("\n" + "=" * 65)
    print(f"FUNCTIONAL STA — TRACK {track}")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Load completed training checkpoint
    # --------------------------------------------------

    (
        completed,
        weights,
        theta_E,
        theta_I,
        history_E,
        history_I
    ) = tr.load_latest_checkpoint(
        training_dir,
        condition,
        TRAINING_BATCH_SIZE,
        TRAINING_SEED
    )

    if completed != FINAL_BATCH:
        raise ValueError(
            f"Track {track} is at batch {completed}, "
            f"expected {FINAL_BATCH}"
        )

    # --------------------------------------------------
    # 2. Adapt checkpoint to original STA format
    # --------------------------------------------------

    # Original STA uses keys such as W_LGN_E.
    # Training uses keys such as LGN_E.
    #
    # Match by pathway name rather than relying on
    # dictionary iteration or matrix position.

    adapted = {}

    for sta_name in sta.WEIGHT_NAMES:

        key = sta_name

        if key.startswith("W_"):
            key = key[2:]

        if key not in weights:
            raise KeyError(
                f"STA weight {sta_name} cannot be "
                f"matched to training weights. "
                f"Available: {list(weights)}"
            )

        adapted[sta_name] = weights[key].copy()

    adapted["threshold_E"] = theta_E.copy()
    adapted["threshold_I"] = theta_I.copy()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    adapter_path = (
        output_dir
        / "sta_compatible_checkpoint.npz"
    )

    # Prevent an accidental rewrite of a previously
    # generated adapter or analysis.
    if adapter_path.exists():
        raise FileExistsError(
            f"STA adapter already exists: {adapter_path}"
        )

    np.savez_compressed(
        adapter_path,
        **adapted
    )

    # Confirm the original STA loader understands it.
    loaded_weights, loaded_E, loaded_I = (
        sta.load_trained_network(adapter_path)
    )

    assert len(loaded_weights) == 6
    assert np.array_equal(loaded_E, theta_E)
    assert np.array_equal(loaded_I, theta_I)

    for index, name in enumerate(sta.WEIGHT_NAMES):
        assert np.array_equal(
            loaded_weights[index],
            adapted[name]
        )

    metadata = {
        "track": track,
        "network_seed": NETWORK_SEED,
        "training_seed": TRAINING_SEED,
        "training_checkpoint": str(checkpoint),
        "completed_learning_batches": completed,
        "sta_seed": sta.SEED,
        "sta_batches": STA_BATCHES,
        "sta_batch_size": sta.STA_BATCH_SIZE,
        "sta_stimuli": (
            STA_BATCHES * sta.STA_BATCH_SIZE
        ),
        "analysis": "functional_spike_triggered_average",
        "plasticity_enabled": False,
    }

    (
        output_dir / "sta_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 3. Run existing STA implementation
    # --------------------------------------------------

    if (
        STA_BATCHES * sta.STA_BATCH_SIZE
        != EXPECTED_STIMULI
    ):
        raise ValueError(
            "STA stimulus count mismatch"
        )

    print("Training checkpoint:", checkpoint)
    print("STA adapter:", adapter_path)
    print("White-noise stimuli:", EXPECTED_STIMULI)
    print("STA seed:", sta.SEED)
    print("Output:", output_dir)

    results = sta.analyze_spike_triggered_average(
        source_project=PROJECT,
        n_batches=STA_BATCHES,
        output_dir=output_dir,
        show=False,
        checkpoint_override=adapter_path
    )

    print("\n" + "=" * 65)
    print(f"TRACK {track} STA COMPLETED")
    print("=" * 65)

    print("Stimuli:", results["n_stimuli"])

    print(
        "Median E STA-weight correlation:",
        np.nanmedian(
            results["RF_correlations_E"]
        )
    )

    print(
        "Median I STA-weight correlation:",
        np.nanmedian(
            results["RF_correlations_I"]
        )
    )

    print(
        "Median E split-half reliability:",
        np.nanmedian(
            results["split_half_reliability"]
        )
    )

    print("Results:", output_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=("A", "B"),
        required=True
    )

    args = parser.parse_args()

    run_track(args.track)


if __name__ == "__main__":
    main()
