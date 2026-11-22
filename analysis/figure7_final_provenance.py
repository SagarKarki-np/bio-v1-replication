
"""
Figure 7 — Final training provenance verification.

Checks:
1. Initial weights SHA-256 checksum and seed.
2. Track A and B training metadata.
3. Correct threshold-calibration sources.
4. Final checkpoint metadata.

Read-only. No simulation or training.
"""

from pathlib import Path
import hashlib
import json
import numpy as np

from analysis import full_paper_train as train


BASE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "paper_protocol"
    / "seed_3002"
)


def check_initialization():

    folder = BASE / "initialization"
    weights_path = folder / "weights.npz"
    metadata_path = folder / "metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    actual_hash = hashlib.sha256(
        weights_path.read_bytes()
    ).hexdigest()

    expected_hash = metadata["weights_sha256"]

    assert actual_hash == expected_hash
    assert int(metadata["seed"]) == 3002

    print("\nINITIALIZATION")
    print("-" * 60)
    print("Seed:", metadata["seed"])
    print("SHA-256 checksum: MATCH")
    print("Initial weights:", weights_path)

    return actual_hash


def check_track(track):

    folder = (
        BASE / "synaptic_training"
        / f"track_{track.lower()}"
    )

    metadata_path = folder / "metadata.json"

    checkpoint = (
        folder / "checkpoints" / "batch_1100.npz"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    expected_source = train.threshold_source(track).resolve()

    recorded_source = Path(
        metadata["threshold_source"]
    ).resolve()

    assert recorded_source == expected_source
    assert expected_source.is_file()

    assert metadata["track"] == track
    assert int(metadata["network_seed"]) == 3002
    assert int(metadata["training_seed"]) == 303002
    assert metadata["parameterization"] == "paper"
    assert metadata["experiment"] == "full_paper_synaptic_training"

    expected_batches = 100 if track == "A" else 680

    assert int(metadata["calibration_batches"]) == expected_batches

    with np.load(checkpoint, allow_pickle=False) as data:

        assert int(data["completed_batches"]) == 1100
        assert int(data["seed"]) == 303002
        assert str(data["parameterization"]) == "paper"

        print(f"\nTRACK {track}")
        print("-" * 60)
        print("Training seed:", int(data["seed"]))
        print("Completed batches:", int(data["completed_batches"]))
        print("Parameterization:", str(data["parameterization"]))

    with np.load(expected_source, allow_pickle=False) as data:

        assert data["threshold_E"].shape == (400, 1)
        assert data["threshold_I"].shape == (100, 1)

        print("Calibration batches:", expected_batches)
        print("Threshold source:", expected_source)
        print("Mean initial E threshold:", data["threshold_E"].mean())
        print("Mean initial I threshold:", data["threshold_I"].mean())

    print("Training metadata: PASS")
    print("Threshold source: PASS")
    print("Final checkpoint: PASS")


def main():

    print("=" * 65)
    print("FIGURE 7 — FINAL PROVENANCE AUDIT")
    print("=" * 65)

    check_initialization()

    check_track("A")
    check_track("B")

    print("\n" + "=" * 65)
    print("ALL CHECKS PASSED")
    print("=" * 65)
    print("No files were modified.")


if __name__ == "__main__":
    main()
