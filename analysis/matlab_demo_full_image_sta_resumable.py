
"""
Resumable full-image STA experiment.

Frozen MATLAB-demo E02_I04 network.
100,000 white-noise presentations.

Reuses:
- Original resumable STA controller
- Validated alternative stimulus generator
- Original V1 simulation and STA calculations

No existing results are modified.
"""

import argparse

from analysis import resumable_full_sta as runner
from analysis import matlab_demo_full_image_sta_test as alternative


CONDITION = "MATLAB_DEMO_FULL_IMAGE"


def alternative_paths(track):

    if track != CONDITION:
        raise ValueError(
            f"Unexpected condition: {track}"
        )

    adapter = alternative.CHECKPOINT

    output = (
        alternative.ROOT
        / "sta_full_image_normalization_test"
        / "full_100000"
    )

    return adapter, output


def main():

    parser = argparse.ArgumentParser()

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

    if not alternative.CHECKPOINT.is_file():
        raise FileNotFoundError(
            alternative.CHECKPOINT
        )

    # Temporarily substitute only the stimulus generator
    # and checkpoint/output paths in this Python process.
    original_paths = runner.paths_for_track
    original_generator = runner.sta.generate_sta_batch

    try:
        runner.paths_for_track = alternative_paths

        runner.sta.generate_sta_batch = (
            alternative.generate_full_image_sta_batch
        )

        print("=" * 65)
        print("MATLAB-DEMO FULL-IMAGE STA EXPERIMENT")
        print("=" * 65)

        print(
            "Checkpoint:",
            alternative.CHECKPOINT
        )

        print(
            "Full image size:",
            alternative.IMAGE_SIZE
        )

        print(
            "STA presentations:",
            runner.TOTAL_STIMULI
        )

        print(
            "STA seed:",
            runner.sta.SEED
        )

        runner.run(
            CONDITION,
            args.checkpoint_every
        )

    finally:
        runner.paths_for_track = original_paths

        runner.sta.generate_sta_batch = (
            original_generator
        )


if __name__ == "__main__":
    main()
