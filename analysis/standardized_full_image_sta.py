
"""
Standardized full-image STA for paper replication.

Supports Track A, Track B and Figure 10 firing-rate conditions.

Uses the validated 128x128 full-image white-noise generator
and existing resumable STA simulation controllers.

This is a documented interpretation of the paper's
LGN pipeline, not a verified exact STA implementation.

Previous per-patch STA results are preserved.
"""

import argparse
from pathlib import Path

from analysis import resumable_full_sta
from analysis import figure10_resumable_sta
from analysis import full_image_sta_generator


CONDITIONS = (
    "track_a",
    "track_b",
    "E01_I02",
    "E05_I10",
    "E10_I20",
    "E20_I40",
)


def select_runner(condition):

    if condition == "track_a":
        return resumable_full_sta, "A"

    if condition == "track_b":
        return resumable_full_sta, "B"

    return figure10_resumable_sta, condition


def standardized_paths(runner, track):

    adapter, old_output = runner.paths_for_track(track)

    new_output = (
        old_output.parent / "full_image_100000"
    )

    if new_output.resolve() == old_output.resolve():
        raise RuntimeError(
            "New output overlaps the original STA output."
        )

    return adapter, new_output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        required=True
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths without running STA"
    )

    args = parser.parse_args()

    if args.checkpoint_every < 1:
        parser.error(
            "--checkpoint-every must be positive"
        )

    runner, track = select_runner(
        args.condition
    )

    # Capture original path resolver before substitution.
    original_paths = runner.paths_for_track
    original_generator = runner.sta.generate_sta_batch

    def new_paths(selected_track):
        adapter, old_output = original_paths(
            selected_track
        )

        new_output = (
            old_output.parent / "full_image_100000"
        )

        if new_output.resolve() == old_output.resolve():
            raise RuntimeError(
                "Output directory collision."
            )

        return adapter, new_output

    adapter, output = new_paths(track)

    if not adapter.is_file():
        raise FileNotFoundError(
            f"Missing trained-network STA adapter: {adapter}"
        )

    old_output = original_paths(track)[1]

    print("=" * 65)
    print("STANDARDIZED FULL-IMAGE STA")
    print("=" * 65)

    print("Condition:", args.condition)
    print("Checkpoint:", adapter)
    print("Original output:", old_output)
    print("New output:", output)
    print("Parent noise image: 128 x 128")
    print("Extracted patch: 16 x 16")
    print("Presentations:", runner.TOTAL_STIMULI)
    print("Batch size:", runner.BATCH_SIZE)
    print("Seed:", runner.sta.SEED)

    if args.check_only:
        print("\nPREFLIGHT PASSED")
        print("No simulation performed.")
        print("No files modified.")
        return

    # Preserve pre-existing output files.
    result_path = (
        output / "STA_100000_results.npz"
    )

    progress_path = (
        output / "sta_progress.pkl"
    )

    if result_path.is_file():
        print(
            "\nCompleted standardized STA already exists."
        )
        print(result_path)
        return

    if output.exists() and not progress_path.is_file():
        if any(output.iterdir()):
            raise FileExistsError(
                "Output contains files but has no resumable "
                f"progress checkpoint: {output}"
            )

    try:
        # The substitution is local to this process.
        runner.paths_for_track = new_paths

        runner.sta.generate_sta_batch = (
            full_image_sta_generator
            .generate_full_image_sta_batch
        )

        runner.run(
            track,
            args.checkpoint_every
        )

    finally:
        runner.paths_for_track = original_paths

        runner.sta.generate_sta_batch = (
            original_generator
        )


if __name__ == "__main__":
    main()
