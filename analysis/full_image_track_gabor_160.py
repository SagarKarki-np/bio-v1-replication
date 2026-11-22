
"""
160x160 Gabor refitting for standardized full-image
STA results from Track A and Track B.

Reuses the validated interpolation/refitting code.
All outputs are saved separately.
"""

import argparse
from pathlib import Path
import numpy as np

from analysis import figure5_interpolation_refit as refit


ROOT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "paper_protocol"
    / "seed_3002"
    / "functional_sta"
)


def paths_for_track(track):

    base = ROOT / f"track_{track.lower()}"

    sta_path = (
        base / "full_image_100000"
        / "STA_100000_results.npz"
    )

    fit_path = (
        base / "full_image_gabor_16"
        / "gabor_all_400_fits.npz"
    )

    output = (
        base / "full_image_gabor_160"
    )

    return sta_path, fit_path, output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=("A", "B"),
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=400
    )

    parser.add_argument(
        "--starts",
        type=int,
        default=12
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2705
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    args = parser.parse_args()

    sta_path, fit_path, output = (
        paths_for_track(args.track)
    )

    if not sta_path.is_file():
        raise FileNotFoundError(sta_path)

    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)

    with np.load(fit_path, allow_pickle=False) as data:
        errors = data["fit_errors"]
        n_stimuli = int(data["n_stimuli"])

    if errors.shape != (400,):
        raise ValueError(
            f"Expected 400 Gabor errors, found {errors.shape}"
        )

    if n_stimuli != 100_000:
        raise ValueError(
            f"Expected 100000 STA presentations, found {n_stimuli}"
        )

    print("=" * 65)
    print("STANDARDIZED FULL-IMAGE 160x160 GABOR REFITTING")
    print("=" * 65)
    print("Track:", args.track)
    print("STA:", sta_path)
    print("Initial Gabor fits:", fit_path)
    print("Output:", output)
    print("Original 16x16 passes:", np.sum(errors < 0.10))
    print("Neurons requested:", args.limit)
    print("New random starts:", args.starts)
    print("Seed:", args.seed)

    if args.check_only:
        print("\nPREFLIGHT PASSED")
        print("No fitting performed.")
        print("No files modified.")
        return

    if args.limit < 1 or args.limit > 400:
        raise ValueError("--limit must be between 1 and 400")

    progress = (
        output / "gabor_160_refit_progress.npz"
    )

    # Set source and output paths for this process only.
    refit.STA_PATH = sta_path
    refit.FIT_PATH = fit_path
    refit.OUTPUT = output
    refit.PROGRESS = progress

    refit.run(
        limit=args.limit,
        n_starts=args.starts,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
