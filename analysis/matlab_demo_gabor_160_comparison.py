
"""
160x160 Gabor refitting for MATLAB-demo STA datasets.

Reuses the original Figure 5 interpolation/refitting code.
Keeps original STA and Gabor results unchanged.
"""

import argparse
from pathlib import Path

from analysis import figure5_interpolation_refit as refit


PROJECT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
    / "E02_I04"
)


def paths_for_condition(condition):

    if condition == "original":

        sta_path = (
            ROOT / "sta_100k"
            / "STA_100000_results.npz"
        )

        fit_path = (
            ROOT / "gabor_100k"
            / "gabor_all_400_fits.npz"
        )

        output = ROOT / "gabor_160_refit"

    elif condition == "full_image":

        base = (
            ROOT
            / "sta_full_image_normalization_test"
        )

        sta_path = (
            base / "full_100000"
            / "STA_100000_results.npz"
        )

        fit_path = (
            base / "gabor_100k"
            / "gabor_all_400_fits.npz"
        )

        output = base / "gabor_160_refit"

    else:
        raise ValueError(condition)

    return sta_path, fit_path, output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--condition",
        choices=["original", "full_image"],
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5
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

    args = parser.parse_args()

    if not 1 <= args.limit <= 400:
        parser.error("--limit must be between 1 and 400")

    sta_path, fit_path, output = (
        paths_for_condition(args.condition)
    )

    for path in (sta_path, fit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Override paths only within this Python process.
    refit.STA_PATH = sta_path
    refit.FIT_PATH = fit_path
    refit.OUTPUT = output
    refit.PROGRESS = (
        output / "gabor_160_refit_progress.npz"
    )

    print("=" * 65)
    print("MATLAB-DEMO 160x160 GABOR REFITTING")
    print("=" * 65)

    print("Condition:", args.condition)
    print("STA:", sta_path)
    print("Original Gabor fits:", fit_path)
    print("Output:", output)
    print("Neurons:", args.limit)
    print("Random starts:", args.starts)
    print("Seed:", args.seed)

    refit.run(
        limit=args.limit,
        n_starts=args.starts,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
