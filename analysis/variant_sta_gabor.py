"""
STA + Gabor pipeline for Track A VARIANT runs (A_d experiments).

Reuses, unchanged, the same code and settings as the original
Track A full-image analysis:
  - standardized 128x128 full-image white-noise STA, 100,000 stimuli
  - gabor_fitting (16x16, 12 starts, seed 2705)
  - figure5_interpolation_refit (160x160 refit, 12 starts, seed 2705)

Only the input/output folders differ. Existing results are never written to.

Stages (run in this order, from the project root):

  python -m analysis.variant_sta_gabor prepare  --variant Ad2
  python -m analysis.variant_sta_gabor sta      --variant Ad2
  python -m analysis.variant_sta_gabor gabor16  --variant Ad2
  python -m analysis.variant_sta_gabor gabor160 --variant Ad2
  python -m analysis.variant_sta_gabor summary  --variant Ad2

'sta' is resumable: re-run the same command to continue.
Variants: Ad2 (A_p=1, A_d=2.0) and plit (A_p=0.001, A_d=0.002).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "results" / "paper_protocol" / "seed_3002"

VARIANTS = {
    "Ad2": "track_a_Ad2",
    "plit": "track_a_Ap0.001_Ad0.002",
}

THRESHOLD = 0.10


def dirs(variant):
    name = VARIANTS[variant]
    training = BASE / "synaptic_training" / name
    out = BASE / "functional_sta" / name
    return {
        "checkpoint": training / "checkpoints" / "batch_1100.npz",
        "out": out,
        "adapter": out / "sta_compatible_checkpoint.npz",
        "sta_dir": out / "full_image_100000",
        "sta": out / "full_image_100000" / "STA_100000_results.npz",
        "g16_dir": out / "full_image_gabor_16",
        "g16": out / "full_image_gabor_16" / "gabor_all_400_fits.npz",
        "g160_dir": out / "full_image_gabor_160",
    }


def prepare(variant):
    from analysis import spike_triggered_average as sta

    p = dirs(variant)
    if not p["checkpoint"].is_file():
        raise FileNotFoundError(p["checkpoint"])
    if p["adapter"].exists():
        print("Adapter already exists:", p["adapter"])
        return

    with np.load(p["checkpoint"], allow_pickle=False) as z:
        if int(z["completed_batches"]) != 1100:
            raise ValueError("Checkpoint is not at batch 1100")
        data = {n: z[n].copy() for n in sta.WEIGHT_NAMES}
        data["threshold_E"] = z["threshold_E"].reshape(-1, 1).copy()
        data["threshold_I"] = z["threshold_I"].reshape(-1, 1).copy()
        print("triplet_depression in checkpoint:",
              float(z["triplet_depression"]))

    p["out"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p["adapter"], **data)

    w, tE, tI = sta.load_trained_network(p["adapter"])
    assert len(w) == 6 and tE.shape == (400, 1) and tI.shape == (100, 1)
    print("Adapter written and validated:", p["adapter"])


def run_sta(variant, checkpoint_every):
    from analysis import resumable_full_sta as runner
    from analysis import full_image_sta_generator as gen

    p = dirs(variant)
    if not p["adapter"].is_file():
        raise FileNotFoundError("Run the 'prepare' stage first.")
    if p["sta"].is_file():
        print("STA already complete:", p["sta"])
        return

    original_paths = runner.paths_for_track
    original_gen = runner.sta.generate_sta_batch
    try:
        runner.paths_for_track = lambda track: (p["adapter"], p["sta_dir"])
        runner.sta.generate_sta_batch = gen.generate_full_image_sta_batch
        print("Variant:", variant, "| output:", p["sta_dir"])
        runner.run("A", checkpoint_every)
    finally:
        runner.paths_for_track = original_paths
        runner.sta.generate_sta_batch = original_gen


def run_gabor16(variant, starts, seed):
    p = dirs(variant)
    if not p["sta"].is_file():
        raise FileNotFoundError("Run the 'sta' stage first.")
    cmd = [
        sys.executable, "-m", "analysis.gabor_fitting",
        "--input", str(p["sta"]),
        "--output-dir", str(p["g16_dir"]),
        "--limit", "400",
        "--starts", str(starts),
        "--seed", str(seed),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT, check=True)


def run_gabor160(variant, starts, seed):
    from analysis import figure5_interpolation_refit as refit

    p = dirs(variant)
    if not p["g16"].is_file():
        raise FileNotFoundError("Run the 'gabor16' stage first.")

    with np.load(p["g16"], allow_pickle=False) as d:
        if d["fit_errors"].shape != (400,) or int(d["n_stimuli"]) != 100_000:
            raise ValueError("Unexpected 16x16 fit file")
        print("16x16 passes:", int((d["fit_errors"] < THRESHOLD).sum()))

    refit.STA_PATH = p["sta"]
    refit.FIT_PATH = p["g16"]
    refit.OUTPUT = p["g160_dir"]
    refit.PROGRESS = p["g160_dir"] / "gabor_160_refit_progress.npz"
    refit.run(limit=400, n_starts=starts, seed=seed)


def summary(variant):
    p = dirs(variant)
    f = p["g160_dir"] / "gabor_160_comparison_400.npz"
    with np.load(f, allow_pickle=False) as d:
        e160 = d["refitted_errors_160"]
        e16 = d["original_errors_16"]
    print(f"{variant}: 16x16 passes {int((e16 < THRESHOLD).sum())}/400 | "
          f"160x160 refit passes {int((e160 < THRESHOLD).sum())}/400")

    ref = (BASE / "functional_sta" / "track_a" / "full_image_gabor_160"
           / "gabor_160_comparison_400.npz")
    if ref.is_file():
        with np.load(ref, allow_pickle=False) as d:
            r = d["refitted_errors_160"]
        print(f"Original Track A (A_d=0.1): 160x160 refit passes "
              f"{int((r < THRESHOLD).sum())}/400")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[
        "prepare", "sta", "gabor16", "gabor160", "summary"])
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2705)
    a = ap.parse_args()

    if a.stage == "prepare":
        prepare(a.variant)
    elif a.stage == "sta":
        run_sta(a.variant, a.checkpoint_every)
    elif a.stage == "gabor16":
        run_gabor16(a.variant, a.starts, a.seed)
    elif a.stage == "gabor160":
        run_gabor160(a.variant, a.starts, a.seed)
    else:
        summary(a.variant)


if __name__ == "__main__":
    main()
