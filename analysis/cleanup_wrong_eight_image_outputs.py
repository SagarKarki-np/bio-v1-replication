"""Remove ONLY known erroneous 8-image Figure 6 / Figure 8 track outputs.

DRY RUN by default. --apply actually removes allowlisted files after proof.
Historical MATLAB-demo original project, Figure 5/7, and checkpoints untouched.
"""
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_protocol" / "seed_3002"


def verify_old_figure6():
    folder = BASE / "figure6_correlations" / "matched_responses"
    found = []
    for track in ("a", "b"):
        p = folder / f"track_{track}_responses.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as d:
            source = str(d["stimulus_path"])
            checkpoint = str(d["checkpoint_path"])
            if not source.lower().replace("/", "\\").endswith("van_hateren_dog.npy"):
                raise RuntimeError(f"REFUSING delete: not known old stimulus: {p}: {source}")
            if f"track_{track}" not in checkpoint.lower():
                raise RuntimeError(f"REFUSING delete: unexpected checkpoint: {p}")
        found.append(p)
    # Correlation image and statistics are derived from those two bad response files.
    downstream = BASE / "figure6_correlations" / "correlation_analysis"
    for filename in ("figure6_track_comparison.png", "figure6_correlation_statistics.npz"):
        p = downstream / filename
        if p.exists():
            # confirm its saved numeric provenance if available
            if p.suffix == ".npz":
                with np.load(p, allow_pickle=False) as d:
                    if "stimulus_path" not in d.files or not str(d["stimulus_path"]).lower().endswith("van_hateren_dog.npy"):
                        raise RuntimeError(f"REFUSING: figure6 statistics provenance not old: {p}")
            if not all((folder / f"track_{t}_responses.npz").exists() for t in ("a", "b")):
                raise RuntimeError("REFUSING: both old Figure 6 response archives required to delete downstream")
            found.append(p)
    return found


def verify_old_figure8():
    found = []
    for track in ("a", "b"):
        folder = BASE / "figure8_selected_stimuli" / f"track_{track}"
        archive = folder / "figure8_selected_stimuli.npz"
        image = folder / "figure8_selected_stimuli.png"
        if not archive.exists() and not image.exists():
            continue
        if not archive.exists():
            raise RuntimeError(f"REFUSING: missing Figure 8 archive for {image}")
        with np.load(archive, allow_pickle=False) as d:
            source = str(d["checkpoint_source"])
            if f"track_{track}" not in source.lower():
                raise RuntimeError(f"REFUSING: unexpected checkpoint {archive}")
            # Older Figure 8 script did NOT record image source; it was verified
            # from terminal outputs showing 8-source-image shape for both tracks.
            if "stimulus_protocol" in d.files:
                raise RuntimeError(f"REFUSING: newer provenance-tagged result: {archive}")
            if "selected_patches" not in d.files or d["selected_patches"].shape != (3, 16, 16):
                raise RuntimeError(f"REFUSING: unexpected Figure 8 archive: {archive}")
        found.append(archive)
        if image.exists():
            found.append(image)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Delete allowlisted files, not just preview")
    args = ap.parse_args()
    targets = verify_old_figure6() + verify_old_figure8()
    print("KNOWN EIGHT-SOURCE-IMAGE OUTPUTS:")
    if not targets:
        print("None found. No deletion needed.")
    for p in targets:
        print(("DELETE" if args.apply else "WOULD DELETE"), p.relative_to(ROOT))
    if args.apply:
        for p in targets:
            p.unlink()
        print(f"Removed {len(targets)} specific files. No other files were touched.")
    else:
        print("Dry run only. Use --apply after verifying the list.")

if __name__ == "__main__":
    main()
