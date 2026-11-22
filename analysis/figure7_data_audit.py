
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
)

WEIGHT_NAMES = (
    "W_E_E",
    "W_E_I",
    "W_I_E",
    "W_I_I",
)

for track in ("a", "b"):

    print("\n" + "=" * 65)
    print(f"FIGURE 7 DATA AUDIT — TRACK {track.upper()}")
    print("=" * 65)

    sta_path = (
        BASE / "functional_sta"
        / f"track_{track}"
        / "full_100000"
        / "STA_100000_results.npz"
    )

    checkpoint_path = (
        BASE / "synaptic_training"
        / f"track_{track}"
        / "checkpoints"
        / "batch_1100.npz"
    )

    print("\nSTA file exists:", sta_path.is_file())
    print("Checkpoint exists:", checkpoint_path.is_file())

    if not sta_path.is_file() or not checkpoint_path.is_file():
        continue

    with np.load(sta_path, allow_pickle=False) as data:

        print("\nSTA arrays:")

        for name in ("STA_E", "STA_I"):

            if name not in data.files:
                print("MISSING:", name)
                continue

            arr = data[name]

            print(
                name,
                "shape:", arr.shape,
                "finite:", np.isfinite(arr).all(),
                "min:", np.min(arr),
                "max:", np.max(arr)
            )

        if "n_stimuli" in data.files:
            print("Stimuli:", int(data["n_stimuli"]))

    with np.load(checkpoint_path, allow_pickle=False) as data:

        print("\nLateral weight matrices:")

        for name in WEIGHT_NAMES:

            arr = data[name]

            print(
                name,
                "shape:", arr.shape,
                "min:", np.min(arr),
                "max:", np.max(arr),
                "mean:", np.mean(arr)
            )

print("\nFigure 7 data audit complete.")
