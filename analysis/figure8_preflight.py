
"""
Figure 8 — Preflight audit for Track A and Track B.

Check existing simulation interfaces, saved STAs,
trained checkpoints and natural-image dataset.

Read-only: no simulation or training.
"""

from pathlib import Path
import inspect
import numpy as np

from figure10_config import SOURCE_PROJECT
from simulation import simulate_v1_batch
from preprocessing import extract_random_patch
from simulation import generate_lgn_population_spikes
from config import (
    DT, N_STEPS, PATCH_SIZE,
    LGN_RATE_SCALE, LGN_MAX_RATE
)


ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
)

WEIGHT_SHAPES = {
    "W_LGN_E": (400, 512),
    "W_LGN_I": (100, 512),
    "W_E_E": (400, 400),
    "W_E_I": (100, 400),
    "W_I_E": (400, 100),
    "W_I_I": (100, 100),
}


def main():

    print("=" * 65)
    print("FIGURE 8 — PREFLIGHT AUDIT")
    print("=" * 65)

    print("\nSIMULATION INTERFACE")

    signature = inspect.signature(simulate_v1_batch)
    print(signature)

    assert "record_inputs" in signature.parameters
    assert "threshold_E" in signature.parameters
    assert "threshold_I" in signature.parameters

    print("record_inputs supported: YES")
    print("Threshold arguments supported: YES")

    print("\nREUSABLE FUNCTIONS")

    print(
        "extract_random_patch:",
        callable(extract_random_patch)
    )

    print(
        "generate_lgn_population_spikes:",
        callable(generate_lgn_population_spikes)
    )

    print("\nCONFIGURATION")

    print("DT:", DT)
    print("N_STEPS:", N_STEPS)
    print("PATCH_SIZE:", PATCH_SIZE)
    print("LGN_RATE_SCALE:", LGN_RATE_SCALE)
    print("LGN_MAX_RATE:", LGN_MAX_RATE)
    print("Presentation duration:", DT * N_STEPS)

    assert PATCH_SIZE == 16
    assert N_STEPS > 0
    assert DT > 0

    print("\nTRACK DATA")

    for track in ("a", "b"):

        sta_path = (
            BASE / "functional_sta"
            / f"track_{track}"
            / "full_100000"
            / "STA_100000_results.npz"
        )

        checkpoint = (
            BASE / "synaptic_training"
            / f"track_{track}"
            / "checkpoints"
            / "batch_1100.npz"
        )

        assert sta_path.is_file(), sta_path
        assert checkpoint.is_file(), checkpoint

        with np.load(sta_path, allow_pickle=False) as data:

            sta = data["STA_E"]

            assert sta.shape == (400, 256)
            assert np.isfinite(sta).all()
            assert int(data["n_stimuli"]) == 100000

        with np.load(checkpoint, allow_pickle=False) as data:

            assert int(data["completed_batches"]) == 1100
            assert str(data["parameterization"]) == "paper"

            for name, shape in WEIGHT_SHAPES.items():
                assert data[name].shape == shape
                assert np.isfinite(data[name]).all()

            assert data["threshold_E"].shape == (400, 1)
            assert data["threshold_I"].shape == (100, 1)

        print(f"Track {track.upper()}: PASS")
        print("  STA:", sta_path)
        print("  Checkpoint:", checkpoint)

    dataset = (
        Path(SOURCE_PROJECT)
        / "data"
        / "van_hateren_dog.npy"
    )

    assert dataset.is_file(), dataset

    images = np.load(dataset, mmap_mode="r")

    print("\nNATURAL-IMAGE DATASET")
    print("Path:", dataset)
    print("Shape:", images.shape)

    assert images.shape == (1024, 1536, 8)

    print("\n" + "=" * 65)
    print("FIGURE 8 PREFLIGHT PASSED")
    print("=" * 65)
    print("No simulation performed.")
    print("No files modified.")


if __name__ == "__main__":
    main()
