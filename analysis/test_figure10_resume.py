
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

base = ROOT / "results" / "figure10_resume_test"

continuous = (
    base / "continuous"
    / "checkpoints" / "batch_0003.npz"
)

resumed = (
    base / "resumed"
    / "checkpoints" / "batch_0003.npz"
)

names = [
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_E", "W_I_I",
    "threshold_E", "threshold_I",
    "history_E_Hz", "history_I_Hz",
    "rng_keys", "rng_pos",
    "rng_has_gauss", "rng_cached_gaussian"
]

all_equal = True

with np.load(continuous, allow_pickle=False) as a:
    with np.load(resumed, allow_pickle=False) as b:

        for name in names:
            equal = np.array_equal(a[name], b[name])
            all_equal &= equal
            print(f"{name}: {equal}")

        assert int(a["completed_batches"]) == 3
        assert int(b["completed_batches"]) == 3

print("\n" + "=" * 50)

if all_equal:
    print("EXACT RESUME MATCH")
else:
    print("RESUME MISMATCH")
    raise AssertionError(
        "Resumed training differs from continuous training."
    )
