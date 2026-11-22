
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Files that may contain per-image neuronal responses
EXTENSIONS = {".npz", ".npy"}

KEYWORDS = (
    "rate", "response", "spike", "figure3",
    "fig3", "sparseness", "evaluation",
    "activity", "natural", "image"
)

print("=" * 65)
print("FIGURE 6 — SEARCH FOR EXISTING RESPONSE DATA")
print("=" * 65)

print("Results folder:", RESULTS)

if not RESULTS.exists():
    raise FileNotFoundError(RESULTS)

matches = 0

for path in sorted(RESULTS.rglob("*")):

    if path.suffix.lower() not in EXTENSIONS:
        continue

    # Skip STA and fitting archives: not Figure 6 responses.
    path_text = str(path).lower()

    if any(x in path_text for x in (
        "full_100000",
        "gabor_fitting",
        "functional_sta"
    )):
        continue

    try:
        with np.load(
            path,
            allow_pickle=False,
            mmap_mode="r"
        ) as data:

            if isinstance(data, np.lib.npyio.NpzFile):
                arrays = [
                    (name, data[name].shape)
                    for name in data.files
                ]
            else:
                arrays = [("(array)", data.shape)]

        relevant = [
            (name, shape)
            for name, shape in arrays
            if (
                len(shape) == 2
                and 100 in shape
                and any(
                    dimension in shape
                    for dimension in (512, 400, 100, 256)
                )
            )
            or any(
                keyword in name.lower()
                for keyword in KEYWORDS
            )
        ]

        if relevant:

            matches += 1
            print(f"\nFILE: {path.relative_to(ROOT)}")

            for name, shape in relevant:
                print(f"   {name}: {shape}")

    except (OSError, ValueError, KeyError) as exc:
        print(f"\nCould not inspect {path.name}: {exc}")

print("\n" + "=" * 65)
print("Matching files:", matches)
print("=" * 65)
