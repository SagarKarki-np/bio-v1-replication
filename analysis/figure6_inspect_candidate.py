
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

path = (
    ROOT
    / "results"
    / "inhibitory_perturbations"
    / "inhibition_intervention.npz"
)

print("=" * 60)
print("FIGURE 6 — CANDIDATE DATA INSPECTION")
print("=" * 60)

with np.load(path, allow_pickle=False) as data:

    print("\nFile:", path)
    print("\nAll saved keys:")

    for key in data.files:
        array = data[key]
        print(f"  {key}: shape={array.shape}, dtype={array.dtype}")

    for key in ("normal_E", "normal_I"):

        array = data[key]

        print(f"\n{key}")
        print("Shape:", array.shape)
        print("Minimum:", np.min(array))
        print("Maximum:", np.max(array))
        print("Mean:", np.mean(array))
        print("Standard deviation:", np.std(array))
        print("Fraction zero:", np.mean(array == 0))
        print("All finite:", np.isfinite(array).all())

        print(
            "First neuron, first 10 images:",
            array[0, :10]
        )

print("\nInspection complete. No data modified.")
