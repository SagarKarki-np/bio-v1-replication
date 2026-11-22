
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gabor_fitting import as_map

PROJECT_DIR = Path(__file__).resolve().parent.parent

STA_FILE = (
    PROJECT_DIR / "results"
    / "STA_100000_training_test_25"
    / "STA_100000_results.npz"
)

FIT_FILE = (
    PROJECT_DIR / "results"
    / "figure5_gabor_100000"
    / "gabor_all_400_fits.npz"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results"
    / "figure5_gabor_boundary_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with np.load(STA_FILE) as data:
    sta = data["STA_E"].copy()

with np.load(FIT_FILE) as data:
    errors = data["fit_errors"].copy()

assert sta.shape == (400, 256)
assert errors.shape == (400,)

mask = np.zeros((16, 16), dtype=bool)

mask[:2, :] = True
mask[-2:, :] = True
mask[:, :2] = True
mask[:, -2:] = True

boundary_energy = np.zeros(400)

for neuron in range(400):

    rf = as_map(sta[neuron])

    energy = rf ** 2
    total = energy.sum()

    boundary_energy[neuron] = (
        energy[mask].sum() / total
        if total > 0 else np.nan
    )

successful = np.isfinite(errors) & (errors < 0.10)
failed = np.isfinite(errors) & (errors >= 0.10)

print("POPULATION BOUNDARY AUDIT")
print("=" * 50)

print("Successful neurons:", successful.sum())
print("Failed neurons:", failed.sum())

print(
    "Median boundary energy (successful):",
    np.nanmedian(boundary_energy[successful])
)

print(
    "Median boundary energy (failed):",
    np.nanmedian(boundary_energy[failed])
)

valid = np.isfinite(boundary_energy) & np.isfinite(errors)

print(
    "Correlation: boundary energy vs fitting error:",
    np.corrcoef(
        boundary_energy[valid],
        errors[valid]
    )[0, 1]
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].boxplot(
    [
        boundary_energy[successful & np.isfinite(boundary_energy)],
        boundary_energy[failed & np.isfinite(boundary_energy)]
    ],
    tick_labels=["Successful", "Failed"]
)

axes[0].set_ylabel("Boundary energy fraction")
axes[0].set_title("Boundary energy by fitting outcome")

axes[1].scatter(
    boundary_energy[valid],
    errors[valid],
    alpha=0.5,
    s=15
)

axes[1].axhline(
    0.10,
    color="red",
    linestyle="--",
    label="Gabor error threshold"
)

axes[1].set_xlabel("Boundary energy fraction")
axes[1].set_ylabel("Gabor fitting error")
axes[1].set_title("Boundary energy vs fitting error")
axes[1].legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "boundary_population_comparison.png",
    dpi=300
)

plt.show()

np.savez_compressed(
    OUTPUT_DIR / "boundary_population_statistics.npz",
    boundary_energy=boundary_energy,
    fitting_errors=errors,
    successful=successful,
    failed=failed
)
