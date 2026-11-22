
"""
Figure 5: Compare boundary energy in learned
weight-derived RFs and functional STAs.

Uses existing 100,000-stimulus results.
No simulation or optimization.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
    / "figure5_weight_sta_boundary_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load arrays
# --------------------------------------------------

with np.load(STA_FILE, allow_pickle=False) as data:
    STA_E = data["STA_E"].copy()
    weight_RF_E = data["weight_RF_E"].copy()

with np.load(FIT_FILE, allow_pickle=False) as data:
    errors = data["fit_errors"].copy()

assert STA_E.shape == (400, 256)
assert weight_RF_E.shape == (400, 256)
assert errors.shape == (400,)


# --------------------------------------------------
# Boundary-energy calculation
# --------------------------------------------------

mask = np.zeros((16, 16), dtype=bool)

mask[:2, :] = True
mask[-2:, :] = True
mask[:, :2] = True
mask[:, -2:] = True


def boundary_energy(rf_vectors):

    fractions = []

    for vector in rf_vectors:

        rf = vector.reshape(16, 16, order="F")

        energy = rf ** 2
        total = np.sum(energy)

        if total > 0:
            fraction = np.sum(energy[mask]) / total
        else:
            fraction = np.nan

        fractions.append(fraction)

    return np.array(fractions)


sta_boundary = boundary_energy(STA_E)
weight_boundary = boundary_energy(weight_RF_E)

successful = np.isfinite(errors) & (errors < 0.10)
failed = np.isfinite(errors) & (errors >= 0.10)

valid = (
    np.isfinite(sta_boundary)
    & np.isfinite(weight_boundary)
)

correlation = np.corrcoef(
    sta_boundary[valid],
    weight_boundary[valid]
)[0, 1]


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\nWEIGHT VERSUS STA BOUNDARY AUDIT")
print("=" * 60)

print("STA-weight boundary correlation:", correlation)

print(
    "Median STA boundary energy:",
    np.nanmedian(sta_boundary)
)

print(
    "Median weight boundary energy:",
    np.nanmedian(weight_boundary)
)

print("\nSUCCESSFUL NEURONS")

print(
    "Median STA boundary energy:",
    np.nanmedian(sta_boundary[successful])
)

print(
    "Median weight boundary energy:",
    np.nanmedian(weight_boundary[successful])
)

print("\nFAILED NEURONS")

print(
    "Median STA boundary energy:",
    np.nanmedian(sta_boundary[failed])
)

print(
    "Median weight boundary energy:",
    np.nanmedian(weight_boundary[failed])
)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].scatter(
    weight_boundary[successful],
    sta_boundary[successful],
    alpha=0.45,
    s=18,
    label="Successful"
)

axes[0].scatter(
    weight_boundary[failed],
    sta_boundary[failed],
    alpha=0.65,
    s=18,
    label="Failed"
)

axes[0].plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    alpha=0.5
)

axes[0].set_xlabel("Weight-derived RF boundary energy")
axes[0].set_ylabel("STA boundary energy")
axes[0].set_title("Weight versus STA boundary localization")
axes[0].legend()

axes[1].boxplot(
    [
        weight_boundary[successful],
        weight_boundary[failed],
        sta_boundary[successful],
        sta_boundary[failed]
    ],
    tick_labels=[
        "Weight\nPass",
        "Weight\nFail",
        "STA\nPass",
        "STA\nFail"
    ]
)

axes[1].set_ylabel("Boundary energy fraction")
axes[1].set_title("Boundary localization by RF type")

plt.tight_layout()

output = OUTPUT_DIR / "weight_sta_boundary_comparison.png"

plt.savefig(output, dpi=300)
plt.close(fig)

np.savez_compressed(
    OUTPUT_DIR / "weight_sta_boundary_statistics.npz",
    sta_boundary=sta_boundary,
    weight_boundary=weight_boundary,
    successful=successful,
    failed=failed,
    correlation=correlation
)

print("\nSaved:", output)
