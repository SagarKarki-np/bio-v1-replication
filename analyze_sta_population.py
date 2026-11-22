
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to the existing NPZ file
results_file = Path(
    r"projects\biologically-grounded-v1-network-replication\results\STA_100000_training_test_25\STA_100000_results.npz"
)

# Load results
data = np.load(results_file)

# Folder where results will be saved
results_dir = results_file.parent
print("Current directory:", Path.cwd())
print("File exists:", results_file.exists())
STA_E = data["STA_E"]
STA_I = data["STA_I"]

weight_RF_E = data["weight_RF_E"]
weight_RF_I = data["weight_RF_I"]

corr_E = data["RF_correlations_E"]
corr_I = data["RF_correlations_I"]

reliability = data["split_half_reliability"]

# Select representative neurons across correlation levels
valid = np.flatnonzero(np.isfinite(corr_E))
ordered = valid[np.argsort(corr_E[valid])]

quantiles = np.linspace(0, 1, 10)
positions = np.round(
    quantiles * (len(ordered) - 1)
).astype(int)

selected = ordered[positions]

print("Selected neurons:", selected)
print("Their correlations:", corr_E[selected])


# Plot 10 neurons in a compact 5 x 4 layout
fig, axes = plt.subplots(
    5, 4,
    figsize=(11, 13),
    constrained_layout=True
)

def normalize_rf(rf):
    rf = rf.reshape(16, 16)
    rf = rf - np.mean(rf)

    scale = np.max(np.abs(rf))

    if scale > 0:
        rf = rf / scale

    return rf


for index, neuron in enumerate(selected):

    row = index // 2
    col = (index % 2) * 2

    weight_rf = normalize_rf(weight_RF_E[neuron])
    sta_rf = normalize_rf(STA_E[neuron])

    axes[row, col].imshow(
        weight_rf,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    axes[row, col + 1].imshow(
        sta_rf,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )

    axes[row, col].set_title(
        f"Neuron {neuron}: Weights",
        fontsize=10
    )

    axes[row, col + 1].set_title(
        f"STA (r={corr_E[neuron]:.3f})",
        fontsize=10
    )

    for ax in axes[row, col:col + 2]:
        ax.axis("off")
        ax.set_aspect("equal")


fig.suptitle(
    "Population Receptive Fields: Weights vs STA",
    fontsize=15
)

output = results_dir / "STA_population_comparison_normalized.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", output)


print("Saved:", output)

# Population statistics
print("\nPOPULATION STATISTICS")
print("Median E correlation:", np.nanmedian(corr_E))
print("Median I correlation:", np.nanmedian(corr_I))
print("Minimum E correlation:", np.nanmin(corr_E))
print("Maximum E correlation:", np.nanmax(corr_E))
print("E correlation 10th percentile:", np.nanpercentile(corr_E, 10))
print("E correlation 90th percentile:", np.nanpercentile(corr_E, 90))
print("Median split-half reliability:", np.nanmedian(reliability))
