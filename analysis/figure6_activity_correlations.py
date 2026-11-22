
"""
Figure 6: Pairwise firing-rate correlations.

Reuses the 100-presentation firing-rate matrices
from our Figure 3 audit.

No simulation or training.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from response_correlations import calculate_pairwise_correlations

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "results"
    / "figure3_seed3001_audit"
    / "figure3_statistics.npz"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "figure6_correlations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Figure 3 results
with np.load(INPUT_FILE, allow_pickle=False) as data:

    ON = data["ON_rates"].copy()
    OFF = data["OFF_rates"].copy()
    E = data["E_rates"].copy()
    I = data["I_rates"].copy()

    duration = float(data["presentation_duration"])
    source = str(data["input_path"])

LGN = np.concatenate([ON, OFF], axis=0)

populations = {
    "LGN": LGN,
    "E": E,
    "I": I
}

assert LGN.shape == (512, 100)
assert E.shape == (400, 100)
assert I.shape == (100, 100)

print("FIGURE 6: ACTIVITY CORRELATIONS")
print("=" * 60)
print("Source file:", INPUT_FILE)
print("Original input path:", source)
print("Presentation duration:", duration)
print("Number of images:", E.shape[1])

# Calculate correlations
results = {}

for name, rates in populations.items():

    C, pairs = calculate_pairwise_correlations(rates)

    valid = pairs[np.isfinite(pairs)]
    silent = np.sum(np.std(rates, axis=1) == 0)

    results[name] = {
        "matrix": C,
        "pairs": pairs,
        "valid": valid
    }

    print(f"\n{name} POPULATION")
    print("-" * 40)
    print("Neurons:", rates.shape[0])
    print("Total pairs:", len(pairs))
    print("Valid pairs:", len(valid))
    print("Constant-response neurons:", silent)

    if len(valid):
        print("Mean correlation:", np.mean(valid))
        print("Median correlation:", np.median(valid))
        print("Mean absolute correlation:", np.mean(np.abs(valid)))
        print("Fraction positive:", np.mean(valid > 0))
        print("Standard deviation:", np.std(valid))

# Plot Figure 6
fig, axes = plt.subplots(
    1, 3,
    figsize=(14, 4.5),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

bins = np.linspace(-1, 1, 51)

for ax, (name, result) in zip(axes, results.items()):

    values = result["valid"]

    if len(values):
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.8,
            edgecolor="none"
        )

        mean = np.mean(values)
        ax.axvline(
            mean,
            linestyle="--",
            color="black",
            label=f"Mean = {mean:.3f}"
        )

        ax.legend(fontsize=9)

    ax.set_title(f"{name} population")
    ax.set_xlabel("Pearson correlation")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Probability density")

fig.suptitle(
    "Figure 6: Pairwise Firing-Rate Correlations",
    fontsize=14
)

figure_path = OUTPUT_DIR / "figure6_correlations.png"
fig.savefig(figure_path, dpi=300)
plt.close(fig)

np.savez_compressed(
    OUTPUT_DIR / "figure6_statistics.npz",
    LGN_correlations=results["LGN"]["pairs"],
    E_correlations=results["E"]["pairs"],
    I_correlations=results["I"]["pairs"],
    C_LGN=results["LGN"]["matrix"],
    C_E=results["E"]["matrix"],
    C_I=results["I"]["matrix"],
    presentation_duration=duration,
    source_file=str(INPUT_FILE),
    original_input_path=source
)

print("\nFigure 6 analysis completed.")
print("Saved:", figure_path)
