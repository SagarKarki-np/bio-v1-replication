
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT / "results" / "figure7_rf_connectivity"
OUTPUT.mkdir(parents=True, exist_ok=True)

STA_PATH = (
    PROJECT / "results"
    / "STA_100000_training_test_25"
    / "STA_100000_results.npz"
)

CHECKPOINT = (
    PROJECT / "results" / "learning"
    / "training_test_25" / "batch_1100.npz"
)

# Use the original archived checkpoint if it is absent
# from the cleaned GitHub project.
if not CHECKPOINT.exists():
    CHECKPOINT = (
        PROJECT.parent / "V1_network_learning"
        / "results" / "learning"
        / "training_test_25" / "batch_1100.npz"
    )

print("STA file:", STA_PATH)
print("Checkpoint:", CHECKPOINT)

if not STA_PATH.exists():
    raise FileNotFoundError(STA_PATH)
if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)


# -------------------------------------------------
# 1. Load experimentally estimated receptive fields
# -------------------------------------------------

with np.load(STA_PATH, allow_pickle=False) as data:
    E_RF = data["STA_E"].copy()
    I_RF = data["STA_I"].copy()

with np.load(CHECKPOINT, allow_pickle=False) as data:
    W_I_E = data["W_I_E"].copy()
    W_E_I = data["W_E_I"].copy()
    W_I_I = data["W_I_I"].copy()
    W_E_E = data["W_E_E"].copy()

assert E_RF.shape == (400, 256)
assert I_RF.shape == (100, 256)

assert W_I_E.shape == (400, 100)
assert W_E_I.shape == (100, 400)
assert W_I_I.shape == (100, 100)
assert W_E_E.shape == (400, 400)


# -------------------------------------------------
# 2. Normalize RF vectors for Pearson correlation
# -------------------------------------------------

def normalize_rfs(rf):
    rf = np.asarray(rf, dtype=float)
    centered = rf - rf.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(centered, axis=1)

    valid = (
        np.all(np.isfinite(rf), axis=1)
        & np.isfinite(norms)
        & (norms > 0)
    )

    normalized = np.zeros_like(centered)
    normalized[valid] = (
        centered[valid] / norms[valid, None]
    )

    return normalized, valid


E_norm, E_valid = normalize_rfs(E_RF)
I_norm, I_valid = normalize_rfs(I_RF)

print("Valid E receptive fields:", E_valid.sum())
print("Valid I receptive fields:", I_valid.sum())


# -------------------------------------------------
# 3. Receptive-field correlation matrices
# -------------------------------------------------

C_EI = np.clip(E_norm @ I_norm.T, -1, 1)
C_EE = np.clip(E_norm @ E_norm.T, -1, 1)
C_II = np.clip(I_norm @ I_norm.T, -1, 1)

C_EI[~(E_valid[:, None] & I_valid[None, :])] = np.nan
C_EE[~(E_valid[:, None] & E_valid[None, :])] = np.nan
C_II[~(I_valid[:, None] & I_valid[None, :])] = np.nan


# -------------------------------------------------
# 4. Match correlations with synaptic connections
# -------------------------------------------------

# Every matrix follows:
# W[postsynaptic, presynaptic]

connections = {
    "I_to_E": (C_EI, W_I_E),
    "E_to_I": (C_EI.T, W_E_I),
    "I_to_I": (C_II, W_I_I),
    "E_to_E": (C_EE, W_E_E),
}


def extract_pairs(corr, weights, same_population=False):
    if same_population:
        # Keep both directed connections (i -> j, j -> i),
        # but remove self-connections.
        mask = ~np.eye(corr.shape[0], dtype=bool)
    else:
        mask = np.ones(corr.shape, dtype=bool)

    mask &= np.isfinite(corr) & np.isfinite(weights)

    return corr[mask], weights[mask]


results = {}

for name, (corr, weights) in connections.items():
    same = name in ("I_to_I", "E_to_E")

    x, y = extract_pairs(
        corr, weights, same_population=same
    )

    results[name] = (x, y)

    print("\n", name)
    print("Valid directed connections:", len(x))
    print("Mean RF correlation:", np.mean(x))
    print("Mean absolute RF correlation:", np.mean(np.abs(x)))
    print("Mean synaptic weight:", np.mean(y))

    if len(x) > 1 and np.std(y) > 0:
        print(
            "Correlation between RF similarity and weight:",
            np.corrcoef(x, y)[0, 1]
        )


# -------------------------------------------------
# 5. Figure 7A and 7B: single excitatory neuron
# -------------------------------------------------

EXAMPLE_E = 0

example_correlations = C_EI[EXAMPLE_E]
example_weights = W_I_E[EXAMPLE_E]

bins = np.arange(-1, 1.021, 0.02)

fig, axes = plt.subplots(
    2, 3, figsize=(17, 10),
    constrained_layout=True
)

ax = axes[0, 0]
ax.hist(
    example_correlations[np.isfinite(example_correlations)],
    bins=bins, color="steelblue"
)
ax.set_title(f"A. E neuron {EXAMPLE_E}: correlations with I RFs")
ax.set_xlabel("RF correlation")
ax.set_ylabel("Number of inhibitory neurons")
ax.set_xlim(-1, 1)

ax = axes[0, 1]
ax.scatter(
    example_correlations,
    example_weights,
    s=20, alpha=0.7
)
ax.set_title(f"B. Inhibitory input to E neuron {EXAMPLE_E}")
ax.set_xlabel("RF correlation")
ax.set_ylabel("I -> E weight")
ax.set_xlim(-1, 1)


# -------------------------------------------------
# 6. Figure 7C-F: population connectivity
# -------------------------------------------------

panel_specs = [
    (axes[0, 2], "I_to_E", "C. I -> E"),
    (axes[1, 0], "E_to_I", "D. E -> I"),
    (axes[1, 1], "I_to_I", "E. I -> I"),
    (axes[1, 2], "E_to_E", "F. E -> E"),
]

for ax, name, title in panel_specs:
    x, y = results[name]

    ax.scatter(x, y, s=2, alpha=0.08)

    ax.set_title(title)
    ax.set_xlabel("Pairwise RF correlation")
    ax.set_ylabel("Synaptic weight")
    ax.set_xlim(-1, 1)

    # Correlation distribution inset
    inset = ax.inset_axes([0.62, 0.62, 0.34, 0.32])
    inset.hist(x, bins=bins, color="grey")
    inset.set_xlim(-1, 1)
    inset.tick_params(labelsize=7)

figure_path = OUTPUT / "figure7_sta_connectivity.png"

fig.savefig(figure_path, dpi=250)
plt.close(fig)


# -------------------------------------------------
# 7. Save results for later audit
# -------------------------------------------------

save_data = {
    "E_RF": E_RF,
    "I_RF": I_RF,
    "C_EI": C_EI,
    "C_EE": C_EE,
    "C_II": C_II,
    "example_E": EXAMPLE_E,
    "example_correlations": example_correlations,
    "example_weights": example_weights,
    "sta_source": str(STA_PATH),
    "checkpoint_source": str(CHECKPOINT),
}

for name, (x, y) in results.items():
    save_data[f"{name}_correlations"] = x
    save_data[f"{name}_weights"] = y

np.savez_compressed(
    OUTPUT / "figure7_statistics.npz",
    **save_data
)

print("\nFIGURE 7 ANALYSIS COMPLETED")
print("Figure:", figure_path)
print("Results:", OUTPUT / "figure7_statistics.npz")
