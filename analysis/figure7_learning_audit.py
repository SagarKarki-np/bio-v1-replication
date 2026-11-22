
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent

ORIGINAL = PROJECT.parent / "V1_network_learning"

INITIAL = (
    ORIGINAL / "results" / "untrained"
    / "batch_001" / "weights.npz"
)

LEARNED = (
    ORIGINAL / "results" / "learning"
    / "training_test_25" / "batch_1100.npz"
)

OUTPUT = PROJECT / "results" / "figure7_learning_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)


# --------------------------------------------------
# 1. Load checkpoint weights
# --------------------------------------------------

def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        return {
            name: data[name].copy()
            for name in WEIGHT_NAMES
        }


initial = load_checkpoint(INITIAL)
learned = load_checkpoint(LEARNED)

print("Initial checkpoint:", INITIAL)
print("Learned checkpoint:", LEARNED)


# --------------------------------------------------
# 2. Reconstruct weight-derived receptive fields
# --------------------------------------------------

def reconstruct_rf_vectors(W_LGN):
    """
    ON minus OFF weights.

    Input:  (neurons, 512)
    Output: (neurons, 256)
    """
    assert W_LGN.shape[1] == 512

    ON = W_LGN[:, :256]
    OFF = W_LGN[:, 256:]

    return ON - OFF


def normalize_rf(RF):
    RF = RF.astype(float)

    centered = RF - RF.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)

    valid = (
        np.all(np.isfinite(RF), axis=1)
        & np.isfinite(norms)
        & (norms > 0)
    )

    X = np.zeros_like(centered)
    X[valid] = centered[valid] / norms[valid, None]

    return X, valid


def rf_correlation_matrices(weights):
    E_RF = reconstruct_rf_vectors(weights["W_LGN_E"])
    I_RF = reconstruct_rf_vectors(weights["W_LGN_I"])

    E, valid_E = normalize_rf(E_RF)
    I, valid_I = normalize_rf(I_RF)

    C_EI = np.clip(E @ I.T, -1, 1)
    C_EE = np.clip(E @ E.T, -1, 1)
    C_II = np.clip(I @ I.T, -1, 1)

    C_EI[~(valid_E[:, None] & valid_I[None, :])] = np.nan
    C_EE[~(valid_E[:, None] & valid_E[None, :])] = np.nan
    C_II[~(valid_I[:, None] & valid_I[None, :])] = np.nan

    return C_EI, C_EE, C_II


# --------------------------------------------------
# 3. Match RF similarity to recurrent weights
# --------------------------------------------------

def get_connection_pairs(weights):
    C_EI, C_EE, C_II = rf_correlation_matrices(weights)

    connections = {
        "I_to_E": (C_EI, weights["W_I_E"]),
        "E_to_I": (C_EI.T, weights["W_E_I"]),
        "I_to_I": (C_II, weights["W_I_I"]),
        "E_to_E": (C_EE, weights["W_E_E"])
    }

    results = {}

    for name, (C, W) in connections.items():
        assert C.shape == W.shape

        mask = np.isfinite(C) & np.isfinite(W)

        if name in ("I_to_I", "E_to_E"):
            np.fill_diagonal(mask, False)

        x = C[mask]
        y = W[mask]

        results[name] = (x, y)

    return results


initial_pairs = get_connection_pairs(initial)
learned_pairs = get_connection_pairs(learned)


# --------------------------------------------------
# 4. Quantify RF-weight relationships
# --------------------------------------------------

def relationship(x, y):
    if (
        x.size < 2
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


print("\nFIGURE 7: LEARNING-EMERGENCE AUDIT")
print("=" * 65)

summary = {}

for name in initial_pairs:
    xi, yi = initial_pairs[name]
    xl, yl = learned_pairs[name]

    r_initial = relationship(xi, yi)
    r_learned = relationship(xl, yl)

    summary[name] = (r_initial, r_learned)

    print(f"\n{name}")
    print("-" * 45)
    print("Number of directed pairs:", len(xi))
    print("Initial RF-weight r:", r_initial)
    print("Learned RF-weight r:", r_learned)
    print("Change in r:", r_learned - r_initial)
    print("Initial mean |RF r|:", np.mean(np.abs(xi)))
    print("Learned mean |RF r|:", np.mean(np.abs(xl)))
    print("Initial mean weight:", np.mean(yi))
    print("Learned mean weight:", np.mean(yl))


# --------------------------------------------------
# 5. Plot binned mean connection strengths
# --------------------------------------------------

bins = np.linspace(-1, 1, 31)
centers = 0.5 * (bins[:-1] + bins[1:])


def binned_mean(x, y):
    means = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)

    for i in range(len(centers)):
        if i == len(centers) - 1:
            mask = (x >= bins[i]) & (x <= bins[i+1])
        else:
            mask = (x >= bins[i]) & (x < bins[i+1])

        counts[i] = mask.sum()

        if counts[i] >= 10:
            means[i] = np.mean(y[mask])

    return means, counts


fig, axes = plt.subplots(
    2, 2, figsize=(12, 9),
    constrained_layout=True
)

for ax, name in zip(axes.flat, initial_pairs):
    xi, yi = initial_pairs[name]
    xl, yl = learned_pairs[name]

    mi, ni = binned_mean(xi, yi)
    ml, nl = binned_mean(xl, yl)

    ax.plot(
        centers, mi,
        "o-", label="Initial", markersize=3
    )
    ax.plot(
        centers, ml,
        "o-", label="Learned", markersize=3
    )

    ri, rl = summary[name]

    ax.set_title(
        f"{name.replace('_', ' ')}"
        f"\nInitial r={ri:.3f}, Learned r={rl:.3f}"
    )
    ax.set_xlabel("RF Pearson correlation")
    ax.set_ylabel("Mean synaptic weight")
    ax.set_xlim(-1, 1)
    ax.legend()
    ax.grid(alpha=0.2)

fig.suptitle(
    "Figure 7 audit: RF similarity and connectivity before/after learning",
    fontsize=14
)

fig.savefig(
    OUTPUT / "figure7_learning_comparison.png",
    dpi=250
)
plt.close(fig)


# --------------------------------------------------
# 6. Save the complete numerical comparison
# --------------------------------------------------

save_data = {
    "initial_checkpoint": str(INITIAL),
    "learned_checkpoint": str(LEARNED)
}

for name in initial_pairs:
    xi, yi = initial_pairs[name]
    xl, yl = learned_pairs[name]

    ri, rl = summary[name]

    save_data[f"{name}_initial_RF_corr"] = xi
    save_data[f"{name}_initial_weights"] = yi

    save_data[f"{name}_learned_RF_corr"] = xl
    save_data[f"{name}_learned_weights"] = yl

    save_data[f"{name}_initial_r"] = ri
    save_data[f"{name}_learned_r"] = rl

np.savez_compressed(
    OUTPUT / "figure7_learning_audit.npz",
    **save_data
)

print("\nAudit completed.")
print("Results:", OUTPUT)
