
"""
Figure 10: Post-training baseline audit.

Compare the newly trained MATLAB-demo E02/I04 network
with the archived original trained network.

Read-only analysis: no checkpoints are modified.
"""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_v1 import L1_LIMITS


# ============================================================
# 1. CHECKPOINT PATHS
# ============================================================

ORIGINAL = (
    PROJECT_ROOT.parent
    / "V1_network_learning"
    / "results"
    / "learning"
    / "training_test_25"
    / "batch_1100.npz"
)

NEW = (
    PROJECT_ROOT
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
    / "E02_I04"
    / "training"
    / "checkpoints"
    / "batch_1100.npz"
)

WEIGHT_KEYS = {
    "LGN_E": "W_LGN_E",
    "LGN_I": "W_LGN_I",
    "E_E": "W_E_E",
    "E_I": "W_E_I",
    "I_I": "W_I_I",
    "I_E": "W_I_E"
}

EXPECTED_SHAPES = {
    "LGN_E": (400, 512),
    "LGN_I": (100, 512),
    "E_E": (400, 400),
    "E_I": (100, 400),
    "I_I": (100, 100),
    "I_E": (400, 100)
}


# ============================================================
# 2. LOAD CHECKPOINTS
# ============================================================

def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        print("\nCheckpoint:", path.name)
        print("Available keys:", data.files)

        weights = {
            name: data[key].copy()
            for name, key in WEIGHT_KEYS.items()
        }

        thresholds = {}

        for name in ("threshold_E", "threshold_I"):
            if name in data.files:
                thresholds[name] = data[name].copy()

        history = {}

        for name in ("history_E_Hz", "history_I_Hz"):
            if name in data.files:
                history[name] = data[name].copy()

        completed = (
            int(data["completed_batches"])
            if "completed_batches" in data.files
            else None
        )

    return weights, thresholds, history, completed


print("\nFIGURE 10 BASELINE AUDIT")
print("=" * 70)

old_W, old_theta, old_history, old_completed = (
    load_checkpoint(ORIGINAL)
)

new_W, new_theta, new_history, new_completed = (
    load_checkpoint(NEW)
)

print("\nOriginal completed batches:", old_completed)
print("New completed batches:", new_completed)

assert new_completed == 1100


# ============================================================
# 3. VALIDATE NEW WEIGHTS
# ============================================================

print("\nNEW CHECKPOINT: WEIGHT VALIDATION")
print("=" * 70)

all_valid = True

for name, W in new_W.items():

    expected_shape = EXPECTED_SHAPES[name]

    shape_ok = W.shape == expected_shape
    finite_ok = np.isfinite(W).all()
    nonnegative_ok = np.all(W >= 0)

    row_sums = W.sum(axis=1)

    l1_ok = np.all(
        row_sums <= L1_LIMITS[name] + 1e-8
    )

    self_connections_ok = True

    if name in ("E_E", "I_I"):
        self_connections_ok = np.all(
            np.diag(W) == 0
        )

    valid = (
        shape_ok
        and finite_ok
        and nonnegative_ok
        and l1_ok
        and self_connections_ok
    )

    all_valid &= valid

    print(f"\n{name}")
    print("  Shape:", W.shape)
    print("  Shape valid:", shape_ok)
    print("  Finite:", finite_ok)
    print("  Nonnegative:", nonnegative_ok)
    print("  L1 valid:", l1_ok)
    print("  Self-connections valid:", self_connections_ok)
    print("  Mean weight:", W.mean())
    print("  Mean row sum:", row_sums.mean())
    print("  Min row sum:", row_sums.min())
    print("  Max row sum:", row_sums.max())


# ============================================================
# 4. COMPARE ORIGINAL AND NEW WEIGHTS
# ============================================================

print("\nORIGINAL VS NEW TRAINED WEIGHTS")
print("=" * 70)

for name in WEIGHT_KEYS:

    A = old_W[name]
    B = new_W[name]

    difference = B - A

    max_difference = np.max(np.abs(difference))
    mean_absolute_difference = np.mean(
        np.abs(difference)
    )

    relative_frobenius = (
        np.linalg.norm(difference)
        / max(np.linalg.norm(A), 1e-12)
    )

    exact_match = np.array_equal(A, B)

    print(f"\n{name}")
    print("  Exact match:", exact_match)
    print("  Maximum absolute difference:", max_difference)
    print("  Mean absolute difference:", mean_absolute_difference)
    print("  Relative Frobenius difference:", relative_frobenius)

    if A.shape == B.shape:

        correlation = np.corrcoef(
            A.ravel(), B.ravel()
        )[0, 1]

        print("  Weight correlation:", correlation)


# ============================================================
# 5. COMPARE FIRING THRESHOLDS
# ============================================================

print("\nFIRING THRESHOLD COMPARISON")
print("=" * 70)

for name in ("threshold_E", "threshold_I"):

    if name not in old_theta or name not in new_theta:
        print(name, ": unavailable in one checkpoint")
        continue

    A = old_theta[name]
    B = new_theta[name]

    print(f"\n{name}")
    print("  Original mean:", A.mean())
    print("  New mean:", B.mean())
    print("  Mean absolute difference:",
          np.mean(np.abs(B - A)))
    print("  Maximum absolute difference:",
          np.max(np.abs(B - A)))


# ============================================================
# 6. NEW TRAINING FIRING RATES
# ============================================================

print("\nNEW TRAINING: FINAL FIRING RATES")
print("=" * 70)

if (
    "history_E_Hz" in new_history
    and "history_I_Hz" in new_history
):

    E = new_history["history_E_Hz"]
    I = new_history["history_I_Hz"]

    print("History length:", len(E))
    print("Last 10 mean E rate:", E[-10:].mean())
    print("Last 10 mean I rate:", I[-10:].mean())

    print("Last 100 mean E rate:", E[-100:].mean())
    print("Last 100 mean I rate:", I[-100:].mean())

    print("Last 100 E rate SD:", E[-100:].std())
    print("Last 100 I rate SD:", I[-100:].std())


# ============================================================
# 7. VERDICT
# ============================================================

print("\n" + "=" * 70)

if all_valid:
    print("NEW CHECKPOINT: STRUCTURAL VALIDATION PASSED")
else:
    print("NEW CHECKPOINT: VALIDATION FAILED")
    raise AssertionError(
        "One or more weight-validation checks failed."
    )

print("Baseline comparison completed.")
