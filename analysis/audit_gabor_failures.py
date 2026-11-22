
"""
Figure 5: Gabor optimization sensitivity audit.

Refit 12 previously unsuccessful excitatory neurons
using additional starting guesses and optimization steps.

No network simulation, training, or STA estimation.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from gabor_fitting import (
    gabor,
    as_map,
    initial_guesses,
    check_spatial_order,
    LOWER,
    UPPER,
)

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

STA_FILE = (
    PROJECT_DIR / "results"
    / "STA_100000_training_test_25"
    / "STA_100000_results.npz"
)

ORIGINAL_FILE = (
    PROJECT_DIR / "results"
    / "figure5_gabor_100000"
    / "gabor_all_400_fits.npz"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results"
    / "figure5_gabor_failure_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "optimization_audit.npz"
FIGURE_FILE = OUTPUT_DIR / "optimization_audit.png"

# Prevent accidental overwriting
if RESULT_FILE.exists() or FIGURE_FILE.exists():
    raise FileExistsError(
        "Audit outputs already exist. Preserve the "
        "previous results before repeating the audit."
    )

# --------------------------------------------------
# 2. Audit parameters
# --------------------------------------------------

N_STARTS = 36
MAX_NFEV = 1000
BASE_SEED = 2705
ERROR_THRESHOLD = 0.10

check_spatial_order()

# --------------------------------------------------
# 3. Load existing results
# --------------------------------------------------

with np.load(STA_FILE, allow_pickle=False) as data:
    STA_E = data["STA_E"].copy()
    n_stimuli = int(data["n_stimuli"])

with np.load(ORIGINAL_FILE, allow_pickle=False) as data:
    original_errors = data["fit_errors"].copy()
    original_params = data["parameters"].copy()
    original_indices = data["neuron_indices_zero_based"].copy()

assert STA_E.shape == (400, 256)
assert n_stimuli == 100000
assert original_errors.shape == (400,)
assert original_params.shape == (400, 8)
assert np.array_equal(original_indices, np.arange(400))

# --------------------------------------------------
# 4. Select 12 failed neurons
# --------------------------------------------------

failed = np.flatnonzero(
    np.isfinite(original_errors)
    & (original_errors >= ERROR_THRESHOLD)
)

ordered = failed[np.argsort(original_errors[failed])]

if len(ordered) < 12:
    raise ValueError("Fewer than 12 failed neurons found.")

near = ordered[:4]

middle_start = len(ordered) // 2 - 2
middle = ordered[middle_start:middle_start + 4]

worst = ordered[-4:]

selected = np.concatenate([near, middle, worst])

print("\nGABOR FAILURE AUDIT")
print("=" * 55)

print("Total original failures:", len(failed))
print("Selected neurons (zero-based):", selected)
print("Original errors:", original_errors[selected])
print("Starting guesses:", N_STARTS)
print("Max evaluations per start:", MAX_NFEV)

# --------------------------------------------------
# 5. Extended fitting function
# --------------------------------------------------

def refit_gabor(rf, neuron):

    rf = np.asarray(rf, dtype=float)

    scale = np.max(np.abs(rf))

    if not np.isfinite(scale) or scale < 1e-12:
        return np.full(8, np.nan), np.nan

    target = rf / scale
    denominator = np.dot(target, target)

    rng = np.random.default_rng(BASE_SEED + int(neuron))

    guesses = initial_guesses(
        target,
        n_starts=N_STARTS,
        rng=rng
    )

    best_error = np.inf
    best_params = np.full(8, np.nan)

    for guess in guesses:

        try:
            result = least_squares(
                fun=lambda p: gabor(p) - target,
                x0=guess,
                bounds=(LOWER, UPPER),
                method="trf",
                max_nfev=MAX_NFEV,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )

        except (ValueError, FloatingPointError):
            continue

        error = (
            np.dot(result.fun, result.fun)
            / denominator
        )

        if np.isfinite(error) and error < best_error:
            best_error = float(error)
            best_params = result.x.copy()

    if not np.isfinite(best_error):
        return np.full(8, np.nan), np.nan

    best_params[0] *= scale

    return best_params, best_error

# --------------------------------------------------
# 6. Refit selected failures
# --------------------------------------------------

new_errors = np.full(len(selected), np.nan)
new_params = np.full((len(selected), 8), np.nan)

for j, neuron in enumerate(selected):

    params, error = refit_gabor(
        STA_E[neuron],
        neuron
    )

    new_errors[j] = error
    new_params[j] = params

    old_error = original_errors[neuron]

    print(
        f"E-{neuron + 1:03d} "
        f"[{j + 1}/{len(selected)}] | "
        f"Old={old_error:.5f} | "
        f"New={error:.5f} | "
        f"Passed={bool(error < ERROR_THRESHOLD)}",
        flush=True
    )

# --------------------------------------------------
# 7. Compare outcomes
# --------------------------------------------------

old = original_errors[selected]
new = new_errors

improvement = old - new

rescued = (
    (old >= ERROR_THRESHOLD)
    & (new < ERROR_THRESHOLD)
)

print("\nAUDIT SUMMARY")
print("=" * 55)

print("Neurons audited:", len(selected))
print("Median original error:", np.median(old))
print("Median new error:", np.nanmedian(new))
print("Median improvement:", np.nanmedian(improvement))
print("Neurons rescued:", int(np.sum(rescued)))
print("Neurons still failing:", int(np.sum(new >= 0.10)))
print("Non-finite new errors:", int(np.sum(~np.isfinite(new))))

# --------------------------------------------------
# 8. Save numerical results
# --------------------------------------------------

np.savez_compressed(
    RESULT_FILE,
    selected_neurons=selected,
    original_errors=old,
    new_errors=new,
    improvement=improvement,
    rescued=rescued,
    original_params=original_params[selected],
    new_params=new_params,
    n_stimuli=n_stimuli,
    n_starts=N_STARTS,
    max_nfev=MAX_NFEV,
    threshold=ERROR_THRESHOLD
)

# --------------------------------------------------
# 9. Visualize each comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    len(selected), 3,
    figsize=(9, 3 * len(selected)),
    constrained_layout=True
)

for row, neuron in enumerate(selected):

    original = as_map(STA_E[neuron])

    old_fit = as_map(
        gabor(original_params[neuron])
    )

    if np.all(np.isfinite(new_params[row])):
        new_fit = as_map(
            gabor(new_params[row])
        )
    else:
        new_fit = np.full((16, 16), np.nan)

    scale = max(np.max(np.abs(original)), 1e-12)

    images = [original, old_fit, new_fit]

    titles = [
        f"E-{neuron + 1:03d}: STA",
        f"Original error={old[row]:.3f}",
        f"Refit error={new[row]:.3f}"
    ]

    for col in range(3):

        axes[row, col].imshow(
            images[col],
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            interpolation="nearest"
        )

        axes[row, col].set_title(
            titles[col],
            fontsize=10
        )

        axes[row, col].axis("off")

fig.savefig(
    FIGURE_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved:")
print(RESULT_FILE)
print(FIGURE_FILE)
