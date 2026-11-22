
"""
Figure 10: Targeted Gabor optimization sensitivity audit.

Refit 12 failed excitatory receptive fields from each condition:
E02/I04 and E05/I10.

Uses the original Figure 5 audit methodology:
36 initial guesses, max_nfev=1000, same parameter bounds.

No training or STA simulation.
Original results are never modified.
"""

from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

from gabor_fitting import (
    gabor,
    initial_guesses,
    check_spatial_order,
    LOWER,
    UPPER,
)

ROOT = Path(__file__).resolve().parent.parent

RESULT_ROOT = (
    ROOT / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
)

CONDITIONS = ("E02_I04", "E05_I10")
N_STARTS = 36
MAX_NFEV = 1000
BASE_SEED = 2705
THRESHOLD = 0.10

check_spatial_order()


def load_condition(condition):
    folder = RESULT_ROOT / condition

    sta_path = (
        folder / "sta_100k" / "STA_100000_results.npz"
    )

    fit_path = (
        folder / "gabor_100k" / "gabor_all_400_fits.npz"
    )

    with np.load(sta_path, allow_pickle=False) as d:
        sta = d["STA_E"].copy()
        n_stimuli = int(d["n_stimuli"])

    with np.load(fit_path, allow_pickle=False) as d:
        errors = d["fit_errors"].copy()
        params = d["parameters"].copy()
        indices = d["neuron_indices_zero_based"].copy()
        n_starts = int(d["n_starts"])
        seed = int(d["seed"])

    assert sta.shape == (400, 256)
    assert params.shape == (400, 8)
    assert errors.shape == (400,)
    assert np.array_equal(indices, np.arange(400))
    assert np.isfinite(errors).all()
    assert n_stimuli == 100000
    assert n_starts == 12
    assert seed == BASE_SEED

    return sta, errors, params


def select_failures(errors):
    """Select 4 near, 4 middle, and 4 worst failures."""

    failed = np.flatnonzero(
        np.isfinite(errors) & (errors >= THRESHOLD)
    )

    ordered = failed[np.argsort(errors[failed])]

    if len(ordered) < 12:
        raise ValueError("Fewer than 12 failed neurons.")

    near = ordered[:4]

    midpoint = len(ordered) // 2
    middle = ordered[midpoint - 2:midpoint + 2]

    worst = ordered[-4:]

    selected = np.concatenate([near, middle, worst])

    assert len(np.unique(selected)) == 12

    return selected


def refit_gabor(rf, neuron):
    """Extended fitting using original Figure 5 settings."""

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
                gtol=1e-6,
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


def audit_condition(condition):
    sta, original_errors, original_params = load_condition(
        condition
    )

    selected = select_failures(original_errors)

    output_dir = (
        RESULT_ROOT / condition / "gabor_optimization_audit"
    )
    output_path = output_dir / "optimization_audit.npz"

    if output_path.exists():
        raise FileExistsError(
            f"Preserving existing audit: {output_path}"
        )

    new_errors = np.full(12, np.nan)
    new_params = np.full((12, 8), np.nan)

    print(f"\n{'=' * 60}")
    print("CONDITION:", condition)
    print("Total original failures:",
          np.sum(original_errors >= THRESHOLD))
    print("Selected neurons:", selected + 1)

    for j, neuron in enumerate(selected):
        params, error = refit_gabor(
            sta[neuron],
            neuron
        )

        new_errors[j] = error
        new_params[j] = params

        print(
            f"E-{neuron + 1:03d} "
            f"[{j + 1}/12] | "
            f"Old={original_errors[neuron]:.5f} | "
            f"New={error:.5f} | "
            f"Rescued={bool(np.isfinite(error) and error < THRESHOLD)}",
            flush=True
        )

    old = original_errors[selected]

    # Compare the extended search with the archived original fit.
    # The extended run is not guaranteed to contain the same
    # initial candidates as the original 12-start optimization.
    combined_best = np.where(
        np.isfinite(new_errors),
        np.minimum(old, new_errors),
        old
    )

    rescued = combined_best < THRESHOLD

    print(f"\n{condition} SUMMARY")
    print("-" * 50)
    print("Neurons audited:", len(selected))
    print("Median original error:", np.median(old))
    print("Median extended error:", np.nanmedian(new_errors))
    print("Median best-available error:", np.median(combined_best))
    print("Neurons rescued:", int(rescued.sum()))
    print("Still failing:", int((~rescued).sum()))

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        selected_neurons=selected,
        original_errors=old,
        extended_errors=new_errors,
        best_available_errors=combined_best,
        rescued=rescued,
        original_params=original_params[selected],
        extended_params=new_params,
        n_starts=N_STARTS,
        max_nfev=MAX_NFEV,
        threshold=THRESHOLD,
    )

    print("Saved:", output_path)

    return int(rescued.sum())


def main():
    # Check both outputs before starting either audit.
    for condition in CONDITIONS:
        output = (
            RESULT_ROOT / condition
            / "gabor_optimization_audit"
            / "optimization_audit.npz"
        )

        if output.exists():
            raise FileExistsError(
                f"Audit already exists: {output}"
            )

    results = {}

    for condition in CONDITIONS:
        results[condition] = audit_condition(condition)

    print("\n" + "=" * 60)
    print("FIGURE 10 OPTIMIZATION AUDIT COMPARISON")
    print("=" * 60)

    for condition, rescued in results.items():
        print(f"{condition}: {rescued}/12 failed fits rescued")

    print("\nOriginal Gabor results remain unchanged.")


if __name__ == "__main__":
    main()
