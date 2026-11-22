
"""
Figure 5: Gabor refitting at 160 x 160.

Uses existing 100,000-stimulus Track A STAs.

Compares:
1. Original Gabor fit at 16 x 16
2. Original Gabor evaluated at 160 x 160
3. Reoptimized Gabor at 160 x 160
4. Reoptimized Gabor evaluated at 16 x 16

No V1 simulations or additional STA stimuli.

Runs are resumable, saving after each neuron.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from analysis import gabor_fitting as gf


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
)

STA_PATH = (
    BASE / "functional_sta"
    / "track_b"
    / "full_100000"
    / "STA_100000_results.npz"
)

FIT_PATH = (
    BASE / "gabor_fitting"
    / "track_b_100000"
    / "gabor_all_400_fits.npz"
)

OUTPUT = (
    BASE / "gabor_fitting"
    / "interpolation_sensitivity_track_b"
)

PROGRESS = OUTPUT / "gabor_160_refit_progress.npz"

SIDE = 16
FINE_SIDE = 160
N_E = 400
THRESHOLD = 0.10


# --------------------------------------------------
# INTERPOLATION
# --------------------------------------------------

fine_axis = np.linspace(0, SIDE - 1, FINE_SIDE)

row_grid, col_grid = np.meshgrid(
    fine_axis, fine_axis, indexing="ij"
)

sample_coordinates = np.array([
    row_grid.ravel(),
    col_grid.ravel()
])


def interpolate_160(field):
    """Endpoint-aligned bilinear interpolation."""

    return map_coordinates(
        np.asarray(field, dtype=float),
        sample_coordinates,
        order=1,
        mode="nearest"
    ).reshape(FINE_SIDE, FINE_SIDE)


X160 = interpolate_160(gf.X_GRID).ravel()
Y160 = interpolate_160(gf.Y_GRID).ravel()


# --------------------------------------------------
# GABOR FUNCTION ON ARBITRARY GRID
# --------------------------------------------------

def gabor_on_grid(params, X, Y):

    A, x0, y0, sx, sy, f, theta, phase = params

    dx = X - x0
    dy = Y - y0

    xr = (
        dx * np.cos(theta)
        + dy * np.sin(theta)
    )

    yr = (
        -dx * np.sin(theta)
        + dy * np.cos(theta)
    )

    envelope = np.exp(
        -0.5 * (
            (xr / sx)**2
            + (yr / sy)**2
        )
    )

    carrier = np.cos(
        2 * np.pi * f * xr + phase
    )

    return A * envelope * carrier


def normalized_error(actual, predicted):

    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    denominator = np.dot(actual, actual)

    if denominator <= 0:
        return np.nan

    difference = actual - predicted

    return float(
        np.dot(difference, difference)
        / denominator
    )


# --------------------------------------------------
# LOAD EXISTING DATA
# --------------------------------------------------

def load_data():

    gf.check_spatial_order()

    with np.load(STA_PATH, allow_pickle=False) as data:

        STA_E = data["STA_E"].copy()
        n_stimuli = int(data["n_stimuli"])

    with np.load(FIT_PATH, allow_pickle=False) as data:

        parameters = data["parameters"].copy()
        errors = data["fit_errors"].copy()

    if n_stimuli != 100_000:
        raise ValueError(
            f"Expected 100000 stimuli, got {n_stimuli}"
        )

    if STA_E.shape != (N_E, SIDE * SIDE):
        raise ValueError(
            f"Unexpected STA shape: {STA_E.shape}"
        )

    if parameters.shape != (N_E, 8):
        raise ValueError(
            "Unexpected parameter shape"
        )

    if errors.shape != (N_E,):
        raise ValueError(
            "Unexpected error shape"
        )

    return STA_E, parameters, errors


# --------------------------------------------------
# FIT ONE INTERPOLATED RECEPTIVE FIELD
# --------------------------------------------------

def fit_interpolated(
    rf16,
    old_params,
    neuron,
    n_starts,
    seed
):

    rf16 = np.asarray(rf16, dtype=float)

    rf160 = interpolate_160(
        gf.as_map(rf16)
    ).ravel()

    scale = float(np.max(np.abs(rf16)))

    if not np.isfinite(scale) or scale < 1e-12:
        return old_params.copy(), np.nan, np.nan

    target160 = rf160 / scale
    target16 = rf16 / scale

    rng = np.random.default_rng(seed + neuron)

    # Use the original fitting initialization method.
    guesses = gf.initial_guesses(
        target16,
        n_starts,
        rng
    )

    # Include the existing 16 x 16 solution as a start.
    old_normalized = old_params.copy()
    old_normalized[0] /= scale

    # Respect the existing parameter bounds.
    old_normalized = np.clip(
        old_normalized,
        gf.LOWER + 1e-9,
        gf.UPPER - 1e-9
    )

    guesses = [old_normalized] + list(guesses)

    denominator = float(
        np.dot(target160, target160)
    )

    # Retain the original solution as a candidate.
    best_params = old_normalized.copy()

    initial_residual = (
        gabor_on_grid(best_params, X160, Y160)
        - target160
    )

    best_error = float(
        np.dot(initial_residual, initial_residual)
        / denominator
    )

    for guess in guesses:

        try:

            fit = least_squares(
                fun=lambda p: (
                    gabor_on_grid(p, X160, Y160)
                    - target160
                ),
                x0=guess,
                bounds=(gf.LOWER, gf.UPPER),
                method="trf",
                max_nfev=250,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )

        except (ValueError, FloatingPointError):
            continue

        residual = fit.fun

        error = float(
            np.dot(residual, residual)
            / denominator
        )

        if np.isfinite(error) and error < best_error:

            best_error = error
            best_params = fit.x.copy()

    # Restore amplitude to original STA units.
    best_params[0] *= scale

    predicted16 = gf.gabor(best_params)

    error16 = normalized_error(
        rf16, predicted16
    )

    return best_params, best_error, error16


# --------------------------------------------------
# RESUMABLE PROGRESS
# --------------------------------------------------

def new_progress():

    return {
        "completed": np.zeros(N_E, dtype=bool),

        "parameters_160": np.full(
            (N_E, 8), np.nan
        ),

        "errors_160": np.full(N_E, np.nan),

        "errors_16": np.full(N_E, np.nan),
    }


def save_progress(state, n_starts, seed):

    temp = PROGRESS.with_name(
        "gabor_160_refit_progress.tmp.npz"
    )

    np.savez_compressed(
        temp,
        **state,
        n_starts=n_starts,
        seed=seed,
        interpolation="endpoint_aligned_bilinear"
    )

    os.replace(temp, PROGRESS)


def load_progress(n_starts, seed):

    if not PROGRESS.exists():
        return new_progress()

    with np.load(PROGRESS, allow_pickle=False) as data:

        stored_starts = int(data["n_starts"])
        stored_seed = int(data["seed"])

        if stored_starts != n_starts or stored_seed != seed:
            raise ValueError(
                "Existing progress uses different fitting "
                "settings. Keep --starts and --seed unchanged."
            )

        state = {
            key: data[key].copy()
            for key in (
                "completed",
                "parameters_160",
                "errors_160",
                "errors_16"
            )
        }

    return state


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

def run(limit, n_starts, seed):

    OUTPUT.mkdir(parents=True, exist_ok=True)

    STA_E, old_params, original_errors = load_data()

    state = load_progress(n_starts, seed)

    old_errors_160 = np.full(N_E, np.nan)

    # Evaluate all original fits on the new grid.
    for neuron in range(N_E):

        rf160 = interpolate_160(
            gf.as_map(STA_E[neuron])
        ).ravel()

        old_errors_160[neuron] = normalized_error(
            rf160,
            gabor_on_grid(
                old_params[neuron],
                X160,
                Y160
            )
        )

    print("\n" + "=" * 65)
    print("FIGURE 5 — 160 x 160 GABOR REOPTIMIZATION")
    print("=" * 65)

    print("STA stimuli:", 100_000)
    print("Neurons requested:", limit)
    print("New random starts:", n_starts)
    print("Plus original fitted parameters: 1")
    print("Already completed:", state["completed"][:limit].sum())

    for neuron in range(limit):

        if state["completed"][neuron]:
            continue

        params, error160, error16 = fit_interpolated(
            STA_E[neuron],
            old_params[neuron],
            neuron,
            n_starts,
            seed
        )

        state["parameters_160"][neuron] = params
        state["errors_160"][neuron] = error160
        state["errors_16"][neuron] = error16
        state["completed"][neuron] = True

        save_progress(
            state, n_starts, seed
        )

        print(
            f"E-{neuron + 1:03d} "
            f"[{neuron + 1}/{limit}] "
            f"old160={old_errors_160[neuron]:.5f} "
            f"refit160={error160:.5f} "
            f"refit16={error16:.5f}",
            flush=True
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    indices = np.arange(limit)

    original = original_errors[indices]

    evaluated = old_errors_160[indices]

    refitted160 = state["errors_160"][indices]

    refitted16 = state["errors_16"][indices]

    print("\n" + "=" * 65)
    print("PAIRED GABOR COMPARISON")
    print("=" * 65)

    for name, errors in (
        ("Original fit, 16 x 16", original),
        ("Original fit, evaluated at 160 x 160", evaluated),
        ("Reoptimized fit, 160 x 160", refitted160),
        ("Reoptimized fit, evaluated at 16 x 16", refitted16)
    ):

        passed = (
            np.isfinite(errors)
            & (errors < THRESHOLD)
        )

        print(f"\n{name}")
        print("Passed:", int(passed.sum()), "/", limit)
        print("Success percentage:", passed.mean() * 100)
        print("Median error:", np.nanmedian(errors))

    print(
        "\nImprovement from reoptimization "
        "on 160 x 160 grid:",
        int(np.sum(
            (evaluated >= THRESHOLD)
            & (refitted160 < THRESHOLD)
        ))
    )

    print(
        "New fits passing on the original grid:",
        int(np.sum(
            (original >= THRESHOLD)
            & (refitted16 < THRESHOLD)
        ))
    )

    summary_path = (
        OUTPUT / f"gabor_160_comparison_{limit}.npz"
    )

    np.savez_compressed(
        summary_path,

        indices=indices,
        original_errors_16=original,
        original_errors_160=evaluated,
        refitted_errors_160=refitted160,
        refitted_errors_16=refitted16,

        parameters_160=(
            state["parameters_160"][indices]
        ),

        passes_original_16=original < THRESHOLD,
        passes_original_160=evaluated < THRESHOLD,
        passes_refitted_160=refitted160 < THRESHOLD,
        passes_refitted_16=refitted16 < THRESHOLD,

        n_stimuli=100_000,
        threshold=THRESHOLD,
        n_starts=n_starts,
        seed=seed
    )

    print("\nSaved:", summary_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=20
    )

    parser.add_argument(
        "--starts",
        type=int,
        default=12
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2705
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= N_E:
        parser.error("--limit must be between 1 and 400")

    if not 1 <= args.starts <= 100:
        parser.error("--starts must be between 1 and 100")

    run(
        args.limit,
        args.starts,
        args.seed
    )


if __name__ == "__main__":
    main()
