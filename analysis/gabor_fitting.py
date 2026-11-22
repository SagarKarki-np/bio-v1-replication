
"""
Figure 5: Two-dimensional Gabor fitting of functional V1 STAs.

Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Fits Gabor functions to existing excitatory STA receptive fields.

IMPORTANT:
    All 16x16 maps and flattened coordinates use MATLAB-compatible
    Fortran (column-major) ordering.

No network simulation or training occurs.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_DIR / "results" / "STA"
    / "STA_2000_results.npz"
)

DEFAULT_OUTPUT = (
    PROJECT_DIR / "results" / "figure5_gabor_corrected"
)

SIDE = 16
N_E = 400
N_PIXELS = SIDE * SIDE

Y_GRID, X_GRID = np.mgrid[0:SIDE, 0:SIDE]

# Crucial: match the column-major order of the saved STAs.
X = X_GRID.flatten(order="F")
Y = Y_GRID.flatten(order="F")

PARAMETER_NAMES = (
    "amplitude", "x0", "y0", "sigma_x", "sigma_y",
    "frequency", "theta", "phase"
)

LOWER = np.array([
    -3.0, -2.0, -2.0, 0.5, 0.5,
    0.04, -np.pi / 2, -np.pi
])

UPPER = np.array([
    3.0, 17.0, 17.0, 16.0, 16.0,
    0.48, np.pi / 2, np.pi
])


# ============================================================
# 2. GABOR MODEL
# ============================================================

def gabor(params):
    """
    Return a Gabor receptive field flattened in Fortran order.

    Parameters:
        amplitude, x0, y0, sigma_x, sigma_y,
        frequency, theta, phase
    """

    A, x0, y0, sx, sy, f, theta, phase = params

    dx = X - x0
    dy = Y - y0

    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)

    envelope = np.exp(
        -0.5 * ((xr / sx)**2 + (yr / sy)**2)
    )

    carrier = np.cos(
        2 * np.pi * f * xr + phase
    )

    return A * envelope * carrier


def as_map(vector):
    """Convert a flattened receptive field to its spatial map."""
    return np.asarray(vector).reshape(
        SIDE, SIDE, order="F"
    )


def check_spatial_order():
    """
    Validate consistency between coordinate flattening
    and spatial reconstruction before fitting.
    """

    test = np.arange(N_PIXELS).reshape(
        SIDE, SIDE, order="F"
    )

    restored = as_map(test.flatten(order="F"))

    assert np.array_equal(test, restored)
    assert np.array_equal(as_map(X), X_GRID)
    assert np.array_equal(as_map(Y), Y_GRID)

    print("Spatial ordering check: PASSED")


# ============================================================
# 3. INITIAL GABOR ESTIMATES
# ============================================================

def initial_guesses(rf, n_starts, rng):
    """
    Estimate RF location from spatial energy and generate
    multiple orientation, phase and frequency starting points.
    """

    # Correct: STA vectors are stored in Fortran order.
    rf_map = as_map(rf)

    energy = gaussian_filter(
        rf_map ** 2, sigma=0.75
    )

    mass = energy.sum()

    if mass <= 0:
        return []

    cx = float((energy * X_GRID).sum() / mass)
    cy = float((energy * Y_GRID).sum() / mass)

    dx = X_GRID - cx
    dy = Y_GRID - cy

    cov_xx = float((energy * dx * dx).sum() / mass)
    cov_yy = float((energy * dy * dy).sum() / mass)
    cov_xy = float((energy * dx * dy).sum() / mass)

    covariance = np.array([
        [cov_xx, cov_xy],
        [cov_xy, cov_yy]
    ])

    _, eigenvectors = np.linalg.eigh(covariance)

    major_angle = np.arctan2(
        eigenvectors[1, -1],
        eigenvectors[0, -1]
    )

    angles = [
        major_angle,
        major_angle + np.pi / 2,
        0.0,
        np.pi / 4,
        -np.pi / 4
    ]

    frequencies = [
        0.12, 0.18, 0.25, 0.33, 0.10
    ]

    phases = [
        0.0, np.pi / 2, 0.0,
        -np.pi / 2, np.pi
    ]

    sx = np.clip(
        np.sqrt(max(cov_xx, 0.8)),
        1.0, 7.0
    )

    sy = np.clip(
        np.sqrt(max(cov_yy, 0.8)),
        1.0, 7.0
    )

    peak = float(np.max(np.abs(rf)))

    guesses = []

    for j in range(n_starts):

        if j < len(angles):
            theta = angles[j]
            f = frequencies[j]
            phase = phases[j]
            x0, y0 = cx, cy

        else:
            theta = rng.uniform(
                -np.pi / 2, np.pi / 2
            )

            f = rng.uniform(0.075, 0.37)
            phase = rng.uniform(-np.pi, np.pi)

            x0 = np.clip(
                cx + rng.normal(0, 1.4),
                -1, 16
            )

            y0 = np.clip(
                cy + rng.normal(0, 1.4),
                -1, 16
            )

        theta = (
            (theta + np.pi / 2) % np.pi
        ) - np.pi / 2

        guess = np.array([
            peak, x0, y0,
            sx, sy, f, theta, phase
        ])

        guesses.append(
            np.clip(
                guess,
                LOWER + 1e-6,
                UPPER - 1e-6
            )
        )

    return guesses


# ============================================================
# 4. FIT ONE RECEPTIVE FIELD
# ============================================================

def fit_one_rf(rf, n_starts=12, seed=2705):
    """
    Fit one Gabor using bounded multistart least squares.

    Normalized error:
        sum((STA - Gabor)^2) / sum(STA^2)

    Successful fit:
        error < 0.10
    """

    rf = np.asarray(rf, dtype=float)

    if rf.shape != (N_PIXELS,):
        return np.full(8, np.nan), np.nan, False

    if not np.isfinite(rf).all():
        return np.full(8, np.nan), np.nan, False

    scale = float(np.max(np.abs(rf)))

    if scale < 1e-12:
        return np.full(8, np.nan), np.nan, False

    target = rf / scale
    denominator = float(np.dot(target, target))

    rng = np.random.default_rng(seed)

    guesses = initial_guesses(
        target, n_starts, rng
    )

    best_params = None
    best_error = np.inf

    for guess in guesses:

        try:
            fitted = least_squares(
                fun=lambda p: gabor(p) - target,
                x0=guess,
                bounds=(LOWER, UPPER),
                method="trf",
                max_nfev=250,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )
        except (ValueError, FloatingPointError):
            continue

        squared_error = float(
            np.dot(fitted.fun, fitted.fun)
        )

        error = squared_error / denominator

        if error < best_error:
            best_error = error
            best_params = fitted.x.copy()

    if best_params is None:
        return np.full(8, np.nan), np.nan, False

    # Restore amplitude to original STA units.
    best_params[0] *= scale

    return (
        best_params,
        best_error,
        bool(best_error < 0.10)
    )


# ============================================================
# 5. STA VERSUS GABOR VISUALIZATION
# ============================================================

def plot_examples(STA, params, errors, indices, path):

    valid = np.flatnonzero(np.isfinite(errors))

    if valid.size == 0:
        print("No valid examples to plot.")
        return

    selected = sorted(
        valid,
        key=lambda j: errors[j]
    )[:8]

    fig, axes = plt.subplots(
        len(selected), 2,
        figsize=(6, 2.5 * len(selected)),
        squeeze=False
    )

    for row, j in enumerate(selected):

        original = as_map(STA[j])

        # Crucial: Gabor output must use the same
        # Fortran spatial reconstruction.
        fitted = as_map(gabor(params[j]))

        limit = max(
            float(np.max(np.abs(original))),
            1e-12
        )

        axes[row, 0].imshow(
            original,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear"
        )

        axes[row, 1].imshow(
            fitted,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear"
        )

        axes[row, 0].set_title(
            f"E-{indices[j] + 1:03d}: STA"
        )

        axes[row, 1].set_title(
            f"Gabor, error={errors[j]:.3f}"
        )

        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. GABOR PARAMETER DISTRIBUTION
# ============================================================

def plot_distribution(nx, ny, passed, path):

    fig, ax = plt.subplots(figsize=(7, 6))

    valid = np.isfinite(nx) & np.isfinite(ny)

    good = valid & passed
    bad = valid & ~passed

    ax.scatter(
        nx[good], ny[good],
        s=22,
        label="Error < 0.10"
    )

    if bad.any():
        ax.scatter(
            nx[bad], ny[bad],
            s=18,
            alpha=0.5,
            label="Error >= 0.10"
        )

    ax.set_xlabel("n_x = sigma_x × frequency")
    ax.set_ylabel("n_y = sigma_y × frequency")
    ax.set_title("Figure 5: Gabor parameter distribution")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5
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
        parser.error("--limit must be between 1 and 400.")

    if not 1 <= args.starts <= 100:
        parser.error("--starts must be between 1 and 100.")

    if not args.input.is_file():
        parser.error(f"STA file not found: {args.input}")

    check_spatial_order()

    with np.load(args.input, allow_pickle=False) as data:
        STA_E = data["STA_E"].copy()
        n_stimuli = int(data["n_stimuli"])

    if STA_E.shape != (N_E, N_PIXELS):
        parser.error(
            f"Unexpected STA shape: {STA_E.shape}"
        )

    print("\nFIGURE 5: CORRECTED GABOR FITTING")
    print("=" * 50)
    print("White-noise presentations:", n_stimuli)
    print("Neurons:", args.limit)
    print("Starting guesses:", args.starts)

    suffix = (
        f"pilot_{args.limit}"
        if args.limit < N_E
        else "all_400"
    )

    output_dir = args.output_dir

    results_path = (
        output_dir / f"gabor_{suffix}_fits.npz"
    )

    examples_path = (
        output_dir / f"gabor_{suffix}_examples.png"
    )

    distribution_path = (
        output_dir / f"gabor_{suffix}_nx_ny.png"
    )

    for path in (
        results_path,
        examples_path,
        distribution_path
    ):
        if path.exists():
            parser.error(
                f"Output already exists: {path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    indices = np.arange(args.limit)

    params = np.full(
        (args.limit, 8),
        np.nan
    )

    errors = np.full(
        args.limit,
        np.nan
    )

    passed = np.zeros(
        args.limit,
        dtype=bool
    )

    for j, neuron in enumerate(indices):

        params[j], errors[j], passed[j] = fit_one_rf(
            STA_E[neuron],
            n_starts=args.starts,
            seed=args.seed + int(neuron)
        )

        print(
            f"E-{neuron + 1:03d} "
            f"[{j + 1}/{args.limit}] "
            f"error={errors[j]:.5f} "
            f"passed={passed[j]}",
            flush=True
        )

    nx = params[:, 3] * params[:, 5]
    ny = params[:, 4] * params[:, 5]

    np.savez_compressed(
        results_path,
        neuron_indices_zero_based=indices,
        parameters=params,
        parameter_names=np.array(PARAMETER_NAMES),
        fit_errors=errors,
        passes_10_percent=passed,
        n_x=nx,
        n_y=ny,
        n_stimuli=n_stimuli,
        n_starts=args.starts,
        seed=args.seed,
        source_sta=str(args.input),
        spatial_order="F",
        error_definition="sum((STA-Gabor)^2)/sum(STA^2)"
    )

    plot_examples(
        STA_E[indices],
        params,
        errors,
        indices,
        examples_path
    )

    plot_distribution(
        nx, ny, passed,
        distribution_path
    )

    print("\nGABOR FITTING SUMMARY")
    print("=" * 50)
    print("Neurons evaluated:", args.limit)
    print("Successful fits:", int(passed.sum()))
    print(
        "Fraction successful:",
        f"{passed.mean() * 100:.2f}%"
    )
    print(
        "Median fitting error:",
        np.nanmedian(errors)
    )

    print("\nPublished reference:")
    print("369/400 E neurons with error < 0.10")

    print("\nSaved:")
    print(results_path)
    print(examples_path)
    print(distribution_path)


if __name__ == "__main__":
    main()
