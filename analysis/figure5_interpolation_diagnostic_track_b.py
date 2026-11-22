
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from analysis import gabor_fitting as gf


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = (
    Path("results")
    / "paper_protocol"
    / "seed_3002"
)

STA_PATH = (
    BASE
    / "functional_sta"
    / "track_b"
    / "full_100000"
    / "STA_100000_results.npz"
)

FIT_PATH = (
    BASE
    / "gabor_fitting"
    / "track_b_100000"
    / "gabor_all_400_fits.npz"
)

OUTPUT = (
    BASE
    / "gabor_fitting"
    / "interpolation_sensitivity_track_b"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD EXISTING RESULTS
# --------------------------------------------------

with np.load(STA_PATH) as data:
    STA_E = data["STA_E"].copy()
    n_stimuli = int(data["n_stimuli"])

with np.load(FIT_PATH) as data:
    parameters = data["parameters"].copy()
    original_errors = data["fit_errors"].copy()

assert STA_E.shape == (400, 256)
assert parameters.shape == (400, 8)
assert n_stimuli == 100000

gf.check_spatial_order()


# --------------------------------------------------
# BILINEAR INTERPOLATION
# --------------------------------------------------

def interpolate_160(field):

    coordinates = np.arange(16)

    interpolator = RegularGridInterpolator(
        (coordinates, coordinates),
        field,
        method="linear"
    )

    fine_coordinates = np.linspace(
        0, 15, 160
    )

    row, col = np.meshgrid(
        fine_coordinates,
        fine_coordinates,
        indexing="ij"
    )

    query_points = np.stack(
        (row, col),
        axis=-1
    )

    return interpolator(query_points)


# --------------------------------------------------
# GABOR ON ANY SPATIAL GRID
# --------------------------------------------------

def gabor_on_grid(params, X, Y):

    A, x0, y0, sx, sy, f, theta, phase = params

    dx = X - x0
    dy = Y - y0

    xr = dx * np.cos(theta) + dy * np.sin(theta)
    yr = -dx * np.sin(theta) + dy * np.cos(theta)

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

    return (
        np.sum((actual - predicted)**2)
        / np.sum(actual**2)
    )


# --------------------------------------------------
# ORIGINAL VERSUS INTERPOLATED ERRORS
# --------------------------------------------------

errors_160 = np.full(400, np.nan)
errors_16_check = np.full(400, np.nan)

X160 = interpolate_160(gf.X_GRID)
Y160 = interpolate_160(gf.Y_GRID)

for neuron in range(400):

    rf16 = STA_E[neuron]

    rf_map = gf.as_map(rf16)

    rf160 = interpolate_160(rf_map)

    params = parameters[neuron]

    if not np.isfinite(params).all():
        continue

    # Original fitted Gabor: 16 x 16
    predicted16 = gf.gabor(params)

    errors_16_check[neuron] = normalized_error(
        rf16,
        predicted16
    )

    # Same fitted parameters: 160 x 160
    predicted160 = gabor_on_grid(
        params,
        X160,
        Y160
    )

    errors_160[neuron] = normalized_error(
        rf160,
        predicted160
    )


# --------------------------------------------------
# VALIDATE THE ORIGINAL CALCULATION
# --------------------------------------------------

np.testing.assert_allclose(
    errors_16_check,
    original_errors,
    rtol=1e-5,
    atol=1e-6
)


# --------------------------------------------------
# COMPARE SUCCESS RATES
# --------------------------------------------------

threshold = 0.10

passed16 = original_errors < threshold
passed160 = errors_160 < threshold

print("\n" + "=" * 60)
print("FIGURE 5 — INTERPOLATION DIAGNOSTIC")
print("=" * 60)

print("STA presentations:", n_stimuli)

print("\nORIGINAL 16 x 16")
print("Successful:", passed16.sum(), "/ 400")
print("Percentage:", passed16.mean() * 100)
print("Median error:", np.nanmedian(original_errors))

print("\nINTERPOLATED 160 x 160")
print("Successful:", passed160.sum(), "/ 400")
print("Percentage:", passed160.mean() * 100)
print("Median error:", np.nanmedian(errors_160))

print("\nCHANGES")
print(
    "Failed -> passed:",
    np.sum(~passed16 & passed160)
)

print(
    "Passed -> failed:",
    np.sum(passed16 & ~passed160)
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

np.savez_compressed(
    OUTPUT / "interpolation_diagnostic.npz",
    original_errors=original_errors,
    interpolated_errors=errors_160,
    passed16=passed16,
    passed160=passed160,
    n_stimuli=n_stimuli
)

print("\nSaved to:", OUTPUT)
