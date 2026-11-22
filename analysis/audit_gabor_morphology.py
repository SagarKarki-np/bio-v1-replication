
"""
Figure 5: Morphology audit of unsuccessful Gabor fits.

Examines:
1. STA versus fitted Gabor
2. Spatial residuals
3. Boundary energy
4. Gabor parameter bounds

No simulation, training, or optimization.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from gabor_fitting import (
    gabor,
    as_map,
    LOWER,
    UPPER,
    PARAMETER_NAMES,
    check_spatial_order,
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

AUDIT_FILE = (
    PROJECT_DIR / "results"
    / "figure5_gabor_failure_audit"
    / "optimization_audit.npz"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results"
    / "figure5_gabor_morphology_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

check_spatial_order()

# --------------------------------------------------
# 2. Load saved arrays
# --------------------------------------------------

with np.load(STA_FILE, allow_pickle=False) as data:
    STA_E = data["STA_E"].copy()

with np.load(AUDIT_FILE, allow_pickle=False) as data:
    neurons = data["selected_neurons"].copy()
    parameters = data["new_params"].copy()
    errors = data["new_errors"].copy()

assert STA_E.shape == (400, 256)
assert parameters.shape == (len(neurons), 8)
assert errors.shape == (len(neurons),)

# --------------------------------------------------
# 3. Boundary-energy measurement
# --------------------------------------------------

def boundary_energy_fraction(rf, width=2):
    """
    Fraction of total squared RF amplitude
    located within 'width' pixels of the edges.
    """

    energy = rf ** 2

    mask = np.zeros((16, 16), dtype=bool)

    mask[:width, :] = True
    mask[-width:, :] = True
    mask[:, :width] = True
    mask[:, -width:] = True

    total = np.sum(energy)

    if total <= 0:
        return np.nan

    return np.sum(energy[mask]) / total


# --------------------------------------------------
# 4. Check fitting parameter bounds
# --------------------------------------------------

def check_parameter_bounds(params):
    """
    Identify parameters within 1% of their
    permitted optimization range.
    """

    span = UPPER - LOWER
    tolerance = 0.01 * span

    near_lower = np.abs(params - LOWER) <= tolerance
    near_upper = np.abs(params - UPPER) <= tolerance

    near = near_lower | near_upper

    return [
        name
        for name, active in zip(PARAMETER_NAMES, near)
        if active
    ]


# --------------------------------------------------
# 5. Process all 12 neurons
# --------------------------------------------------

boundary_fractions = []
bound_flags = []
residual_errors = []

records = []

print("\nGABOR MORPHOLOGY AUDIT")
print("=" * 65)

for j, neuron in enumerate(neurons):

    sta = as_map(STA_E[neuron])

    params = parameters[j]

    if not np.isfinite(params).all():
        raise ValueError(
            f"Non-finite Gabor parameters for neuron {neuron}"
        )

    fitted = as_map(gabor(params))

    residual = sta - fitted

    boundary = boundary_energy_fraction(sta)

    bounds = check_parameter_bounds(params)

    calculated_error = (
        np.sum(residual ** 2)
        / np.sum(sta ** 2)
    )

    if not np.isclose(
        calculated_error,
        errors[j],
        rtol=1e-4,
        atol=1e-6
    ):
        raise ValueError(
            f"Error mismatch for neuron {neuron}: "
            f"{calculated_error} vs {errors[j]}"
        )

    boundary_fractions.append(boundary)
    bound_flags.append(len(bounds) > 0)
    residual_errors.append(calculated_error)

    records.append({
        "neuron": int(neuron),
        "sta": sta,
        "fitted": fitted,
        "residual": residual,
        "error": calculated_error,
        "boundary": boundary,
        "bounds": bounds
    })

    print(
        f"E-{neuron + 1:03d} | "
        f"Error={calculated_error:.4f} | "
        f"Boundary energy={boundary:.2%} | "
        f"Near bounds={bounds}"
    )


# --------------------------------------------------
# 6. Create compact figures
# --------------------------------------------------

# Four figures, each showing three neurons.
# Columns: STA, fitted Gabor, residual, residual energy

for start in range(0, len(records), 3):

    subset = records[start:start + 3]

    fig, axes = plt.subplots(
        len(subset),
        4,
        figsize=(11, 2.8 * len(subset)),
        squeeze=False,
        constrained_layout=True
    )

    for row, record in enumerate(subset):

        sta = record["sta"]
        fitted = record["fitted"]
        residual = record["residual"]

        scale = max(np.max(np.abs(sta)), 1e-12)

        images = [
            sta,
            fitted,
            residual,
            residual ** 2
        ]

        titles = [
            f'E-{record["neuron"] + 1:03d}: STA',
            f'Gabor (error={record["error"]:.3f})',
            "Residual (STA - Gabor)",
            "Squared residual"
        ]

        for col, image in enumerate(images):

            if col < 3:
                axes[row, col].imshow(
                    image,
                    cmap="RdBu_r",
                    vmin=-scale,
                    vmax=scale,
                    interpolation="nearest"
                )
            else:
                axes[row, col].imshow(
                    image,
                    cmap="inferno",
                    vmin=0,
                    interpolation="nearest"
                )

            axes[row, col].set_title(
                titles[col],
                fontsize=9
            )

            axes[row, col].set_aspect("equal")
            axes[row, col].axis("off")

    figure_number = start // 3 + 1

    path = OUTPUT_DIR / (
        f"morphology_part_{figure_number}.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", path)


# --------------------------------------------------
# 7. Save numerical diagnostics
# --------------------------------------------------

boundary_fractions = np.array(boundary_fractions)
bound_flags = np.array(bound_flags)
residual_errors = np.array(residual_errors)

np.savez_compressed(
    OUTPUT_DIR / "morphology_statistics.npz",
    neuron_indices=neurons,
    errors=residual_errors,
    boundary_energy=boundary_fractions,
    parameter_near_bounds=bound_flags,
    parameters=parameters
)

print("\nMORPHOLOGY SUMMARY")
print("=" * 65)

print(
    "Median boundary energy:",
    np.median(boundary_fractions)
)

print(
    "Neurons near parameter bounds:",
    int(np.sum(bound_flags)),
    "/",
    len(neurons)
)

print(
    "Neurons with >50% boundary energy:",
    int(np.sum(boundary_fractions > 0.5))
)

print(
    "Median Gabor error:",
    np.median(residual_errors)
)

print("\nMorphology audit completed.")
