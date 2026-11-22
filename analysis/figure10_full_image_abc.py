
"""
Final Figure 10A-C analysis.

Combines:
- Previously saved natural-image evaluations
- Functional RFs from 100,000-stimulus STA runs

Does not retrain or simulate any network.
Does not modify original results.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]

ROOT = (
    PROJECT
    / "results"
    / "paper_protocol"
    / "seed_3002"
)

SWEEP = (
    ROOT
    / "figure10_firing_rate_sweep"
    / "paper"
)

SUMMARY = SWEEP / "summary"

OUTPUT = SUMMARY / "full_image_abc"

CONDITIONS = (
    "E01_I02",
    "E02_I04",
    "E05_I10",
    "E10_I20",
    "E20_I40",
)


def sta_path(condition):
    if condition == "E02_I04":
        return (
            ROOT
            / "functional_sta"
            / "track_a"
            / "full_image_100000"
            / "STA_100000_results.npz"
        )

    return (
        SWEEP
        / condition
        / "functional_sta"
        / "full_image_100000"
        / "STA_100000_results.npz"
    )


def pairwise_rms_pearson(rf):
    rf = np.asarray(rf, dtype=np.float64)

    if rf.ndim != 2 or not np.isfinite(rf).all():
        raise ValueError("Invalid receptive fields")

    centered = rf - rf.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)

    if np.any(norms == 0):
        raise ValueError("Zero-variance receptive field")

    normalized = centered / norms[:, None]
    correlations = normalized @ normalized.T

    upper = correlations[
        np.triu_indices(rf.shape[0], k=1)
    ]

    return float(np.sqrt(np.mean(upper ** 2)))


def load_data():
    df = pd.read_csv(
        SUMMARY / "figure10_sweep_summary.csv"
    )

    if (
        len(df) != 5
        or df["condition"].duplicated().any()
        or set(df["condition"]) != set(CONDITIONS)
    ):
        raise ValueError(
            "Unexpected Figure 10 summary conditions"
        )

    df = df.set_index("condition").loc[
        list(CONDITIONS)
    ].reset_index()

    numerical_columns = [
        "target_e", "target_i",
        "observed_e", "observed_i",
        "e_temporal", "e_population",
        "i_temporal", "i_population",
    ]

    if not np.isfinite(
        df[numerical_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Nonfinite summary values")

    rms_e = []
    rms_i = []
    reliability = []

    for condition in CONDITIONS:
        path = sta_path(condition)

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            if int(data["n_stimuli"]) != 100_000:
                raise ValueError(
                    f"Incomplete STA: {condition}"
                )

            E = data["STA_E"]
            I = data["STA_I"]

            if E.shape != (400, 256):
                raise ValueError(
                    f"Incorrect E STA shape: {condition}"
                )

            if I.shape != (100, 256):
                raise ValueError(
                    f"Incorrect I STA shape: {condition}"
                )

            rms_e.append(pairwise_rms_pearson(E))
            rms_i.append(pairwise_rms_pearson(I))

            reliability.append(
                float(np.nanmedian(
                    data["split_half_reliability"]
                ))
            )

    df["functional_rms_e"] = rms_e
    df["functional_rms_i"] = rms_i
    df["e_split_half_reliability"] = reliability

    return df


def prepare_plot():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xlabel("Target excitatory firing rate (Hz)")
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig, ax


def plot_rf_correlations(df, output):
    fig, ax = prepare_plot()

    x = df["target_e"].to_numpy()

    ax.plot(
        x,
        df["functional_rms_e"],
        "o-",
        label="Excitatory RFs",
        linewidth=2
    )

    ax.plot(
        x,
        df["functional_rms_i"],
        "s-",
        label="Inhibitory RFs",
        linewidth=2
    )

    ax.set_ylabel("Pairwise RF correlation (RMS)")
    ax.set_title(
        "Figure 10A - Functional receptive-field correlations"
    )
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(df, output):
    fig, ax = prepare_plot()

    x = df["target_e"].to_numpy()

    ax.plot(
        x, df["e_temporal"],
        "o-", label="Excitatory", linewidth=2
    )

    ax.plot(
        x, df["i_temporal"],
        "s-", label="Inhibitory", linewidth=2
    )

    ax.set_ylabel("Temporal sparseness")
    ax.set_title("Figure 10B - Temporal sparseness")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_population(df, output):
    fig, ax = prepare_plot()

    x = df["target_e"].to_numpy()

    ax.plot(
        x, df["e_population"],
        "o-", label="Excitatory", linewidth=2
    )

    ax.plot(
        x, df["i_population"],
        "s-", label="Inhibitory", linewidth=2
    )

    ax.set_ylabel("Population sparseness")
    ax.set_title("Figure 10C - Population sparseness")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data()

    # Compare full-image with original per-patch STA.
    previous_rms = {
        "E01_I02": 0.257690104,
        "E02_I04": 0.208919802,
        "E05_I10": 0.184045181,
        "E10_I20": 0.239617670,
        "E20_I40": 0.429794543,
    }

    df["previous_per_patch_rms_e"] = (
        df["condition"].map(previous_rms)
    )

    df["rms_e_difference"] = (
        df["functional_rms_e"]
        - df["previous_per_patch_rms_e"]
    )

    if not np.isfinite(
        df["functional_rms_e"].to_numpy()
    ).all():
        raise ValueError("Nonfinite full-image RMS values")

    if not np.isfinite(
        df["rms_e_difference"].to_numpy()
    ).all():
        raise ValueError("Nonfinite RMS differences")

    files = {
        "csv": OUTPUT / "figure10_abc_final.csv",
        "comparison": OUTPUT / "rf_preprocessing_comparison.csv",
        "rf": OUTPUT / "figure10A_functional_rf.png",
        "temporal": OUTPUT / "figure10B_temporal.png",
        "population": OUTPUT / "figure10C_population.png",
    }

    if any(path.exists() for path in files.values()):
        raise FileExistsError(
            "One or more final outputs already exist. "
            "Refusing to overwrite."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    df.to_csv(files["csv"], index=False)

    df[[
        "condition",
        "previous_per_patch_rms_e",
        "functional_rms_e",
        "rms_e_difference",
    ]].to_csv(
        files["comparison"],
        index=False
    )

    plot_rf_correlations(df, files["rf"])
    plot_temporal(df, files["temporal"])
    plot_population(df, files["population"])

    print("=" * 70)
    print("FINAL FIGURE 10A-C RESULTS")
    print("=" * 70)

    columns = [
        "condition",
        "observed_e", "observed_i",
        "functional_rms_e", "functional_rms_i",
        "e_temporal", "e_population",
        "i_temporal", "i_population",
        "e_split_half_reliability",
    ]

    print(df[columns].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    ))

    print("\nSaved files:")
    for path in files.values():
        print(path)

    print("\nNo model training or simulation performed.")


if __name__ == "__main__":
    main()
