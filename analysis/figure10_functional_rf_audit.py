
"""
Figure 10A: Read-only functional RF correlation audit.

Uses existing Track A 100,000-stimulus STA.
No new simulation or training.
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results" / "paper_protocol"
    / "seed_3002"
)

STA_PATH = (
    BASE / "functional_sta" / "track_a"
    / "full_100000" / "STA_100000_results.npz"
)

EVAL_PATH = (
    BASE / "figure10_firing_rate_sweep"
    / "paper" / "E02_I04"
    / "evaluation" / "paper_1000_source"
    / "response_evaluation.npz"
)


def rms_pairwise_pearson(rf):
    """
    Compute RMS Pearson correlation over unique
    unordered neuron pairs, excluding self-pairs.
    """

    rf = np.asarray(rf, dtype=np.float64)

    if rf.ndim != 2:
        raise ValueError("RFs must be 2D")

    if rf.shape != (400, 256):
        raise ValueError(
            f"Unexpected E RF shape: {rf.shape}"
        )

    centered = rf - rf.mean(
        axis=1, keepdims=True
    )

    norms = np.linalg.norm(
        centered, axis=1
    )

    valid = (
        np.isfinite(centered).all(axis=1)
        & np.isfinite(norms)
        & (norms > 0)
    )

    if valid.sum() < 2:
        raise RuntimeError(
            "Too few valid receptive fields"
        )

    normalized = (
        centered[valid]
        / norms[valid, None]
    )

    corr = normalized @ normalized.T

    indices = np.triu_indices(
        corr.shape[0], k=1
    )

    pairs = corr[indices]

    return {
        "rms": float(
            np.sqrt(np.mean(pairs ** 2))
        ),
        "mean": float(np.mean(pairs)),
        "mean_absolute": float(
            np.mean(np.abs(pairs))
        ),
        "n_valid": int(valid.sum()),
        "n_pairs": int(pairs.size),
        "min": float(pairs.min()),
        "max": float(pairs.max()),
    }


def main():

    print("=" * 65)
    print("FIGURE 10A — FUNCTIONAL RF AUDIT")
    print("=" * 65)

    if not STA_PATH.is_file():
        raise FileNotFoundError(STA_PATH)

    if not EVAL_PATH.is_file():
        raise FileNotFoundError(EVAL_PATH)

    with np.load(
        STA_PATH, allow_pickle=False
    ) as data:

        print("STA keys:", data.files)

        if "STA_E" not in data.files:
            raise KeyError("STA_E missing")

        rf = data["STA_E"].copy()

    print("STA source:", STA_PATH)
    print("STA shape:", rf.shape)

    result = rms_pairwise_pearson(rf)

    with np.load(
        EVAL_PATH, allow_pickle=False
    ) as data:

        weight_rms = float(
            data["weight_rf_rms"]
        )

    print("\nFUNCTIONAL STA CORRELATION")
    print("-" * 45)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nCOMPARISON")
    print("-" * 45)
    print("Functional STA RMS:", result["rms"])
    print("Weight-derived RMS:", weight_rms)
    print(
        "Difference:",
        result["rms"] - weight_rms
    )

    print("\nREAD-ONLY AUDIT COMPLETED")


if __name__ == "__main__":
    main()
