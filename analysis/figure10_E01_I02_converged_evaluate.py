
"""
Figure 10A-C: Paper-protocol response evaluation.

Initial condition: E01_I02.

Uses:
- Correct 1000-image PaperImageDataset
- Trained Figure 10 paper checkpoint
- 100 natural-image presentations
- Published-form Treves-Rolls sparseness
- Preliminary weight-derived RF correlation

No synaptic learning or threshold updates.
Does not modify training checkpoints.
"""

from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from config import DT, N_STEPS, TAU_M, V_MIN, V_RESET
from paper_image_dataset import PaperImageDataset
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch

from analysis.sparseness import sparseness


CONDITION = "E01_I02"
SEED = 2027
N_PRESENTATIONS = 100
NOISE_STD = 1.0

BASE = (
    PROJECT / "results" / "paper_protocol"
    / "seed_3002"
)

CHECKPOINT = (
    BASE / "convergence_experiments"
    / CONDITION
    / "training" / "checkpoints"
    / "batch_1100.npz"
)

IMAGE_DIR = PROJECT / "data" / "paper_van_hateren"

OUTPUT = (
    BASE / "convergence_experiments"
    / CONDITION
    / "evaluation" / "paper_1000_source"
)

RESULT_PATH = OUTPUT / "response_evaluation.npz"

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)


def summarize_sparseness(counts, name):
    """
    counts: neurons x presentations.

    Temporal: across presentations, axis=1.
    Population: across neurons, axis=0.
    """

    temporal = sparseness(
        counts, axis=1,
        finite_correction=False
    )

    population = sparseness(
        counts, axis=0,
        finite_correction=False
    )

    print(f"\n{name} SPARSENESS")
    print("-" * 45)

    print(
        "Temporal mean:",
        np.nanmean(temporal)
    )

    print(
        "Population mean:",
        np.nanmean(population)
    )

    print(
        "Silent neurons:",
        np.isnan(temporal).sum()
    )

    print(
        "Silent presentations:",
        np.isnan(population).sum()
    )

    return temporal, population


def weight_rf_rms(W_LGN_E):
    """
    Diagnostic ONLY: weight-derived signed RFs.

    It is not a substitute for functional STA
    unless methodological equivalence is validated.
    """

    if W_LGN_E.shape != (400, 512):
        raise ValueError(
            f"Wrong LGN->E shape: {W_LGN_E.shape}"
        )

    rf = (
        W_LGN_E[:, :256]
        - W_LGN_E[:, 256:]
    )

    centered = rf - rf.mean(
        axis=1, keepdims=True
    )

    norms = np.linalg.norm(
        centered, axis=1
    )

    valid = (
        np.isfinite(norms)
        & (norms > 0)
        & np.isfinite(centered).all(axis=1)
    )

    normalized = (
        centered[valid]
        / norms[valid, None]
    )

    correlation = normalized @ normalized.T

    upper = np.triu_indices(
        correlation.shape[0], k=1
    )

    pair_values = correlation[upper]

    rms = np.sqrt(
        np.mean(pair_values ** 2)
    )

    print("\nWEIGHT-DERIVED RF DIAGNOSTIC")
    print("-" * 45)
    print("Valid E neurons:", valid.sum())
    print("Valid pairs:", pair_values.size)
    print("RMS correlation:", rms)

    return float(rms), int(pair_values.size)


def main():

    print("=" * 65)
    print("FIGURE 10 — PAPER EVALUATION")
    print("=" * 65)

    print("Condition:", CONDITION)
    print("Protocol: paper_1000_source")
    print("Checkpoint:", CHECKPOINT)

    if RESULT_PATH.exists():
        raise FileExistsError(
            "Evaluation already exists: "
            + str(RESULT_PATH)
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    with np.load(
        CHECKPOINT, allow_pickle=False
    ) as data:

        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"

        assert np.isclose(
            float(data["target_E_Hz"]), 1.0
        )

        assert np.isclose(
            float(data["target_I_Hz"]), 2.0
        )

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

    assert theta_E.shape == (400, 1)
    assert theta_I.shape == (100, 1)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(IMAGE_DIR)

    images = PaperImageDataset(str(IMAGE_DIR))

    if tuple(images.shape) != (1024, 1536, 1000):
        raise RuntimeError(
            f"Wrong dataset: {images.shape}"
        )

    print("Images:", images.shape)

    # Fixed seed permits later conditions to receive
    # the same LGN stimulus realization.
    np.random.seed(SEED)

    LGN_spikes = generate_lgn_batch(
        images,
        batch_size=N_PRESENTATIONS,
        n_steps=N_STEPS,
        dt=DT
    )

    assert LGN_spikes.shape == (
        512, N_PRESENTATIONS, N_STEPS
    )

    # Preserve identical simulation-noise RNG state
    # for future condition comparisons.
    noise_state = np.random.get_state()

    np.random.set_state(noise_state)

    E_spikes, I_spikes = simulate_v1_batch(
        LGN_spikes,
        weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=theta_E,
        threshold_I=theta_I
    )

    E_counts = E_spikes.sum(axis=2)
    I_counts = I_spikes.sum(axis=2)

    assert E_counts.shape == (400, N_PRESENTATIONS)
    assert I_counts.shape == (100, N_PRESENTATIONS)

    duration = DT * N_STEPS

    E_rate = E_counts.mean() / duration
    I_rate = I_counts.mean() / duration

    print("\nFIRING RATES")
    print("-" * 45)
    print("Target E: 1.0 Hz")
    print("Target I: 2.0 Hz")
    print("Observed E:", E_rate, "Hz")
    print("Observed I:", I_rate, "Hz")

    E_temporal, E_population = (
        summarize_sparseness(
            E_counts, "EXCITATORY"
        )
    )

    I_temporal, I_population = (
        summarize_sparseness(
            I_counts, "INHIBITORY"
        )
    )

    rf_rms, rf_pairs = weight_rf_rms(
        weights[0]
    )

    # Save outputs without modifying the trained model.
    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(
        RESULT_PATH,
        condition=CONDITION,
        protocol="paper_1000_source",
        checkpoint=str(CHECKPOINT),
        dataset=str(IMAGE_DIR),
        dataset_shape=np.array(images.shape),

        seed=SEED,
        n_presentations=N_PRESENTATIONS,
        duration=duration,
        noise_std=NOISE_STD,

        E_counts=E_counts,
        I_counts=I_counts,
        LGN_counts=LGN_spikes.sum(axis=2),

        E_rate=E_rate,
        I_rate=I_rate,

        E_temporal=E_temporal,
        E_population=E_population,
        I_temporal=I_temporal,
        I_population=I_population,

        weight_rf_rms=rf_rms,
        weight_rf_pairs=rf_pairs,

        rf_method="weight_derived_diagnostic"
    )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("Saved:", RESULT_PATH)
    print("No training or threshold updates performed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
