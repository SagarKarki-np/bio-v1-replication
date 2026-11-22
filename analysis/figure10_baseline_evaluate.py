
"""
Figure 10A-C: Evaluate the paper-protocol 2/4 Hz baseline.

Compare Track A with the already evaluated E01_I02 network.

Uses the identical:
- 1000-image dataset
- 100 natural-image presentations
- Random seed
- Simulation settings
- Sparseness functions

Read-only simulation; no plasticity or threshold updates.
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

from analysis.figure10_paper_evaluate import (
    summarize_sparseness,
    weight_rf_rms,
)

BASE = (
    PROJECT / "results" / "paper_protocol"
    / "seed_3002"
)

CHECKPOINT = (
    BASE / "synaptic_training"
    / "track_a" / "checkpoints"
    / "batch_1100.npz"
)

IMAGE_DIR = PROJECT / "data" / "paper_van_hateren"

OUTPUT = (
    BASE / "figure10_firing_rate_sweep"
    / "paper" / "E02_I04"
    / "evaluation" / "paper_1000_source"
)

RESULT_PATH = OUTPUT / "response_evaluation.npz"

LOW_RATE_RESULT = (
    BASE / "figure10_firing_rate_sweep"
    / "paper" / "E01_I02"
    / "evaluation" / "paper_1000_source"
    / "response_evaluation.npz"
)

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I",
    "W_E_E", "W_E_I",
    "W_I_I", "W_I_E"
)

SEED = 2027
N_PRESENTATIONS = 100
NOISE_STD = 1.0


def main():

    print("=" * 65)
    print("FIGURE 10 — BASELINE 2/4 Hz EVALUATION")
    print("=" * 65)

    print("Protocol: paper_1000_source")
    print("Condition: E02_I04")
    print("Baseline: Track A")

    if RESULT_PATH.exists():
        raise FileExistsError(
            f"Evaluation already exists: {RESULT_PATH}"
        )

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    if not LOW_RATE_RESULT.is_file():
        raise FileNotFoundError(LOW_RATE_RESULT)

    # -----------------------------------------
    # 1. Load trained Track A checkpoint
    # -----------------------------------------

    with np.load(
        CHECKPOINT, allow_pickle=False
    ) as data:

        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

        if "target_E_Hz" in data.files:
            assert np.isclose(
                float(data["target_E_Hz"]), 2.0
            )

        if "target_I_Hz" in data.files:
            assert np.isclose(
                float(data["target_I_Hz"]), 4.0
            )

    assert theta_E.shape == (400, 1)
    assert theta_I.shape == (100, 1)

    print("\nCheckpoint:", CHECKPOINT)
    print("Mean E threshold:", theta_E.mean())
    print("Mean I threshold:", theta_I.mean())

    # -----------------------------------------
    # 2. Load 1000-image paper dataset
    # -----------------------------------------

    images = PaperImageDataset(str(IMAGE_DIR))

    if tuple(images.shape) != (1024, 1536, 1000):
        raise RuntimeError(
            f"Incorrect image dataset: {images.shape}"
        )

    print("Images:", images.shape)

    # -----------------------------------------
    # 3. Reproduce identical evaluation stimuli
    # -----------------------------------------

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

    LGN_counts = LGN_spikes.sum(axis=2)

    # -----------------------------------------
    # 4. Frozen-network evaluation
    # -----------------------------------------

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
    print("Target E: 2.0 Hz")
    print("Target I: 4.0 Hz")
    print("Observed E:", E_rate, "Hz")
    print("Observed I:", I_rate, "Hz")

    E_temporal, E_population = summarize_sparseness(
        E_counts, "EXCITATORY"
    )

    I_temporal, I_population = summarize_sparseness(
        I_counts, "INHIBITORY"
    )

    rf_rms, rf_pairs = weight_rf_rms(weights[0])

    # -----------------------------------------
    # 5. Confirm common LGN inputs
    # -----------------------------------------

    with np.load(
        LOW_RATE_RESULT, allow_pickle=False
    ) as low:

        low_lgn_counts = low["LGN_counts"]

        same_lgn_counts = np.array_equal(
            low_lgn_counts,
            LGN_counts
        )

        if not same_lgn_counts:
            raise RuntimeError(
                "LGN inputs differ between evaluations. "
                "Results will not be saved."
            )

        low_values = {
            "E_rate": float(low["E_rate"]),
            "I_rate": float(low["I_rate"]),
            "E_temporal": float(
                np.nanmean(low["E_temporal"])
            ),
            "E_population": float(
                np.nanmean(low["E_population"])
            ),
            "I_temporal": float(
                np.nanmean(low["I_temporal"])
            ),
            "I_population": float(
                np.nanmean(low["I_population"])
            ),
            "RF_RMS": float(low["weight_rf_rms"]),
        }

    print("\nIdentical LGN spike counts:", same_lgn_counts)
    print(
        "Note: matching spike counts provide a useful "
        "check, but do not independently prove "
        "identical spike timing."
    )

    # -----------------------------------------
    # 6. Compare both conditions
    # -----------------------------------------

    baseline_values = {
        "E_rate": float(E_rate),
        "I_rate": float(I_rate),
        "E_temporal": float(np.nanmean(E_temporal)),
        "E_population": float(np.nanmean(E_population)),
        "I_temporal": float(np.nanmean(I_temporal)),
        "I_population": float(np.nanmean(I_population)),
        "RF_RMS": float(rf_rms),
    }

    print("\n" + "=" * 65)
    print("FIGURE 10 — E01_I02 versus E02_I04")
    print("=" * 65)

    print(
        f"{'Measurement':<22}"
        f"{'E01_I02':>14}"
        f"{'E02_I04':>14}"
    )

    print("-" * 50)

    for key in low_values:
        print(
            f"{key:<22}"
            f"{low_values[key]:>14.6f}"
            f"{baseline_values[key]:>14.6f}"
        )

    print("\nSilent presentations:")
    print(
        "E population:",
        int(np.isnan(E_population).sum())
    )
    print(
        "I population:",
        int(np.isnan(I_population).sum())
    )

    # -----------------------------------------
    # 7. Save results
    # -----------------------------------------

    OUTPUT.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULT_PATH,
        condition="E02_I04",
        track="A",
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
        LGN_counts=LGN_counts,

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

    print("\nEVALUATION COMPLETED")
    print("Saved:", RESULT_PATH)
    print("Training checkpoints unchanged.")


if __name__ == "__main__":
    main()
