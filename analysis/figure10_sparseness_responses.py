
"""
Figure 10B-C: Matched natural-image response evaluation.

Evaluate E02/I04 and E05/I10 using the original
Figure 3 response-generation methodology.

No synaptic plasticity.
No threshold adaptation.
No training.
Existing checkpoints are read-only.
"""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DT, N_STEPS, TAU_M, V_MIN, V_RESET,
)
from figure10_config import SOURCE_PROJECT
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
    / "matlab_demo"
)

CONDITIONS = ("E20_I40",)
WEIGHT_NAMES = (
    "W_LGN_E",
    "W_LGN_I",
    "W_E_E",
    "W_E_I",
    "W_I_I",
    "W_I_E",
)

SEED = 2027
N_PRESENTATIONS = 100
NOISE_STD = 1.0


def load_network(condition):
    checkpoint = (
        RESULT_ROOT
        / condition
        / "training"
        / "checkpoints"
        / "batch_1100.npz"
    )

    with np.load(checkpoint, allow_pickle=False) as data:
        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )
        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

        if "completed_batches" in data.files:
            assert int(data["completed_batches"]) == 1100

        if "parameterization" in data.files:
            assert str(data["parameterization"]) == "matlab_demo"

    assert theta_E.shape == (400, 1)
    assert theta_I.shape == (100, 1)

    return weights, theta_E, theta_I


def main():
    print("\nFIGURE 10B-C: MATCHED RESPONSE EVALUATION")
    print("=" * 60)

    output_paths = {
        condition: (
            RESULT_ROOT
            / condition
            / "sparseness_evaluation"
            / "matched_responses.npz"
        )
        for condition in CONDITIONS
    }

    # Preserve any previously saved evaluations.
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(
                f"Existing result will not be overwritten: {path}"
            )

    networks = {
        condition: load_network(condition)
        for condition in CONDITIONS
    }

    images = np.load(
        Path(SOURCE_PROJECT)
        / "data"
        / "van_hateren_dog.npy",
        mmap_mode="r"
    )

    print("Image dataset shape:", images.shape)

    # Same stimulus-generation seed as Figure 3.
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

    print("LGN spikes shape:", LGN_spikes.shape)

    # Identical RNG state for simulation noise.
    noise_state = np.random.get_state()

    for condition in CONDITIONS:
        weights, theta_E, theta_I = networks[condition]

        print(f"\nEvaluating {condition}...", flush=True)

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

        output = output_paths[condition]
        output.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output,
            E_counts=E_counts,
            I_counts=I_counts,
            LGN_counts=LGN_spikes.sum(axis=2),
            condition=np.array(condition),
            seed=np.array(SEED),
            n_presentations=np.array(N_PRESENTATIONS),
            duration=np.array(N_STEPS * DT),
            noise_std=np.array(NOISE_STD)
        )

        duration = N_STEPS * DT

        print("E count shape:", E_counts.shape)
        print("I count shape:", I_counts.shape)
        print("Mean E rate (Hz):", E_counts.mean() / duration)
        print("Mean I rate (Hz):", I_counts.mean() / duration)
        print("Saved:", output)

    print("\nBoth evaluations completed.")
    print("Network weights and thresholds unchanged.")


if __name__ == "__main__":
    main()
