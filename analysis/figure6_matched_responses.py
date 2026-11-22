
"""
Figure 6: Matched activity-response evaluation.

Evaluate the fully trained Track A and Track B
networks using 100 matched natural-image presentations.

Uses the established Figure 3/10 response procedure.

No training.
No synaptic plasticity.
No threshold adaptation.
Checkpoints are read-only.
Existing results are never overwritten.
"""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    DT, N_STEPS, TAU_M, V_MIN, V_RESET
)
from figure10_config import SOURCE_PROJECT
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


BASE = (
    ROOT / "results"
    / "paper_protocol"
    / "seed_3002"
)

OUTPUT_DIR = (
    BASE / "figure6_correlations"
    / "matched_responses"
)

TRACKS = ("a", "b")

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


def checkpoint_path(track):

    return (
        BASE
        / "synaptic_training"
        / f"track_{track}"
        / "checkpoints"
        / "batch_1100.npz"
    )


def load_network(track):

    path = checkpoint_path(track)

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

    assert theta_E.shape == (400, 1)
    assert theta_I.shape == (100, 1)

    return weights, theta_E, theta_I


def main():

    print("=" * 65)
    print("FIGURE 6 — MATCHED NATURAL-IMAGE RESPONSES")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_paths = {
        track: OUTPUT_DIR / f"track_{track}_responses.npz"
        for track in TRACKS
    }

    # Protect previous results.
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(
                f"Result already exists: {path}"
            )

    # Load both networks before simulation.
    networks = {
        track: load_network(track)
        for track in TRACKS
    }

    dataset_path = (
        Path(SOURCE_PROJECT)
        / "data"
        / "van_hateren_dog.npy"
    )

    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    images = np.load(
        dataset_path,
        mmap_mode="r"
    )

    assert images.shape == (1024, 1536, 8)

    print("Stimulus dataset:", dataset_path)
    print("Dataset shape:", images.shape)

    print("\nGenerating shared LGN responses...")

    # Match the Figure 3/10 stimulus-generation seed.
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

    # Capture RNG state after LGN generation.
    # Restore it before running each V1 network.
    noise_state = np.random.get_state()

    duration = N_STEPS * DT

    print("LGN spikes:", LGN_spikes.shape)
    print("Presentation duration:", duration)
    print(
        "Mean LGN rate (Hz):",
        LGN_counts.mean() / duration
    )

    for track in TRACKS:

        print(
            f"\nEvaluating Track {track.upper()}...",
            flush=True
        )

        weights, theta_E, theta_I = networks[track]

        # Both tracks begin with the same RNG state.
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

        assert LGN_counts.shape == (512, 100)
        assert E_counts.shape == (400, 100)
        assert I_counts.shape == (100, 100)

        output = output_paths[track]

        np.savez_compressed(
            output,
            LGN_counts=LGN_counts,
            E_counts=E_counts,
            I_counts=I_counts,
            duration=np.array(duration),
            seed=np.array(SEED),
            n_presentations=np.array(N_PRESENTATIONS),
            noise_std=np.array(NOISE_STD),
            checkpoint_path=np.array(
                str(checkpoint_path(track))
            ),
            stimulus_path=np.array(
                str(dataset_path)
            ),
            track=np.array(track),
            parameterization=np.array("paper"),
            completed_batches=np.array(1100),
        )

        print("LGN:", LGN_counts.shape)
        print("E:", E_counts.shape)
        print("I:", I_counts.shape)

        print(
            "Mean E rate (Hz):",
            E_counts.mean() / duration
        )

        print(
            "Mean I rate (Hz):",
            I_counts.mean() / duration
        )

        print("Saved:", output)

    print("\n" + "=" * 65)
    print("FIGURE 6 RESPONSE GENERATION COMPLETE")
    print("=" * 65)
    print("Both networks evaluated on shared LGN inputs.")
    print("No training or checkpoint changes.")


if __name__ == "__main__":
    main()
