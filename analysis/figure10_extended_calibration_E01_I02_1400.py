
"""
Extended paper-mode threshold calibration.

Continue seed 3002 from its existing batch-100
calibrated thresholds, with frozen synaptic weights.

The original calibration files remain unchanged.
"""

import json
from pathlib import Path

import numpy as np

from analysis import figure10_full_calibration as fc


NETWORK_SEED = 3002
ETA = 0.1

START_BATCH = 100
MAX_BATCH = 1400
BATCH_SIZE = 100

TARGET_E_HZ = 1.0
TARGET_I_HZ = 2.0

DURATION = fc.N_STEPS * fc.DT

SOURCE = (
    Path("results/paper_protocol")
    / f"seed_{NETWORK_SEED}"
    / "calibration/E01_I02"
    / "calibrated_thresholds.npz"
)

OUTPUT = (
    Path("results/paper_protocol")
    / f"seed_{NETWORK_SEED}"
    / "extended_calibration"
    / "E01_I02_continuation_1400"
)

CHECKPOINT = OUTPUT / "latest_checkpoint.npz"


def save_checkpoint(
    completed, theta_E, theta_I,
    rates_E, rates_I,
    histories_E, histories_I
):
    # Store NumPy's random state without pickle.
    rng = np.random.get_state()

    np.savez_compressed(
        CHECKPOINT,
        completed_batches=completed,
        threshold_E=theta_E,
        threshold_I=theta_I,
        rates_E=np.asarray(rates_E),
        rates_I=np.asarray(rates_I),
        mean_threshold_E=np.asarray(histories_E),
        mean_threshold_I=np.asarray(histories_I),
        rng_name=np.str_(rng[0]),
        rng_keys=rng[1],
        rng_pos=rng[2],
        rng_has_gauss=rng[3],
        rng_cached_gaussian=rng[4]
    )


def main():

    weights, weights_path = fc.load_initial_weights(
        parameterization="paper",
        network_seed=NETWORK_SEED
    )

    images = fc.PaperImageDataset(
        "data/paper_van_hateren"
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load checkpoint or initialize continuation.
    # --------------------------------------------------

    if CHECKPOINT.exists():

        print("RESUMING EXTENDED CALIBRATION")

        with np.load(
            CHECKPOINT, allow_pickle=False
        ) as data:

            completed = int(data["completed_batches"])

            theta_E = data["threshold_E"].copy()
            theta_I = data["threshold_I"].copy()

            rates_E = data["rates_E"].tolist()
            rates_I = data["rates_I"].tolist()

            histories_E = (
                data["mean_threshold_E"].tolist()
            )

            histories_I = (
                data["mean_threshold_I"].tolist()
            )

            rng_state = (
                str(data["rng_name"].item()),
                data["rng_keys"].copy(),
                int(data["rng_pos"]),
                int(data["rng_has_gauss"]),
                float(data["rng_cached_gaussian"])
            )

        np.random.set_state(rng_state)

    else:

        if not SOURCE.exists():
            raise FileNotFoundError(SOURCE)

        if any(OUTPUT.iterdir()):
            raise FileExistsError(
                "Output directory already contains files "
                "but no resumable checkpoint."
            )

        with np.load(
            SOURCE, allow_pickle=False
        ) as data:

            theta_E = data["threshold_E"].copy()
            theta_I = data["threshold_I"].copy()

        completed = START_BATCH

        rates_E = []
        rates_I = []

        histories_E = []
        histories_I = []

        # Independent continuation stimulus/noise stream.
        np.random.seed(203102)

        print("STARTING FROM BATCH 100")

    assert theta_E.shape == (fc.N_E, 1)
    assert theta_I.shape == (fc.N_I, 1)

    if not START_BATCH <= completed <= MAX_BATCH:
        raise ValueError("Invalid checkpoint batch.")

    print("=" * 60)
    print("EXTENDED PAPER CALIBRATION")
    print("=" * 60)
    print("Network seed:", NETWORK_SEED)
    print("Learning rate:", ETA)
    print("Starting batch:", completed)
    print("Maximum batch:", MAX_BATCH)
    print("Weights:", weights_path)
    print("Mean theta E:", theta_E.mean())
    print("Mean theta I:", theta_I.mean())

    target_E_count = TARGET_E_HZ * DURATION
    target_I_count = TARGET_I_HZ * DURATION

    stable_windows = 0
    converged = False

    # Evaluate any already completed full windows.
    for end in range(10, len(rates_E) + 1, 10):
        e_mean = np.mean(rates_E[end-10:end])
        i_mean = np.mean(rates_I[end-10:end])

        valid = (
            0.9 <= e_mean <= 1.1
            and 1.8 <= i_mean <= 2.2
        )

        stable_windows = (
            stable_windows + 1 if valid else 0
        )

    if stable_windows >= 3:
        converged = True

    # --------------------------------------------------
    # Continue calibration
    # --------------------------------------------------

    for batch_number in range(
        completed + 1, MAX_BATCH + 1
    ):

        if converged:
            break

        LGN_spikes = fc.generate_lgn_batch(
            images,
            batch_size=BATCH_SIZE,
            n_steps=fc.N_STEPS,
            dt=fc.DT
        )

        E_spikes, I_spikes = fc.simulate_v1_batch(
            LGN_spikes,
            weights,
            dt=fc.DT,
            tau_m=fc.TAU_M,
            noise_std=fc.NOISE_STD,
            v_min=fc.V_MIN,
            v_reset=fc.V_RESET,
            threshold_E=theta_E,
            threshold_I=theta_I,
            record_inputs=False
        )

        counts_E = (
            E_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        counts_I = (
            I_spikes.sum(axis=2)
            .mean(axis=1, keepdims=True)
        )

        rate_E = float(
            counts_E.mean() / DURATION
        )

        rate_I = float(
            counts_I.mean() / DURATION
        )

        # Same published count-based threshold rule.
        theta_E = fc.update_thresholds(
            theta_E,
            counts_E,
            target_E_count,
            ETA
        )

        theta_I = fc.update_thresholds(
            theta_I,
            counts_I,
            target_I_count,
            ETA
        )

        if not (
            np.isfinite(theta_E).all()
            and np.isfinite(theta_I).all()
        ):
            raise RuntimeError(
                "Nonfinite threshold detected."
            )

        rates_E.append(rate_E)
        rates_I.append(rate_I)

        histories_E.append(float(theta_E.mean()))
        histories_I.append(float(theta_I.mean()))

        completed = batch_number

        if completed % 10 == 0:

            avg_E = float(
                np.mean(rates_E[-10:])
            )

            avg_I = float(
                np.mean(rates_I[-10:])
            )

            valid = (
                0.9 <= avg_E <= 1.1
                and 1.8 <= avg_I <= 2.2
            )

            stable_windows = (
                stable_windows + 1 if valid else 0
            )

            print(
                f"Batch {completed:3d}/{MAX_BATCH} | "
                f"E: {avg_E:.3f} Hz | "
                f"I: {avg_I:.3f} Hz | "
                f"theta E: {theta_E.mean():.3f} | "
                f"theta I: {theta_I.mean():.3f} | "
                f"stable: {stable_windows}/3",
                flush=True
            )

            save_checkpoint(
                completed,
                theta_E, theta_I,
                rates_E, rates_I,
                histories_E, histories_I
            )

            if stable_windows >= 3:
                converged = True
                print(
                    "CONVERGENCE CRITERION SATISFIED",
                    flush=True
                )
                break

    # --------------------------------------------------
    # Save final results
    # --------------------------------------------------

    np.savez_compressed(
        OUTPUT / "calibrated_thresholds.npz",
        threshold_E=theta_E,
        threshold_I=theta_I
    )

    np.savez_compressed(
        OUTPUT / "calibration_history.npz",
        batches=np.arange(
            START_BATCH + 1, completed + 1
        ),
        observed_E_Hz=np.asarray(rates_E),
        observed_I_Hz=np.asarray(rates_I),
        mean_threshold_E=np.asarray(histories_E),
        mean_threshold_I=np.asarray(histories_I)
    )

    metadata = {
        "experiment": "extended_calibration",
        "network_seed": NETWORK_SEED,
        "eta": ETA,
        "initial_batch": START_BATCH,
        "final_batch": completed,
        "maximum_batch": MAX_BATCH,
        "update_rule": "count",
        "weights_frozen": True,
        "continuation_random_seed": 203102,
        "exact_original_RNG_continuation": False,
        "converged": converged
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("EXTENDED CALIBRATION SUMMARY")
    print("=" * 60)

    print("Final batch:", completed)

    if rates_E:
        print(
            "Last 10 E mean:",
            np.mean(rates_E[-10:])
        )

        print(
            "Last 10 I mean:",
            np.mean(rates_I[-10:])
        )

    print("Final theta E:", theta_E.mean())
    print("Final theta I:", theta_I.mean())
    print("Converged:", converged)
    print("Results:", OUTPUT)


if __name__ == "__main__":
    main()




