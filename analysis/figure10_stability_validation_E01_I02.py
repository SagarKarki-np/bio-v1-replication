
"""
Independent validation of batch-1490 homeostatic convergence.

Seed 3002, eta = 0.1.
Runs batches 681-1590 with frozen synaptic weights.

Reads the original batch-1490 checkpoint.
Never modifies the existing calibration results.
"""

import json
from pathlib import Path

import numpy as np

from analysis import figure10_full_calibration as fc


SEED = 3002
ETA = 0.1
START_BATCH = 1490
VALIDATION_BATCHES = 100
BATCH_SIZE = 100

E_TARGET = 1.0
I_TARGET = 2.0
DURATION = fc.N_STEPS * fc.DT

ROOT = (
    Path("results/paper_protocol")
    / f"seed_{SEED}"
)

SOURCE = (
    ROOT
    / "extended_calibration"
    / "eta_0.10"
    / "latest_checkpoint.npz"
)

OUTPUT = ROOT / "stability_validation" / "E01_I02_batch_1490"

CHECKPOINT = OUTPUT / "latest_checkpoint.npz"

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I", "W_E_E",
    "W_E_I", "W_I_I", "W_I_E"
)



# E1/I2 source-specific configuration
_source_candidates = [
    name for name, value in list(globals().items())
    if isinstance(value, Path)
    and value.name == "latest_checkpoint.npz"
    and "extended_calibration" in value.parts
]
if len(_source_candidates) != 1:
    raise RuntimeError(
        "Could not uniquely identify original calibration source."
    )
globals()[_source_candidates[0]] = (
    ROOT / "extended_calibration"
    / "E01_I02_auto_convergence"
    / "latest_checkpoint.npz"
)

def save_progress(completed, theta_E, theta_I,
                  rates_E, rates_I,
                  history_E, history_I):

    rng = np.random.get_state()

    np.savez_compressed(
        CHECKPOINT,
        completed_batches=completed,
        threshold_E=theta_E,
        threshold_I=theta_I,
        rates_E=np.asarray(rates_E),
        rates_I=np.asarray(rates_I),
        mean_threshold_E=np.asarray(history_E),
        mean_threshold_I=np.asarray(history_I),
        rng_name=np.str_(rng[0]),
        rng_keys=rng[1],
        rng_pos=rng[2],
        rng_has_gauss=rng[3],
        rng_cached_gaussian=rng[4]
    )


def load_state(path):

    with np.load(path, allow_pickle=False) as data:

        completed = int(data["completed_batches"])

        theta_E = data["threshold_E"].copy()
        theta_I = data["threshold_I"].copy()

        rng = (
            str(data["rng_name"].item()),
            data["rng_keys"].copy(),
            int(data["rng_pos"]),
            int(data["rng_has_gauss"]),
            float(data["rng_cached_gaussian"])
        )

        return completed, theta_E, theta_I, rng


def main():

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    # The source MUST be the completed batch-1490 state.
    source_batch, source_E, source_I, source_rng = (
        load_state(SOURCE)
    )

    if source_batch != START_BATCH:
        raise ValueError(
            f"Expected source batch 1490, got {source_batch}"
        )

    if source_E.shape != (fc.N_E, 1):
        raise ValueError("Invalid E threshold shape")

    if source_I.shape != (fc.N_I, 1):
        raise ValueError("Invalid I threshold shape")

    weights, _ = fc.load_initial_weights(
        parameterization="paper",
        network_seed=SEED
    )

    images = fc.PaperImageDataset(
        "data/paper_van_hateren"
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT.exists():

        completed, theta_E, theta_I, rng = (
            load_state(CHECKPOINT)
        )

        with np.load(
            CHECKPOINT, allow_pickle=False
        ) as data:

            rates_E = data["rates_E"].tolist()
            rates_I = data["rates_I"].tolist()

            history_E = (
                data["mean_threshold_E"].tolist()
            )

            history_I = (
                data["mean_threshold_I"].tolist()
            )

        print("RESUMING VALIDATION")

    else:

        if any(OUTPUT.iterdir()):
            raise FileExistsError(
                "Validation output exists without "
                "a resumable checkpoint."
            )

        completed = START_BATCH
        theta_E = source_E.copy()
        theta_I = source_I.copy()
        rng = source_rng

        rates_E = []
        rates_I = []
        history_E = []
        history_I = []

        print("STARTING NEW VALIDATION")

    np.random.set_state(rng)

    if not START_BATCH <= completed <= 1590:
        raise ValueError("Invalid validation batch number")

    print("=" * 60)
    print("HOMEOSTATIC STABILITY VALIDATION")
    print("=" * 60)

    print("Starting from batch:", completed)
    print("Eta:", ETA)
    print("Starting source theta E:", source_E.mean())
    print("Starting source theta I:", source_I.mean())

    for batch in range(completed + 1, 1591):

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

        rate_E = float(counts_E.mean() / DURATION)
        rate_I = float(counts_I.mean() / DURATION)

        theta_E = fc.update_thresholds(
            theta_E, counts_E,
            E_TARGET * DURATION, ETA
        )

        theta_I = fc.update_thresholds(
            theta_I, counts_I,
            I_TARGET * DURATION, ETA
        )

        if not (
            np.isfinite(theta_E).all()
            and np.isfinite(theta_I).all()
        ):
            raise RuntimeError(
                "Nonfinite threshold detected"
            )

        rates_E.append(rate_E)
        rates_I.append(rate_I)

        history_E.append(float(theta_E.mean()))
        history_I.append(float(theta_I.mean()))

        if batch % 10 == 0:

            avg_E = np.mean(rates_E[-10:])
            avg_I = np.mean(rates_I[-10:])

            valid = (
                1.8 <= avg_E <= 2.2
                and 3.6 <= avg_I <= 4.4
            )

            print(
                f"Batch {batch:3d}/1590 | "
                f"E: {avg_E:.3f} Hz | "
                f"I: {avg_I:.3f} Hz | "
                f"theta E: {theta_E.mean():.3f} | "
                f"theta I: {theta_I.mean():.3f} | "
                f"valid: {valid}",
                flush=True
            )

            save_progress(
                batch,
                theta_E, theta_I,
                rates_E, rates_I,
                history_E, history_I
            )

    # --------------------------------------------------
    # Analyze 10 independent 10-batch windows
    # --------------------------------------------------

    valid_windows = []

    for start in range(0, VALIDATION_BATCHES, 10):

        e = np.mean(rates_E[start:start + 10])
        i = np.mean(rates_I[start:start + 10])

        valid_windows.append(
            (0.9 * E_TARGET <= e <= 1.1 * E_TARGET and 0.9 * I_TARGET <= i <= 1.1 * I_TARGET)
        )

    n_valid = sum(valid_windows)
    final_three_valid = all(valid_windows[-3:])

    mean_E = float(np.mean(rates_E))
    mean_I = float(np.mean(rates_I))

    drift_E = float(
        theta_E.mean() - source_E.mean()
    )

    drift_I = float(
        theta_I.mean() - source_I.mean()
    )

    # Change across the last 50 validation batches.
    late_drift_E = float(
        history_E[-1] - history_E[-51]
    )

    late_drift_I = float(
        history_I[-1] - history_I[-51]
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("Mean E rate:", mean_E)
    print("Mean I rate:", mean_I)

    print("Valid windows:", n_valid, "/10")
    print("Final three valid:", final_three_valid)

    print("Full validation theta E change:", drift_E)
    print("Full validation theta I change:", drift_I)

    print("Last 50 theta E change:", late_drift_E)
    print("Last 50 theta I change:", late_drift_I)

    print("Final mean theta E:", theta_E.mean())
    print("Final mean theta I:", theta_I.mean())

    # Save complete results without changing source.
    np.savez_compressed(
        OUTPUT / "validation_history.npz",
        batches=np.arange(681, 781),
        rates_E=np.asarray(rates_E),
        rates_I=np.asarray(rates_I),
        mean_threshold_E=np.asarray(history_E),
        mean_threshold_I=np.asarray(history_I),
        valid_windows=np.asarray(valid_windows)
    )

    metadata = {
        "experiment": "stability_validation",
        "source_batch": 1490,
        "end_batch": 1590,
        "seed": SEED,
        "eta": ETA,
        "weights_frozen": True,
        "valid_windows": int(n_valid),
        "final_three_windows_valid": bool(
            final_three_valid
        ),
        "mean_E_Hz": mean_E,
        "mean_I_Hz": mean_I,
        "threshold_drift_E": drift_E,
        "threshold_drift_I": drift_I,
        "last_50_threshold_drift_E": late_drift_E,
        "last_50_threshold_drift_I": late_drift_I
    }

    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\nVALIDATION COMPLETE")
    print("Results:", OUTPUT)


if __name__ == "__main__":
    main()
