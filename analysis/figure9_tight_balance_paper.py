
"""
Figure 9 — Tight excitation/inhibition balance.

Adapted from the existing Figure 9 MATLAB-demo analysis.

Uses the correct 1000-source-image paper dataset and
the trained Track A or Track B checkpoint.

No training or plasticity. Read-only input files.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from config import (
    DT, N_STEPS, TAU_M, V_MIN, V_RESET,
    PATCH_SIZE, LGN_RATE_SCALE, LGN_MAX_RATE
)
from preprocessing import extract_random_patch
from paper_image_dataset import PaperImageDataset
from simulation import (
    generate_lgn_population_spikes,
    simulate_v1_batch
)
from analysis.excitation_inhibition import correlation_at_lag


BASE = (
    PROJECT / "results" / "paper_protocol"
    / "seed_3002"
)

WEIGHT_NAMES = (
    "W_LGN_E", "W_LGN_I", "W_E_E",
    "W_E_I", "W_I_I", "W_I_E"
)

NEURON = 0
N_CANDIDATES = 1000
N_SELECTED = 10

MAX_LAG_MS = 20
EXCLUDE_FIRST_MS = 100

SEED = 2029
NOISE_STD = 0.0


def main(track):

    print("=" * 65)
    print(f"FIGURE 9 — TIGHT E/I BALANCE — TRACK {track.upper()}")
    print("=" * 65)

    checkpoint = (
        BASE / "synaptic_training"
        / f"track_{track}"
        / "checkpoints" / "batch_1100.npz"
    )

    sta_path = (
        BASE / "functional_sta"
        / f"track_{track}"
        / "full_100000"
        / "STA_100000_results.npz"
    )

    image_dir = (
        PROJECT / "data" / "paper_van_hateren"
    )

    output = (
        BASE / "figure9_tight_balance"
        / "paper_1000_source"
        / f"track_{track}"
    )

    figure_path = output / "figure9_tight_balance.png"
    stats_path = output / "figure9_statistics.npz"

    # Refuse to overwrite earlier results.
    if figure_path.exists() or stats_path.exists():
        raise FileExistsError(
            f"Figure 9 results already exist: {output}"
        )

    # -------------------------------------------
    # 1. Load the trained network
    # -------------------------------------------

    with np.load(
        checkpoint, allow_pickle=False
    ) as data:

        assert int(data["completed_batches"]) == 1100
        assert str(data["parameterization"]) == "paper"

        weights = tuple(
            data[name].copy()
            for name in WEIGHT_NAMES
        )

        threshold_E = data["threshold_E"].copy()
        threshold_I = data["threshold_I"].copy()

    assert threshold_E.shape == (400, 1)
    assert threshold_I.shape == (100, 1)

    # -------------------------------------------
    # 2. Load the paper STA
    # -------------------------------------------

    with np.load(
        sta_path, allow_pickle=False
    ) as data:

        assert int(data["n_stimuli"]) == 100000
        assert data["STA_E"].shape == (400, 256)

        sta = data["STA_E"][NEURON].copy()

    assert np.isfinite(sta).all()

    # -------------------------------------------
    # 3. Load the correct paper dataset
    # -------------------------------------------

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    images = PaperImageDataset(str(image_dir))

    if tuple(images.shape) != (1024, 1536, 1000):
        raise RuntimeError(
            f"Incorrect dataset: {images.shape}"
        )

    print("\nPROTOCOL: paper_1000_source")
    print("IMAGE_DIR:", image_dir)
    print("Images:", images.shape)
    print("Checkpoint:", checkpoint)
    print("STA:", sta_path)

    # Match the original MATLAB-demo analysis seed.
    np.random.seed(SEED)

    # -------------------------------------------
    # 4. Generate 1000 candidate patches
    # -------------------------------------------

    patches = np.empty(
        (N_CANDIDATES, PATCH_SIZE, PATCH_SIZE)
    )

    for i in range(N_CANDIDATES):

        image_idx = np.random.randint(
            images.shape[2]
        )

        patches[i] = extract_random_patch(
            images[:, :, image_idx],
            PATCH_SIZE,
            20
        )

    # MATLAB column-major spatial ordering.
    vectors = np.stack([
        p.flatten(order="F")
        for p in patches
    ])

    sta_centered = sta - sta.mean()
    sta_norm = np.linalg.norm(sta_centered)

    centered = (
        vectors
        - vectors.mean(axis=1, keepdims=True)
    )

    norms = np.linalg.norm(centered, axis=1)

    if sta_norm <= 0:
        raise ValueError("STA has zero variance.")

    valid = np.isfinite(norms) & (norms > 0)

    correlations = np.full(
        N_CANDIDATES, np.nan
    )

    correlations[valid] = (
        centered[valid] @ sta_centered
    ) / (norms[valid] * sta_norm)

    if np.sum(valid) < N_SELECTED:
        raise ValueError(
            "Insufficient valid natural-image patches."
        )

    # Select ten highest positive RF correlations.
    selected_indices = np.argsort(
        np.nan_to_num(
            correlations,
            nan=-np.inf
        )
    )[-N_SELECTED:][::-1]

    selected_patches = patches[selected_indices]
    selected_correlations = correlations[selected_indices]

    print("\nSELECTED PATCH CORRELATIONS")
    print(selected_correlations)

    # -------------------------------------------
    # 5. Generate LGN spikes
    # -------------------------------------------

    LGN_spikes = np.zeros(
        (512, N_SELECTED, N_STEPS),
        dtype=bool
    )

    for i, patch in enumerate(selected_patches):

        vector = patch.flatten(order="F")
        scaled = vector * LGN_RATE_SCALE

        rates = np.concatenate([
            np.maximum(scaled, 0),
            np.maximum(-scaled, 0)
        ])

        rates = np.clip(
            rates,
            0,
            LGN_MAX_RATE
        )

        LGN_spikes[:, i, :] = (
            generate_lgn_population_spikes(
                rates, DT, N_STEPS
            )
        )

    print("\nLGN spikes:", LGN_spikes.shape)

    # -------------------------------------------
    # 6. Simulate trained network
    # -------------------------------------------

    _, _, traces = simulate_v1_batch(
        LGN_spikes,
        weights,
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I,
        record_inputs=True
    )

    exc_input = (
        traces["LGN_E"][NEURON]
        + traces["E_E"][NEURON]
    )

    inh_input = traces["I_E"][NEURON]

    # -------------------------------------------
    # 7. Offline voltage reconstruction
    # -------------------------------------------

    decay = np.exp(-DT / TAU_M)

    V_exc = np.zeros_like(exc_input)
    V_inh_mag = np.zeros_like(inh_input)

    ve = np.zeros(N_SELECTED)
    vi = np.zeros(N_SELECTED)

    for t in range(N_STEPS):

        ve = decay * ve + exc_input[:, t]
        vi = decay * vi + inh_input[:, t]

        V_exc[:, t] = ve
        V_inh_mag[:, t] = vi

    # These are offline voltage contributions.
    # They do not include spike reset/clipping.

    # -------------------------------------------
    # 8. Exclude first 100 ms
    # -------------------------------------------

    start = int(round(
        EXCLUDE_FIRST_MS / (DT * 1000)
    ))

    E_voltage = V_exc[:, start:]
    I_voltage = V_inh_mag[:, start:]

    E_synaptic = exc_input[:, start:]
    I_synaptic = inh_input[:, start:]

    # -------------------------------------------
    # 9. Calculate lagged correlations
    # -------------------------------------------

    max_lag_steps = int(round(
        MAX_LAG_MS / (DT * 1000)
    ))

    lags = np.arange(
        -max_lag_steps,
        max_lag_steps + 1
    )

    lag_ms = lags * DT * 1000

    voltage_corr = np.array([
        correlation_at_lag(
            E_voltage, I_voltage, lag
        )
        for lag in lags
    ]).T

    synaptic_corr = np.array([
        correlation_at_lag(
            E_synaptic, I_synaptic, lag
        )
        for lag in lags
    ]).T

    mean_voltage = np.nanmean(
        voltage_corr, axis=0
    )

    mean_synaptic = np.nanmean(
        synaptic_corr, axis=0
    )

    peak_voltage = int(
        np.nanargmax(mean_voltage)
    )

    peak_synaptic = int(
        np.nanargmax(mean_synaptic)
    )

    zero = int(
        np.argmin(np.abs(lag_ms))
    )

    print("\nFIGURE 9 RESULTS")
    print("=" * 65)

    print(
        "Valid voltage traces:",
        np.isfinite(voltage_corr).all(axis=1).sum(),
        "/", N_SELECTED
    )

    print(
        "Valid synaptic traces:",
        np.isfinite(synaptic_corr).all(axis=1).sum(),
        "/", N_SELECTED
    )

    print(
        "Voltage peak lag:",
        lag_ms[peak_voltage], "ms"
    )

    print(
        "Voltage peak correlation:",
        mean_voltage[peak_voltage]
    )

    print(
        "Voltage zero-lag correlation:",
        mean_voltage[zero]
    )

    print(
        "Synaptic peak lag:",
        lag_ms[peak_synaptic], "ms"
    )

    print(
        "Synaptic peak correlation:",
        mean_synaptic[peak_synaptic]
    )

    print(
        "Synaptic zero-lag correlation:",
        mean_synaptic[zero]
    )

    # -------------------------------------------
    # 10. Plot Figure 9
    # -------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        constrained_layout=True
    )

    panels = [
        (
            voltage_corr,
            mean_voltage,
            "Voltage-component cross-correlation"
        ),
        (
            synaptic_corr,
            mean_synaptic,
            "Instantaneous synaptic-input cross-correlation"
        )
    ]

    for ax, (curves, average, title) in zip(
        axes, panels
    ):

        for curve in curves:
            ax.plot(
                lag_ms, curve,
                alpha=0.3
            )

        ax.plot(
            lag_ms,
            average,
            color="black",
            linewidth=2.5,
            label="Mean of 10"
        )

        ax.axvline(
            0,
            color="gray",
            linestyle="--"
        )

        ax.set_title(title)

        ax.set_xlabel(
            "Lag (ms); positive = inhibition follows excitation"
        )

        ax.set_ylabel("Pearson correlation")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle(
        f"Figure 9: Track {track.upper()}, "
        "tight E/I balance"
    )

    # -------------------------------------------
    # 11. Save results
    # -------------------------------------------

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    np.savez_compressed(
        stats_path,

        selected_indices=selected_indices,
        selected_correlations=selected_correlations,
        selected_patches=selected_patches,

        lag_ms=lag_ms,

        voltage_correlations=voltage_corr,
        synaptic_correlations=synaptic_corr,

        mean_voltage=mean_voltage,
        mean_synaptic=mean_synaptic,

        V_exc=V_exc,
        V_inh_magnitude=V_inh_mag,

        exc_input=exc_input,
        inh_input=inh_input,

        neuron=NEURON,
        seed=SEED,
        noise_std=NOISE_STD,
        excluded_initial_ms=EXCLUDE_FIRST_MS,

        checkpoint_source=str(checkpoint),
        sta_source=str(sta_path),

        stimulus_protocol="paper_1000_source",
        stimulus_path=str(image_dir),
        stimulus_shape=np.array(images.shape),

        track=track,
        presentation_duration=DT * N_STEPS,
    )

    print("\nFigure 9 analysis completed.")
    print("Figure:", figure_path)
    print("Statistics:", stats_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--track",
        choices=("a", "b"),
        required=True
    )

    args = parser.parse_args()
    main(args.track)
