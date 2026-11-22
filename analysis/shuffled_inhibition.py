
"""
Shuffled inhibitory-connectivity and rate-matching controls.

LGN-V1 network replication
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Experiments
-----------
1. Compare intact inhibition with 10 row-wise shuffled
   I-to-E connectivity matrices.

2. Test five inhibitory scaling factors on the final
   (tenth) shuffled connectivity matrix.

3. Refine the firing-rate match using the original
   bisection procedure.

Numerical reproducibility
-------------------------
- LGN and simulation RNG seed: 2035
- Inhibitory shuffle RNG seed: 2036
- Same LGN spike trains in every condition
- Same NumPy RNG state restored before simulation
- Same condition ordering and original formulas
- No synaptic learning during evaluation
- The original weights and thresholds remain unchanged

Outputs
-------
shuffled_inhibition_control.npz
rate_matched_shuffled_inhibition.npz
refined_rate_matched_inhibition.npz

The original data and checkpoint are read-only.
"""

from pathlib import Path
import argparse

import numpy as np

from config import (
    DT,
    N_STEPS,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
    N_E,
    N_I,
    N_LGN,
)

from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch

from analysis.inhibitory_perturbations import (
    load_trained_network,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "shuffled_inhibition"
)

INPUT_SEED = 2035
SHUFFLE_SEED = 2036

EVALUATION_BATCH_SIZE = 100
N_SHUFFLES = 10

COARSE_SCALES = np.array([
    1.0, 1.2, 1.4, 1.6, 1.8
])

REFINEMENT_LOW = 1.8
REFINEMENT_HIGH = 3.0

RATE_TOLERANCE = 0.02
MAX_REFINEMENT_ITERATIONS = 12

METRIC_LABELS = [
    "E firing rate",
    "Active E fraction",
    "Population sparseness",
    "Mean response correlation",
    "Mean absolute correlation",
]


# ============================================================
# 2. SHUFFLE INHIBITORY CONNECTIONS
# ============================================================

def shuffle_inhibitory_weights(weights, rng):
    """
    Independently shuffle each row of W_I_E.

    The sixth matrix represents:
        inhibitory presynaptic -> excitatory postsynaptic

    Each row corresponds to one excitatory neuron.

    Row permutations preserve the exact incoming
    inhibitory weight sum for each excitatory neuron.

    The original matrices are not modified.

    Parameters
    ----------
    weights : sequence of six weight matrices
    rng : numpy.random.Generator

    Returns
    -------
    tuple
        Six copied matrices, with W_I_E shuffled.
    """

    if len(weights) != 6:
        raise ValueError(
            "Expected six synaptic weight matrices."
        )

    W = [
        matrix.copy()
        for matrix in weights
    ]

    original = W[5]

    shuffled = np.empty_like(original)

    for row in range(original.shape[0]):

        shuffled[row] = original[
            row,
            rng.permutation(original.shape[1])
        ]

    # Preserve the original weight-sum check.
    assert np.allclose(
        original.sum(axis=1),
        shuffled.sum(axis=1)
    )

    W[5] = shuffled

    return tuple(W)


# ============================================================
# 3. SCALE SHUFFLED INHIBITION
# ============================================================

def scale_shuffled_inhibition(weights, scale):
    """
    Multiply only W_I_E by scale.

    Returns independent copies of all six matrices.
    """

    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            "Scale must be finite and nonnegative."
        )

    modified = [
        matrix.copy()
        for matrix in weights
    ]

    modified[5] *= scale

    return tuple(modified)


# ============================================================
# 4. EVALUATE ONE FROZEN NETWORK
# ============================================================

def evaluate_network(
    weights,
    LGN_spikes,
    threshold_E,
    threshold_I,
    noise_state
):
    """
    Evaluate a frozen network using matched LGN
    spike trains and matched simulation random draws.

    Returns
    -------
    ndarray, shape (5,)
        [0] Mean E firing rate (Hz)
        [1] Active E fraction
        [2] Treves-Rolls population sparseness
        [3] Mean signed E response correlation
        [4] Mean absolute E response correlation

    The calculations preserve the original script.
    """

    np.random.set_state(noise_state)

    E_spikes, _ = simulate_v1_batch(
        LGN_spikes,
        tuple(weights),
        dt=DT,
        tau_m=TAU_M,
        noise_std=NOISE_STD,
        v_min=V_MIN,
        v_reset=V_RESET,
        threshold_E=threshold_E,
        threshold_I=threshold_I
    )

    counts = E_spikes.sum(axis=2)

    duration = N_STEPS * DT

    mean_rate = counts.mean() / duration

    active = np.mean(counts > 0)

    # Original Treves-Rolls calculation.
    mean_r = counts.mean(axis=0)
    mean_r2 = (counts ** 2).mean(axis=0)

    valid = mean_r2 > 0

    sparsity = (
        1 - mean_r[valid] ** 2 / mean_r2[valid]
    ) / (1 - 1 / counts.shape[0])

    # Original response-correlation calculation.
    active_neurons = counts.std(axis=1) > 0

    X = counts[active_neurons]

    if X.shape[0] >= 2:
        C = np.corrcoef(X)

        upper = C[
            np.triu_indices_from(C, k=1)
        ]

        mean_corr = np.mean(upper)
        abs_corr = np.mean(np.abs(upper))
    else:
        mean_corr = np.nan
        abs_corr = np.nan

    mean_sparsity = (
        np.mean(sparsity)
        if sparsity.size > 0
        else np.nan
    )

    return np.array([
        mean_rate,
        active,
        mean_sparsity,
        mean_corr,
        abs_corr
    ])


# ============================================================
# 5. RUN TEN SHUFFLED-CONTROL NETWORKS
# ============================================================

def run_shuffled_controls(
    weights,
    evaluate,
    n_shuffles=N_SHUFFLES,
    seed=SHUFFLE_SEED
):
    """
    Run independently shuffled inhibitory networks.

    Parameters
    ----------
    weights : tuple
        Original trained weights.

    evaluate : callable
        Function accepting one weight tuple
        and returning five population metrics.

    n_shuffles : int
        Number of shuffled networks.

    seed : int
        NumPy Generator seed for row permutations.

    Returns
    -------
    normal : ndarray, shape (5,)
    shuffled_results : ndarray, shape (n_shuffles, 5)
    shuffled_base : tuple
        Final shuffled network for rate matching.
    """

    if n_shuffles < 1:
        raise ValueError(
            "At least one shuffle is required."
        )

    # Baseline is evaluated first, as in the original.
    normal = evaluate(weights)

    print("\nNormal:", normal)

    rng = np.random.default_rng(seed)

    shuffled_results = []

    shuffled_base = None

    for k in range(n_shuffles):

        W = shuffle_inhibitory_weights(
            weights,
            rng
        )

        result = evaluate(W)

        shuffled_results.append(result)

        # Preserve the final shuffled network.
        shuffled_base = W

        print(
            f"Shuffle {k + 1}:",
            result
        )

    shuffled_results = np.array(
        shuffled_results
    )

    print("\nNORMAL VS SHUFFLED INHIBITION")

    for j, label in enumerate(METRIC_LABELS):

        print(f"\n{label}")
        print("Normal:", normal[j])

        print(
            "Shuffled mean:",
            shuffled_results[:, j].mean()
        )

        print(
            "Shuffled SD:",
            shuffled_results[:, j].std()
        )

    return (
        normal,
        shuffled_results,
        shuffled_base
    )


# ============================================================
# 6. COARSE FIRING-RATE MATCH
# ============================================================

def run_coarse_rate_matching(
    shuffled_base,
    evaluate,
    normal,
    scales=COARSE_SCALES
):
    """
    Test candidate inhibitory scaling factors on
    the final shuffled connectivity network.

    This preserves the original five-scale search.
    """

    target_rate = normal[0]

    rate_matching_results = []

    print("\nRATE MATCHING CONTROL")

    for scale in scales:

        modified = scale_shuffled_inhibition(
            shuffled_base,
            scale
        )

        result = evaluate(modified)

        rate_matching_results.append(result)

        print(
            f"Scale {scale:.2f} | "
            f"E rate {result[0]:.4f} Hz | "
            f"Sparseness {result[2]:.4f} | "
            f"Correlation {result[3]:.4f}"
        )

    rate_matching_results = np.array(
        rate_matching_results
    )

    differences = np.abs(
        rate_matching_results[:, 0] - target_rate
    )

    best = np.argmin(differences)

    print("\nCLOSEST RATE MATCH")

    print("Normal rate:", target_rate)
    print("Best inhibitory scale:", scales[best])

    print(
        "Shuffled rate:",
        rate_matching_results[best, 0]
    )

    print("\nNORMAL VS RATE-MATCHED SHUFFLE")

    for j, label in enumerate(METRIC_LABELS):

        print(f"\n{label}")
        print("Normal:", normal[j])

        print(
            "Shuffled:",
            rate_matching_results[best, j]
        )

    return rate_matching_results, int(best)


# ============================================================
# 7. REFINED FIRING-RATE MATCH
# ============================================================

def refine_rate_matching(
    shuffled_base,
    evaluate,
    normal,
    low=REFINEMENT_LOW,
    high=REFINEMENT_HIGH,
    tolerance=RATE_TOLERANCE,
    max_iterations=MAX_REFINEMENT_ITERATIONS
):
    """
    Refine inhibitory scaling using the original
    bisection-search procedure.

    The original method assumes increasing the
    inhibitory scale decreases E firing rate
    sufficiently to bracket the target.

    Uses the same lower and upper bounds and
    stopping criterion.
    """

    target_rate = normal[0]

    def evaluate_at_scale(scale):

        modified = scale_shuffled_inhibition(
            shuffled_base,
            scale
        )

        return evaluate(modified)

    print("\nREFINED FIRING-RATE MATCHING")

    low_result = evaluate_at_scale(low)
    high_result = evaluate_at_scale(high)

    print("Lower bound:", low, low_result[0])
    print("Upper bound:", high, high_result[0])

    if not (
        low_result[0] >= target_rate
        and high_result[0] <= target_rate
    ):
        raise ValueError(
            "Target firing rate is not bracketed. "
            "Adjust low and high."
        )

    best_scale = None
    best_result = None

    for iteration in range(max_iterations):

        mid = (low + high) / 2

        result = evaluate_at_scale(mid)

        print(
            f"Iteration {iteration + 1}: "
            f"Scale={mid:.5f}, "
            f"E rate={result[0]:.5f} Hz"
        )

        best_scale = mid
        best_result = result

        if abs(result[0] - target_rate) <= tolerance:
            break

        if result[0] > target_rate:
            low = mid
        else:
            high = mid

    print("\nFINAL RATE-MATCHED COMPARISON")
    print("Matched scale:", best_scale)

    for j, label in enumerate(METRIC_LABELS):

        print(f"\n{label}")
        print("Normal:", normal[j])
        print("Shuffled:", best_result[j])

        print(
            "Difference:",
            best_result[j] - normal[j]
        )

    return best_scale, best_result


# ============================================================
# 8. COMPLETE EXPERIMENT
# ============================================================

def analyze_shuffled_inhibition(
    source_project,
    learned_run="training_test_25",
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Execute all three original shuffled-inhibition
    controls with matched visual inputs and noise.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser()

    checkpoint_path = (
        source_project
        / "results"
        / "learning"
        / learned_run
        / "batch_1100.npz"
    )

    image_path = (
        source_project
        / "data"
        / "van_hateren_dog.npy"
    )

    # --------------------------------------------------------
    # A. Load the trained network
    # --------------------------------------------------------

    (
        weights,
        threshold_E,
        threshold_I
    ) = load_trained_network(
        checkpoint_path
    )

    print("Loaded checkpoint:", checkpoint_path)

    print(
        "Threshold shapes:",
        threshold_E.shape,
        threshold_I.shape
    )

    # --------------------------------------------------------
    # B. Generate matched LGN input
    # --------------------------------------------------------

    np.random.seed(INPUT_SEED)

    images = np.load(image_path)

    LGN_spikes = generate_lgn_batch(
        images,
        EVALUATION_BATCH_SIZE,
        N_STEPS,
        DT
    )

    expected_shape = (
        N_LGN,
        EVALUATION_BATCH_SIZE,
        N_STEPS
    )

    if LGN_spikes.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}; "
            f"got {LGN_spikes.shape}"
        )

    print("LGN spike shape:", LGN_spikes.shape)

    noise_state = np.random.get_state()

    # --------------------------------------------------------
    # C. Evaluation function sharing identical inputs
    # --------------------------------------------------------

    def evaluate(W):

        return evaluate_network(
            weights=W,
            LGN_spikes=LGN_spikes,
            threshold_E=threshold_E,
            threshold_I=threshold_I,
            noise_state=noise_state
        )

    # --------------------------------------------------------
    # D. Ten shuffled inhibition controls
    # --------------------------------------------------------

    (
        normal,
        shuffled_results,
        shuffled_base
    ) = run_shuffled_controls(
        weights=weights,
        evaluate=evaluate,
        n_shuffles=N_SHUFFLES,
        seed=SHUFFLE_SEED
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_dir / "shuffled_inhibition_control.npz",
        normal=normal,
        shuffled=shuffled_results,
        labels=np.array(METRIC_LABELS)
    )

    # --------------------------------------------------------
    # E. Coarse rate matching
    # --------------------------------------------------------

    (
        rate_matching_results,
        best_index
    ) = run_coarse_rate_matching(
        shuffled_base=shuffled_base,
        evaluate=evaluate,
        normal=normal,
        scales=COARSE_SCALES
    )

    np.savez_compressed(
        output_dir / "rate_matched_shuffled_inhibition.npz",
        scales=COARSE_SCALES,
        results=rate_matching_results,
        best_scale=COARSE_SCALES[best_index],
        normal=normal
    )

    # --------------------------------------------------------
    # F. Refined rate matching
    # --------------------------------------------------------

    best_scale, best_result = refine_rate_matching(
        shuffled_base=shuffled_base,
        evaluate=evaluate,
        normal=normal,
        low=REFINEMENT_LOW,
        high=REFINEMENT_HIGH,
        tolerance=RATE_TOLERANCE,
        max_iterations=MAX_REFINEMENT_ITERATIONS
    )

    np.savez_compressed(
        output_dir / "refined_rate_matched_inhibition.npz",
        normal=normal,
        shuffled=best_result,
        matched_scale=best_scale,
        target_rate=normal[0]
    )

    print("\nShuffled inhibition analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "normal": normal,
        "shuffled_results": shuffled_results,
        "coarse_results": rate_matching_results,
        "coarse_best_index": best_index,
        "refined_best_scale": best_scale,
        "refined_best_result": best_result,
    }


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run shuffled and rate-matched "
            "inhibitory-connectivity controls."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to archived original V1 project"
    )

    parser.add_argument(
        "--learned-run",
        type=str,
        default="training_test_25",
        help="Original learning run directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="New output directory"
    )

    args = parser.parse_args()

    analyze_shuffled_inhibition(
        source_project=args.source_project,
        learned_run=args.learned_run,
        output_dir=args.output_dir
    )
