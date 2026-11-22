
"""
Structural grating-matching analysis across V1 learning.

Replicates the original final grating analysis using
feedforward LGN-to-excitatory synaptic weights.

Checkpoints
-----------
0    : untrained network
100  : learned batch 100
1100 : learned batch 1100

For each excitatory neuron, the analysis:

1. Builds an ON-minus-OFF spatial receptive field.
2. Mean-centers and L2-normalizes that field.
3. Projects it onto orthonormal cosine/sine grating pairs.
4. Measures the projection magnitude.
5. Records the maximum score across all gratings.

This is a structural RF-matching analysis, not a
spiking-network orientation-selectivity experiment.

The original grating definitions, NumPy array
operations, checkpoint ordering and saved field names
are retained to support exact archive comparison.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from config import N_E, N_LGN, PATCH_SIZE


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "final_analysis"
)

N_PIXELS = PATCH_SIZE ** 2

ANGLES = np.arange(0, 180, 15)
FREQUENCIES = np.arange(0.5, 4.5, 0.5)

CHECKPOINTS = (0, 100, 1100)


# ============================================================
# 2. ORIGINAL CHECKPOINT PATHS
# ============================================================

def get_checkpoint_paths(
    source_project,
    learned_run="training_test_25"
):
    """
    Build paths to the original three checkpoints.
    """

    source_project = Path(
        source_project
    ).expanduser().resolve()

    learning_dir = (
        source_project
        / "results"
        / "learning"
        / learned_run
    )

    return {
        0: (
            source_project
            / "results"
            / "untrained"
            / "batch_001"
            / "weights.npz"
        ),
        100: learning_dir / "batch_0100.npz",
        1100: learning_dir / "batch_1100.npz"
    }


# ============================================================
# 3. LOAD WEIGHT-DERIVED RECEPTIVE FIELDS
# ============================================================

def load_receptive_fields(path):
    """
    Load the LGN-to-E matrix and construct signed
    spatial receptive fields.

    ON and OFF channels use Fortran-order reshaping.

    Returns
    -------
    RF : ndarray, shape (N_E, PATCH_SIZE, PATCH_SIZE)
    """

    with np.load(
        path,
        allow_pickle=False
    ) as data:

        W = data["W_LGN_E"].copy()

    expected_shape = (N_E, N_LGN)

    if W.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}; got {W.shape}"
        )

    RF = np.zeros(
        (N_E, PATCH_SIZE, PATCH_SIZE)
    )

    for neuron in range(N_E):

        ON = W[neuron, :N_PIXELS].reshape(
            PATCH_SIZE,
            PATCH_SIZE,
            order="F"
        )

        OFF = W[neuron, N_PIXELS:].reshape(
            PATCH_SIZE,
            PATCH_SIZE,
            order="F"
        )

        RF[neuron] = ON - OFF

    return RF


# ============================================================
# 4. GENERATE SINUSOIDAL GRATINGS
# ============================================================

def create_coordinate_grid():
    """
    Generate the centered coordinate grid.

    Preserves the original meshgrid argument order:
        y, x = np.meshgrid(..., indexing="ij")
    """

    y, x = np.meshgrid(
        np.arange(PATCH_SIZE),
        np.arange(PATCH_SIZE),
        indexing="ij"
    )

    center = (PATCH_SIZE - 1) / 2

    x = x - center
    y = y - center

    return x, y


def create_grating(
    angle,
    phase,
    frequency,
    x,
    y
):
    """
    Generate the original oriented sinusoidal grating.

    Frequency is measured in cycles per patch.
    """

    theta = np.deg2rad(angle)

    coordinate = (
        -x * np.sin(theta)
        + y * np.cos(theta)
    )

    return np.cos(
        2 * np.pi * frequency
        * coordinate / PATCH_SIZE
        + phase
    )


# ============================================================
# 5. BUILD ORTHONORMAL TEMPLATE PAIRS
# ============================================================

def build_template_pairs(
    angles=ANGLES,
    frequencies=FREQUENCIES
):
    """
    Create orthonormal cosine/sine template pairs.

    Preserves:
        frequency-major ordering
        angle-minor ordering
        column mean subtraction
        np.linalg.matrix_rank check
        np.linalg.qr(mode="reduced")

    Returns
    -------
    list of ndarray
        Each template has shape (N_PIXELS, 2).
    """

    x, y = create_coordinate_grid()

    template_pairs = []

    for frequency in frequencies:

        for angle in angles:

            cos_grating = create_grating(
                angle,
                0,
                frequency,
                x,
                y
            )

            sin_grating = create_grating(
                angle,
                np.pi / 2,
                frequency,
                x,
                y
            )

            G_cos = cos_grating.reshape(-1)
            G_sin = sin_grating.reshape(-1)

            G_cos = G_cos - G_cos.mean()
            G_sin = G_sin - G_sin.mean()

            templates = np.column_stack(
                (G_cos, G_sin)
            )

            if np.linalg.matrix_rank(templates) < 2:
                raise ValueError(
                    f"Degenerate grating: "
                    f"{angle}, {frequency}"
                )

            Q, R = np.linalg.qr(
                templates,
                mode="reduced"
            )

            template_pairs.append(Q)

    return template_pairs


# ============================================================
# 6. CALCULATE GRATING-MATCHING SCORES
# ============================================================

def calculate_matching(
    RF,
    template_pairs,
    angles=ANGLES,
    frequencies=FREQUENCIES
):
    """
    Calculate the maximum normalized projection
    magnitude for each excitatory receptive field.

    Returns
    -------
    max_scores : ndarray, shape (N_E,)
    responses : ndarray
        Shape (N_E, len(frequencies), len(angles))

    A zero-norm RF receives zero responses.
    """

    if RF.shape != (
        N_E,
        PATCH_SIZE,
        PATCH_SIZE
    ):
        raise ValueError(
            f"Unexpected RF shape: {RF.shape}"
        )

    if len(template_pairs) != (
        len(frequencies) * len(angles)
    ):
        raise ValueError(
            "Number of grating templates is incorrect."
        )

    flat_RF = RF.reshape(N_E, N_PIXELS)

    # Original spatial mean removal.
    flat_RF = (
        flat_RF
        - flat_RF.mean(axis=1, keepdims=True)
    )

    norms = np.linalg.norm(
        flat_RF,
        axis=1,
        keepdims=True
    )

    normalized_RF = np.divide(
        flat_RF,
        norms,
        out=np.zeros_like(flat_RF),
        where=norms > 0
    )

    responses = np.zeros((
        N_E,
        len(frequencies),
        len(angles)
    ))

    index = 0

    for f_idx in range(len(frequencies)):

        for a_idx in range(len(angles)):

            Q = template_pairs[index]

            projections = normalized_RF @ Q

            responses[:, f_idx, a_idx] = (
                np.linalg.norm(
                    projections,
                    axis=1
                )
            )

            index += 1

    max_scores = responses.max(
        axis=(1, 2)
    )

    return max_scores, responses


# ============================================================
# 7. SUMMARIZE CHECKPOINT SCORES
# ============================================================

def summarize_checkpoint(batch, max_scores):
    """
    Print original descriptive score statistics.
    """

    print(f"\nBATCH {batch}")

    print("Mean:", max_scores.mean())
    print("Median:", np.median(max_scores))
    print("Minimum:", max_scores.min())
    print("Maximum:", max_scores.max())


def compare_learning_changes(scores):
    """
    Calculate per-neuron score changes.

    All three checkpoint comparisons are preserved.
    """

    change_0_100 = (
        scores[100] - scores[0]
    )

    change_100_1100 = (
        scores[1100] - scores[100]
    )

    change_0_1100 = (
        scores[1100] - scores[0]
    )

    print("\nCHANGE FROM BATCH 100 TO 1100")
    print("Mean change:", change_100_1100.mean())

    print(
        "Neurons improved:",
        np.sum(change_100_1100 > 0)
    )

    print(
        "Neurons decreased:",
        np.sum(change_100_1100 < 0)
    )

    print("\nCHANGE FROM BATCH 0 TO 1100")
    print("Mean change:", change_0_1100.mean())

    print(
        "Neurons improved:",
        np.sum(change_0_1100 > 0)
    )

    print(
        "Neurons decreased:",
        np.sum(change_0_1100 < 0)
    )

    return {
        "change_0_100": change_0_100,
        "change_100_1100": change_100_1100,
        "change_0_1100": change_0_1100,
    }


# ============================================================
# 8. PLOT GRATING-MATCHING DISTRIBUTIONS
# ============================================================

def plot_distributions(
    scores,
    save_path,
    show=False
):
    """
    Reproduce original checkpoint distributions.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 31)

    for batch in CHECKPOINTS:

        ax.hist(
            scores[batch],
            bins=bins,
            alpha=0.45,
            label=f"Batch {batch}"
        )

    ax.set_xlabel(
        "Maximum normalized grating-matching score"
    )

    ax.set_ylabel("Number of neurons")

    ax.set_title(
        "Grating Matching During V1 Learning"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 9. PLOT PAIRED CHANGES
# ============================================================

def plot_paired_scores(
    scores,
    save_path,
    show=False
):
    """
    Compare batch 100 and batch 1100 scores.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        scores[100],
        scores[1100],
        alpha=0.6,
        s=18
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Batch 100 matching score")
    ax.set_ylabel("Batch 1100 matching score")

    ax.set_title(
        "Individual Neuron Grating Matching"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# 10. RUN COMPLETE ANALYSIS
# ============================================================

def analyze_grating_matching(
    source_project,
    learned_run="training_test_25",
    output_dir=DEFAULT_OUTPUT_DIR,
    show=False
):
    """
    Run structural grating matching at checkpoints
    0, 100 and 1100.

    The original checkpoint files are read-only.
    """

    paths = get_checkpoint_paths(
        source_project,
        learned_run
    )

    output_dir = Path(
        output_dir
    ).expanduser()

    # Build the same templates for all checkpoints.
    template_pairs = build_template_pairs()

    print("Template pairs:", len(template_pairs))

    scores = {}
    all_responses = {}

    for batch, path in paths.items():

        RF = load_receptive_fields(path)

        max_scores, responses = calculate_matching(
            RF,
            template_pairs
        )

        scores[batch] = max_scores
        all_responses[batch] = responses

        summarize_checkpoint(
            batch,
            max_scores
        )

    changes = compare_learning_changes(
        scores
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve all original archive field names.
    np.savez_compressed(
        output_dir / "final_grating_analysis.npz",

        angles=ANGLES,
        frequencies=FREQUENCIES,

        scores_0=scores[0],
        scores_100=scores[100],
        scores_1100=scores[1100],

        responses_0=all_responses[0],
        responses_100=all_responses[100],
        responses_1100=all_responses[1100]
    )

    plot_distributions(
        scores,
        output_dir / "grating_matching_0_100_1100.png",
        show=show
    )

    plot_paired_scores(
        scores,
        output_dir / "grating_matching_100_vs_1100.png",
        show=show
    )

    print("\nFinal grating analysis completed!")
    print("Results saved to:", output_dir)

    return {
        "scores": scores,
        "responses": all_responses,
        "changes": changes,
    }


# ============================================================
# 11. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analyze structural grating matching across "
            "V1 learning checkpoints."
        )
    )

    parser.add_argument(
        "--source-project",
        required=True,
        type=str,
        help="Path to original V1 project"
    )

    parser.add_argument(
        "--learned-run",
        type=str,
        default="training_test_25"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR)
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )

    args = parser.parse_args()

    analyze_grating_matching(
        source_project=args.source_project,
        learned_run=args.learned_run,
        output_dir=args.output_dir,
        show=args.show
    )
