
"""
Figure 10 robustness experiment configuration.

Ruslim et al. (2025)
PLOS Computational Biology
DOI: 10.1371/journal.pcbi.1013644

Purpose
-------
Configure firing-rate robustness experiments without
modifying the validated baseline model.

Two parameterizations are kept separate:

1. matlab_demo:
   Follows the numerical conventions of the released
   MATLAB demonstration and our existing replication.

2. paper:
   Uses the depression and homeostatic parameters
   reported in the published paper.

The Figure 10 firing-rate sweep also requires
learning-rate adjustments. Only explicitly verified
scaling rules are represented here.

No simulation is performed in this module.
"""

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_PROJECT = (
    PROJECT_DIR.parent / "V1_network_learning"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "figure10_robustness"
    / "firing_rate_sweep"
)


# ============================================================
# 2. EXPERIMENT CONDITIONS
# ============================================================

EXCITATORY_TARGETS_HZ = (
    1.0,
    2.0,
    5.0,
    10.0,
    20.0
)

INHIBITORY_RATE_MULTIPLIER = 2.0

BASELINE_TARGET_E_HZ = 2.0


# ============================================================
# 3. CALIBRATION PARAMETERS
# ============================================================

INITIAL_THRESHOLD_E = 10.0
INITIAL_THRESHOLD_I = 10.0

FULL_CALIBRATION_BATCHES = 100
FULL_BATCH_SIZE = 100

PILOT_CALIBRATION_BATCHES = 3
PILOT_BATCH_SIZE = 8

CALIBRATION_SEED = 42

# Synaptic weights are frozen throughout calibration.
CALIBRATION_UPDATES_WEIGHTS = False

CALIBRATION_CHECKPOINT_EVERY = 10


# ============================================================
# 4. PARAMETERIZATION
# ============================================================

@dataclass(frozen=True)
class Parameterization:
    name: str
    homeostatic_eta_E: float
    homeostatic_eta_I: float
    depression_scale: float


PARAMETERIZATIONS = {

    "matlab_demo": Parameterization(
        name="matlab_demo",
        homeostatic_eta_E=1.0,
        homeostatic_eta_I=1.0,
        depression_scale=1.0
    ),

    "paper": Parameterization(
        name="paper",
        homeostatic_eta_E=0.1,
        homeostatic_eta_I=0.1,
        depression_scale=0.05
    )
}


# ============================================================
# 5. EXPERIMENT CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FiringRateCondition:

    target_E_Hz: float
    target_I_Hz: float

    parameterization: str

    eta_threshold_E: float
    eta_threshold_I: float

    triplet_potentiation: float
    triplet_depression: float

    lgn_to_e_lr_multiplier: float
    e_to_e_lr_multiplier: float

    @property
    def condition_name(self):

        e = int(self.target_E_Hz)
        i = int(self.target_I_Hz)

        return f"E{e:02d}_I{i:02d}"


# ============================================================
# 6. CONSTRUCT ONE CONDITION
# ============================================================

def make_condition(
    target_E_Hz,
    parameterization="matlab_demo"
):
    """
    Construct a firing-rate sweep condition.

    The inhibitory target is twice the E target.

    The depression coefficient is coupled to the
    excitatory target according to the selected
    parameterization.

    Learning-rate multipliers implement only the
    two scaling relationships explicitly verified
    from the published Figure 10 methods.
    """

    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {parameterization}"
        )

    if target_E_Hz <= 0:
        raise ValueError(
            "Target firing rate must be positive."
        )

    params = PARAMETERIZATIONS[parameterization]

    target_I_Hz = (
        target_E_Hz * INHIBITORY_RATE_MULTIPLIER
    )

    triplet_depression = (
        target_E_Hz * params.depression_scale
    )

    rate_ratio = (
        BASELINE_TARGET_E_HZ / target_E_Hz
    )

    return FiringRateCondition(
        target_E_Hz=float(target_E_Hz),
        target_I_Hz=float(target_I_Hz),

        parameterization=parameterization,

        eta_threshold_E=params.homeostatic_eta_E,
        eta_threshold_I=params.homeostatic_eta_I,

        triplet_potentiation=1.0,
        triplet_depression=triplet_depression,

        lgn_to_e_lr_multiplier=rate_ratio,
        e_to_e_lr_multiplier=rate_ratio ** 2
    )


# ============================================================
# 7. GENERATE COMPLETE SWEEP
# ============================================================

def make_firing_rate_sweep(
    parameterization="matlab_demo"
):
    """
    Return all five firing-rate conditions.
    """

    return [
        make_condition(rate, parameterization)
        for rate in EXCITATORY_TARGETS_HZ
    ]


# ============================================================
# 8. OUTPUT PATHS
# ============================================================

def condition_output_dir(condition):
    """
    Return a unique directory for one condition.

    MATLAB-demo and paper configurations are
    never written to the same directory.
    """

    return (
        OUTPUT_DIR
        / condition.parameterization
        / condition.condition_name
    )


# ============================================================
# 9. CONFIGURATION DIAGNOSTICS
# ============================================================

if __name__ == "__main__":

    for parameterization in PARAMETERIZATIONS:

        print()
        print("=" * 70)
        print("PARAMETERIZATION:", parameterization)
        print("=" * 70)

        for condition in make_firing_rate_sweep(
            parameterization
        ):

            print(
                f"{condition.condition_name} | "
                f"E={condition.target_E_Hz:.1f} Hz | "
                f"I={condition.target_I_Hz:.1f} Hz | "
                f"A_d={condition.triplet_depression:.3f} | "
                f"LGN->E LR x"
                f"{condition.lgn_to_e_lr_multiplier:.3f} | "
                f"E->E LR x"
                f"{condition.e_to_e_lr_multiplier:.3f}"
            )

        print()
        print("Threshold learning rates:")
        params = PARAMETERIZATIONS[parameterization]

        print("E:", params.homeostatic_eta_E)
        print("I:", params.homeostatic_eta_I)
