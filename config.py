
"""
Configuration for the LGN-V1 spiking neural network.

Reference
---------
Ruslim et al. (2025), PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1013644

Weight convention
-----------------
W_POST_PRE[postsynaptic_neuron, presynaptic_neuron]

All synaptic weight matrices use this convention.

Time constants and simulation durations are expressed in seconds.
Firing rates are expressed in Hz.
"""

# ============================================================
# 1. NETWORK ARCHITECTURE
# ============================================================

PATCH_SIZE = 16

N_LGN_ON = PATCH_SIZE ** 2
N_LGN_OFF = PATCH_SIZE ** 2
N_LGN = N_LGN_ON + N_LGN_OFF

N_E = 400
N_I = 100


# ============================================================
# 2. SIMULATION PARAMETERS
# ============================================================

DT = 0.001
PRESENTATION_DURATION = 0.4

N_STEPS = int(round(PRESENTATION_DURATION / DT))
BATCH_SIZE = 100

TAU_M = 0.010

V_RESET = 0.0
V_MIN = -10.0
NOISE_STD = 1.0

INITIAL_THRESHOLD_E = 10.0
INITIAL_THRESHOLD_I = 10.0


# ============================================================
# 3. HOMEOSTASIS
# ============================================================

TARGET_RATE_E = 2.0
TARGET_RATE_I = 4.0

ETA_THRESHOLD_E = 1.0
ETA_THRESHOLD_I = 1.0


# ============================================================
# 4. STDP PARAMETERS
# ============================================================

TAU_STDP_SHORT = 0.020
TAU_STDP_LONG = 0.050

TRIPLET_POTENTIATION = 1.0
TRIPLET_DEPRESSION = TARGET_RATE_E

SYMMETRIC_STDP_COEFFICIENT = 0.5


# ============================================================
# 5. SYNAPTIC LEARNING RATES
# ============================================================

# Naming convention: SOURCE_TO_TARGET

ETA_LGN_TO_E = 0.0002
ETA_LGN_TO_I = 0.003

ETA_E_TO_E = 0.0001
ETA_E_TO_I = 0.04

ETA_I_TO_E = 0.09
ETA_I_TO_I = 0.06


# ============================================================
# 6. INCOMING L1 WEIGHT LIMITS
# ============================================================

# L1 normalization is applied to incoming synaptic weights
# for each postsynaptic neuron.

L1_LGN_TO_E = 100.0
L1_LGN_TO_I = 80.0

L1_E_TO_E = 10.0
L1_E_TO_I = 240.0

L1_I_TO_E = 120.0
L1_I_TO_I = 120.0


# ============================================================
# 7. WEIGHT INITIALIZATION
# ============================================================

INITIAL_WEIGHT_MEAN = 1.0
INITIAL_WEIGHT_STD = 0.5

STRONG_CONNECTION_PROBABILITY = 0.2
WEAK_CONNECTION_DIVISOR = 100.0


# ============================================================
# 8. LGN PREPROCESSING
# ============================================================

CENTER_SIGMA = 1.0
SURROUND_SIGMA = 1.5
DIVISIVE_SIGMA = 1.5

LGN_RATE_SCALE = 70.0
LGN_MAX_RATE = 100.0


# ============================================================
# 9. TRAINING CONFIGURATION
# ============================================================

CALIBRATION_BATCHES = 100

# Published training description; exact interpretation
# differs from the released MATLAB demo loop.
MAIN_LEARNING_BATCHES = 1200

# MATLAB demo executes indices 101 through 1200,
# corresponding to 1100 main-learning batches.
DEMO_MAIN_LEARNING_BATCHES = 1100

DEMO_RATE_HALVING_INDICES = (400, 800)


# ============================================================
# 10. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42
