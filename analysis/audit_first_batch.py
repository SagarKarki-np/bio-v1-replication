
"""
Compare first-batch outputs across project implementations.

Run this script separately for the original and
cleaned projects. Both use the archived initial weights,
the same image data and random seed.

No files are modified.
"""

from pathlib import Path
import argparse
import hashlib
import sys

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--implementation", required=True)
parser.add_argument("--source", required=True)
args = parser.parse_args()

implementation = Path(args.implementation).resolve()
source = Path(args.source).resolve()

# Import functions from the selected project.
sys.path.insert(0, str(implementation))

from config import (
    BATCH_SIZE, N_STEPS, DT, TAU_M,
    NOISE_STD, V_MIN, V_RESET, N_E, N_I
)
from lgn_spikes import generate_lgn_batch
from simulation import simulate_v1_batch


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def rng_digest():
    state = np.random.get_state()
    payload = (
        state[1].tobytes()
        + repr((state[0], state[2],
                state[3], state[4])).encode()
    )
    return hashlib.sha256(payload).hexdigest()


weight_path = (
    source / "results" / "untrained"
    / "batch_001" / "weights.npz"
)

image_path = (
    source / "data" / "van_hateren_dog.npy"
)

names = (
    "W_LGN_E", "W_LGN_I", "W_E_E",
    "W_E_I", "W_I_I", "W_I_E"
)

with np.load(weight_path, allow_pickle=False) as data:
    weights = tuple(data[k].copy() for k in names)

images = np.load(image_path, mmap_mode="r")

threshold_E = np.full((N_E, 1), 10.0)
threshold_I = np.full((N_I, 1), 10.0)

np.random.seed(42)

print("\nIMPLEMENTATION:", implementation)
print("Source data:", source)
print("Parameters:",
      BATCH_SIZE, N_STEPS, DT, TAU_M,
      NOISE_STD, V_MIN, V_RESET)

print("\nInitial RNG hash:", rng_digest())

# First 100-image batch.
LGN_spikes = generate_lgn_batch(
    images,
    BATCH_SIZE,
    N_STEPS,
    DT
)

print("LGN shape:", LGN_spikes.shape)
print("LGN hash:", digest(LGN_spikes))
print("LGN spike count:", int(LGN_spikes.sum()))
print("RNG hash after LGN:", rng_digest())

E_spikes, I_spikes = simulate_v1_batch(
    LGN_spikes,
    weights,
    DT,
    TAU_M,
    NOISE_STD,
    V_MIN,
    V_RESET,
    threshold_E=threshold_E,
    threshold_I=threshold_I
)

duration = N_STEPS * DT

print("\nE spikes hash:", digest(E_spikes))
print("I spikes hash:", digest(I_spikes))

print(
    "E rate:",
    E_spikes.sum() / (N_E * BATCH_SIZE * duration)
)
print(
    "I rate:",
    I_spikes.sum() / (N_I * BATCH_SIZE * duration)
)

print("RNG hash after V1:", rng_digest())
