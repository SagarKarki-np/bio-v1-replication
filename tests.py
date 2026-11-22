
import numpy as np

from train_v1 import update_homeostatic_thresholds
from analysis.figure10_homeostasis import update_figure10_thresholds

rng = np.random.default_rng(42)

E_spikes = rng.random((400, 8, 400)) < 0.002
I_spikes = rng.random((100, 8, 400)) < 0.004

theta_E = np.full((400, 1), 34.49)
theta_I = np.full((100, 1), 37.47)

old_E, old_I = update_homeostatic_thresholds(
    E_spikes, I_spikes,
    theta_E, theta_I,
    1.0, 1.0
)

new_E, new_I = update_figure10_thresholds(
    E_spikes, I_spikes,
    theta_E, theta_I,
    target_E_Hz=2.0,
    target_I_Hz=4.0,
    eta_theta_E=1.0,
    eta_theta_I=1.0,
    dt=0.001
)

print("E exact:", np.array_equal(old_E, new_E))
print("I exact:", np.array_equal(old_I, new_I))

assert np.array_equal(old_E, new_E)
assert np.array_equal(old_I, new_I)

print("EXACT HOMEOSTASIS MATCH")
