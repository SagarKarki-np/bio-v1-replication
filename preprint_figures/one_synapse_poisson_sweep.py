"""
One-synapse Poisson sweep: where do triplet-STDP LTP and LTD balance for the three A_p/A_d rules?
Same trace convention as plasticity.update_stdp_timestep (decay -> weight change -> add S/tau),
1 ms steps, 0.4 s presentations, traces reset to zero at each presentation start, pre rate 20 Hz,
independent Bernoulli post spikes. Weight-change terms are the ones in plasticity.triplet_stdp_matrix
(checked against the repository function below).

Usage (from project root):
  python preprint_figures/one_synapse_poisson_sweep.py run  <post_Hz> <n_presentations>   # appends a part file
  python preprint_figures/one_synapse_poisson_sweep.py merge                                # builds the final CSV + table
LTP needs >= 2 post spikes in a window, so low post rates need many presentations (rare-event sampling).
"""
import csv, sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(".").resolve()))
from plasticity import triplet_stdp_matrix

OUT = Path("preprint_figures")
DT, DUR = 0.001, 0.4
STEPS = int(round(DUR / DT)); TS, TL, PRE = 0.020, 0.050, 20.0
RULES = {"demo (A_p=1, A_d=2.0)": (1.0, 2.0),
         "paper-literal, 1/tau traces (A_p=0.001, A_d=0.002)": (0.001, 0.002),
         "hybrid used in paper branch (A_p=1, A_d=0.1)": (1.0, 0.1)}
ds, dl = np.exp(-DT / TS), np.exp(-DT / TL)

def check_against_repo():
    r = np.random.default_rng(1); a = [r.random((1, 7)) for _ in range(5)]
    assert np.isclose(triplet_stdp_matrix(*a, 1.0, 0.0)[0, 0], float(((a[1] * a[4]) * a[2]).sum()))
    assert np.isclose(-triplet_stdp_matrix(*a, 0.0, 1.0)[0, 0], float((a[3] * a[0]).sum()))

def simulate(nu, B, seed):
    rng = np.random.default_rng(seed)
    ltp = np.zeros(B); ltd = np.zeros(B)
    Upre = np.zeros(B); Ups = np.zeros(B); Upl = np.zeros(B)
    for _ in range(STEPS):
        Upre *= ds; Ups *= ds; Upl *= dl
        Sp = rng.random(B) < PRE * DT; So = rng.random(B) < nu * DT
        ltp += So * Upl * Upre; ltd += Ups * Sp
        Upre += Sp / TS; Ups += So / TS; Upl += So / TL
    return ltp, ltd

if sys.argv[1] == "run":
    check_against_repo()
    nu, B = float(sys.argv[2]), int(sys.argv[3])
    ltp, ltd = simulate(nu, B, seed=int(round(nu * 1000)) + 7)
    part = {"nu": nu, "B": B, "ltp": ltp.mean() / DUR, "ltp_se": ltp.std(ddof=1) / np.sqrt(B) / DUR,
            "ltd": ltd.mean() / DUR, "ltd_se": ltd.std(ddof=1) / np.sqrt(B) / DUR}
    (OUT / f"one_synapse_part_{nu}.json").write_text(json.dumps(part))
    print(part)
else:
    parts = sorted((json.loads(p.read_text()) for p in OUT.glob("one_synapse_part_*.json")), key=lambda d: d["nu"])
    rows = []
    for d in parts:
        for name, (Ap, Ad) in RULES.items():
            P, D = Ap * d["ltp"], Ad * d["ltd"]
            rows.append([name, d["nu"], d["B"], P, Ap * d["ltp_se"], D, Ad * d["ltd_se"], P / D, d["nu"] / (P / D)])
    with open(OUT / "one_synapse_poisson_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rule", "post_Hz", "n_presentations", "LTP_per_s", "LTP_se", "LTD_per_s", "LTD_se", "LTP/LTD", "implied_balance_Hz"])
        w.writerows(rows)
    for name, (Ap, Ad) in RULES.items():
        print("\n" + name, "| nominal balance A_d/A_p =", Ad / Ap, "Hz")
        print(f"{'post Hz':>8} {'LTP/s':>10} {'LTD/s':>10} {'LTP/LTD':>8} {'implied nu*':>12}")
        for r_ in rows:
            if r_[0] == name: print(f"{r_[1]:8.2f} {r_[3]:10.4g} {r_[5]:10.4g} {r_[7]:8.3f} {r_[8]:12.4f}")
