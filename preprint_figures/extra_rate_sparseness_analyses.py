"""
Read-only extra analyses on the saved 100-presentation evaluations (Fig. 10 paper sweep, seed 3002).
 1. Presentation-level bootstrap CIs on mean E and I rates
 2. Poisson floor for Treves-Rolls sparseness (independent Poisson counts, same per-neuron mean count)
 3. Silent-presentation sensitivity (exclude / 0 / 1) for population sparseness
Uncorrected Treves-Rolls, as in the manuscript. Writes CSV only to preprint_figures/.
"""
import csv
from pathlib import Path
import numpy as np

BASE = Path("results/paper_protocol/seed_3002/figure10_firing_rate_sweep/paper")
OUT = Path("preprint_figures")
CONDS = ["E01_I02", "E02_I04", "E05_I10", "E10_I20", "E20_I40"]
NBOOT, NSIM = 10000, 2000
rng = np.random.default_rng(20261122)

def tr(x, axis):
    x = np.asarray(x, float)
    m = x.mean(axis=axis); m2 = (x * x).mean(axis=axis)
    out = np.full(m.shape, np.nan)
    ok = m2 > 0
    out[ok] = 1 - m[ok] ** 2 / m2[ok]
    return out

def ci(a):
    return np.percentile(a, [2.5, 97.5])

rows_rate, rows_sp, rows_silent = [], [], []
for c in CONDS:
    z = np.load(BASE / c / "evaluation" / "paper_1000_source" / "response_evaluation.npz", allow_pickle=False)
    dur = float(z["duration"])
    for pop, key in (("E", "E_counts"), ("I", "I_counts")):
        X = z[key].astype(float)            # neurons x presentations
        n, p = X.shape
        per_pres = X.mean(axis=0) / dur     # mean rate in each presentation (Hz)
        mean_rate = per_pres.mean()
        boots = np.array([per_pres[rng.integers(0, p, p)].mean() for _ in range(NBOOT)])
        lo, hi = ci(boots)
        rows_rate.append([c, pop, round(mean_rate, 3), round(lo, 3), round(hi, 3)])

        # observed sparseness
        T = tr(X, 1); P = tr(X, 0)
        obs_T, obs_P = np.nanmean(T), np.nanmean(P)
        # bootstrap: temporal over neurons, population over presentations
        bT = np.array([np.nanmean(T[rng.integers(0, n, n)]) for _ in range(NBOOT)])
        bP = np.array([np.nanmean(P[rng.integers(0, p, p)]) for _ in range(NBOOT)])
        # Poisson floor: independent Poisson counts, per-neuron mean count preserved
        lam = X.mean(axis=1, keepdims=True)
        simT, simP = [], []
        for _ in range(NSIM):
            S = rng.poisson(np.repeat(lam, p, axis=1)).astype(float)
            simT.append(np.nanmean(tr(S, 1))); simP.append(np.nanmean(tr(S, 0)))
        simT, simP = np.array(simT), np.array(simP)
        rows_sp.append([c, pop, round(obs_T, 4), *np.round(ci(bT), 4), round(simT.mean(), 4), *np.round(ci(simT), 4),
                        round(obs_P, 4), *np.round(ci(bP), 4), round(simP.mean(), 4), *np.round(ci(simP), 4)])

        # silent presentations (population sparseness undefined)
        und = np.isnan(P)
        rows_silent.append([c, pop, int(und.sum()), round(np.nanmean(P), 4),
                            round(np.mean(np.nan_to_num(P, nan=0.0)), 4), round(np.mean(np.nan_to_num(P, nan=1.0)), 4)])

def write(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

write("rates_bootstrap_ci.csv", ["condition", "pop", "mean_rate_Hz", "ci_lo", "ci_hi"], rows_rate)
write("sparseness_poisson_floor.csv",
      ["condition", "pop", "temporal_obs", "T_ci_lo", "T_ci_hi", "T_poisson_mean", "T_pois_lo", "T_pois_hi",
       "population_obs", "P_ci_lo", "P_ci_hi", "P_poisson_mean", "P_pois_lo", "P_pois_hi"], rows_sp)
write("silent_presentations.csv", ["condition", "pop", "n_undefined", "P_exclude", "P_zero", "P_one"], rows_silent)

print("RATES"); [print(r) for r in rows_rate]
print("SPARSENESS (obs, CI | Poisson floor mean, 95% range) temporal then population")
for r in rows_sp: print(r[0], r[1], "T", r[2], r[3:5], "|", r[5], r[6:8], " P", r[8], r[9:11], "|", r[11], r[12:14])
print("SILENT"); [print(r) for r in rows_silent]
