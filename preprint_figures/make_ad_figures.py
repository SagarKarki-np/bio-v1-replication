"""Figures/tables for the A_d experiment. Reads existing results only; writes to preprint_figures/."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/paper_protocol/seed_3002/functional_sta")
OUT = Path("preprint_figures")
RUNS = [  # label, folder, colour
    ("Track A, A_d = 0.1 (original)", "track_a", "#2a78d6"),
    ("Track A, A_d = 2.0", "track_a_Ad2", "#eb6834"),
    ("Track B, A_d = 0.1 (converged calibration)", "track_b", "#52514e"),
]
THS = [0.08, 0.10, 0.12, 0.15]
PUBLISHED = 369

def load(folder):
    z = np.load(BASE / folder / "full_image_gabor_160" / "gabor_160_comparison_400.npz", allow_pickle=False)
    return z["refitted_errors_160"], z["original_errors_16"]

data = {f: load(f) for _, f, _ in RUNS}

# ---- table: acceptance vs error threshold ----
with open(OUT / "gabor_acceptance_vs_threshold.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["condition", "resolution"] + [f"error<{t}" for t in THS])
    for lab, f, _ in RUNS:
        e160, e16 = data[f]
        w.writerow([lab, "160x160"] + [int((e160 < t).sum()) for t in THS])
        w.writerow([lab, "16x16"] + [int((e16 < t).sum()) for t in THS])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---- Figure A: bar chart of accepted fits (160x160, error < 0.10) ----
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)
vals = [int((data[f][0] < 0.10).sum()) for _, f, _ in RUNS]
short = ["Track A\nA_d = 0.1", "Track A\nA_d = 2.0", "Track B\nA_d = 0.1"]
bars = ax.bar(range(3), vals, color=[c for _, _, c in RUNS], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v}", ha="center", va="bottom", fontsize=10)
ax.axhline(PUBLISHED, color="#52514e", ls="--", lw=1)
ax.text(2.45, PUBLISHED + 6, "published: 369", ha="right", va="bottom", fontsize=9, color="#52514e")
ax.set_xticks(range(3)); ax.set_xticklabels(short)
ax.set_ylim(0, 420); ax.set_ylabel("Accepted Gabor fits (of 400)")
ax.set_title("Gabor acceptance, 160×160 refit, error < 0.10 (seed 3002, one run)", fontsize=10, loc="left")
fig.text(0.01, 0.01, "Triplet-literal variant (A_p = 0.001, A_d = 0.002; symmetric pathways unchanged):\nLGN→E and E→E weights changed ~2%; STA not analysed.", fontsize=7.2, color="#52514e")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(OUT / "fig_ad_acceptance_bars.png"); plt.close(fig)

# ---- Figure B: CDF of fit errors ----
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)
for lab, f, c in RUNS:
    e = np.sort(data[f][0])
    ax.plot(e, np.arange(1, len(e) + 1), color=c, lw=2, label=lab)
ax.axvline(0.10, color="#52514e", ls=":", lw=1)
ax.set_xlim(0, 0.3); ax.set_ylim(0, 400)
ax.set_xlabel("Gabor fit error (160×160 refit)"); ax.set_ylabel("Neurons with error ≤ x")
ax.set_title("Cumulative distribution of fit errors", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_ad_error_cdf.png"); plt.close(fig)
print("done", vals)
