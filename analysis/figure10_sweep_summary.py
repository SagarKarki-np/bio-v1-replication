
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT / "results" / "paper_protocol"
    / "seed_3002" / "figure10_firing_rate_sweep"
    / "paper"
)

OUT = BASE / "summary"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("E01_I02", 1, 2),
    ("E02_I04", 2, 4),
    ("E05_I10", 5, 10),
    ("E10_I20", 10, 20),
    ("E20_I40", 20, 40),
]

rows = []
reference_lgn = None

for name, target_e, target_i in CONDITIONS:

    path = (
        BASE / name / "evaluation"
        / "paper_1000_source"
        / "response_evaluation.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:

        if str(data["condition"]) != name:
            raise ValueError(f"Condition mismatch: {name}")

        if str(data["protocol"]) != "paper_1000_source":
            raise ValueError(f"Protocol mismatch: {name}")

        lgn = data["LGN_counts"]

        if reference_lgn is None:
            reference_lgn = lgn.copy()
            same_lgn = True
        else:
            same_lgn = np.array_equal(
                lgn, reference_lgn
            )

        temporal_e = data["E_temporal"]
        population_e = data["E_population"]

        temporal_i = data["I_temporal"]
        population_i = data["I_population"]

        row = {
            "condition": name,
            "target_e": target_e,
            "target_i": target_i,
            "observed_e": float(data["E_rate"]),
            "observed_i": float(data["I_rate"]),
            "e_temporal": float(
                np.nanmean(temporal_e)
            ),
            "e_population": float(
                np.nanmean(population_e)
            ),
            "i_temporal": float(
                np.nanmean(temporal_i)
            ),
            "i_population": float(
                np.nanmean(population_i)
            ),
            "rf_weight_rms": float(
                data["weight_rf_rms"]
            ),
            "silent_e": int(
                np.count_nonzero(
                    data["E_counts"].sum(axis=0) == 0
                )
            ),
            "silent_i": int(
                np.count_nonzero(
                    data["I_counts"].sum(axis=0) == 0
                )
            ),
            "valid_e_population": int(
                np.isfinite(population_e).sum()
            ),
            "valid_i_population": int(
                np.isfinite(population_i).sum()
            ),
            "same_lgn_counts": same_lgn,
        }

        rows.append(row)

        print(
            name,
            "| E:", round(row["observed_e"], 3),
            "| I:", round(row["observed_i"], 3),
            "| LGN counts match:", same_lgn,
            "| valid E pop:",
            row["valid_e_population"],
            "| valid I pop:",
            row["valid_i_population"]
        )

with open(
    OUT / "figure10_sweep_summary.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

x = np.array([r["target_e"] for r in rows])

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.5)
)

axes[0].plot(
    x, [r["rf_weight_rms"] for r in rows],
    "o-"
)
axes[0].set_title(
    "A: Weight-derived RF diagnostic"
)
axes[0].set_ylabel("RMS correlation")

axes[1].plot(
    x, [r["e_temporal"] for r in rows],
    "o-", label="E"
)
axes[1].plot(
    x, [r["i_temporal"] for r in rows],
    "s-", label="I"
)
axes[1].set_title("B: Temporal sparseness")
axes[1].legend()

axes[2].plot(
    x, [r["e_population"] for r in rows],
    "o-", label="E"
)
axes[2].plot(
    x, [r["i_population"] for r in rows],
    "s-", label="I"
)
axes[2].set_title("C: Population sparseness")
axes[2].legend()

for ax in axes:
    ax.set_xlabel("Target excitatory rate (Hz)")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)

fig.suptitle(
    "Figure 10A-C: Preliminary single-seed sweep"
)

fig.tight_layout()
fig.savefig(
    OUT / "figure10_sweep_preliminary.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSummary saved to:", OUT)
print(
    "All LGN count matrices identical:",
    all(r["same_lgn_counts"] for r in rows)
)
print(
    "NOTE: LGN count equality does not prove "
    "identical spike timing."
)
print(
    "NOTE: RF correlation is weight-derived, "
    "not functional STA."
)
