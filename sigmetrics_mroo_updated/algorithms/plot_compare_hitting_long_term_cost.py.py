from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    ALGORITHM_DIR
    / "new_results"
    / "beta_m_sweep_regime_weights_rho_7"
    / "regime_length_2160_eps_0p1"
    / "beta_m_all_results.csv"
)

PLOT_DIR = (
    RESULT_FILE.parent
    / "plots_beta_over_m"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# KEEP BEST ROWS
# ============================================================

if "selected_best" in results.columns:
    results = results[
        results["selected_best"] == True
    ].copy()


# ============================================================
# CHOOSE ALGORITHM
# ============================================================

ALGORITHM = "MROO"

data = results[
    results["algorithm"] == ALGORITHM
].copy()

data = (
    data
    .sort_values("beta_over_m")
    .reset_index(drop=True)
)


# ============================================================
# PRINT HIT / LONG-TERM BALANCE
# ============================================================

data["long_to_hit_ratio"] = (
    data["long_term_cost"]
    / data["hitting_cost"]
)

data["absolute_difference"] = (
    data["long_term_cost"]
    - data["hitting_cost"]
).abs()


print(
    "\nMROO HITTING VS LONG-TERM COST:"
)

print(
    data[
        [
            "beta_over_m",
            "hitting_cost",
            "long_term_cost",
            "long_to_hit_ratio",
            "absolute_difference",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# GROUPED BAR PLOT
# ============================================================

x = np.arange(
    len(data)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(10, 5.5)
)


bars_hit = ax.bar(
    x - width / 2,
    data["hitting_cost"],
    width,
    label="Hitting Cost",
)


bars_long = ax.bar(
    x + width / 2,
    data["long_term_cost"],
    width,
    label="Long-Term Cost",
)


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        f"{value:g}"
        for value in data["beta_over_m"]
    ]
)


ax.set_xlabel(
    r"$\beta/m$"
)

ax.set_ylabel(
    "Cost"
)

ax.set_title(
    f"{ALGORITHM}: Hitting vs Long-Term Cost"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


output_file = (
    PLOT_DIR
    / "mroo_hitting_vs_long_term_by_beta_over_m.png"
)


fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)


print(
    "\nSaved:",
    output_file
)


plt.close(fig)