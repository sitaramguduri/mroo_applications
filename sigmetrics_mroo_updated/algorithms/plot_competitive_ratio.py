from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
)


# ============================================================
# RESULT FILES
# ============================================================

OPT_FILE = (
    RESULT_DIR
    / "offline_opt_full_horizon_results.csv"
)

GREEDY_FILE = (
    RESULT_DIR
    / "greedy_full_horizon_results.csv"
)

MROO_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_results.csv"
)

SMROO_SUM_FILE = (
    RESULT_DIR
    / "smroo_sum_full_horizon_results.csv"
)

SMROO_MAX_FILE = (
    RESULT_DIR
    / "smroo_max_full_horizon_results.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

RATIO_FILE = (
    RESULT_DIR
    / "full_horizon_competitive_ratios.csv"
)

PLOT_FILE = (
    RESULT_DIR
    / "full_horizon_competitive_ratios.png"
)


# ============================================================
# LOAD RESULTS
# ============================================================

opt = pd.read_csv(
    OPT_FILE
)

greedy = pd.read_csv(
    GREEDY_FILE
)

mroo = pd.read_csv(
    MROO_FILE
)

smroo_sum = pd.read_csv(
    SMROO_SUM_FILE
)

smroo_max = pd.read_csv(
    SMROO_MAX_FILE
)


# ============================================================
# SELECT MROO RESULT
#
# If multiple configurations exist,
# use the one with minimum total cost.
# ============================================================

mroo_best = (
    mroo
    .sort_values(
        by="total_cost"
    )
    .iloc[0]
)


# ============================================================
# OFFLINE OPT TOTAL COST
# ============================================================

opt_total = (
    opt
    .iloc[0]
    ["total_cost"]
)


# ============================================================
# COMPUTE COMPETITIVE RATIOS
# ============================================================

ratio_df = pd.DataFrame({

    "Algorithm": [
        "OPT",
        "Greedy",
        "S-MROO-SUM",
        "S-MROO-MAX",
        "MROO",
    ],

    "Total Cost": [
        opt_total,
        greedy.iloc[0]["total_cost"],
        smroo_sum.iloc[0]["total_cost"],
        smroo_max.iloc[0]["total_cost"],
        mroo_best["total_cost"],
    ],
})


ratio_df[
    "Competitive Ratio"
] = (

    ratio_df[
        "Total Cost"
    ]

    /

    opt_total
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n========================================"
)

print(
    "FULL-HORIZON COMPETITIVE RATIOS"
)

print(
    "========================================"
)

print(
    ratio_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RATIOS
# ============================================================

ratio_df.to_csv(
    RATIO_FILE,
    index=False
)


# ============================================================
# COLORS
#
# MROO gets a distinct color.
# ============================================================

BAR_COLORS = [
    "gray",        # OPT
    "lightgray",   # Greedy
    "silver",      # S-MROO-SUM
    "darkgray",    # S-MROO-MAX
    "tab:blue",    # MROO
]


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    ratio_df[
        "Algorithm"
    ],
    ratio_df[
        "Competitive Ratio"
    ],
    color=BAR_COLORS,
)


# ============================================================
# OFFLINE OPTIMAL REFERENCE LINE
# ============================================================

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.5,
    color="black",
    label="Offline OPT = 1",
)


# ============================================================
# ADD RATIO VALUE ABOVE EACH BAR
# ============================================================

for bar, ratio in zip(
    bars,
    ratio_df[
        "Competitive Ratio"
    ],
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2.0,

        bar.get_height(),

        f"{ratio:.3f}",

        ha="center",
        va="bottom",
        fontsize=11,
    )


# ============================================================
# LABELS
# ============================================================

plt.ylabel(
    "Competitive Ratio"
)

plt.xlabel(
    "Algorithm"
)

plt.title(
    "Full-Horizon Competitive Ratio"
)


plt.grid(
    axis="y",
    alpha=0.3
)

plt.legend()


plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


print(
    "\nSaved ratios to:",
    RATIO_FILE
)

print(
    "Saved plot to:",
    PLOT_FILE
)