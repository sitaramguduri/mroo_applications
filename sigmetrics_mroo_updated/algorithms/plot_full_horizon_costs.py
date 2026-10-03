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

SMROO_MAX_FILE = (
    RESULT_DIR
    / "smroo_max_full_horizon_results.csv"
)

SMROO_SUM_FILE = (
    RESULT_DIR
    / "smroo_sum_full_horizon_results.csv"
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

smroo_max = pd.read_csv(
    SMROO_MAX_FILE
)

smroo_sum = pd.read_csv(
    SMROO_SUM_FILE
)


# ============================================================
# IF MROO HAS MULTIPLE CONFIGURATIONS:
#
# use the one with minimum total cost
# ============================================================

mroo_best = (
    mroo
    .sort_values(
        by="total_cost"
    )
    .iloc[0]
)


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Algorithm": [
        "OPT",
        "Greedy",
        "S-MROO-SUM",
        "S-MROO-MAX",
        "MROO",
    ],

    "Hitting Cost": [
        opt.iloc[0]["hitting_cost"],
        greedy.iloc[0]["hitting_cost"],
        smroo_sum.iloc[0]["hitting_cost"],
        smroo_max.iloc[0]["hitting_cost"],
        mroo_best["hitting_cost"],
    ],

    "Long-Term Cost": [
        opt.iloc[0]["long_term_cost"],
        greedy.iloc[0]["long_term_cost"],
        smroo_sum.iloc[0]["long_term_cost"],
        smroo_max.iloc[0]["long_term_cost"],
        mroo_best["long_term_cost"],
    ],

    "Total Cost": [
        opt.iloc[0]["total_cost"],
        greedy.iloc[0]["total_cost"],
        smroo_sum.iloc[0]["total_cost"],
        smroo_max.iloc[0]["total_cost"],
        mroo_best["total_cost"],
    ],
})


print(
    "\nFull-horizon cost comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SAVE TABLE
# ============================================================

COMPARISON_FILE = (
    RESULT_DIR
    / "full_horizon_cost_comparison.csv"
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


# ============================================================
# BAR COLORS
#
# Make MROO visually distinct.
# ============================================================

BAR_COLORS = [
    "gray",        # OPT
    "lightgray",   # Greedy
    "silver",      # S-MROO-SUM
    "darkgray",    # S-MROO-MAX
    "tab:blue",    # MROO
]


# ============================================================
# GENERIC BAR PLOT FUNCTION
# ============================================================

def plot_cost(
    column,
    ylabel,
    title,
    filename,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        comparison["Algorithm"],
        comparison[column],
        color=BAR_COLORS,
    )

    plt.ylabel(
        ylabel
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.title(
        title
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_file = (
        RESULT_DIR
        / filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "Saved:",
        output_file
    )


# ============================================================
# HITTING COST
# ============================================================

plot_cost(
    column="Hitting Cost",
    ylabel="Cumulative Hitting Cost",
    title="Full-Horizon Hitting Cost",
    filename="full_horizon_hitting_cost.png",
)


# ============================================================
# LONG-TERM COST
# ============================================================

plot_cost(
    column="Long-Term Cost",
    ylabel="Cumulative Long-Term Cost",
    title="Full-Horizon Long-Term Cost",
    filename="full_horizon_long_term_cost.png",
)


# ============================================================
# TOTAL COST
# ============================================================

plot_cost(
    column="Total Cost",
    ylabel="Cumulative Total Cost",
    title="Full-Horizon Total Cost",
    filename="full_horizon_total_cost.png",
)