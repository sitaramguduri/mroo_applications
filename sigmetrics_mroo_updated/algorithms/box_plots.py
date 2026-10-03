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


GREEDY_FILE = (
    RESULT_DIR
    / "greedy_boxplot_results.csv"
)

SMROO_SUM_FILE = (
    RESULT_DIR
    / "smroo_sum_boxplot_results.csv"
)

SMROO_MAX_FILE = (
    RESULT_DIR
    / "smroo_max_boxplot_results.csv"
)

MROO_FILE = (
    RESULT_DIR
    / "mroo_boxplot_results.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

greedy = pd.read_csv(
    GREEDY_FILE
)

smroo_sum = pd.read_csv(
    SMROO_SUM_FILE
)

smroo_max = pd.read_csv(
    SMROO_MAX_FILE
)

mroo = pd.read_csv(
    MROO_FILE
)


# ============================================================
# USE ONLY FIRST 200 MROO RUNS
# ============================================================

mroo = (
    mroo
    .iloc[:200]
    .copy()
)


# ============================================================
# CHECK NUMBER OF RUNS
# ============================================================

print("Number of runs:")
print("Greedy      :", len(greedy))
print("S-MROO-SUM :", len(smroo_sum))
print("S-MROO-MAX :", len(smroo_max))
print("MROO        :", len(mroo))


# ============================================================
# ALGORITHM ORDER
# ============================================================

algorithm_names = [
    "Greedy",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "MROO",
]


# ============================================================
# GENERIC BOXPLOT FUNCTION
# ============================================================

def plot_cost_boxplot(
    column,
    ylabel,
    title,
    output_name,
):

    data = [
        greedy[column].dropna(),
        smroo_sum[column].dropna(),
        smroo_max[column].dropna(),
        mroo[column].dropna(),
    ]


    plt.figure(
        figsize=(9, 6)
    )


    plt.boxplot(
        data,
        tick_labels=algorithm_names,
        showmeans=True,
        showfliers=True,
    )


    plt.ylabel(
        ylabel,
        fontsize=12
    )

    plt.xlabel(
        "Algorithm",
        fontsize=12
    )

    plt.title(
        title,
        fontsize=14
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    output_file = (
        RESULT_DIR
        / output_name
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
# 1. HITTING COST
# ============================================================

plot_cost_boxplot(
    column="hitting_cost",
    ylabel="Cumulative Hitting Cost",
    title="Hitting Cost Comparison",
    output_name="boxplot_hitting_cost.png",
)


# ============================================================
# 2. LONG-TERM COST
# ============================================================

plot_cost_boxplot(
    column="long_term_cost",
    ylabel="Cumulative Long-Term Cost",
    title="Long-Term Cost Comparison",
    output_name="boxplot_long_term_cost.png",
)


# ============================================================
# 3. TOTAL COST
# ============================================================

plot_cost_boxplot(
    column="total_cost",
    ylabel="Cumulative Total Cost",
    title="Total Cost Comparison",
    output_name="boxplot_total_cost.png",
)
# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = pd.DataFrame({

    "Algorithm": algorithm_names,

    "N": [
        len(greedy),
        len(smroo_sum),
        len(smroo_max),
        len(mroo),
    ],

    "Mean Hitting": [
        greedy["hitting_cost"].mean(),
        smroo_sum["hitting_cost"].mean(),
        smroo_max["hitting_cost"].mean(),
        mroo["hitting_cost"].mean(),
    ],

    "Mean Long-Term": [
        greedy["long_term_cost"].mean(),
        smroo_sum["long_term_cost"].mean(),
        smroo_max["long_term_cost"].mean(),
        mroo["long_term_cost"].mean(),
    ],

    "Mean Total": [
        greedy["total_cost"].mean(),
        smroo_sum["total_cost"].mean(),
        smroo_max["total_cost"].mean(),
        mroo["total_cost"].mean(),
    ],

})


print("\nSummary:")
print(
    summary.to_string(
        index=False
    )
)