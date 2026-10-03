import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ALGORITHM_DIR.parent

RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "beta_m_sweep_theoretical_lambda"
)

MASTER_FILE = (
    RESULT_DIR
    / "beta_m_all_results.csv"
)

PLOT_DIR = (
    RESULT_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(MASTER_FILE)

print("\nLoaded:", MASTER_FILE)
print("Rows:", len(df))


# ============================================================
# KEEP ALGORITHMS IN FIXED ORDER
# ============================================================

ALGORITHM_ORDER = [
    "OPT",
    "Greedy",
    "MROO",
    "S-MROO-SUM",
    "S-MROO-MAX",
]

df["algorithm"] = pd.Categorical(
    df["algorithm"],
    categories=ALGORITHM_ORDER,
    ordered=True,
)

df = df.sort_values(
    by=["beta", "m", "algorithm"]
).reset_index(drop=True)


# ============================================================
# SCENARIO LABELS
# ============================================================

settings = (
    df[["beta", "m", "beta_over_m"]]
    .drop_duplicates()
    .sort_values(by=["beta", "m"])
    .reset_index(drop=True)
)

settings["label"] = settings.apply(
    lambda row: (
        rf"$\beta$={int(row['beta'])}, "
        rf"$m$={int(row['m'])}"
    ),
    axis=1,
)

scenario_labels = settings["label"].tolist()


# ============================================================
# MERGE LABELS INTO MAIN DF
# ============================================================

df = df.merge(
    settings[["beta", "m", "label"]],
    on=["beta", "m"],
    how="left",
)


# ============================================================
# IF cost_ratio_to_opt IS MISSING, COMPUTE IT
# ============================================================

if "cost_ratio_to_opt" not in df.columns:
    opt_costs = (
        df[df["algorithm"] == "OPT"]
        .set_index(["beta", "m"])["total_cost"]
        .to_dict()
    )

    df["cost_ratio_to_opt"] = df.apply(
        lambda row: row["total_cost"] / opt_costs[(row["beta"], row["m"])],
        axis=1,
    )


# ============================================================
# PLOT 1: GROUPED BAR PLOTS FOR COSTS
# ============================================================

def plot_grouped_cost(
    cost_column,
    title,
    ylabel,
    filename,
):
    plot_df = (
        df.pivot_table(
            index="label",
            columns="algorithm",
            values=cost_column,
            aggfunc="first",
        )
        .reindex(index=scenario_labels)
        .reindex(columns=ALGORITHM_ORDER)
    )

    x = np.arange(len(plot_df.index))
    n_algs = len(ALGORITHM_ORDER)
    width = 0.15

    plt.figure(figsize=(18, 7))

    for i, alg in enumerate(ALGORITHM_ORDER):
        values = plot_df[alg].to_numpy(dtype=float)
        offset = (i - (n_algs - 1) / 2.0) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=alg,
        )

    plt.xticks(
        x,
        plot_df.index,
        rotation=45,
        ha="right",
    )

    plt.ylabel(ylabel)
    plt.xlabel(r"Scenario $(\beta, m)$")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_file = PLOT_DIR / filename
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Saved:", out_file)


plot_grouped_cost(
    cost_column="hitting_cost",
    title="Theoretical Lambda: Hitting Cost Comparison",
    ylabel="Cumulative Hitting Cost",
    filename="theoretical_hitting_cost.png",
)

plot_grouped_cost(
    cost_column="long_term_cost",
    title="Theoretical Lambda: Long-Term Cost Comparison",
    ylabel="Cumulative Long-Term Cost",
    filename="theoretical_long_term_cost.png",
)

plot_grouped_cost(
    cost_column="total_cost",
    title="Theoretical Lambda: Total Cost Comparison",
    ylabel="Cumulative Total Cost",
    filename="theoretical_total_cost.png",
)


# ============================================================
# PLOT 2: COMPETITIVE RATIO GROUPED BAR PLOT
# ============================================================

def plot_ratio_comparison():
    ratio_df = (
        df.pivot_table(
            index="label",
            columns="algorithm",
            values="cost_ratio_to_opt",
            aggfunc="first",
        )
        .reindex(index=scenario_labels)
        .reindex(columns=ALGORITHM_ORDER)
    )

    x = np.arange(len(ratio_df.index))
    n_algs = len(ALGORITHM_ORDER)
    width = 0.15

    plt.figure(figsize=(18, 7))

    for i, alg in enumerate(ALGORITHM_ORDER):
        values = ratio_df[alg].to_numpy(dtype=float)
        offset = (i - (n_algs - 1) / 2.0) * width

        plt.bar(
            x + offset,
            values,
            width=width,
            label=alg,
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="OPT ratio = 1",
    )

    plt.xticks(
        x,
        ratio_df.index,
        rotation=45,
        ha="right",
    )

    plt.ylabel("Competitive Ratio / Cost Ratio to OPT")
    plt.xlabel(r"Scenario $(\beta, m)$")
    plt.title("Theoretical Lambda: Competitive Ratio Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_file = PLOT_DIR / "theoretical_competitive_ratio.png"
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Saved:", out_file)


plot_ratio_comparison()


# ============================================================
# HELPER: BUILD MATRIX FOR A GIVEN ALGORITHM
# ============================================================

def build_matrix(metric_column, algorithm_name):
    subset = (
        df[df["algorithm"] == algorithm_name]
        .pivot_table(
            index="beta",
            columns="m",
            values=metric_column,
            aggfunc="first",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    return subset


# ============================================================
# PLOT 3: HEATMAP OF METRIC FOR ONE ALGORITHM
# ============================================================

def plot_heatmap(
    matrix,
    title,
    filename,
    cmap="viridis",
    value_fmt=".3f",
):
    beta_vals = matrix.index.to_list()
    m_vals = matrix.columns.to_list()
    values = matrix.to_numpy(dtype=float)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )

    plt.colorbar(im)

    plt.xticks(
        np.arange(len(m_vals)),
        [str(int(x)) for x in m_vals],
    )
    plt.yticks(
        np.arange(len(beta_vals)),
        [str(int(x)) for x in beta_vals],
    )

    plt.xlabel(r"$m$")
    plt.ylabel(r"$\beta$")
    plt.title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(
                j,
                i,
                format(values[i, j], value_fmt),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(values[i, j]) > np.nanmax(np.abs(values)) * 0.5 else "black",
            )

    plt.tight_layout()

    out_file = PLOT_DIR / filename
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Saved:", out_file)


# ============================================================
# HEATMAPS OF TOTAL COST RATIO FOR EACH ONLINE ALGORITHM
# ============================================================

for alg in ["MROO", "S-MROO-SUM", "S-MROO-MAX", "Greedy"]:
    mat = build_matrix(
        metric_column="cost_ratio_to_opt",
        algorithm_name=alg,
    )

    plot_heatmap(
        matrix=mat,
        title=f"{alg}: Cost Ratio to OPT (Theoretical Lambda)",
        filename=f"heatmap_ratio_{alg.lower().replace('-', '_')}.png",
        cmap="viridis",
        value_fmt=".3f",
    )


# ============================================================
# PLOT 4: REGION PLOTS
#
# Positive value means MROO is better.
# Negative value means S-MROO is better.
# ============================================================

def plot_region_where_mroo_is_better(
    smroo_name,
    filename,
):
    mroo_mat = build_matrix(
        metric_column="cost_ratio_to_opt",
        algorithm_name="MROO",
    )

    smroo_mat = build_matrix(
        metric_column="cost_ratio_to_opt",
        algorithm_name=smroo_name,
    )

    # positive => MROO better
    region_mat = smroo_mat - mroo_mat

    beta_vals = region_mat.index.to_list()
    m_vals = region_mat.columns.to_list()
    values = region_mat.to_numpy(dtype=float)

    vmax = np.nanmax(np.abs(values))
    vmin = -vmax

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im)
    cbar.set_label(
        f"{smroo_name} ratio - MROO ratio"
    )

    plt.xticks(
        np.arange(len(m_vals)),
        [str(int(x)) for x in m_vals],
    )
    plt.yticks(
        np.arange(len(beta_vals)),
        [str(int(x)) for x in beta_vals],
    )

    plt.xlabel(r"$m$")
    plt.ylabel(r"$\beta$")
    plt.title(
        f"Region Plot: Where MROO Beats {smroo_name}"
    )

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]

            if val > 0:
                label = f"+{val:.3f}"
            else:
                label = f"{val:.3f}"

            plt.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    plt.tight_layout()

    out_file = PLOT_DIR / filename
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("Saved:", out_file)

    # also print actual region table
    print(f"\nWhere MROO beats {smroo_name}:")
    region_df = region_mat.copy()

    better_points = []
    for beta in region_df.index:
        for m in region_df.columns:
            delta = region_df.loc[beta, m]
            if delta > 0:
                better_points.append((beta, m, delta))

    if len(better_points) == 0:
        print("  No tested region where MROO is better.")
    else:
        for beta, m, delta in better_points:
            print(
                f"  beta={beta}, m={m}, "
                f"improvement={delta:.6f}"
            )


plot_region_where_mroo_is_better(
    smroo_name="S-MROO-SUM",
    filename="region_mroo_vs_smroo_sum.png",
)

plot_region_where_mroo_is_better(
    smroo_name="S-MROO-MAX",
    filename="region_mroo_vs_smroo_max.png",
)