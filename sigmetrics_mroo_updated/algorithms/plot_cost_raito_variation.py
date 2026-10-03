from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    RESULT_FILE
)

print("Loaded:", RESULT_FILE)


# ============================================================
# FILTER BEST CONFIGURATIONS
# ============================================================

if "selected_best" in df.columns:
    df = df[
        df["selected_best"] == True
    ].copy()


df = df[
    df["weight_mode"] == "regime_switching"
].copy()

df = df[
    df["regime_length"] == 2160
].copy()

df = df[
    df["regime_eps"] == 0.1
].copy()


# ============================================================
# ALGORITHM ORDER
# ============================================================

ALGORITHM_ORDER = [
    "Greedy",
    "MROO",
    "S-MROO-MAX",
    "S-MROO-SUM",
    "OPT",
]


# ============================================================
# HELPER
# ============================================================

def make_pivot(value_column):

    pivot = (
        df.pivot_table(
            index="beta_over_m",
            columns="algorithm",
            values=value_column,
            aggfunc="first",
        )
    )

    pivot = pivot.reindex(
        columns=ALGORITHM_ORDER
    )

    pivot = pivot.sort_index()

    return pivot


# ============================================================
# PLOT 1:
# COST / OPT VS beta/m
# ============================================================

ratio_pivot = make_pivot(
    "cost_ratio_to_opt"
)


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


for alg in ALGORITHM_ORDER:

    if alg not in ratio_pivot.columns:
        continue

    y = ratio_pivot[alg]

    if y.notna().sum() == 0:
        continue

    ax.plot(
        ratio_pivot.index,
        y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=alg,
    )


ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.0,
    color="black",
)


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"$\beta/m$"
)

ax.set_ylabel(
    "Cost / OPT"
)

ax.set_title(
    r"Cost Ratio to OPT vs. $\beta/m$"
)

ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()


output_file = (
    PLOT_DIR
    / "cost_ratio_vs_beta_over_m.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved:",
    output_file,
)

plt.close(fig)


# ============================================================
# PLOT 2:
# HITTING COST VS beta/m
# ============================================================

hitting_pivot = make_pivot(
    "hitting_cost"
)


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


for alg in ALGORITHM_ORDER:

    if alg not in hitting_pivot.columns:
        continue

    y = hitting_pivot[alg]

    if y.notna().sum() == 0:
        continue

    ax.plot(
        hitting_pivot.index,
        y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=alg,
    )


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"$\beta/m$"
)

ax.set_ylabel(
    "Hitting Cost"
)

ax.set_title(
    r"Hitting Cost vs. $\beta/m$"
)

ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()


output_file = (
    PLOT_DIR
    / "hitting_cost_vs_beta_over_m.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved:",
    output_file,
)

plt.close(fig)


# ============================================================
# PLOT 3:
# LONG-TERM COST VS beta/m
# ============================================================

long_term_pivot = make_pivot(
    "long_term_cost"
)


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


for alg in ALGORITHM_ORDER:

    if alg not in long_term_pivot.columns:
        continue

    y = long_term_pivot[alg]

    if y.notna().sum() == 0:
        continue

    ax.plot(
        long_term_pivot.index,
        y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=alg,
    )


ax.set_xscale(
    "log"
)

# Long-term cost spans several orders of magnitude,
# so log y-scale is much easier to read.
ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$\beta/m$"
)

ax.set_ylabel(
    "Long-Term Cost"
)

ax.set_title(
    r"Long-Term Cost vs. $\beta/m$"
)

ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()


output_file = (
    PLOT_DIR
    / "long_term_cost_vs_beta_over_m.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved:",
    output_file,
)

plt.close(fig)


# ============================================================
# PLOT 4:
# TOTAL COST VS beta/m
# ============================================================

total_pivot = make_pivot(
    "total_cost"
)


fig, ax = plt.subplots(
    figsize=(9, 5.5)
)


for alg in ALGORITHM_ORDER:

    if alg not in total_pivot.columns:
        continue

    y = total_pivot[alg]

    if y.notna().sum() == 0:
        continue

    ax.plot(
        total_pivot.index,
        y,
        marker="o",
        linewidth=2,
        markersize=6,
        label=alg,
    )


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$\beta/m$"
)

ax.set_ylabel(
    "Total Cost"
)

ax.set_title(
    r"Total Cost vs. $\beta/m$"
)

ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()

fig.tight_layout()


output_file = (
    PLOT_DIR
    / "total_cost_vs_beta_over_m.png"
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print(
    "Saved:",
    output_file,
)

plt.close(fig)


print("\nDone.")