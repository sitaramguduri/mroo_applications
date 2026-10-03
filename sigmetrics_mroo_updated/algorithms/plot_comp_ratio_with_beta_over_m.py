from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    ALGORITHM_DIR
    / "new_results"
    / "beta_m_sweep_regime_weights_kappa_0p1"
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
# KEEP ONLY BEST ROWS
# ============================================================

if "selected_best" in df.columns:
    df = df[
        df["selected_best"] == True
    ].copy()


# ============================================================
# OPTIONAL FILTERS
# Keep only one experiment family if needed
# ============================================================

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
# PIVOT TABLE FOR COMPETITIVE RATIO
# ============================================================

ratio_pivot = (
    df.pivot_table(
        index="beta_over_m",
        columns="algorithm",
        values="cost_ratio_to_opt",
        aggfunc="first",
    )
)

ratio_pivot = ratio_pivot.reindex(
    columns=ALGORITHM_ORDER
)

ratio_pivot = ratio_pivot.sort_index()

print("\nCompetitive ratio table:")
print(ratio_pivot)


# ============================================================
# CLASSIFY REGIONS
#
# comparable_tol = 0.02 means:
# if MROO is within 2% of the best competing online method,
# call it "Comparable"
# ============================================================

comparable_tol = 0.02

summary_rows = []

for x in ratio_pivot.index:

    mroo_val = ratio_pivot.loc[x, "MROO"]

    competitor_vals = []

    for alg in ["Greedy", "S-MROO-MAX", "S-MROO-SUM"]:
        val = ratio_pivot.loc[x, alg]
        if pd.notna(val):
            competitor_vals.append(val)

    best_other = min(competitor_vals)

    if mroo_val < best_other - 1e-12:
        region = "MROO best"

    elif mroo_val <= best_other * (1.0 + comparable_tol):
        region = "Comparable"

    else:
        region = "MROO worse"

    summary_rows.append({
        "beta_over_m": x,
        "MROO": mroo_val,
        "best_other_online": best_other,
        "gap_to_best_other": mroo_val - best_other,
        "region": region,
    })

summary_df = pd.DataFrame(summary_rows)

print("\nRegion summary:")
print(summary_df.to_string(index=False))


# ============================================================
# COMPUTE REGION SPANS FOR BACKGROUND SHADING
# ============================================================

xvals = summary_df["beta_over_m"].to_numpy(dtype=float)
regions = summary_df["region"].tolist()

if len(xvals) == 1:
    left_edges = np.array([xvals[0] / 1.5])
    right_edges = np.array([xvals[0] * 1.5])
else:
    midpoints = np.sqrt(xvals[:-1] * xvals[1:])
    left_edges = np.empty_like(xvals)
    right_edges = np.empty_like(xvals)

    left_edges[0] = xvals[0] / np.sqrt(xvals[1] / xvals[0])
    right_edges[-1] = xvals[-1] * np.sqrt(xvals[-1] / xvals[-2])

    left_edges[1:] = midpoints
    right_edges[:-1] = midpoints


# ============================================================
# PLOT 1: COMPETITIVE RATIO VS beta/m
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

# ------------------------------------------------------------
# Background shading by region
# ------------------------------------------------------------

for i in range(len(xvals)):

    region = regions[i]

    if region == "MROO best":
        color = "lightgreen"
        alpha = 0.20

    elif region == "Comparable":
        color = "lightyellow"
        alpha = 0.25

    else:
        color = "mistyrose"
        alpha = 0.20

    ax.axvspan(
        left_edges[i],
        right_edges[i],
        color=color,
        alpha=alpha,
        zorder=0,
    )

# ------------------------------------------------------------
# Plot all algorithms
# ------------------------------------------------------------

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
        label=alg,
    )

# OPT line at 1
ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.2,
    color="black",
    label="OPT = 1",
)

ax.set_xscale("log")

ax.set_xlabel(r"$\beta / m$")
ax.set_ylabel("Competitive Ratio")
ax.set_title(
    "Competitive Ratio vs $\\beta/m$\n"
    "Region Highlight: where MROO is best vs comparable"
)

ax.grid(
    which="both",
    axis="both",
    alpha=0.25,
)

ax.legend()
fig.tight_layout()

plot_file_1 = (
    PLOT_DIR
    / "competitive_ratio_vs_beta_over_m_regions.png"
)

fig.savefig(
    plot_file_1,
    dpi=300,
    bbox_inches="tight",
)

print("Saved:", plot_file_1)

plt.close(fig)


# ============================================================
# PLOT 2: MROO advantage over best competing online method
#
# positive  -> MROO worse
# negative  -> MROO better
#
# We plot:
#   MROO ratio - best other online ratio
# so below zero means MROO is better
# ============================================================

summary_df["mroo_minus_best_other"] = (
    summary_df["MROO"]
    - summary_df["best_other_online"]
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    summary_df["beta_over_m"],
    summary_df["mroo_minus_best_other"],
    marker="o",
    linewidth=2,
)

ax.axhline(
    y=0.0,
    linestyle="--",
    linewidth=1.2,
    color="black",
)

for _, row in summary_df.iterrows():
    ax.annotate(
        row["region"],
        xy=(
            row["beta_over_m"],
            row["mroo_minus_best_other"],
        ),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )

ax.set_xscale("log")
ax.set_xlabel(r"$\beta / m$")
ax.set_ylabel("MROO ratio - best other online ratio")
ax.set_title(
    "How far MROO is from the best competing online method\n"
    "(negative means MROO is better)"
)

ax.grid(
    which="both",
    axis="both",
    alpha=0.25,
)

fig.tight_layout()

plot_file_2 = (
    PLOT_DIR
    / "mroo_gap_to_best_other_vs_beta_over_m.png"
)

fig.savefig(
    plot_file_2,
    dpi=300,
    bbox_inches="tight",
)

print("Saved:", plot_file_2)

plt.close(fig)


# ============================================================
# SAVE REGION SUMMARY
# ============================================================

summary_file = (
    PLOT_DIR
    / "mroo_region_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print("Saved:", summary_file)