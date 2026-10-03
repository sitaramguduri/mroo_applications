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

RESULT_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_results.csv"
)

PLOT_FILE = (
    RESULT_DIR
    / "mroo_cost_vs_eta.png"
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(
    RESULT_FILE
)

results = (
    results
    .sort_values(
        by="eta"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# GET T AND THEORETICAL ETA
# ============================================================

T = int(
    results["T"].iloc[0]
)

eta_theory = (
    T ** (-1.0 / 3.0)
)


print(
    "T =",
    T
)

print(
    "Theoretical eta = T^(-1/3) =",
    eta_theory
)


# ============================================================
# PRINT DATA
# ============================================================

print(
    "\nResults:"
)

print(
    results[
        [
            "eta",
            "hitting_cost",
            "long_term_cost",
            "total_cost",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


# ============================================================
# COST CURVES
# ============================================================

ax.plot(
    results["eta"],
    results["hitting_cost"],
    marker="o",
    linewidth=2,
    markersize=6,
    label="Hitting Cost",
)


ax.plot(
    results["eta"],
    results["long_term_cost"],
    marker="s",
    linewidth=2,
    markersize=6,
    label="Long-Term Cost",
)


ax.plot(
    results["eta"],
    results["total_cost"],
    marker="^",
    linewidth=2,
    markersize=7,
    label="Total Cost",
)


# ============================================================
# HIGHLIGHT THEORETICAL ETA
# ============================================================

ax.axvline(
    x=eta_theory,
    linestyle="--",
    linewidth=1.8,
    label=r"Theoretical $\eta=T^{-1/3}$",
)


# ============================================================
# FIND THE RESULT CORRESPONDING TO THEORETICAL ETA
# ============================================================

theory_index = (
    results["eta"] - eta_theory
).abs().idxmin()

theory_row = results.loc[
    theory_index
]


# Highlight theoretical total-cost point
ax.scatter(
    theory_row["eta"],
    theory_row["total_cost"],
    s=120,
    marker="*",
    zorder=5,
)


# ============================================================
# ANNOTATE THE THEORETICAL POINT
# ============================================================

ax.annotate(
    (
        r"$\eta=T^{-1/3}$"
        "\n"
        f"{eta_theory:.3g}"
    ),
    xy=(
        theory_row["eta"],
        theory_row["total_cost"],
    ),
    xytext=(-95, 35),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.2,
    },
    fontsize=10,
)


# ============================================================
# AXIS SETTINGS
# ============================================================

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"Learning Rate $\eta$",
    fontsize=12,
)

ax.set_ylabel(
    "Full-Horizon Cost",
    fontsize=12,
)

ax.set_title(
    r"MROO Cost Sensitivity to Learning Rate $\eta$",
    fontsize=13,
)


# ============================================================
# GRID AND LEGEND
# ============================================================

ax.grid(
    alpha=0.3,
)

ax.legend(
    fontsize=10,
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


print(
    "\nSaved plot:",
    PLOT_FILE
)