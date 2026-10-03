from pathlib import Path

import numpy as np
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

KAPPA_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_kappa_all_eta.csv"
)

PLOT_FILE = (
    RESULT_DIR
    / "mroo_kappa_shadow_eta_comparison.png"
)


# ============================================================
# LOAD DATA
# ============================================================

kappa_df = pd.read_csv(
    KAPPA_FILE
)


# ============================================================
# KAPPA COLUMNS
# ============================================================

kappa_columns = [
    col
    for col in kappa_df.columns
    if col.startswith("kappa_")
    and col != "kappa_init"
]

print(
    "Kappa columns:",
    kappa_columns
)


# ============================================================
# HORIZON + THEORETICAL ETA
# ============================================================

T = int(
    kappa_df["t"].max()
)

eta_theory = (
    T ** (-1.0 / 3.0)
)

print(
    "T =",
    T
)

print(
    "Theoretical eta =",
    eta_theory
)


# ============================================================
# ETA VALUES TO SHOW
# ============================================================

TARGET_ETAS = [
    1e-5,
    1e-3,
    eta_theory,
]


available_etas = np.sort(
    kappa_df["eta"].unique()
)


selected_etas = []

for target in TARGET_ETAS:

    closest_eta = available_etas[
        np.argmin(
            np.abs(
                available_etas - target
            )
        )
    ]

    selected_etas.append(
        closest_eta
    )


# ============================================================
# TIME IN HOURS
# ============================================================

kappa_df["hours"] = (
    kappa_df["t"]
    * 10.0
    / 3600.0
)


# ============================================================
# SMOOTHING
#
# 360 x 10 sec = 1 hour
# ============================================================

ROLLING_WINDOW = 360


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    sharex=True,
)


# ============================================================
# EACH ETA
# ============================================================

for ax, eta in zip(
    axes,
    selected_etas,
):

    eta_df = (
        kappa_df[
            np.isclose(
                kappa_df["eta"],
                eta,
                rtol=1e-8,
                atol=1e-12,
            )
        ]
        .copy()
        .sort_values(
            "t"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SMOOTH INDIVIDUAL KAPPA COORDINATES
    # ========================================================

    smoothed = pd.DataFrame(
        index=eta_df.index
    )

    for col in kappa_columns:

        smoothed[col] = (
            eta_df[col]
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
                center=True,
            )
            .mean()
        )


    # ========================================================
    # MEAN / MIN / MAX ACROSS KAPPA DIMENSIONS
    # ========================================================

    kappa_mean = (
        smoothed[
            kappa_columns
        ]
        .mean(
            axis=1
        )
    )

    kappa_min = (
        smoothed[
            kappa_columns
        ]
        .min(
            axis=1
        )
    )

    kappa_max = (
        smoothed[
            kappa_columns
        ]
        .max(
            axis=1
        )
    )


    # ========================================================
    # SHADOW REGION
    # ========================================================

    ax.fill_between(
        eta_df["hours"],
        kappa_min,
        kappa_max,
        alpha=0.20,
        label=r"Range of $\kappa_i$",
    )


    # ========================================================
    # MEAN LINE
    # ========================================================

    ax.plot(
        eta_df["hours"],
        kappa_mean,
        linewidth=2,
        label=r"Mean $\kappa_t$",
    )


    # ========================================================
    # TITLE
    # ========================================================

    if np.isclose(
        eta,
        eta_theory,
        rtol=1e-6,
        atol=1e-12,
    ):

        title = (
            rf"$\eta=T^{{-1/3}}"
            rf"\approx{eta:.3g}$"
            " (Theoretical)"
        )

    else:

        title = (
            rf"$\eta={eta:.0e}$"
        )


    ax.set_title(
        title,
        fontsize=11,
    )


    ax.set_ylabel(
        r"$\kappa_t$",
        fontsize=11,
    )


    ax.grid(
        alpha=0.25
    )


# ============================================================
# X AXIS
# ============================================================

axes[-1].set_xlabel(
    "Time (hours)",
    fontsize=11,
)


# ============================================================
# SHARED LEGEND
# ============================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(
        0.5,
        0.985,
    ),
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "MROO Dual Variable Evolution Across Learning Rates",
    fontsize=13,
    y=1.02,
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.95,
    ]
)


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
    "\nSaved:",
    PLOT_FILE
)