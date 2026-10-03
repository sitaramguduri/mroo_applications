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
    / "mroo_kappa_eta_comparison.png"
)


# ============================================================
# LOAD DATA
# ============================================================

kappa_df = pd.read_csv(
    KAPPA_FILE
)


# ============================================================
# IDENTIFY KAPPA COLUMNS
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
# GET HORIZON LENGTH
#
# kappa history includes initial point, so:
# T = max(t)
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
# ETA VALUES TO PLOT
# ============================================================

TARGET_ETAS = [
    1e-5,
    1e-3,
    eta_theory,
]


# ============================================================
# FIND ACTUAL STORED ETA VALUES
#
# This avoids floating-point matching problems.
# ============================================================

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


print(
    "Selected eta values:",
    selected_etas
)


# ============================================================
# TIME CONVERSION
#
# Each time step = 10 seconds.
# ============================================================

kappa_df["hours"] = (
    kappa_df["t"]
    * 10.0
    / 3600.0
)


# ============================================================
# SMOOTHING
#
# Rolling window in number of time steps.
#
# 360 steps × 10 sec = 1 hour.
# So this is approximately a 1-hour rolling average.
# ============================================================

ROLLING_WINDOW = 360


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    sharex=True,
)


# ============================================================
# PLOT EACH ETA
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
            by="t"
        )
    )


    # ========================================================
    # ROLLING AVERAGE
    # ========================================================

    for col in kappa_columns:

        eta_df[
            f"{col}_smooth"
        ] = (
            eta_df[col]
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
            )
            .mean()
        )


    # ========================================================
    # DRAW LINES
    # ========================================================

    for col in kappa_columns:

        ax.plot(
            eta_df["hours"],
            eta_df[
                f"{col}_smooth"
            ],
            linewidth=1.5,
            label=col,
        )


    # ========================================================
    # PANEL TITLE
    # ========================================================

    if np.isclose(
        eta,
        eta_theory,
        rtol=1e-6,
        atol=1e-12,
    ):

        panel_title = (
            rf"$\eta=T^{{-1/3}}"
            rf"\approx{eta:.3g}$"
            " (Theoretical)"
        )

    else:

        panel_title = (
            rf"$\eta={eta:.0e}$"
        )


    ax.set_title(
        panel_title,
        fontsize=11,
    )

    ax.set_ylabel(
        r"$\kappa_t$",
        fontsize=11,
    )

    ax.grid(
        alpha=0.3,
    )


# ============================================================
# X AXIS
# ============================================================

axes[-1].set_xlabel(
    "Time (hours)",
    fontsize=11,
)


# ============================================================
# LEGEND
#
# One shared legend is cleaner than three separate legends.
# ============================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(kappa_columns),
    bbox_to_anchor=(
        0.5,
        0.985,
    ),
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    r"MROO Dual Variable Evolution for Different Learning Rates",
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
    "\nSaved plot:",
    PLOT_FILE
)