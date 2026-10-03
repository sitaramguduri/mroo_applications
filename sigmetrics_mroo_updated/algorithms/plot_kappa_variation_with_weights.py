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
    / "beta_m_sweep_regime_weights_rho_7"
    / "regime_length_2160_eps_0p1"
)

SETTING_DIR = (
    RESULT_DIR
    / "beta_180_m_15"
)


# ============================================================
# FIND KAPPA FILE
# ============================================================

kappa_files = list(
    SETTING_DIR.glob(
        "mroo_full_horizon_lambda1_*_eta_*_init_*.csv"
    )
)


if len(kappa_files) == 0:

    raise FileNotFoundError(
        f"No MROO kappa file found in {SETTING_DIR}"
    )


if len(kappa_files) > 1:

    print(
        "\nMultiple kappa files found:"
    )

    for file in kappa_files:

        print(
            " ",
            file.name,
        )

    raise RuntimeError(
        "More than one kappa file found. "
        "Choose the desired configuration explicitly."
    )


KAPPA_FILE = kappa_files[0]


print(
    "\nLoading:"
)

print(
    KAPPA_FILE
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    KAPPA_FILE
)


print(
    "\nColumns:"
)

print(
    data.columns.tolist()
)


# ============================================================
# PARAMETERS
# ============================================================

beta = 180.0
m = 15.0

beta_over_m = (
    beta / m
)


REGIME_LENGTH = 2160

EPS_WEIGHT = 0.1


# ============================================================
# COMPUTE DELTA KAPPA
# ============================================================

data["delta_kappa_1"] = (
    data["kappa_0"].diff()
)

data["delta_kappa_2"] = (
    data["kappa_1"].diff()
)

data["delta_kappa_3"] = (
    data["kappa_2"].diff()
)


# ============================================================
# IDENTIFY ACTIVE WEIGHT REGIME
# ============================================================

data["regime"] = (
    (
        data["t"].astype(int)
        // REGIME_LENGTH
    )
    % 3
)


# 1 means w_1 dominant
# 2 means w_2 dominant
# 3 means w_3 dominant

data["dominant_w"] = (
    data["regime"] + 1
)


# ============================================================
# BUILD ACTUAL WEIGHTS
# ============================================================

norm = np.sqrt(
    1.0
    + 2.0 * EPS_WEIGHT**2
)


high_weight = (
    1.0
    / norm
)

low_weight = (
    EPS_WEIGHT
    / norm
)


data["w_1"] = np.where(
    data["regime"] == 0,
    high_weight,
    low_weight,
)

data["w_2"] = np.where(
    data["regime"] == 1,
    high_weight,
    low_weight,
)

data["w_3"] = np.where(
    data["regime"] == 2,
    high_weight,
    low_weight,
)


# ============================================================
# SMOOTH DELTA KAPPA
# ============================================================

ROLLING_WINDOW = 100


for i in range(
    1,
    4,
):

    data[
        f"delta_kappa_{i}_smooth"
    ] = (
        data[
            f"delta_kappa_{i}"
        ]
        .rolling(
            window=ROLLING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )


# ============================================================
# PLOT DIRECTORY
# ============================================================

PLOT_DIR = (
    RESULT_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FIGURE 1
#
# WEIGHTS + DELTA KAPPA
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 7),
    sharex=True,
)


# ------------------------------------------------------------
# TOP PANEL: w_t
# ------------------------------------------------------------

axes[0].plot(
    data["t"],
    data["w_1"],
    label=r"$w_{t,1}$",
    linewidth=1.2,
)

axes[0].plot(
    data["t"],
    data["w_2"],
    label=r"$w_{t,2}$",
    linewidth=1.2,
)

axes[0].plot(
    data["t"],
    data["w_3"],
    label=r"$w_{t,3}$",
    linewidth=1.2,
)


axes[0].set_ylabel(
    r"$w_t$"
)

axes[0].set_title(
    "Memory-Weight Regimes"
)

axes[0].legend(
    ncol=3
)

axes[0].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# BOTTOM PANEL: DELTA KAPPA
# ------------------------------------------------------------

axes[1].plot(
    data["t"],
    data["delta_kappa_1_smooth"],
    label=r"$\Delta\kappa_1$",
    linewidth=1.2,
)

axes[1].plot(
    data["t"],
    data["delta_kappa_2_smooth"],
    label=r"$\Delta\kappa_2$",
    linewidth=1.2,
)

axes[1].plot(
    data["t"],
    data["delta_kappa_3_smooth"],
    label=r"$\Delta\kappa_3$",
    linewidth=1.2,
)


axes[1].axhline(
    0,
    linestyle="--",
    linewidth=0.8,
    color="black",
)


axes[1].set_xlabel(
    "Time step"
)

axes[1].set_ylabel(
    r"$\Delta\kappa_t$"
)

axes[1].set_title(
    "MROO Dual Updates"
)

axes[1].legend(
    ncol=3
)

axes[1].grid(
    alpha=0.2
)


# ============================================================
# REGIME BOUNDARIES
# ============================================================

for boundary in range(
    REGIME_LENGTH,
    int(data["t"].max()),
    REGIME_LENGTH,
):

    axes[0].axvline(
        boundary,
        linestyle="--",
        linewidth=0.5,
        alpha=0.2,
    )

    axes[1].axvline(
        boundary,
        linestyle="--",
        linewidth=0.5,
        alpha=0.2,
    )


fig.suptitle(
    "MROO Response to Time-Varying Memory Weights\n"
    f"beta = {beta:g}, "
    f"m = {m:g}, "
    f"beta/m = {beta_over_m:g}, "
    f"regime length = {REGIME_LENGTH}"
)


fig.tight_layout()


OUTPUT_FILE = (
    PLOT_DIR
    / "weights_vs_delta_kappa_beta_180_m_15.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


plt.close(
    fig
)


# ============================================================
# FIGURE 2
#
# AVERAGE DELTA KAPPA BY ACTIVE WEIGHT
# ============================================================

regime_average = (
    data
    .groupby(
        "dominant_w"
    )[
        [
            "delta_kappa_1",
            "delta_kappa_2",
            "delta_kappa_3",
        ]
    ]
    .mean()
)


print(
    "\nAverage Delta kappa by dominant weight:"
)

print(
    regime_average
)


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


x = np.arange(
    len(
        regime_average.index
    )
)

bar_width = 0.24


for j, column in enumerate(
    [
        "delta_kappa_1",
        "delta_kappa_2",
        "delta_kappa_3",
    ]
):

    offset = (
        j - 1
    ) * bar_width

    ax.bar(
        x + offset,
        regime_average[
            column
        ],
        width=bar_width,
        label=rf"$\Delta\kappa_{j+1}$",
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        rf"$w_{int(i)}$ dominant"
        for i in regime_average.index
    ]
)


ax.set_xlabel(
    "Dominant memory-weight coordinate"
)

ax.set_ylabel(
    r"Mean $\Delta\kappa_i$"
)

ax.set_title(
    "Average Dual Update by Memory-Weight Regime"
)


ax.axhline(
    0,
    linestyle="--",
    linewidth=0.8,
    color="black",
)


ax.legend()

ax.grid(
    axis="y",
    alpha=0.2,
)


fig.tight_layout()


OUTPUT_FILE = (
    PLOT_DIR
    / "average_delta_kappa_by_weight_beta_180_m_15.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


plt.close(
    fig
)