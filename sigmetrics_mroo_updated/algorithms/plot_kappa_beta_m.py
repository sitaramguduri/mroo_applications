from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

beta = 1800.0
m = 15.0

beta_over_m = (
    beta / m
)


eta = data[
    "eta"
].iloc[0]

lambda_1 = data[
    "lambda_1"
].iloc[0]

kappa_init = data[
    "kappa_init"
].iloc[0]


# ============================================================
# PLOT KAPPA OVER TIME
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 5.5)
)


ax.plot(
    data["t"],
    data["kappa_0"],
    label=r"$\kappa_1$",
    linewidth=1.0,
)

ax.plot(
    data["t"],
    data["kappa_1"],
    label=r"$\kappa_2$",
    linewidth=1.0,
)

ax.plot(
    data["t"],
    data["kappa_2"],
    label=r"$\kappa_3$",
    linewidth=1.0,
)


# ============================================================
# REGIME BOUNDARIES
# ============================================================

REGIME_LENGTH = 2160


for boundary in range(
    REGIME_LENGTH,
    int(data["t"].max()),
    REGIME_LENGTH,
):

    ax.axvline(
        boundary,
        linestyle="--",
        linewidth=0.5,
        alpha=0.25,
    )


# ============================================================
# LABELS
# ============================================================

ax.set_title(
    "MROO Dual Variables Over Time\n"
    f"beta = {beta:g}, "
    f"m = {m:g}, "
    f"beta/m = {beta_over_m:g}"
)


ax.set_xlabel(
    "Time step"
)

ax.set_ylabel(
    r"$\kappa_t$"
)


ax.legend()


ax.grid(
    alpha=0.2
)


fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

PLOT_DIR = (
    RESULT_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_FILE = (
    PLOT_DIR
    / "kappa_over_time_beta_1800_m_15.png"
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