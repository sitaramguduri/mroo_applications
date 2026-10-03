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

KAPPA_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_kappa_history.csv"
)

PLOT_FILE = (
    RESULT_DIR
    / "mroo_kappa_subplots.png"
)


# ============================================================
# LOAD DATA
# ============================================================

kappa_df = pd.read_csv(
    KAPPA_FILE
)

kappa_columns = [
    col
    for col in kappa_df.columns
    if col.startswith("kappa_")
]

# convert to hours
kappa_df["hours"] = (
    kappa_df["t"] * 10.0 / 3600.0
)

# ============================================================
# DOWNSAMPLE
# ============================================================

STEP = 100   # keep every 100th point

plot_df = kappa_df.iloc[::STEP].copy()


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    len(kappa_columns),
    1,
    figsize=(12, 8),
    sharex=True,
)

if len(kappa_columns) == 1:
    axes = [axes]

for ax, col in zip(axes, kappa_columns):

    ax.plot(
        plot_df["hours"],
        plot_df[col],
        linewidth=1.2,
    )

    ax.set_ylabel(col)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Time (hours)")

fig.suptitle("MROO Dual Variables Over Time", y=0.98)

plt.tight_layout()

plt.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("Saved:", PLOT_FILE)