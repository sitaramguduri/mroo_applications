from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

BASE_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)

PLOT_DIR = (
    BASE_RESULT_DIR
    / "comparison_shadow_T_17280"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS TO COMPARE
# ============================================================

EXPERIMENTS = [
    {
        "folder": "beta_1p5_m_15_T_17280",
        "beta": 1.5,
        "m": 15.0,
    },
    {
        "folder": "beta_15_m_15_T_17280",
        "beta": 15.0,
        "m": 15.0,
    },
    {
        "folder": "beta_180_m_15_T_17280",
        "beta": 180.0,
        "m": 15.0,
    },
    {
        "folder": "beta_900_m_15_T_17280",
        "beta": 900.0,
        "m": 15.0,
    },
    {
        "folder": "beta_1800_m_15_T_17280",
        "beta": 1800.0,
        "m": 15.0,
    },
]


# ============================================================
# ALGORITHMS
# ============================================================

ALGORITHM_ORDER = [
    "MROO",
    "S-MROO-MAX",
    "S-MROO-SUM",
    "OFFLINE-OPT",
    "GREEDY",
]

ALGORITHM_LABELS = {
    "MROO": "MROO",
    "S-MROO-MAX": "S-MROO-MAX",
    "S-MROO-SUM": "S-MROO-SUM",
    "OFFLINE-OPT": "OPT",
    "GREEDY": "GRD",
}


# ============================================================
# LOAD ALL DATA
# ============================================================

frames = []

for setting_id, experiment in enumerate(EXPERIMENTS):

    result_file = (
        BASE_RESULT_DIR
        / experiment["folder"]
        / "combined_algorithm_window_results.csv"
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"Could not find:\n{result_file}"
        )

    df = pd.read_csv(result_file).copy()

    df["setting_id"] = setting_id
    df["setting_label"] = (
        rf"$\beta={experiment['beta']:g}$"
        "\n"
        rf"$m={experiment['m']:g}$"
    )

    frames.append(df)

all_df = pd.concat(
    frames,
    ignore_index=True,
)


# ============================================================
# AGGREGATE
# ============================================================

summary = (
    all_df
    .groupby(
        ["setting_id", "setting_label", "algorithm"],
        as_index=False,
    )
    .agg(
        mean_hitting=("hitting_cost", "mean"),
        std_hitting=("hitting_cost", "std"),
        mean_long_term=("long_term_cost", "mean"),
        std_long_term=("long_term_cost", "std"),
        mean_total=("total_cost", "mean"),
        std_total=("total_cost", "std"),
        n_windows=("total_cost", "count"),
    )
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(19, 6.5),
)

plot_info = [
    ("mean_hitting", "std_hitting", "Hitting Cost"),
    ("mean_long_term", "std_long_term", "Long-Term Cost"),
    ("mean_total", "std_total", "Total Cost"),
]


for ax, (mean_col, std_col, title) in zip(axes, plot_info):

    plotted_y = []

    for algorithm in ALGORITHM_ORDER:

        alg_df = (
            summary[
                summary["algorithm"] == algorithm
            ]
            .sort_values("setting_id")
        )

        if len(alg_df) == 0:
            continue

        x = alg_df["setting_id"].to_numpy(dtype=int)
        y = alg_df[mean_col].to_numpy(dtype=float)
        s = alg_df[std_col].fillna(0.0).to_numpy(dtype=float)

        plotted_y.extend((y - s).tolist())
        plotted_y.extend((y + s).tolist())

        line, = ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            markeredgewidth=1.3,
            linewidth=2.8,
            label=ALGORITHM_LABELS[algorithm],
            zorder=3,
        )

        # lighter shaded band
        ax.fill_between(
            x,
            y - s,
            y + s,
            color=line.get_color(),
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )

        # add thin dashed upper/lower bounds
        ax.plot(
            x,
            y - s,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            color=line.get_color(),
            zorder=2,
        )

        ax.plot(
            x,
            y + s,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            color=line.get_color(),
            zorder=2,
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=10,
    )

    ax.set_ylabel(
        title,
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel(
        r"$(\beta,m)$ Setting",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xticks(
        np.arange(len(EXPERIMENTS))
    )

    ax.set_xticklabels(
        [
            rf"$\beta={e['beta']:g}$" + "\n" + rf"$m={e['m']:g}$"
            for e in EXPERIMENTS
        ],
        fontsize=10,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        width=1.5,
        length=5,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # tighter y-range, especially useful for hitting cost
    if len(plotted_y) > 0:
        y_min = min(plotted_y)
        y_max = max(plotted_y)
        pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)

    legend = ax.legend(
        loc="best",
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )
    legend.get_frame().set_linewidth(1.2)


fig.suptitle(
    "Cost Comparison Across Problem Settings\n"
    r"$T=17280$ (48-hour windows), Mean $\pm$ Standard Deviation",
    fontsize=17,
    fontweight="bold",
)

fig.tight_layout(
    rect=[0, 0, 1, 0.90]
)

output_file = (
    PLOT_DIR
    / "cost_shadow_comparison_T_17280_clearer.png"
)

fig.savefig(
    output_file,
    dpi=400,
    bbox_inches="tight",
)

print("Saved:", output_file)

plt.show()