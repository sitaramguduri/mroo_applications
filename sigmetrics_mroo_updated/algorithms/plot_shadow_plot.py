from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
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
    / "rho_7"
    / "1min_24hour"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET_M = 100.0

WINDOW_SIZE = 1440

N_WINDOWS = 100


# ============================================================
# 3. EXCLUDED FOLDERS
#
# Add folders here if you do NOT want them in the plot.
# ============================================================

EXCLUDED_FOLDERS = {

    "beta_500_m_100_T_1440",
    "beta_2000_m_100_T_1440",
    "beta_50_m_100_T_1440",
    "beta_100_m_100_T_1440",
    "beta_600_m_100_T_1440",
    #  "beta_1000_m_100_T_1440",

}


# ============================================================
# 4. ALGORITHMS TO PLOT
# ============================================================

ALGORITHM_ORDER = [
    "OFFLINE-OPT",
    "GREEDY",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
]


ALGORITHM_LABELS = {
    "OFFLINE-OPT": "OPT",
    "GREEDY": "GRD",
    "S-MROO-SUM": "S-MROO-SUM",
    "S-MROO-MAX": "S-MROO-MAX",
    "DMD": "DMD",
    "MROO": "MROO",
}


# ============================================================
# 5. COSTS
# ============================================================

COSTS = [
    (
        "hitting_cost",
        "Hitting Cost",
    ),
    (
        "long_term_cost",
        "Long-Term Cost",
    ),
    (
        "total_cost",
        "Total Cost",
    ),
]


# ============================================================
# 6. HELPER: PARSE FOLDER NAME
#
# Example:
#
# beta_50_m_100_T_1440
# beta_7p5_m_100_T_1440
# ============================================================

def parse_number(
    value,
):

    return float(
        value.replace(
            "p",
            ".",
        )
    )


def parse_setting_folder(
    folder_name,
):

    parts = folder_name.split(
        "_"
    )

    try:

        beta_index = parts.index(
            "beta"
        )

        m_index = parts.index(
            "m"
        )

        T_index = parts.index(
            "T"
        )


        beta = parse_number(
            parts[
                beta_index + 1
            ]
        )


        m = parse_number(
            parts[
                m_index + 1
            ]
        )


        T = int(
            parts[
                T_index + 1
            ]
        )


        return (
            beta,
            m,
            T,
        )


    except Exception:

        return None


# ============================================================
# 7. FIND ALL m = TARGET_M SETTINGS
# ============================================================

setting_folders = []


for folder in BASE_RESULT_DIR.iterdir():


    # --------------------------------------------------------
    # Skip explicitly excluded folders
    # --------------------------------------------------------

    if folder.name in EXCLUDED_FOLDERS:

        print(
            "Excluding:",
            folder.name,
        )

        continue


    # --------------------------------------------------------
    # Must be a directory
    # --------------------------------------------------------

    if not folder.is_dir():

        continue


    # --------------------------------------------------------
    # Parse folder
    # --------------------------------------------------------

    parsed = parse_setting_folder(
        folder.name
    )


    if parsed is None:

        continue


    beta, m, T = parsed


    # --------------------------------------------------------
    # Only use desired m
    # --------------------------------------------------------

    if not np.isclose(
        m,
        TARGET_M,
    ):

        continue


    # --------------------------------------------------------
    # Only use desired window size
    # --------------------------------------------------------

    if T != WINDOW_SIZE:

        continue


    # --------------------------------------------------------
    # Check combined results
    # --------------------------------------------------------

    result_file = (
        folder
        / "combined_algorithm_window_results.csv"
    )


    if not result_file.exists():

        print(
            "Skipping because combined result "
            "file does not exist:"
        )

        print(
            folder
        )

        continue


    setting_folders.append(
        (
            beta,
            m,
            T,
            folder,
        )
    )


# ============================================================
# 8. SORT BY beta/m
# ============================================================

setting_folders = sorted(
    setting_folders,
    key=lambda item:
        item[0] / item[1],
)


# ============================================================
# 9. PRINT SETTINGS
# ============================================================

print(
    "\n========================================"
)

print(
    f"SETTINGS FOUND FOR m = {TARGET_M:g}"
)

print(
    "========================================"
)


for (
    beta,
    m,
    T,
    folder,
) in setting_folders:


    print(
        f"beta={beta:g}, "
        f"m={m:g}, "
        f"beta/m={beta / m:g}, "
        f"T={T}"
    )


if len(
    setting_folders
) == 0:

    raise RuntimeError(
        f"No result folders found for "
        f"m={TARGET_M:g} and "
        f"T={WINDOW_SIZE}."
    )


# ============================================================
# 10. LOAD ALL RESULTS
# ============================================================

all_results = []


for (
    beta,
    m,
    T,
    folder,
) in setting_folders:


    result_file = (
        folder
        / "combined_algorithm_window_results.csv"
    )


    print(
        "\nReading:"
    )

    print(
        result_file
    )


    df = pd.read_csv(
        result_file
    )


    # --------------------------------------------------------
    # Use beta/m metadata from folder
    # --------------------------------------------------------

    df[
        "beta"
    ] = beta


    df[
        "m"
    ] = m


    df[
        "beta_over_m"
    ] = (
        beta / m
    )


    df[
        "source_folder"
    ] = folder.name


    all_results.append(
        df
    )


combined_df = pd.concat(
    all_results,
    ignore_index=True,
    sort=False,
)


# ============================================================
# 11. VERIFY NUMBER OF WINDOWS
# ============================================================

print(
    "\n========================================"
)

print(
    "ROW COUNTS"
)

print(
    "========================================"
)


counts = (

    combined_df

    .groupby(
        [
            "beta_over_m",
            "algorithm",
        ]
    )

    .size()

    .reset_index(
        name="n_windows"
    )

)


print(
    counts.to_string(
        index=False
    )
)


bad_counts = counts[
    counts[
        "n_windows"
    ]
    != N_WINDOWS
]


if len(
    bad_counts
) > 0:

    print(
        "\nWARNING: Some configurations "
        f"do not have exactly {N_WINDOWS} windows:"
    )

    print(
        bad_counts.to_string(
            index=False
        )
    )


# ============================================================
# 12. BUILD MEAN / STD SUMMARY
# ============================================================

summary = (

    combined_df

    .groupby(
        [
            "algorithm",
            "beta",
            "m",
            "beta_over_m",
        ],
        as_index=False,
    )

    .agg(

        n_windows=(
            "window_id",
            "count",
        ),

        mean_hitting=(
            "hitting_cost",
            "mean",
        ),

        std_hitting=(
            "hitting_cost",
            "std",
        ),

        mean_long_term=(
            "long_term_cost",
            "mean",
        ),

        std_long_term=(
            "long_term_cost",
            "std",
        ),

        mean_total=(
            "total_cost",
            "mean",
        ),

        std_total=(
            "total_cost",
            "std",
        ),

    )

)


summary = (

    summary

    .sort_values(
        [
            "algorithm",
            "beta_over_m",
        ]
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 13. PRINT SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "SUMMARY"
)

print(
    "========================================"
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 14. MAP COST COLUMNS
# ============================================================

SUMMARY_COLUMNS = {

    "hitting_cost":
        (
            "mean_hitting",
            "std_hitting",
        ),

    "long_term_cost":
        (
            "mean_long_term",
            "std_long_term",
        ),

    "total_cost":
        (
            "mean_total",
            "std_total",
        ),

}


# ============================================================
# 15. CREATE 2 x 3 FIGURE
#
# Row 1:
#     linear y-axis
#
# Row 2:
#     logarithmic y-axis
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        16,
        9,
    ),
)


# ============================================================
# 16. HELPER TO DRAW ONE PANEL
# ============================================================

def draw_panel(
    ax,
    cost_name,
    ylabel,
    log_scale=False,
):


    mean_column, std_column = (
        SUMMARY_COLUMNS[
            cost_name
        ]
    )


    for algorithm in ALGORITHM_ORDER:


        algorithm_df = (

            summary[
                summary[
                    "algorithm"
                ]
                == algorithm
            ]

            .sort_values(
                "beta_over_m"
            )

        )


        if len(
            algorithm_df
        ) == 0:

            continue


        x = (
            algorithm_df[
                "beta_over_m"
            ]
            .to_numpy(
                dtype=float
            )
        )


        y = (
            algorithm_df[
                mean_column
            ]
            .to_numpy(
                dtype=float
            )
        )


        std = (
            algorithm_df[
                std_column
            ]
            .to_numpy(
                dtype=float
            )
        )


        # ----------------------------------------------------
        # Mean line
        # ----------------------------------------------------

        line = ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label=ALGORITHM_LABELS[
                algorithm
            ],
        )[0]


        # ----------------------------------------------------
        # Shadow region
        #
        # On a log axis, lower values must be positive.
        # ----------------------------------------------------

        if log_scale:

            lower = np.maximum(
                y - std,
                1e-8,
            )

        else:

            lower = (
                y - std
            )


        upper = (
            y + std
        )


        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.15,
            color=line.get_color(),
        )


    # --------------------------------------------------------
    # Set logarithmic scale
    # --------------------------------------------------------

    if log_scale:

        ax.set_yscale(
            "log"
        )


    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$\beta / m$",
        fontsize=14,
        fontweight="bold",
    )


    ax.set_ylabel(
        ylabel,
        fontsize=14,
        fontweight="bold",
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.4,
    )


    ax.set_axisbelow(
        True
    )


    # --------------------------------------------------------
    # Ticks
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=10,
    )


    for label in ax.get_xticklabels():

        label.set_fontweight(
            "bold"
        )


    for label in ax.get_yticklabels():

        label.set_fontweight(
            "bold"
        )


    # --------------------------------------------------------
    # Border
    # --------------------------------------------------------

    for spine in ax.spines.values():

        spine.set_linewidth(
            1.2
        )


# ============================================================
# 17. FIRST ROW: CURRENT LINEAR PLOTS
# ============================================================

for column_index, (
    cost_name,
    ylabel,
) in enumerate(
    COSTS
):


    draw_panel(
        axes[
            0,
            column_index
        ],
        cost_name,
        ylabel,
        log_scale=False,
    )


# ============================================================
# 18. SECOND ROW: LOG-SCALE PLOTS
# ============================================================

for column_index, (
    cost_name,
    ylabel,
) in enumerate(
    COSTS
):


    draw_panel(
        axes[
            1,
            column_index
        ],
        cost_name,
        ylabel,
        log_scale=True,
    )


# ============================================================
# 19. ROW LABELS / TITLES
# ============================================================

axes[
    0,
    0
].text(
    -0.18,
    0.5,
    "Linear Scale",
    transform=axes[
        0,
        0
    ].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=14,
    fontweight="bold",
)


axes[
    1,
    0
].text(
    -0.18,
    0.5,
    "Log Scale",
    transform=axes[
        1,
        0
    ].transAxes,
    rotation=90,
    va="center",
    ha="center",
    fontsize=14,
    fontweight="bold",
)


# ============================================================
# 20. COLUMN TITLES
# ============================================================

axes[
    0,
    0
].set_title(
    "Hitting Cost",
    fontsize=14,
    fontweight="bold",
)


axes[
    0,
    1
].set_title(
    "Long-Term Cost",
    fontsize=14,
    fontweight="bold",
)


axes[
    0,
    2
].set_title(
    "Total Cost",
    fontsize=14,
    fontweight="bold",
)


# ============================================================
# 21. LEGEND
# ============================================================

handles, labels = (
    axes[
        0,
        2
    ]
    .get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        0.965,
    ),
    ncol=len(
        ALGORITHM_ORDER
    ),
    fontsize=10,
    frameon=False,
)


# ============================================================
# 22. MAIN TITLE
# ============================================================

fig.suptitle(
    (
        "Cost vs. "
        r"$\beta/m$"
        f" for m={TARGET_M:g}"
    ),
    fontsize=16,
    fontweight="bold",
    y=0.995,
)


# ============================================================
# 23. SPACING
# ============================================================

fig.subplots_adjust(
    left=0.08,
    right=0.99,
    bottom=0.08,
    top=0.89,
    hspace=0.35,
    wspace=0.28,
)


# ============================================================
# 24. SAVE PLOT
# ============================================================

OUTPUT_FILE = (
    BASE_RESULT_DIR
    / (
        f"shadow_cost_vs_beta_over_m"
        f"_m_{TARGET_M:g}"
        f"_T_{WINDOW_SIZE}"
        f"_linear_and_log.png"
    )
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# 25. SAVE SUMMARY CSV
# ============================================================

SUMMARY_OUTPUT_FILE = (
    BASE_RESULT_DIR
    / (
        f"shadow_cost_vs_beta_over_m"
        f"_m_{TARGET_M:g}"
        f"_T_{WINDOW_SIZE}"
        f"_linear_and_log_summary.csv"
    )
)


summary.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False,
)


# ============================================================
# 26. PRINT OUTPUTS
# ============================================================

print(
    "\n========================================"
)

print(
    "LINEAR + LOG SHADOW PLOT SAVED"
)

print(
    "========================================"
)


print(
    OUTPUT_FILE
)


print(
    "\nSummary CSV:"
)


print(
    SUMMARY_OUTPUT_FILE
)


# ============================================================
# 27. SHOW
# ============================================================

plt.show()