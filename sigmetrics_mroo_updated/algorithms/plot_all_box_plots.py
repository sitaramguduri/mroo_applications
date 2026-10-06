from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
    / "matrix_memory_v1"
    / "1min_24hour"
    / "matrix_memory"
)


SAVED_CONFIG_DIR = (
    BASE_RESULT_DIR
    / "saved_plot_configurations"
)


SAVED_WINDOW_FILE = (
    SAVED_CONFIG_DIR
    / "saved_window_results.csv"
)


SAVED_SUMMARY_FILE = (
    SAVED_CONFIG_DIR
    / "saved_configuration_summary.csv"
)


OUTPUT_DIR = (
    SAVED_CONFIG_DIR
    / "final_ratio_plots"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. PLOT SETTINGS
# ============================================================

ALGORITHM_ORDER = [

    "OFFLINE-OPT",

    "GREEDY",

    "S-MROO-SUM",

    "S-MROO-MAX",

    "DMD",

    "MROO",
]


# ALGORITHM_LABELS = {

#     "OFFLINE-OPT":
#         "OPT",

#     "GREEDY":
#         "GREEDY",

#     "S-MROO-SUM":
#         "S-MROO-SUM",

#     "S-MROO-MAX":
#         "S-MROO-MAX",

#     "DMD":
#         "DMD",

#     "MROO":
#         "MROO",
# }
ALGORITHM_LABELS = {
    "OFFLINE-OPT": "OPT",
    "GREEDY": "GREEDY",
    "S-MROO-SUM": "ROBD",
    "S-MROO-MAX": "S-MROO",
    "DMD": "DMD",
    "MROO": "DMROO",
}

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
# 3. OPTIONAL: CHOOSE ONLY SPECIFIC SAVED CONFIGURATIONS
#
# None means:
#
#     use every saved configuration in the master file.
#
# Or specify:
#
# SAVE_NAMES = [
#     "beta50_m100_good_v1",
#     "beta100_m100_good_v1",
#     "beta200_m100_good_v1",
#     "beta400_m100_good_v1",
#     "beta600_m100_good_v1",
# ]
#
# ============================================================

SAVE_NAMES = None


# ============================================================
# 4. LOAD SAVED WINDOW RESULTS
# ============================================================

if not SAVED_WINDOW_FILE.exists():

    raise FileNotFoundError(
        "\nCould not find saved window results:\n"
        f"{SAVED_WINDOW_FILE}"
    )


df = pd.read_csv(
    SAVED_WINDOW_FILE,
    low_memory=False,
)


print(
    "\n========================================"
)

print(
    "SAVED BETA/M RATIO BOXPLOTS"
)

print(
    "========================================"
)


print(
    "\nReading:"
)

print(
    SAVED_WINDOW_FILE
)


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {

    "save_name",

    "algorithm",

    "window_id",

    "beta",

    "m",

    "hitting_cost",

    "long_term_cost",

    "total_cost",
}


missing_columns = (
    required_columns
    -
    set(
        df.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "\nSaved result file is missing columns:\n"
        f"{sorted(missing_columns)}"
    )


# ============================================================
# 6. BETA/M RATIO
# ============================================================

if (
    "beta_m_ratio"
    not in df.columns
):

    df[
        "beta_m_ratio"
    ] = (
        df[
            "beta"
        ].astype(float)
        /
        df[
            "m"
        ].astype(float)
    )


df[
    "beta_m_ratio"
] = pd.to_numeric(
    df[
        "beta_m_ratio"
    ],
    errors="coerce",
)


# ============================================================
# 7. OPTIONAL SAVE-NAME FILTER
# ============================================================

if SAVE_NAMES is not None:

    df = (
        df[
            df[
                "save_name"
            ].isin(
                SAVE_NAMES
            )
        ]
        .copy()
    )


    missing_saves = (

        set(
            SAVE_NAMES
        )

        -

        set(
            df[
                "save_name"
            ].unique()
        )
    )


    if missing_saves:

        raise RuntimeError(
            "\nRequested SAVE_NAME values were not found:\n"
            f"{sorted(missing_saves)}"
        )


# ============================================================
# 8. SHOW SAVED CONFIGURATIONS
# ============================================================

saved_configs = (

    df[
        [
            "save_name",
            "beta",
            "m",
            "beta_m_ratio",
        ]
    ]

    .drop_duplicates()

    .sort_values(
        [
            "beta_m_ratio",
            "beta",
            "m",
        ]
    )

    .reset_index(
        drop=True
    )
)


print(
    "\n========================================"
)

print(
    "CONFIGURATIONS FOUND"
)

print(
    "========================================"
)


print(
    saved_configs.to_string(
        index=False
    )
)


# ============================================================
# 9. IMPORTANT SAFETY CHECK
#
# We expect ONE manually-approved SAVE_NAME per beta/m ratio.
#
# Otherwise two different configurations at the same ratio
# could accidentally be mixed into the same boxplot.
# ============================================================

configs_per_ratio = (

    saved_configs

    .groupby(
        "beta_m_ratio"
    )[
        "save_name"
    ]

    .nunique()
)


ambiguous_ratios = (

    configs_per_ratio[
        configs_per_ratio
        >
        1
    ]
)


if len(
    ambiguous_ratios
) > 0:

    print(
        "\n========================================"
    )

    print(
        "MULTIPLE SAVED CONFIGURATIONS FOUND"
    )

    print(
        "========================================"
    )


    print(
        "\nMore than one SAVE_NAME exists for these beta/m ratios:"
    )


    print(
        ambiguous_ratios
    )


    print(
        "\nConfigurations:"
    )


    ambiguous_df = (

        saved_configs[
            saved_configs[
                "beta_m_ratio"
            ].isin(
                ambiguous_ratios.index
            )
        ]
    )


    print(
        ambiguous_df.to_string(
            index=False
        )
    )


    raise RuntimeError(
        "\nDo not mix multiple manually saved configurations "
        "for the same beta/m ratio.\n\n"
        "Set SAVE_NAMES near the top of this script to select "
        "exactly one configuration for each ratio."
    )


# ============================================================
# 10. REMOVE DUPLICATED WINDOW ROWS
#
# Unique identity:
#
#     save_name
#     algorithm
#     window_id
#
# ============================================================

df = (

    df

    .drop_duplicates(

        subset=[
            "save_name",
            "algorithm",
            "window_id",
        ],

        keep="last",
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 11. VERIFY 100 WINDOWS PER ALGORITHM / RATIO
# ============================================================

window_counts = (

    df

    .groupby(
        [
            "save_name",
            "beta_m_ratio",
            "algorithm",
        ]
    )[
        "window_id"
    ]

    .nunique()

    .reset_index(
        name="n_windows"
    )
)


print(
    "\n========================================"
)

print(
    "WINDOW COUNTS"
)

print(
    "========================================"
)


print(
    window_counts.to_string(
        index=False
    )
)


bad_counts = (

    window_counts[
        window_counts[
            "n_windows"
        ]
        !=
        100
    ]
)


if len(
    bad_counts
) > 0:

    raise RuntimeError(
        "\nSome saved configurations do not contain exactly "
        "100 windows:\n\n"
        f"{bad_counts.to_string(index=False)}"
    )


# ============================================================
# 12. VERIFY ALL ALGORITHMS EXIST AT EACH RATIO
# ============================================================

ratio_values = (

    df[
        "beta_m_ratio"
    ]

    .dropna()

    .unique()
)


ratio_values = np.sort(
    ratio_values.astype(float)
)


for ratio in ratio_values:

    ratio_df = (

        df[
            np.isclose(
                df[
                    "beta_m_ratio"
                ].astype(float),
                ratio,
                rtol=1e-12,
                atol=1e-15,
            )
        ]
    )


    found_algorithms = set(
        ratio_df[
            "algorithm"
        ].unique()
    )


    missing_algorithms = (

        set(
            ALGORITHM_ORDER
        )

        -

        found_algorithms
    )


    if missing_algorithms:

        raise RuntimeError(
            f"\nFor beta/m={ratio:g}, the following "
            "algorithms are missing:\n"
            f"{sorted(missing_algorithms)}"
        )


print(
    "\nVerified: every beta/m ratio has all algorithms."
)


# ============================================================
# 13. SHOW COST SUMMARY
# ============================================================

cost_summary = (

    df

    .groupby(
        [
            "beta_m_ratio",
            "algorithm",
        ],
        as_index=False,
    )

    .agg(

        mean_hitting_cost=(
            "hitting_cost",
            "mean",
        ),

        std_hitting_cost=(
            "hitting_cost",
            "std",
        ),

        mean_long_term_cost=(
            "long_term_cost",
            "mean",
        ),

        std_long_term_cost=(
            "long_term_cost",
            "std",
        ),

        mean_total_cost=(
            "total_cost",
            "mean",
        ),

        std_total_cost=(
            "total_cost",
            "std",
        ),
    )
)


cost_summary[
    "_algorithm_order"
] = (
    cost_summary[
        "algorithm"
    ]
    .map(
        {
            algorithm:
                index

            for index, algorithm
            in enumerate(
                ALGORITHM_ORDER
            )
        }
    )
)


cost_summary = (

    cost_summary

    .sort_values(
        [
            "beta_m_ratio",
            "_algorithm_order",
        ]
    )

    .drop(
        columns=[
            "_algorithm_order",
        ]
    )

    .reset_index(
        drop=True
    )
)


print(
    "\n========================================"
)

print(
    "SAVED CONFIGURATION COST SUMMARY"
)

print(
    "========================================"
)


print(
    cost_summary.to_string(
        index=False
    )
)


SUMMARY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "saved_beta_m_ratio_cost_summary.csv"
)


cost_summary.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False,
)


# ============================================================
# 14. BOX POSITION SETTINGS
#
# Example with six algorithms:
#
# ratio=1
#
#        OPT GRD SUM MAX DMD MROO
#         |   |   |   |   |   |
#
# ratio=2
#
#        OPT GRD SUM MAX DMD MROO
#
# ============================================================

n_algorithms = len(
    ALGORITHM_ORDER
)


group_centers = np.arange(
    len(
        ratio_values
    ),
    dtype=float,
)


group_width = 0.78


box_width = (
    group_width
    /
    n_algorithms
)


offsets = (

    (
        np.arange(
            n_algorithms
        )

        -

        (
            n_algorithms
            -
            1
        )
        /
        2.0
    )

    *

    box_width
)


# ============================================================
# 15. ALGORITHM COLORS
#
# We deliberately let matplotlib choose its normal color
# cycle and assign one consistent color per algorithm.
# ============================================================

default_colors = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()[
        "color"
    ]
)


algorithm_colors = {

    algorithm:
        default_colors[
            index
            %
            len(
                default_colors
            )
        ]

    for index, algorithm
    in enumerate(
        ALGORITHM_ORDER
    )
}


# ============================================================
# 16. CREATE THREE-PANEL BOXPLOT
# ============================================================

figure_width = max(
    15.5,
    2.4
    *
    len(
        ratio_values
    ),
)


fig, axes = plt.subplots(

    1,

    3,

    figsize=(
        figure_width,
        5.8,
    ),
)


# ============================================================
# 17. DRAW EACH COST PANEL
# ============================================================

for (
    ax,
    (
        cost_column,
        ylabel,
    ),
) in zip(
    axes,
    COSTS,
):


    # --------------------------------------------------------
    # One algorithm at a time so each algorithm gets the same
    # visual appearance across every beta/m ratio.
    # --------------------------------------------------------

    for (
        algorithm_index,
        algorithm,
    ) in enumerate(
        ALGORITHM_ORDER
    ):


        algorithm_data = []

        positions = []


        for (
            ratio_index,
            ratio,
        ) in enumerate(
            ratio_values
        ):


            mask = (

                np.isclose(

                    df[
                        "beta_m_ratio"
                    ].astype(float),

                    ratio,

                    rtol=1e-12,

                    atol=1e-15,
                )

                &

                (
                    df[
                        "algorithm"
                    ]
                    ==
                    algorithm
                )
            )


            values = (

                df.loc[
                    mask,
                    cost_column,
                ]

                .dropna()

                .astype(float)

                .to_numpy()
            )


            if len(
                values
            ) == 0:

                raise RuntimeError(
                    f"No {cost_column} values found for "
                    f"{algorithm}, beta/m={ratio:g}."
                )


            algorithm_data.append(
                values
            )


            positions.append(

                group_centers[
                    ratio_index
                ]

                +

                offsets[
                    algorithm_index
                ]
            )


        # ----------------------------------------------------
        # Draw all boxes for this algorithm
        # ----------------------------------------------------

        box_result = ax.boxplot(

            algorithm_data,

            positions=positions,

            widths=
                box_width
                *
                0.82,

            patch_artist=True,

            showmeans=False,

            showfliers=True,

            manage_ticks=False,
        )


        # ----------------------------------------------------
        # Same color for same algorithm across all ratios
        # ----------------------------------------------------

        for patch in box_result[
            "boxes"
        ]:

            patch.set_facecolor(
                algorithm_colors[
                    algorithm
                ]
            )

            patch.set_alpha(
                0.70
            )


        for median in box_result[
            "medians"
        ]:

            median.set_linewidth(
                1.4
            )


    # --------------------------------------------------------
    # X axis = beta/m ratios
    # --------------------------------------------------------

    ax.set_xticks(
        group_centers
    )


    ax.set_xticklabels(

        [
            f"{ratio:g}"

            for ratio
            in ratio_values
        ],

        fontsize=11,
    )


    ax.set_xlabel(

        r"$\beta/m$",

        fontsize=13,

        fontweight="bold",
    )


    ax.set_ylabel(

        ylabel,

        fontsize=13,

        fontweight="bold",
    )


    ax.tick_params(

        axis="y",

        labelsize=10,
    )


    ax.grid(

        axis="y",

        linestyle="--",

        linewidth=0.7,

        alpha=0.45,
    )


    ax.set_axisbelow(
        True
    )


# ============================================================
# 18. LEGEND
# ============================================================

legend_handles = [

    Patch(

        facecolor=
            algorithm_colors[
                algorithm
            ],

        alpha=0.70,

        label=
            ALGORITHM_LABELS[
                algorithm
            ],
    )

    for algorithm
    in ALGORITHM_ORDER
]


fig.legend(

    handles=
        legend_handles,

    loc=
        "upper center",

    bbox_to_anchor=(
        0.5,
        0.985,
    ),

    ncol=
        len(
            ALGORITHM_ORDER
        ),

    fontsize=10,

    frameon=True,
)


# ============================================================
# 19. TITLE
# ============================================================

fig.suptitle(

    (
        "Algorithm Cost Distributions Across "
        r"$\beta/m$ Ratios"
    ),

    fontsize=15,

    fontweight="bold",

    y=1.04,
)


fig.subplots_adjust(

    left=0.055,

    right=0.995,

    bottom=0.14,

    top=0.84,

    wspace=0.27,
)


# ============================================================
# 20. SAVE FIGURE
# ============================================================

COMBINED_PNG_FILE = (

    OUTPUT_DIR
    /
    "saved_beta_m_ratio_algorithm_boxplots.png"
)


COMBINED_PDF_FILE = (

    OUTPUT_DIR
    /
    "saved_beta_m_ratio_algorithm_boxplots.pdf"
)


fig.savefig(

    COMBINED_PNG_FILE,

    dpi=300,

    bbox_inches="tight",
)


fig.savefig(

    COMBINED_PDF_FILE,

    bbox_inches="tight",
)


# ============================================================
# 21. ALSO MAKE ONE LARGE FIGURE PER COST
#
# These are useful for the paper if the 3-panel version is
# too compressed.
# ============================================================

individual_plot_files = []


for (
    cost_column,
    ylabel,
) in COSTS:


    fig_single, ax = plt.subplots(

        figsize=(
            max(
                10.0,
                1.6
                *
                len(
                    ratio_values
                ),
            ),
            6.0,
        )
    )


    for (
        algorithm_index,
        algorithm,
    ) in enumerate(
        ALGORITHM_ORDER
    ):


        algorithm_data = []

        positions = []


        for (
            ratio_index,
            ratio,
        ) in enumerate(
            ratio_values
        ):


            mask = (

                np.isclose(

                    df[
                        "beta_m_ratio"
                    ].astype(float),

                    ratio,

                    rtol=1e-12,

                    atol=1e-15,
                )

                &

                (
                    df[
                        "algorithm"
                    ]
                    ==
                    algorithm
                )
            )


            values = (

                df.loc[
                    mask,
                    cost_column,
                ]

                .dropna()

                .astype(float)

                .to_numpy()
            )


            algorithm_data.append(
                values
            )


            positions.append(

                group_centers[
                    ratio_index
                ]

                +

                offsets[
                    algorithm_index
                ]
            )


        box_result = ax.boxplot(

            algorithm_data,

            positions=
                positions,

            widths=
                box_width
                *
                0.82,

            patch_artist=True,

            showmeans=False,

            showfliers=True,

            manage_ticks=False,
        )


        for patch in box_result[
            "boxes"
        ]:

            patch.set_facecolor(
                algorithm_colors[
                    algorithm
                ]
            )

            patch.set_alpha(
                0.70
            )


        for median in box_result[
            "medians"
        ]:

            median.set_linewidth(
                1.5
            )


    ax.set_xticks(
        group_centers
    )


    ax.set_xticklabels(

        [
            f"{ratio:g}"

            for ratio
            in ratio_values
        ],

        fontsize=12,
    )


    ax.set_xlabel(

        r"$\beta/m$",

        fontsize=14,

        fontweight="bold",
    )


    ax.set_ylabel(

        ylabel,

        fontsize=14,

        fontweight="bold",
    )


    ax.grid(

        axis="y",

        linestyle="--",

        linewidth=0.7,

        alpha=0.45,
    )


    ax.set_axisbelow(
        True
    )


    ax.legend(

        handles=
            legend_handles,

        loc=
            "best",

        fontsize=10,

        frameon=True,
    )


    ax.set_title(

        (
            f"{ylabel} Across "
            r"$\beta/m$ Ratios"
        ),

        fontsize=14,

        fontweight="bold",
    )


    fig_single.tight_layout()


    filename_label = (

        cost_column

        .replace(
            "_cost",
            "",
        )
    )


    output_png = (

        OUTPUT_DIR
        /
        (
            "saved_beta_m_ratio_"
            f"{filename_label}"
            "_boxplot.png"
        )
    )


    output_pdf = (

        OUTPUT_DIR
        /
        (
            "saved_beta_m_ratio_"
            f"{filename_label}"
            "_boxplot.pdf"
        )
    )


    fig_single.savefig(

        output_png,

        dpi=300,

        bbox_inches="tight",
    )


    fig_single.savefig(

        output_pdf,

        bbox_inches="tight",
    )


    individual_plot_files.append(
        (
            output_png,
            output_pdf,
        )
    )


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "PLOTS COMPLETE"
)

print(
    "========================================"
)


print(
    "\nRatios plotted:"
)

print(
    ratio_values
)


print(
    "\nCombined 3-panel PNG:"
)

print(
    COMBINED_PNG_FILE
)


print(
    "\nCombined 3-panel PDF:"
)

print(
    COMBINED_PDF_FILE
)


print(
    "\nCost summary:"
)

print(
    SUMMARY_OUTPUT_FILE
)


print(
    "\nIndividual plots:"
)


for (
    png_file,
    pdf_file,
) in individual_plot_files:

    print(
        "\n",
        png_file
    )

    print(
        pdf_file
    )


plt.show()