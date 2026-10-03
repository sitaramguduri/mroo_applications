from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# BASE PATH
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

BASE_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)

PLOT_DIR = (
    BASE_RESULT_DIR
    / "comparison_plots_T_17280"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHOOSE THE SETTINGS TO COMPARE
#
# Each entry becomes one ROW in the final figure.
# We are using only window size 17280 here.
# ============================================================

EXPERIMENTS = [

    # {
    #     "folder": "beta_1p5_m_15_T_17280",
    #     "label": r"$\beta=1.5,\ m=15,\ T=17280$",
    # },

    {
        "folder": "beta_15_m_15_T_17280",
        "label": r"$\beta=15,\ m=15,\ T=17280$",
    },

    {
        "folder": "beta_900_m_15_T_17280",
        "label": r"$\beta=900,\ m=15,\ T=17280$",
    },

    {
        "folder": "beta_1800_m_15_T_17280",
        "label": r"$\beta=1800,\ m=15,\ T=17280$",
    },

]


# ============================================================
# ALGORITHM ORDER
#
# DMD ADDED HERE
# ============================================================

ALGORITHM_ORDER = [

    "MROO",

    "S-MROO-MAX",

    "S-MROO-SUM",

    "DMD",

    "OFFLINE-OPT",

    "GREEDY",

]


# ============================================================
# SHORT LABELS FOR PLOTS
#
# DMD ADDED HERE
# ============================================================

ALGORITHM_LABELS = {

    "MROO":
        "MROO",

    "S-MROO-MAX":
        "S-MROO-MAX",

    "S-MROO-SUM":
        "S-MROO-SUM",

    "DMD":
        "DMD",

    "OFFLINE-OPT":
        "OPT",

    "GREEDY":
        "GRD",

}


# ============================================================
# COST METRICS
# ============================================================

COST_COLUMNS = [

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
# LOAD DATA
# ============================================================

all_results = []


for experiment in EXPERIMENTS:


    result_file = (

        BASE_RESULT_DIR

        / experiment[
            "folder"
        ]

        / "combined_algorithm_window_results.csv"

    )


    if not result_file.exists():

        raise FileNotFoundError(

            f"Could not find:\n"

            f"{result_file}"

        )


    df = pd.read_csv(
        result_file
    )


    df[
        "experiment_folder"
    ] = experiment[
        "folder"
    ]


    df[
        "experiment_label"
    ] = experiment[
        "label"
    ]


    all_results.append(
        df
    )


    print(
        "\nLoaded:",
        result_file,
    )


    if "algorithm" in df.columns:

        print(
            df[
                "algorithm"
            ]
            .value_counts()
        )


all_df = pd.concat(

    all_results,

    ignore_index=True,

)


# ============================================================
# OPTIONAL CHECK
# ============================================================

if "window_size" in all_df.columns:


    unique_windows = sorted(

        all_df[
            "window_size"
        ]

        .dropna()

        .unique()

        .tolist()

    )


    print(
        "\nWindow sizes found:",
        unique_windows,
    )


    if unique_windows != [17280]:

        print(
            "WARNING: Data contains "
            "window sizes other than 17280."
        )


# ============================================================
# VERIFY DMD IS PRESENT
# ============================================================

print(
    "\n========================================"
)

print(
    "ALGORITHMS FOUND"
)

print(
    "========================================"
)


print(
    sorted(
        all_df[
            "algorithm"
        ]
        .unique()
        .tolist()
    )
)


if "DMD" not in all_df["algorithm"].unique():

    print(
        "\nWARNING: DMD is not present in the "
        "combined result files."
    )

    print(
        "Make sure main_sweep.py has been rerun "
        "after adding DMD."
    )


# ============================================================
# PLOT
#
# ONE ROW PER (beta, m)
# THREE COLUMNS:
#
#   Hitting Cost
#   Long-Term Cost
#   Total Cost
# ============================================================

n_rows = len(
    EXPERIMENTS
)

n_cols = 3


fig, axes = plt.subplots(

    n_rows,

    n_cols,

    figsize=(
        18,
        5 * n_rows,
    ),

    squeeze=False,

)


# ============================================================
# DRAW EACH ROW
# ============================================================

for row_idx, experiment in enumerate(
    EXPERIMENTS
):


    experiment_df = all_df[

        all_df[
            "experiment_folder"
        ]

        == experiment[
            "folder"
        ]

    ].copy()


    if len(
        experiment_df
    ) == 0:

        raise ValueError(

            f"No data found for "

            f"{experiment['folder']}"

        )


    # --------------------------------------------------------
    # Keep only algorithms actually present
    # --------------------------------------------------------

    algorithm_order = [

        algorithm

        for algorithm
        in ALGORITHM_ORDER

        if algorithm
        in experiment_df[
            "algorithm"
        ]
        .unique()

    ]


    print(
        "\nExperiment:",
        experiment[
            "folder"
        ],
    )


    print(
        "Algorithms plotted:",
        algorithm_order,
    )


    # ========================================================
    # THREE COST PANELS
    # ========================================================

    for col_idx, (
        cost_column,
        title_name,
    ) in enumerate(
        COST_COLUMNS
    ):


        ax = axes[
            row_idx,
            col_idx,
        ]


        box_data = []

        tick_labels = []


        # ----------------------------------------------------
        # Gather 100-window cost values per algorithm
        # ----------------------------------------------------

        for algorithm in algorithm_order:


            values = (

                experiment_df[

                    experiment_df[
                        "algorithm"
                    ]

                    == algorithm

                ][
                    cost_column
                ]

                .dropna()

                .to_numpy()

            )


            if len(
                values
            ) > 0:


                box_data.append(
                    values
                )


                tick_labels.append(

                    ALGORITHM_LABELS.get(

                        algorithm,

                        algorithm,

                    )

                )


        # ----------------------------------------------------
        # Boxplot
        # ----------------------------------------------------

        ax.boxplot(

            box_data,

            tick_labels=
                tick_labels,

            showmeans=True,

        )


        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(

            f"{title_name}\n"

            f"{experiment['label']}",

            fontsize=12,

            fontweight="bold",

        )


        # ----------------------------------------------------
        # X label
        # ----------------------------------------------------

        ax.set_xlabel(

            "Algorithm",

            fontweight="bold",

        )


        # ----------------------------------------------------
        # Y label
        # ----------------------------------------------------

        ax.set_ylabel(

            title_name,

            fontweight="bold",

        )


        # ----------------------------------------------------
        # X-axis formatting
        # ----------------------------------------------------

        ax.tick_params(

            axis="x",

            rotation=20,

            labelsize=9,

        )


        for label in ax.get_xticklabels():

            label.set_fontweight(
                "bold"
            )


        # ----------------------------------------------------
        # Y-axis formatting
        # ----------------------------------------------------

        for label in ax.get_yticklabels():

            label.set_fontweight(
                "bold"
            )


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(

            axis="y",

            linestyle="--",

            linewidth=0.7,

            alpha=0.3,

        )


        ax.set_axisbelow(
            True
        )


        # ----------------------------------------------------
        # Thicker borders
        # ----------------------------------------------------

        for spine in ax.spines.values():

            spine.set_linewidth(
                1.2
            )


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(

    "Cost Comparison Across Different "
    "$(\\beta,m)$ Settings\n"
    "Window Size = 17280",

    fontsize=16,

    fontweight="bold",

)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(

    rect=[
        0,
        0,
        1,
        0.97,
    ]

)


# ============================================================
# OUTPUT FILE
# ============================================================

output_file = (

    PLOT_DIR

    / "grid_boxplots_beta_m_T_17280_with_dmd.png"

)


# ============================================================
# SAVE
# ============================================================

fig.savefig(

    output_file,

    dpi=300,

    bbox_inches="tight",

)


print(
    "\nSaved:",
    output_file,
)


# ============================================================
# SHOW
# ============================================================

plt.show()