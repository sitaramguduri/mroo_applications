from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CHOOSE EXPERIMENT
# ============================================================
#
# Change ONLY this line when you want to plot another setting.
#
# Examples:
#   "beta_30_m_150_T_1440"
#   "beta_75_m_150_T_1440"
#   "beta_150_m_150_T_1440"
#
SETTING_FOLDER = "beta_900_m_150_T_1440"


# ============================================================
# 2. BASE PATHS
# ============================================================
#
# This script should be placed directly inside:
#
# algorithms/
#
# Example:
#
# algorithms/
# ├── plot_completed_boxplots.py
# ├── mroo.py
# ├── dmd.py
# └── new_results/
#

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# 3. RESULTS ROOT
# ============================================================

RESULT_ROOT = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "rho_7"
    / "1min_24hour"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)


# ============================================================
# 4. SELECTED SETTING DIRECTORY
# ============================================================

SETTING_DIR = (
    RESULT_ROOT
    / SETTING_FOLDER
)


# ============================================================
# 5. INPUT FILE
# ============================================================

COMBINED_WINDOW_FILE = (
    SETTING_DIR
    / "combined_algorithm_window_results.csv"
)


# ============================================================
# 6. OUTPUT DIRECTORY
# ============================================================
#
# Plots will be saved inside:
#
# beta_30_m_150_T_1440/
# └── boxplots/
#

OUTPUT_DIR = (
    SETTING_DIR
    / "boxplots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 7. COMPLETION FILES
# ============================================================
#
# An algorithm is considered completed if its corresponding
# DONE file exists.
#
DONE_FILES = {
    "MROO":
        "mroo_window_DONE.txt",

    "S-MROO-MAX":
        "smroo_max_window_DONE.txt",

    "S-MROO-SUM":
        "smroo_sum_window_DONE.txt",

    "DMD":
        "dmd_window_DONE.txt",

    "OFFLINE-OPT":
        "offline_opt_window_DONE.txt",

    "GREEDY":
        "greedy_window_DONE.txt",
}


# ============================================================
# 8. DISPLAY ORDER
# ============================================================

ALGORITHM_ORDER = [
    "OFFLINE-OPT",
    "GREEDY",
    "DMD",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "MROO",
]


# ============================================================
# 9. COSTS TO PLOT
# ============================================================

COSTS = {
    "hitting_cost":
        "Hitting Cost",

    "long_term_cost":
        "Long-Term Cost",

    "total_cost":
        "Total Cost",
}


# ============================================================
# 10. FIND COMPLETED ALGORITHMS
# ============================================================

def get_completed_algorithms():

    completed = []

    for (
        algorithm,
        done_filename,
    ) in DONE_FILES.items():

        done_file = (
            SETTING_DIR
            / done_filename
        )

        if done_file.exists():

            completed.append(
                algorithm
            )

    return completed


# ============================================================
# 11. LOAD COMBINED RESULTS
# ============================================================

def load_results():

    if not RESULT_ROOT.exists():

        raise FileNotFoundError(
            "Result root does not exist:\n"
            f"{RESULT_ROOT}"
        )

    if not SETTING_DIR.exists():

        raise FileNotFoundError(
            "Setting folder does not exist:\n"
            f"{SETTING_DIR}"
        )

    if not COMBINED_WINDOW_FILE.exists():

        raise FileNotFoundError(
            "Combined window result file does not exist:\n"
            f"{COMBINED_WINDOW_FILE}"
        )

    df = pd.read_csv(
        COMBINED_WINDOW_FILE
    )

    required_columns = {
        "algorithm",
        "hitting_cost",
        "long_term_cost",
        "total_cost",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Combined result file is missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# 12. FILTER TO COMPLETED ALGORITHMS
# ============================================================

def get_completed_results(
    df,
):

    completed_algorithms = (
        get_completed_algorithms()
    )

    if not completed_algorithms:

        raise RuntimeError(
            "No completed algorithms were found in:\n"
            f"{SETTING_DIR}"
        )

    print(
        "\nCompleted algorithms detected:"
    )

    for algorithm in completed_algorithms:

        print(
            f"  - {algorithm}"
        )

    available_algorithms = set(
        df[
            "algorithm"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    # --------------------------------------------------------
    # Check if DONE exists but combined CSV is missing data
    # --------------------------------------------------------

    missing_from_combined = [

        algorithm

        for algorithm
        in completed_algorithms

        if algorithm
        not in available_algorithms

    ]

    if missing_from_combined:

        print(
            "\nWARNING:"
        )

        print(
            "These algorithms have DONE files but are not "
            "present in combined_algorithm_window_results.csv:"
        )

        for algorithm in missing_from_combined:

            print(
                f"  - {algorithm}"
            )

    # --------------------------------------------------------
    # Preferred algorithm order
    # --------------------------------------------------------

    algorithms_to_plot = [

        algorithm

        for algorithm
        in ALGORITHM_ORDER

        if (
            algorithm
            in completed_algorithms

            and

            algorithm
            in available_algorithms
        )

    ]

    # --------------------------------------------------------
    # Include any unexpected algorithm names too
    # --------------------------------------------------------

    for algorithm in completed_algorithms:

        if (
            algorithm
            in available_algorithms

            and

            algorithm
            not in algorithms_to_plot
        ):

            algorithms_to_plot.append(
                algorithm
            )

    if not algorithms_to_plot:

        raise RuntimeError(
            "No completed algorithms have data in the "
            "combined result file."
        )

    filtered_df = (

        df[
            df[
                "algorithm"
            ].isin(
                algorithms_to_plot
            )
        ]

        .copy()

    )

    return (
        filtered_df,
        algorithms_to_plot,
    )


# ============================================================
# 13. PRINT NUMBER OF WINDOWS
# ============================================================

def print_window_counts(
    df,
    algorithms,
):

    print(
        "\nRows used for plotting:"
    )

    for algorithm in algorithms:

        count = len(

            df[
                df[
                    "algorithm"
                ]
                == algorithm
            ]

        )

        print(
            f"  {algorithm}: {count}"
        )


# ============================================================
# 14. PLOT SINGLE COST
# ============================================================

def plot_one_cost(
    df,
    algorithms,
    cost_column,
    title,
):

    data = []

    labels = []

    for algorithm in algorithms:

        values = (

            df.loc[
                df[
                    "algorithm"
                ]
                == algorithm,

                cost_column,
            ]

            .dropna()

            .astype(float)

            .to_numpy()

        )

        if len(values) == 0:

            print(
                f"Skipping {algorithm} "
                f"for {cost_column}: "
                "no valid values."
            )

            continue

        data.append(
            values
        )

        labels.append(
            algorithm
        )

    if not data:

        print(
            f"No valid data available "
            f"for {cost_column}."
        )

        return

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_title(
        f"{title}\n"
        f"{SETTING_FOLDER}"
    )

    ax.set_xlabel(
        "Algorithm"
    )

    ax.set_ylabel(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    output_file = (

        OUTPUT_DIR

        / (
            f"boxplot_"
            f"{cost_column}_"
            f"{SETTING_FOLDER}.png"
        )

    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# 15. PLOT ALL THREE COSTS
# ============================================================

def plot_all_costs(
    df,
    algorithms,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            6,
        ),
    )

    for (
        ax,
        (
            cost_column,
            title,
        ),
    ) in zip(
        axes,
        COSTS.items(),
    ):

        data = []

        labels = []

        for algorithm in algorithms:

            values = (

                df.loc[
                    df[
                        "algorithm"
                    ]
                    == algorithm,

                    cost_column,
                ]

                .dropna()

                .astype(float)

                .to_numpy()

            )

            if len(values) == 0:

                continue

            data.append(
                values
            )

            labels.append(
                algorithm
            )

        if not data:

            ax.set_visible(
                False
            )

            continue

        ax.boxplot(
            data,
            tick_labels=labels,
            showmeans=True,
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Algorithm"
        )

        ax.set_ylabel(
            title
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        ax.tick_params(
            axis="x",
            rotation=20,
        )

    fig.suptitle(
        f"Algorithm Cost Comparison\n"
        f"{SETTING_FOLDER}"
    )

    plt.tight_layout()

    output_file = (

        OUTPUT_DIR

        / (
            f"boxplot_all_costs_"
            f"{SETTING_FOLDER}.png"
        )

    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "COMPLETED-ALGORITHM BOXPLOT GENERATOR"
    )

    print(
        "========================================"
    )

    print(
        "\nAlgorithms folder:"
    )

    print(
        ALGORITHM_DIR
    )

    print(
        "\nResults root:"
    )

    print(
        RESULT_ROOT
    )

    print(
        "\nSelected setting:"
    )

    print(
        SETTING_FOLDER
    )

    print(
        "\nSetting directory:"
    )

    print(
        SETTING_DIR
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_results()

    # --------------------------------------------------------
    # Detect completed algorithms
    # --------------------------------------------------------

    (
        completed_df,
        algorithms,
    ) = get_completed_results(
        df
    )

    print(
        "\nAlgorithms being plotted:"
    )

    for algorithm in algorithms:

        print(
            f"  - {algorithm}"
        )

    # --------------------------------------------------------
    # Print window counts
    # --------------------------------------------------------

    print_window_counts(
        completed_df,
        algorithms,
    )

    # --------------------------------------------------------
    # Individual plots
    # --------------------------------------------------------

    print(
        "\nGenerating individual cost plots..."
    )

    for (
        cost_column,
        title,
    ) in COSTS.items():

        plot_one_cost(
            completed_df,
            algorithms,
            cost_column,
            title,
        )

    # --------------------------------------------------------
    # Combined plot
    # --------------------------------------------------------

    print(
        "\nGenerating combined cost plot..."
    )

    plot_all_costs(
        completed_df,
        algorithms,
    )

    print(
        "\n========================================"
    )

    print(
        "BOXPLOTS COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()