from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CHOOSE EXPERIMENT FOLDER
# ============================================================

SETTING_FOLDER = "beta_200_m_100_T_1440"


# ============================================================
# 2. PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

RESULT_ROOT = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "rho_7"
    / "1min_24hour"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)

SETTING_DIR = (
    RESULT_ROOT
    / SETTING_FOLDER
)

SUMMARY_FILE = (
    SETTING_DIR
    / "combined_algorithm_summary.csv"
)


# ============================================================
# 3. DISPLAY NAMES / ORDER
# ============================================================

DISPLAY_NAMES = {
    "OFFLINE-OPT": "OPT",
    "GREEDY": "GRD",
    "ROBD-SUM": "ROBD-SUM",
    "ROBD-MAX": "ROBD-MAX",
    "S-MROO-SUM": "S-MROO-SUM",
    "S-MROO-MAX": "S-MROO-MAX",
    "DMD": "DMD",
    "MROO": "MROO",
}


ALGORITHM_ORDER = [
    "OFFLINE-OPT",
    "GREEDY",
    "ROBD-SUM",
    "ROBD-MAX",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
]


# ============================================================
# 4. COST DEFINITIONS
# ============================================================

COST_ROWS = [
    (
        "Hitting Cost",
        "mean_hitting_cost",
        "std_hitting_cost",
    ),
    (
        "Long-term Cost",
        "mean_long_term_cost",
        "std_long_term_cost",
    ),
    (
        "Total Cost",
        "mean_total_cost",
        "std_total_cost",
    ),
]


# ============================================================
# 5. WHICH ALGORITHMS COUNT AS ONLINE
# ============================================================
#
# OFFLINE-OPT is excluded when choosing the bold minimum.
#

ONLINE_ALGORITHMS = {
    "GREEDY",
    "ROBD-SUM",
    "ROBD-MAX",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
}


# ============================================================
# 6. NUMBER FORMAT
# ============================================================

DECIMALS = 2


def format_cost(
    mean_value,
    std_value,
    bold=False,
):

    text = (
        f"{mean_value:.{DECIMALS}f}"
        f" $\\pm$ "
        f"{std_value:.{DECIMALS}f}"
    )

    if bold:

        return (
            "\\textbf{"
            + text
            + "}"
        )

    return text


# ============================================================
# 7. LOAD SUMMARY
# ============================================================

def load_summary():

    if not SUMMARY_FILE.exists():

        raise FileNotFoundError(
            "Could not find:\n"
            f"{SUMMARY_FILE}"
        )

    df = pd.read_csv(
        SUMMARY_FILE
    )

    required_columns = {
        "algorithm",
        "mean_hitting_cost",
        "std_hitting_cost",
        "mean_long_term_cost",
        "std_long_term_cost",
        "mean_total_cost",
        "std_total_cost",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "combined_algorithm_summary.csv "
            "is missing columns:\n"
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# 8. SELECT AVAILABLE ALGORITHMS
# ============================================================

def get_algorithm_rows(
    df,
):

    rows = {}

    available = set(
        df[
            "algorithm"
        ]
        .astype(str)
        .tolist()
    )

    for algorithm in ALGORITHM_ORDER:

        if algorithm not in available:

            continue

        algorithm_df = (
            df[
                df[
                    "algorithm"
                ]
                == algorithm
            ]
        )

        if len(
            algorithm_df
        ) != 1:

            raise RuntimeError(
                f"Expected exactly one summary row for "
                f"{algorithm}, but found "
                f"{len(algorithm_df)}."
            )

        rows[
            algorithm
        ] = (
            algorithm_df
            .iloc[0]
        )

    if not rows:

        raise RuntimeError(
            "No recognized algorithms were found."
        )

    return rows


# ============================================================
# 9. BUILD DISPLAY TABLE
# ============================================================

def build_display_table(
    algorithm_rows,
):

    algorithms = list(
        algorithm_rows.keys()
    )

    table = []

    for (
        row_label,
        mean_column,
        std_column,
    ) in COST_ROWS:

        # ----------------------------------------------------
        # Find minimum online cost for this metric
        # ----------------------------------------------------

        online_means = {}

        for algorithm in algorithms:

            if (
                algorithm
                not in ONLINE_ALGORITHMS
            ):

                continue

            row = (
                algorithm_rows[
                    algorithm
                ]
            )

            mean_value = float(
                row[
                    mean_column
                ]
            )

            if np.isfinite(
                mean_value
            ):

                online_means[
                    algorithm
                ] = mean_value

        if online_means:

            minimum_online_cost = min(
                online_means.values()
            )

        else:

            minimum_online_cost = None

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------

        output_row = {
            "Metrics":
                row_label
        }

        for algorithm in algorithms:

            row = (
                algorithm_rows[
                    algorithm
                ]
            )

            mean_value = float(
                row[
                    mean_column
                ]
            )

            std_value = float(
                row[
                    std_column
                ]
            )

            bold = False

            if (
                algorithm
                in ONLINE_ALGORITHMS

                and

                minimum_online_cost
                is not None

                and

                np.isclose(
                    mean_value,
                    minimum_online_cost,
                )
            ):

                bold = True

            output_row[
                DISPLAY_NAMES.get(
                    algorithm,
                    algorithm,
                )
            ] = format_cost(
                mean_value,
                std_value,
                bold=bold,
            )

        table.append(
            output_row
        )

    return pd.DataFrame(
        table
    )


# ============================================================
# 10. PRINT LATEX TABLE
# ============================================================

def print_latex_table(
    display_df,
):

    columns = list(
        display_df.columns
    )

    num_algorithms = (
        len(columns)
        - 1
    )

    alignment = (
        "l"
        +
        "c" * num_algorithms
    )

    print(
        "\n========================================"
    )

    print(
        "LATEX TABLE"
    )

    print(
        "========================================\n"
    )

    print(
        "\\begin{table}[t]"
    )

    print(
        "\\centering"
    )

    print(
        "\\small"
    )

    print(
        f"\\begin{{tabular}}{{{alignment}}}"
    )

    print(
        "\\toprule"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = (
        "Metrics"
        +
        " & "
        +
        " & ".join(
            columns[
                1:
            ]
        )
        +
        " \\\\"
    )

    print(
        header
    )

    print(
        "\\midrule"
    )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    for _, row in display_df.iterrows():

        values = [

            str(
                row[
                    column
                ]
            )

            for column
            in columns

        ]

        print(
            " & ".join(
                values
            )
            +
            " \\\\"
        )

    print(
        "\\bottomrule"
    )

    print(
        "\\end{tabular}"
    )

    print(
        "\\caption{Cost comparison of different "
        "algorithms over 100 continuous 24-hour workload "
        "windows. Each entry reports the mean cost with "
        "standard deviation. Minimum costs among online "
        "algorithms are highlighted in bold.}"
    )

    print(
        "\\label{tab:cost_comparison}"
    )

    print(
        "\\end{table}"
    )


# ============================================================
# 11. PRINT SIMPLE TERMINAL TABLE
# ============================================================

def print_terminal_table(
    display_df,
):

    # Remove LaTeX formatting for terminal display.

    terminal_df = (
        display_df
        .copy()
    )

    for column in terminal_df.columns[
        1:
    ]:

        terminal_df[
            column
        ] = (
            terminal_df[
                column
            ]
            .str.replace(
                "\\textbf{",
                "",
                regex=False,
            )
            .str.replace(
                "}",
                "",
                regex=False,
            )
            .str.replace(
                "$\\pm$",
                "±",
                regex=False,
            )
        )

    print(
        "\n========================================"
    )

    print(
        "COST TABLE"
    )

    print(
        "========================================\n"
    )

    print(
        terminal_df.to_string(
            index=False
        )
    )


# ============================================================
# 12. SAVE FORMATTED CSV
# ============================================================

def save_formatted_csv(
    display_df,
):

    output_file = (
        SETTING_DIR
        /
        "cost_comparison_table.csv"
    )

    csv_df = (
        display_df.copy()
    )

    for column in csv_df.columns[
        1:
    ]:

        csv_df[
            column
        ] = (
            csv_df[
                column
            ]
            .str.replace(
                "\\textbf{",
                "",
                regex=False,
            )
            .str.replace(
                "}",
                "",
                regex=False,
            )
            .str.replace(
                "$\\pm$",
                "±",
                regex=False,
            )
        )

    csv_df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nSaved formatted table:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Setting folder =",
        SETTING_FOLDER,
    )

    print(
        "Summary file =",
        SUMMARY_FILE,
    )

    df = load_summary()

    algorithm_rows = (
        get_algorithm_rows(
            df
        )
    )

    print(
        "\nAlgorithms included:"
    )

    for algorithm in algorithm_rows:

        print(
            " ",
            DISPLAY_NAMES.get(
                algorithm,
                algorithm,
            ),
        )

    display_df = (
        build_display_table(
            algorithm_rows
        )
    )

    print_terminal_table(
        display_df
    )

    print_latex_table(
        display_df
    )

    save_formatted_csv(
        display_df
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()