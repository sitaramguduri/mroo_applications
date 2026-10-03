from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
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


# ------------------------------------------------------------
# Choose the experiment folder to rebuild
# ------------------------------------------------------------

SETTING_FOLDER = "beta_75_m_150_T_1440"


SETTING_DIR = (
    BASE_RESULT_DIR
    / SETTING_FOLDER
)


N_WINDOWS = 100


# ============================================================
# 2. ALGORITHMS
# ============================================================

ALGORITHMS = [
    "OFFLINE-OPT",
    # "GREEDY",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
]


# ============================================================
# 3. RAW RESULT FILES
# ============================================================

RESULT_FILES = {

    "MROO":
        "mroo_window_results.csv",

    "S-MROO-MAX":
        "smroo_max_window_results.csv",

    "S-MROO-SUM":
        "smroo_sum_window_results.csv",

    "DMD":
        "dmd_window_results.csv",

    "OFFLINE-OPT":
        "offline_opt_window_results.csv",

    "GREEDY":
        "greedy_window_results.csv",

}


# ============================================================
# 4. SUMMARY FILES
# ============================================================

SUMMARY_FILES = {

    "MROO":
        "mroo_window_summary.csv",

    "S-MROO-MAX":
        "smroo_max_window_summary.csv",

    "S-MROO-SUM":
        "smroo_sum_window_summary.csv",

    "DMD":
        "dmd_window_summary.csv",

    "OFFLINE-OPT":
        "offline_opt_window_summary.csv",

    "GREEDY":
        "greedy_window_summary.csv",

}


# ============================================================
# 5. OUTPUT FILES
# ============================================================

COMBINED_WINDOW_FILE = (
    SETTING_DIR
    / "combined_algorithm_window_results.csv"
)

COMBINED_SUMMARY_FILE = (
    SETTING_DIR
    / "combined_algorithm_summary.csv"
)


# ============================================================
# 6. EXTRACT BETA / M / T FROM FOLDER NAME
# ============================================================

match = re.fullmatch(
    r"beta_([^_]+)_m_([^_]+)_T_([^_]+)",
    SETTING_FOLDER,
)

if match is None:

    raise RuntimeError(
        "Could not parse SETTING_FOLDER. "
        "Expected format: beta_<beta>_m_<m>_T_<T>"
    )


BETA = float(
    match.group(1)
)

M = float(
    match.group(2)
)

WINDOW_SIZE = int(
    float(
        match.group(3)
    )
)


print(
    "\n========================================"
)

print(
    "REBUILDING COMBINED ALGORITHM RESULTS"
)

print(
    "========================================"
)

print(
    "Setting directory =",
    SETTING_DIR,
)

print(
    "beta =",
    BETA,
)

print(
    "m =",
    M,
)

print(
    "T =",
    WINDOW_SIZE,
)


# ============================================================
# 7. HELPER:
# FIND MROO VECTOR-KAPPA COLUMNS
# ============================================================

def get_mroo_kappa_component_columns(df):

    prefix = "kappa_init_"

    columns = []

    for column in df.columns:

        column_string = str(
            column
        )

        if not column_string.startswith(
            prefix
        ):

            continue

        suffix = column_string[
            len(prefix):
        ]

        if suffix.isdigit():

            columns.append(
                column
            )


    return sorted(

        columns,

        key=lambda column:
            int(
                str(column)[
                    len(prefix):
                ]
            ),

    )


# ============================================================
# 8. LOAD RAW FILES
# ============================================================

def load_algorithm_files(
    algorithm,
):

    result_file = (
        SETTING_DIR
        / RESULT_FILES[
            algorithm
        ]
    )

    summary_file = (
        SETTING_DIR
        / SUMMARY_FILES[
            algorithm
        ]
    )


    if not result_file.exists():

        raise FileNotFoundError(
            f"\nMissing result file for {algorithm}:\n"
            f"{result_file}"
        )


    if not summary_file.exists():

        raise FileNotFoundError(
            f"\nMissing summary file for {algorithm}:\n"
            f"{summary_file}"
        )


    window_df = pd.read_csv(
        result_file,
        low_memory=False,
    )

    summary_df = pd.read_csv(
        summary_file,
        low_memory=False,
    )


    if len(window_df) == 0:

        raise RuntimeError(
            f"{algorithm} result file is empty."
        )


    if len(summary_df) == 0:

        raise RuntimeError(
            f"{algorithm} summary file is empty."
        )


    return (
        window_df,
        summary_df,
    )


# ============================================================
# 9. MROO SELECTION
# ============================================================

def select_mroo(
    window_df,
    summary_df,
):

    if "mean_total_cost" not in summary_df.columns:

        raise RuntimeError(
            "MROO summary has no mean_total_cost column."
        )


    best_config = (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


    lambda_1 = float(
        best_config[
            "lambda_1"
        ]
    )

    eta = float(
        best_config[
            "eta"
        ]
    )


    mask = (

        np.isclose(
            window_df[
                "lambda_1"
            ].astype(float),
            lambda_1,
        )

        &

        np.isclose(
            window_df[
                "eta"
            ].astype(float),
            eta,
        )

    )


    # --------------------------------------------------------
    # NEW VECTOR KAPPA FORMAT
    # --------------------------------------------------------

    kappa_columns = (
        get_mroo_kappa_component_columns(
            window_df
        )
    )


    if len(kappa_columns) > 0:

        for column in kappa_columns:

            if column not in best_config.index:

                raise RuntimeError(
                    "MROO summary is missing "
                    f"{column}"
                )


            target = float(
                best_config[
                    column
                ]
            )


            mask &= np.isclose(
                window_df[
                    column
                ].astype(float),
                target,
            )


    # --------------------------------------------------------
    # OLD SCALAR FORMAT
    # --------------------------------------------------------

    else:

        kappa = float(
            best_config[
                "kappa_init"
            ]
        )


        mask &= np.isclose(
            window_df[
                "kappa_init"
            ].astype(float),
            kappa,
        )


    selected = (
        window_df[
            mask
        ]
        .copy()
    )


    if len(selected) != N_WINDOWS:

        raise RuntimeError(
            f"MROO: expected {N_WINDOWS} selected windows, "
            f"found {len(selected)}."
        )


    return (
        selected,
        best_config,
    )


# ============================================================
# 10. S-MROO SELECTION
# ============================================================

def select_smroo(
    window_df,
    summary_df,
    algorithm,
):

    if "mean_total_cost" not in summary_df.columns:

        raise RuntimeError(
            f"{algorithm} summary has no mean_total_cost."
        )


    best_config = (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


    parameter_pair = int(
        best_config[
            "parameter_pair"
        ]
    )


    selected = (
        window_df[
            window_df[
                "parameter_pair"
            ].astype(int)
            == parameter_pair
        ]
        .copy()
    )


    if len(selected) != N_WINDOWS:

        raise RuntimeError(
            f"{algorithm}: expected "
            f"{N_WINDOWS} windows, "
            f"found {len(selected)}."
        )


    return (
        selected,
        best_config,
    )


# ============================================================
# 11. DMD SELECTION
# ============================================================

def select_dmd(
    window_df,
    summary_df,
):

    best_config = (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


    eta = float(
        best_config[
            "eta"
        ]
    )


    kappa_init = float(
        best_config[
            "kappa_init"
        ]
    )


    mask = (

        np.isclose(
            window_df[
                "eta"
            ].astype(float),
            eta,
        )

        &

        np.isclose(
            window_df[
                "kappa_init"
            ].astype(float),
            kappa_init,
        )

    )


    selected = (
        window_df[
            mask
        ]
        .copy()
    )


    if len(selected) != N_WINDOWS:

        raise RuntimeError(
            f"DMD: expected {N_WINDOWS} windows, "
            f"found {len(selected)}."
        )


    return (
        selected,
        best_config,
    )


# ============================================================
# 12. OFFLINE OPT
# ============================================================

def select_offline_opt(
    window_df,
    summary_df,
):

    if len(summary_df) != 1:

        raise RuntimeError(
            "Expected exactly one OFFLINE-OPT summary row, "
            f"found {len(summary_df)}."
        )


    if len(window_df) != N_WINDOWS:

        raise RuntimeError(
            f"OFFLINE-OPT: expected {N_WINDOWS} windows, "
            f"found {len(window_df)}."
        )


    return (
        window_df.copy(),
        summary_df.iloc[0],
    )


# ============================================================
# 13. GREEDY
# ============================================================

def select_greedy(
    window_df,
    summary_df,
):

    if len(summary_df) != 1:

        raise RuntimeError(
            "Expected exactly one GREEDY summary row, "
            f"found {len(summary_df)}."
        )


    if len(window_df) != N_WINDOWS:

        raise RuntimeError(
            f"GREEDY: expected {N_WINDOWS} windows, "
            f"found {len(window_df)}."
        )


    return (
        window_df.copy(),
        summary_df.iloc[0],
    )


# ============================================================
# 14. ADD METADATA
# ============================================================

def add_metadata(
    df,
    algorithm,
):

    df = df.copy()


    df[
        "algorithm"
    ] = algorithm


    df[
        "beta"
    ] = BETA


    df[
        "m"
    ] = M


    df[
        "beta_over_m"
    ] = (
        BETA
        / M
    )


    # --------------------------------------------------------
    # Put identifying columns first
    # --------------------------------------------------------

    first_columns = [
        "algorithm",
        "beta",
        "m",
        "beta_over_m",
    ]


    remaining_columns = [
        column
        for column
        in df.columns
        if column not in first_columns
    ]


    return df[
        first_columns
        + remaining_columns
    ]


# ============================================================
# 15. BUILD ONE COMBINED SUMMARY ROW
# ============================================================

def build_summary_row(
    algorithm,
    selected,
    best_config,
):

    row = {

        "algorithm":
            algorithm,

        "beta":
            BETA,

        "m":
            M,

        "beta_over_m":
            BETA / M,

        "n_windows":
            len(
                selected
            ),

        "mean_hitting_cost":
            float(
                selected[
                    "hitting_cost"
                ].mean()
            ),

        "std_hitting_cost":
            float(
                selected[
                    "hitting_cost"
                ].std()
            ),

        "mean_long_term_cost":
            float(
                selected[
                    "long_term_cost"
                ].mean()
            ),

        "std_long_term_cost":
            float(
                selected[
                    "long_term_cost"
                ].std()
            ),

        "mean_total_cost":
            float(
                selected[
                    "total_cost"
                ].mean()
            ),

        "std_total_cost":
            float(
                selected[
                    "total_cost"
                ].std()
            ),

    }


    # --------------------------------------------------------
    # Copy useful configuration metadata
    # --------------------------------------------------------

    optional_columns = [

        "lambda_1",

        "lambda_2",

        "eta",

        "kappa_init",

        "parameter_pair",

        "gamma_R_q",

        "L_q",

        "R",

        "solver_status",

        "max_objective_difference",

    ]


    for column in optional_columns:

        if column not in best_config.index:

            row[
                column
            ] = np.nan

            continue


        value = best_config[
            column
        ]


        if pd.isna(
            value
        ):

            row[
                column
            ] = np.nan


        elif column == "solver_status":

            row[
                column
            ] = str(
                value
            )


        elif column == "kappa_init":

            # MROO can now store vectors as strings.
            try:

                row[
                    column
                ] = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                row[
                    column
                ] = str(
                    value
                )


        elif column in {
            "parameter_pair",
            "R",
        }:

            row[
                column
            ] = int(
                value
            )


        else:

            row[
                column
            ] = float(
                value
            )


    # --------------------------------------------------------
    # Preserve MROO vector kappa components
    # --------------------------------------------------------

    kappa_component_columns = [
        column
        for column
        in best_config.index
        if (
            str(column).startswith(
                "kappa_init_"
            )
            and
            str(column)[
                len("kappa_init_"):
            ].isdigit()
        )
    ]


    kappa_component_columns = sorted(

        kappa_component_columns,

        key=lambda column:
            int(
                str(column)[
                    len("kappa_init_"):
                ]
            ),

    )


    for column in kappa_component_columns:

        value = best_config[
            column
        ]


        row[
            column
        ] = (
            np.nan
            if pd.isna(
                value
            )
            else float(
                value
            )
        )


    return row


# ============================================================
# 16. VERIFY SAME WINDOWS
# ============================================================

def verify_same_windows(
    combined_df,
):

    required = [
        "window_id",
        "start_index",
        "end_index",
    ]


    missing = [
        column
        for column
        in required
        if column not in combined_df.columns
    ]


    if missing:

        print(
            "\nWARNING: cannot verify identical windows. "
            f"Missing columns: {missing}"
        )

        return


    reference = None
    reference_algorithm = None


    for algorithm in ALGORITHMS:

        current = (
            combined_df[
                combined_df[
                    "algorithm"
                ]
                == algorithm
            ][
                required
            ]

            .sort_values(
                "window_id"
            )

            .reset_index(
                drop=True
            )
        )


        if len(current) != N_WINDOWS:

            raise RuntimeError(
                f"{algorithm} has {len(current)} windows, "
                f"expected {N_WINDOWS}."
            )


        if reference is None:

            reference = current

            reference_algorithm = (
                algorithm
            )

            continue


        if not current.equals(
            reference
        ):

            raise RuntimeError(
                f"{algorithm} does not use the same "
                f"windows as {reference_algorithm}."
            )


    print(
        "\nVerified: all algorithms use "
        "the same window definitions."
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    selected_frames = []

    summary_rows = []


    # ========================================================
    # LOAD AND SELECT ALL EXISTING ALGORITHMS
    # ========================================================

    for algorithm in ALGORITHMS:

        print(
            "\n----------------------------------------"
        )

        print(
            "Loading",
            algorithm,
        )


        (
            window_df,
            summary_df,
        ) = load_algorithm_files(
            algorithm
        )


        # ----------------------------------------------------
        # Select configuration
        # ----------------------------------------------------

        if algorithm == "MROO":

            (
                selected,
                best_config,
            ) = select_mroo(
                window_df,
                summary_df,
            )


        elif algorithm in {
            "S-MROO-SUM",
            "S-MROO-MAX",
        }:

            (
                selected,
                best_config,
            ) = select_smroo(
                window_df,
                summary_df,
                algorithm,
            )


        elif algorithm == "DMD":

            (
                selected,
                best_config,
            ) = select_dmd(
                window_df,
                summary_df,
            )


        elif algorithm == "OFFLINE-OPT":

            (
                selected,
                best_config,
            ) = select_offline_opt(
                window_df,
                summary_df,
            )


        elif algorithm == "GREEDY":

            (
                selected,
                best_config,
            ) = select_greedy(
                window_df,
                summary_df,
            )


        else:

            raise RuntimeError(
                f"Unknown algorithm: {algorithm}"
            )


        # ----------------------------------------------------
        # Add algorithm / beta / m metadata
        # ----------------------------------------------------

        selected = add_metadata(
            selected,
            algorithm,
        )


        selected_frames.append(
            selected
        )


        summary_row = (
            build_summary_row(
                algorithm,
                selected,
                best_config,
            )
        )


        summary_rows.append(
            summary_row
        )


        print(
            "Selected windows =",
            len(
                selected
            ),
        )


        if (
            "mean_total_cost"
            in summary_row
        ):

            print(
                "Mean total cost =",
                summary_row[
                    "mean_total_cost"
                ],
            )


        if algorithm == "MROO":

            print(
                "Selected MROO lambda_1 =",
                best_config[
                    "lambda_1"
                ],
            )

            print(
                "Selected MROO eta =",
                best_config[
                    "eta"
                ],
            )

            kappa_columns = (
                get_mroo_kappa_component_columns(
                    summary_df
                )
            )

            if kappa_columns:

                print(
                    "Selected MROO kappa =",
                    [
                        float(
                            best_config[
                                column
                            ]
                        )
                        for column
                        in kappa_columns
                    ],
                )

            else:

                print(
                    "Selected MROO kappa =",
                    best_config[
                        "kappa_init"
                    ],
                )


    # ========================================================
    # COMBINE ALL SIX
    # ========================================================

    combined_df = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )


    combined_df = (
        combined_df

        .sort_values(
            [
                "algorithm",
                "window_id",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    expected_rows = (
        len(
            ALGORITHMS
        )
        * N_WINDOWS
    )


    if len(
        combined_df
    ) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} combined rows, "
            f"found {len(combined_df)}."
        )


    # ========================================================
    # VERIFY SAME WINDOWS
    # ========================================================

    verify_same_windows(
        combined_df
    )


    # ========================================================
    # WRITE COMBINED WINDOW FILE
    # ========================================================

    combined_df.to_csv(
        COMBINED_WINDOW_FILE,
        index=False,
    )


    # ========================================================
    # WRITE COMBINED SUMMARY FILE
    # ========================================================

    combined_summary_df = (
        pd.DataFrame(
            summary_rows
        )

        .sort_values(
            "algorithm"
        )

        .reset_index(
            drop=True
        )
    )


    combined_summary_df.to_csv(
        COMBINED_SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # PRINT FINAL CHECK
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "COMBINED RESULTS REBUILT"
    )

    print(
        "========================================"
    )


    print(
        "\nRows by algorithm:"
    )


    print(
        combined_df[
            "algorithm"
        ]
        .value_counts()
        .reindex(
            ALGORITHMS
        )
    )


    print(
        "\nTotal combined rows =",
        len(
            combined_df
        ),
    )


    print(
        "\nWindow result file:"
    )

    print(
        COMBINED_WINDOW_FILE
    )


    print(
        "\nSummary file:"
    )

    print(
        COMBINED_SUMMARY_FILE
    )


    print(
        "\nCombined summary:\n"
    )


    display_columns = [

        "algorithm",

        "n_windows",

        "lambda_1",

        "lambda_2",

        "eta",

        "kappa_init",

        "kappa_init_0",

        "kappa_init_1",

        "kappa_init_2",

        "parameter_pair",

        "mean_hitting_cost",

        "mean_long_term_cost",

        "mean_total_cost",

    ]


    display_columns = [
        column
        for column
        in display_columns
        if column
        in combined_summary_df.columns
    ]


    print(
        combined_summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":

    main()