from pathlib import Path
import ast

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
    / "matrix_memory_v1"
    / "1min_24hour"
    / "matrix_memory"
)


# ------------------------------------------------------------
# Legacy directory fallback
# ------------------------------------------------------------

if not BASE_RESULT_DIR.exists():

    legacy_dir = (
        ALGORITHM_DIR
        / "matrix_memory_v1"
        / "1min_24hour"
        / "matrix_weights"
    )

    if legacy_dir.exists():

        BASE_RESULT_DIR = (
            legacy_dir
        )


# ============================================================
# 2. EXPERIMENT TO PLOT
# ============================================================

SETTING_FOLDER = (
    "beta_200_m_100_T_1440"
)

N_WINDOWS = 100


SETTING_DIR = (
    BASE_RESULT_DIR
    / SETTING_FOLDER
)


# ============================================================
# 3. CONFIGURATIONS TO PLOT
#
# CHANGE ONLY THIS SECTION when you want a different
# algorithm configuration in the main comparison.
# ============================================================


# ------------------------------------------------------------
# MROO
# ------------------------------------------------------------

MROO_ETA = 1e-7

# Scalar:
#     [3.0] means effective [3, 3, 3]
#
# Or explicitly:
#     [3.0, 3.0, 3.0]
#
MROO_KAPPA = [
    3.0,
]


# ------------------------------------------------------------
# DMD
# ------------------------------------------------------------

DMD_ETA = 1e-7

# DMD normally stores scalar kappa_init.
DMD_KAPPA = 3.0


# ------------------------------------------------------------
# S-MROO-SUM
# ------------------------------------------------------------

SMROO_SUM_LAMBDA_1 = 8

SMROO_SUM_LAMBDA_2 = 346


# ------------------------------------------------------------
# S-MROO-MAX
# ------------------------------------------------------------

SMROO_MAX_LAMBDA_1 = 0.4

SMROO_MAX_LAMBDA_2 = 0


# ============================================================
# 4. RESULT FILES
# ============================================================

HISTORY_DIR = (
    SETTING_DIR
    / "history"
)


HISTORY_RESULT_FILE = (
    HISTORY_DIR
    / "all_algorithm_window_results.csv"
)


HISTORY_SUMMARY_FILE = (
    HISTORY_DIR
    / "all_algorithm_summary.csv"
)


# ------------------------------------------------------------
# Current-comparison fallback
# ------------------------------------------------------------

CURRENT_COMPARISON_FILE = (
    SETTING_DIR
    / "current_comparison"
    / "combined_algorithm_window_results.csv"
)


LEGACY_COMPARISON_FILE = (
    SETTING_DIR
    / "combined_algorithm_window_results.csv"
)


# ------------------------------------------------------------
# Raw MROO fallback for tuning plots
# ------------------------------------------------------------

MROO_RESULT_FILE = (
    SETTING_DIR
    / "mroo_window_results.csv"
)


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

def config_isclose(
    values,
    target,
):

    return np.isclose(
        np.asarray(
            values,
            dtype=float,
        ),
        float(
            target
        ),
        rtol=1e-12,
        atol=1e-15,
    )


def get_kappa_columns(
    frame,
):

    prefix = (
        "kappa_init_"
    )

    columns = [

        column

        for column
        in frame.columns

        if (
            str(
                column
            ).startswith(
                prefix
            )
            and
            str(
                column
            )[
                len(
                    prefix
                ):
            ].isdigit()
        )
    ]

    return sorted(

        columns,

        key=lambda column:

            int(
                str(
                    column
                )[
                    len(
                        prefix
                    ):
                ]
            ),
    )


def parse_kappa(
    value,
):

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        return (
            np.asarray(
                value,
                dtype=float,
            )
            .reshape(-1)
        )


    text = str(
        value
    ).strip()


    # Scalar value
    try:

        return np.asarray(
            [
                float(
                    text
                )
            ],
            dtype=float,
        )

    except ValueError:

        pass


    parsed = (
        ast.literal_eval(
            text
        )
    )


    return (
        np.asarray(
            parsed,
            dtype=float,
        )
        .reshape(-1)
    )


def expand_kappa(
    value,
    dimension,
):

    values = (
        np.asarray(
            value,
            dtype=float,
        )
        .reshape(-1)
    )


    if values.size == 1:

        return np.full(
            dimension,
            float(
                values[0]
            ),
            dtype=float,
        )


    if values.size != dimension:

        raise RuntimeError(
            "Kappa dimension mismatch. "
            f"Requested {values.size} values "
            f"but data has dimension {dimension}."
        )


    return values


def kappa_string(
    values,
):

    values = (
        np.asarray(
            values,
            dtype=float,
        )
        .reshape(-1)
    )


    return (
        "["
        +
        ", ".join(
            f"{value:.6g}"
            for value
            in values
        )
        +
        "]"
    )


def first_non_nan(
    frame,
    column,
):

    if (
        column
        not in frame.columns
    ):

        return np.nan


    values = (
        frame[
            column
        ]
        .dropna()
    )


    if len(
        values
    ) == 0:

        return np.nan


    return (
        values.iloc[0]
    )


# ============================================================
# 6. LOAD PERSISTENT HISTORY
# ============================================================

if HISTORY_RESULT_FILE.exists():

    all_results_df = (
        pd.read_csv(
            HISTORY_RESULT_FILE,
            low_memory=False,
        )
    )

    DATA_SOURCE = (
        HISTORY_RESULT_FILE
    )


else:

    # --------------------------------------------------------
    # Fallback for experiments generated before history/
    # existed.
    # --------------------------------------------------------

    if CURRENT_COMPARISON_FILE.exists():

        all_results_df = (
            pd.read_csv(
                CURRENT_COMPARISON_FILE,
                low_memory=False,
            )
        )

        DATA_SOURCE = (
            CURRENT_COMPARISON_FILE
        )


    elif LEGACY_COMPARISON_FILE.exists():

        all_results_df = (
            pd.read_csv(
                LEGACY_COMPARISON_FILE,
                low_memory=False,
            )
        )

        DATA_SOURCE = (
            LEGACY_COMPARISON_FILE
        )


    else:

        raise FileNotFoundError(
            "\nCould not find any result file.\n"
            f"Tried:\n"
            f"{HISTORY_RESULT_FILE}\n"
            f"{CURRENT_COMPARISON_FILE}\n"
            f"{LEGACY_COMPARISON_FILE}"
        )


print(
    "\n========================================"
)

print(
    "MATRIX-MEMORY PLOTTER"
)

print(
    "========================================"
)

print(
    "\nSetting:"
)

print(
    SETTING_DIR
)

print(
    "\nReading:"
)

print(
    DATA_SOURCE
)


# ============================================================
# 7. VALIDATE DATA
# ============================================================

required_columns = {

    "algorithm",

    "window_id",

    "start_index",

    "end_index",

    "hitting_cost",

    "long_term_cost",

    "total_cost",

    "beta",

    "m",
}


missing_columns = (
    required_columns
    -
    set(
        all_results_df.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "Result file is missing columns: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# 8. EXPERIMENT INFORMATION
# ============================================================

beta_values = (

    all_results_df[
        "beta"
    ]

    .dropna()

    .astype(float)

    .unique()
)


m_values = (

    all_results_df[
        "m"
    ]

    .dropna()

    .astype(float)

    .unique()
)


if len(
    beta_values
) != 1:

    raise RuntimeError(
        "Expected one beta value, found "
        f"{beta_values}."
    )


if len(
    m_values
) != 1:

    raise RuntimeError(
        "Expected one m value, found "
        f"{m_values}."
    )


BETA = float(
    beta_values[0]
)


M = float(
    m_values[0]
)


if (
    "experiment_window_size"
    in all_results_df.columns
):

    window_sizes = (

        all_results_df[
            "experiment_window_size"
        ]

        .dropna()

        .astype(int)

        .unique()
    )

else:

    window_sizes = []


if len(
    window_sizes
) == 1:

    WINDOW_SIZE = int(
        window_sizes[0]
    )


else:

    try:

        WINDOW_SIZE = int(

            SETTING_FOLDER

            .split(
                "_T_"
            )[-1]
        )

    except Exception as error:

        raise RuntimeError(
            "Could not determine window size."
        ) from error


WINDOW_HOURS = (
    WINDOW_SIZE
    /
    60.0
)


print(
    "\n========================================"
)

print(
    "EXPERIMENT"
)

print(
    "========================================"
)

print(
    "beta        =",
    BETA,
)

print(
    "m           =",
    M,
)

print(
    "beta/m      =",
    BETA / M,
)

print(
    "window size =",
    WINDOW_SIZE,
)

print(
    "hours       =",
    WINDOW_HOURS,
)


print(
    "\nStored rows by algorithm:"
)

print(
    all_results_df[
        "algorithm"
    ]
    .value_counts()
)


# ============================================================
# 9. SHOW AVAILABLE CONFIGURATIONS
# ============================================================

print(
    "\n========================================"
)

print(
    "AVAILABLE STORED CONFIGURATIONS"
)

print(
    "========================================"
)


for algorithm in [
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
]:

    algorithm_df = (

        all_results_df[
            all_results_df[
                "algorithm"
            ]
            ==
            algorithm
        ]

        .copy()
    )


    print(
        f"\n{algorithm}:"
    )


    if len(
        algorithm_df
    ) == 0:

        print(
            "  NONE"
        )

        continue


    if algorithm in {
        "S-MROO-SUM",
        "S-MROO-MAX",
    }:

        columns = [

            column

            for column
            in [
                "lambda_1",
                "lambda_2",
            ]

            if column
            in algorithm_df.columns
        ]


    elif algorithm == "DMD":

        columns = [

            column

            for column
            in [
                "eta",
                "kappa_init",
            ]

            if column
            in algorithm_df.columns
        ]


    else:

        columns = [

            column

            for column
            in [
                "lambda_1",
                "eta",
            ]

            if column
            in algorithm_df.columns
        ]


        columns.extend(
            get_kappa_columns(
                algorithm_df
            )
        )


        if (
            len(
                get_kappa_columns(
                    algorithm_df
                )
            ) == 0
            and
            "kappa_init"
            in algorithm_df.columns
        ):

            columns.append(
                "kappa_init"
            )


    if (
        "config_id"
        in algorithm_df.columns
    ):

        columns.append(
            "config_id"
        )


    if len(
        columns
    ) > 0:

        available = (

            algorithm_df[
                columns
            ]

            .drop_duplicates()

            .reset_index(
                drop=True
            )
        )


        print(
            available.to_string(
                index=False
            )
        )


# ============================================================
# 10. SELECT S-MROO CONFIGURATION
# ============================================================

def select_smroo(
    frame,
    algorithm,
    lambda_1,
    lambda_2,
):

    algorithm_df = (

        frame[
            frame[
                "algorithm"
            ]
            ==
            algorithm
        ]

        .copy()
    )


    if len(
        algorithm_df
    ) == 0:

        raise RuntimeError(
            f"No {algorithm} results exist."
        )


    required = {
        "lambda_1",
        "lambda_2",
    }


    if not required.issubset(
        algorithm_df.columns
    ):

        raise RuntimeError(
            f"{algorithm} results do not contain "
            "lambda_1/lambda_2."
        )


    mask = (

        config_isclose(

            algorithm_df[
                "lambda_1"
            ],

            lambda_1,
        )

        &

        config_isclose(

            algorithm_df[
                "lambda_2"
            ],

            lambda_2,
        )
    )


    selected = (

        algorithm_df[
            mask
        ]

        .copy()

        .drop_duplicates(

            subset=[
                "window_id",
            ],

            keep="last",
        )

        .sort_values(
            "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        selected
    ) != N_WINDOWS:

        available = (

            algorithm_df[
                [
                    "lambda_1",
                    "lambda_2",
                ]
            ]

            .drop_duplicates()

            .sort_values(
                [
                    "lambda_1",
                    "lambda_2",
                ]
            )
        )


        raise RuntimeError(
            f"\n{algorithm}: configuration not available.\n"
            f"Requested:\n"
            f"  lambda_1 = {lambda_1}\n"
            f"  lambda_2 = {lambda_2}\n"
            f"Expected {N_WINDOWS} rows but found "
            f"{len(selected)}.\n\n"
            "Available configurations:\n"
            f"{available.to_string(index=False)}"
        )


    return selected


# ============================================================
# 11. SELECT DMD CONFIGURATION
# ============================================================

def select_dmd(
    frame,
    eta,
    kappa,
):

    algorithm_df = (

        frame[
            frame[
                "algorithm"
            ]
            ==
            "DMD"
        ]

        .copy()
    )


    if len(
        algorithm_df
    ) == 0:

        raise RuntimeError(
            "No DMD results exist."
        )


    if (
        "eta"
        not in algorithm_df.columns
        or
        "kappa_init"
        not in algorithm_df.columns
    ):

        raise RuntimeError(
            "DMD results require eta and kappa_init."
        )


    mask = (

        config_isclose(

            algorithm_df[
                "eta"
            ],

            eta,
        )

        &

        config_isclose(

            algorithm_df[
                "kappa_init"
            ],

            kappa,
        )
    )


    selected = (

        algorithm_df[
            mask
        ]

        .copy()

        .drop_duplicates(

            subset=[
                "window_id",
            ],

            keep="last",
        )

        .sort_values(
            "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        selected
    ) != N_WINDOWS:

        available = (

            algorithm_df[
                [
                    "eta",
                    "kappa_init",
                ]
            ]

            .drop_duplicates()

            .sort_values(
                [
                    "eta",
                    "kappa_init",
                ]
            )
        )


        raise RuntimeError(
            "\nDMD configuration not available.\n"
            f"Requested:\n"
            f"  eta   = {eta}\n"
            f"  kappa = {kappa}\n"
            f"Expected {N_WINDOWS} rows but found "
            f"{len(selected)}.\n\n"
            "Available configurations:\n"
            f"{available.to_string(index=False)}"
        )


    return selected


# ============================================================
# 12. SELECT MROO CONFIGURATION
# ============================================================

def select_mroo(
    frame,
    eta,
    kappa,
):

    algorithm_df = (

        frame[
            frame[
                "algorithm"
            ]
            ==
            "MROO"
        ]

        .copy()
    )


    if len(
        algorithm_df
    ) == 0:

        raise RuntimeError(
            "No MROO results exist."
        )


    if (
        "eta"
        not in algorithm_df.columns
    ):

        raise RuntimeError(
            "MROO results do not contain eta."
        )


    mask = (
        config_isclose(

            algorithm_df[
                "eta"
            ],

            eta,
        )
    )


    kappa_columns = (
        get_kappa_columns(
            algorithm_df
        )
    )


    # --------------------------------------------------------
    # New MROO format:
    # kappa_init_0, kappa_init_1, ...
    # --------------------------------------------------------

    if len(
        kappa_columns
    ) > 0:

        target_kappa = (
            expand_kappa(
                kappa,
                len(
                    kappa_columns
                ),
            )
        )


        for (
            index,
            column,
        ) in enumerate(
            kappa_columns
        ):

            mask &= (

                config_isclose(

                    algorithm_df[
                        column
                    ],

                    target_kappa[
                        index
                    ],
                )
            )


    # --------------------------------------------------------
    # Serialized kappa fallback
    # --------------------------------------------------------

    elif (
        "kappa_init"
        in algorithm_df.columns
    ):

        target = (
            np.asarray(
                kappa,
                dtype=float,
            )
            .reshape(-1)
        )


        def row_matches_kappa(
            value,
        ):

            if pd.isna(
                value
            ):

                return False


            current = (
                parse_kappa(
                    value
                )
            )


            if target.size == 1:

                target_effective = (
                    np.full(
                        current.size,
                        float(
                            target[0]
                        ),
                    )
                )

            else:

                target_effective = (
                    target
                )


            return bool(

                current.shape
                ==
                target_effective.shape

                and

                np.allclose(

                    current,

                    target_effective,

                    rtol=1e-12,

                    atol=1e-15,
                )
            )


        mask &= (

            algorithm_df[
                "kappa_init"
            ]

            .map(
                row_matches_kappa
            )

            .to_numpy(
                dtype=bool
            )
        )


    else:

        raise RuntimeError(
            "MROO results contain neither "
            "kappa_init nor kappa_init_i columns."
        )


    selected = (

        algorithm_df[
            mask
        ]

        .copy()
    )


    # --------------------------------------------------------
    # There may theoretically be >1 lambda_1 with same
    # eta/kappa.
    #
    # If config_id exists and multiple configurations match,
    # show them instead of silently picking one.
    # --------------------------------------------------------

    if (
        "config_id"
        in selected.columns
    ):

        matching_config_ids = (

            selected[
                "config_id"
            ]

            .dropna()

            .unique()
        )


        if len(
            matching_config_ids
        ) > 1:

            available = (

                selected[
                    [
                        column
                        for column
                        in [
                            "config_id",
                            "lambda_1",
                            "eta",
                        ]
                        +
                        kappa_columns
                        if column
                        in selected.columns
                    ]
                ]

                .drop_duplicates()
            )


            raise RuntimeError(
                "\nMore than one MROO configuration "
                "matches the requested eta/kappa.\n"
                "The stored data differ in another "
                "parameter such as lambda_1.\n\n"
                f"{available.to_string(index=False)}"
            )


    selected = (

        selected

        .drop_duplicates(

            subset=[
                "window_id",
            ],

            keep="last",
        )

        .sort_values(
            "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        selected
    ) != N_WINDOWS:

        display_columns = [

            column

            for column
            in (
                [
                    "lambda_1",
                    "eta",
                ]
                +
                kappa_columns
            )

            if column
            in algorithm_df.columns
        ]


        if (
            len(
                kappa_columns
            ) == 0
            and
            "kappa_init"
            in algorithm_df.columns
        ):

            display_columns.append(
                "kappa_init"
            )


        available = (

            algorithm_df[
                display_columns
            ]

            .drop_duplicates()
        )


        raise RuntimeError(
            "\nMROO configuration not available.\n"
            f"Requested:\n"
            f"  eta   = {eta}\n"
            f"  kappa = {kappa}\n"
            f"Expected {N_WINDOWS} rows but found "
            f"{len(selected)}.\n\n"
            "Available configurations:\n"
            f"{available.to_string(index=False)}"
        )


    return selected


# ============================================================
# 13. SELECT NON-TUNED ALGORITHM
# ============================================================

def select_simple_algorithm(
    frame,
    algorithm,
):

    algorithm_df = (

        frame[
            frame[
                "algorithm"
            ]
            ==
            algorithm
        ]

        .copy()

        .drop_duplicates(

            subset=[
                "window_id",
            ],

            keep="last",
        )

        .sort_values(
            "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        algorithm_df
    ) != N_WINDOWS:

        raise RuntimeError(
            f"{algorithm}: expected {N_WINDOWS} rows "
            f"but found {len(algorithm_df)}."
        )


    return algorithm_df


# ============================================================
# 14. SELECT REQUESTED CONFIGURATIONS
# ============================================================

offline_df = (
    select_simple_algorithm(
        all_results_df,
        "OFFLINE-OPT",
    )
)


greedy_df = (
    select_simple_algorithm(
        all_results_df,
        "GREEDY",
    )
)


smroo_sum_df = (
    select_smroo(

        all_results_df,

        "S-MROO-SUM",

        SMROO_SUM_LAMBDA_1,

        SMROO_SUM_LAMBDA_2,
    )
)


smroo_max_df = (
    select_smroo(

        all_results_df,

        "S-MROO-MAX",

        SMROO_MAX_LAMBDA_1,

        SMROO_MAX_LAMBDA_2,
    )
)


dmd_df = (
    select_dmd(

        all_results_df,

        DMD_ETA,

        DMD_KAPPA,
    )
)


mroo_df = (
    select_mroo(

        all_results_df,

        MROO_ETA,

        MROO_KAPPA,
    )
)


# ============================================================
# 15. BUILD MAIN COMPARISON TABLE
# ============================================================

comparison_frames = {

    "OFFLINE-OPT":
        offline_df,

    "GREEDY":
        greedy_df,

    "S-MROO-SUM":
        smroo_sum_df,

    "S-MROO-MAX":
        smroo_max_df,

    "DMD":
        dmd_df,

    "MROO":
        mroo_df,
}


comparison_df = (
    pd.concat(
        comparison_frames.values(),
        ignore_index=True,
        sort=False,
    )
)


PLOT_ORDER = [

    "OFFLINE-OPT",

    "GREEDY",

    "S-MROO-SUM",

    "S-MROO-MAX",

    "DMD",

    "MROO",
]


ALGORITHM_LABELS = {

    "OFFLINE-OPT":
        "OPT",

    "GREEDY":
        "GRD",

    "S-MROO-SUM":
        "S-MROO\nSUM",

    "S-MROO-MAX":
        "S-MROO\nMAX",

    "DMD":
        "DMD",

    "MROO":
        "MROO",
}


# ============================================================
# 16. VERIFY SAME WINDOWS
# ============================================================

required_window_columns = [

    "window_id",

    "start_index",

    "end_index",
]


reference_windows = None


for algorithm in PLOT_ORDER:

    algorithm_df = (
        comparison_frames[
            algorithm
        ]
    )


    windows = (

        algorithm_df[
            required_window_columns
        ]

        .sort_values(
            "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if reference_windows is None:

        reference_windows = (
            windows
        )


    elif not windows.equals(
        reference_windows
    ):

        raise RuntimeError(
            f"{algorithm} did not use the same "
            "window definitions as the other algorithms."
        )


print(
    "\nVerified: all selected configurations "
    "use exactly the same windows."
)


# ============================================================
# 17. PRINT REQUESTED CONFIGURATIONS
# ============================================================

print(
    "\n========================================"
)

print(
    "SELECTED CONFIGURATIONS"
)

print(
    "========================================"
)


print(
    "\nS-MROO-SUM"
)

print(
    "  lambda_1 =",
    SMROO_SUM_LAMBDA_1,
)

print(
    "  lambda_2 =",
    SMROO_SUM_LAMBDA_2,
)


print(
    "\nS-MROO-MAX"
)

print(
    "  lambda_1 =",
    SMROO_MAX_LAMBDA_1,
)

print(
    "  lambda_2 =",
    SMROO_MAX_LAMBDA_2,
)


print(
    "\nDMD"
)

print(
    "  eta       =",
    DMD_ETA,
)

print(
    "  kappa     =",
    DMD_KAPPA,
)


print(
    "\nMROO"
)

print(
    "  eta       =",
    MROO_ETA,
)

print(
    "  kappa     =",
    MROO_KAPPA,
)


# ------------------------------------------------------------
# Print stored MROO lambda if one exists
# ------------------------------------------------------------

mroo_lambda_1 = (
    first_non_nan(
        mroo_df,
        "lambda_1",
    )
)


if not pd.isna(
    mroo_lambda_1
):

    print(
        "  lambda_1  =",
        mroo_lambda_1,
    )


# ============================================================
# 18. COST SUMMARY
# ============================================================

summary_rows = []


for algorithm in PLOT_ORDER:

    algorithm_df = (
        comparison_frames[
            algorithm
        ]
    )


    row = {

        "algorithm":
            algorithm,

        "n_windows":
            len(
                algorithm_df
            ),

        "lambda_1":
            first_non_nan(
                algorithm_df,
                "lambda_1",
            ),

        "lambda_2":
            first_non_nan(
                algorithm_df,
                "lambda_2",
            ),

        "eta":
            first_non_nan(
                algorithm_df,
                "eta",
            ),

        "kappa_init":
            first_non_nan(
                algorithm_df,
                "kappa_init",
            ),

        "mean_hitting_cost":
            algorithm_df[
                "hitting_cost"
            ].mean(),

        "std_hitting_cost":
            algorithm_df[
                "hitting_cost"
            ].std(),

        "mean_long_term_cost":
            algorithm_df[
                "long_term_cost"
            ].mean(),

        "std_long_term_cost":
            algorithm_df[
                "long_term_cost"
            ].std(),

        "mean_total_cost":
            algorithm_df[
                "total_cost"
            ].mean(),

        "std_total_cost":
            algorithm_df[
                "total_cost"
            ].std(),
    }


    summary_rows.append(
        row
    )


summary_df = (
    pd.DataFrame(
        summary_rows
    )
)


print(
    "\n========================================"
)

print(
    "ALGORITHM COST SUMMARY"
)

print(
    "========================================"
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 19. COST PLOT DEFINITIONS
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
# 20. MAIN ALGORITHM BOXPLOTS
# ============================================================

fig_box, axes_box = (
    plt.subplots(

        1,

        3,

        figsize=(
            15.5,
            4.8,
        ),
    )
)


for (
    ax,
    (
        cost_column,
        ylabel,
    ),
) in zip(
    axes_box,
    COSTS,
):


    box_data = [

        comparison_frames[
            algorithm
        ][
            cost_column
        ]

        .dropna()

        .to_numpy()

        for algorithm
        in PLOT_ORDER
    ]


    ax.boxplot(

        box_data,

        tick_labels=[

            ALGORITHM_LABELS[
                algorithm
            ]

            for algorithm
            in PLOT_ORDER
        ],

        patch_artist=True,

        showmeans=False,

        widths=0.58,
    )


    ax.set_ylabel(

        ylabel,

        fontsize=13,

        fontweight="bold",
    )


    ax.tick_params(

        axis="x",

        labelsize=10,
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


fig_box.suptitle(

    (
        "Matrix-Memory Cost Comparison "
        f"(beta={BETA:g}, "
        f"m={M:g}, "
        f"T={WINDOW_SIZE})"
    ),

    fontsize=14,

    fontweight="bold",
)


fig_box.subplots_adjust(

    left=0.065,

    right=0.99,

    bottom=0.20,

    top=0.86,

    wspace=0.28,
)


ALGORITHM_BOXPLOT_FILE = (

    SETTING_DIR

    /
    (
        "matrix_memory_boxplot_selected_configs"
        f"_beta_{BETA:g}"
        f"_m_{M:g}"
        f"_T_{WINDOW_SIZE}.png"
    )
)


fig_box.savefig(

    ALGORITHM_BOXPLOT_FILE,

    dpi=300,

    bbox_inches="tight",
)


# ============================================================
# 21. MAIN ALGORITHM VIOLIN PLOTS
# ============================================================

fig_violin, axes_violin = (
    plt.subplots(

        1,

        3,

        figsize=(
            15.5,
            4.8,
        ),
    )
)


for (
    ax,
    (
        cost_column,
        ylabel,
    ),
) in zip(
    axes_violin,
    COSTS,
):


    violin_data = [

        comparison_frames[
            algorithm
        ][
            cost_column
        ]

        .dropna()

        .to_numpy()

        for algorithm
        in PLOT_ORDER
    ]


    positions = np.arange(

        1,

        len(
            PLOT_ORDER
        )
        + 1,
    )


    ax.violinplot(

        violin_data,

        positions=positions,

        widths=0.75,

        showmeans=False,

        showmedians=True,

        showextrema=True,
    )


    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(

        [

            ALGORITHM_LABELS[
                algorithm
            ]

            for algorithm
            in PLOT_ORDER
        ]
    )


    ax.set_ylabel(

        ylabel,

        fontsize=13,

        fontweight="bold",
    )


    ax.tick_params(

        axis="x",

        labelsize=10,
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


fig_violin.suptitle(

    (
        "Matrix-Memory Cost Distribution "
        f"(beta={BETA:g}, "
        f"m={M:g}, "
        f"T={WINDOW_SIZE})"
    ),

    fontsize=14,

    fontweight="bold",
)


fig_violin.subplots_adjust(

    left=0.065,

    right=0.99,

    bottom=0.20,

    top=0.86,

    wspace=0.28,
)


ALGORITHM_VIOLIN_FILE = (

    SETTING_DIR

    /
    (
        "matrix_memory_violin_selected_configs"
        f"_beta_{BETA:g}"
        f"_m_{M:g}"
        f"_T_{WINDOW_SIZE}.png"
    )
)


fig_violin.savefig(

    ALGORITHM_VIOLIN_FILE,

    dpi=300,

    bbox_inches="tight",
)


# ============================================================
# 22. SAVE SELECTED DATA
#
# This is useful because it records exactly which 600 rows
# generated the current plot.
# ============================================================

SELECTED_RESULT_FILE = (

    SETTING_DIR

    /
    "plot_selected_algorithm_window_results.csv"
)


SELECTED_SUMMARY_FILE = (

    SETTING_DIR

    /
    "plot_selected_algorithm_summary.csv"
)


comparison_df.to_csv(

    SELECTED_RESULT_FILE,

    index=False,
)


summary_df.to_csv(

    SELECTED_SUMMARY_FILE,

    index=False,
)


# ============================================================
# 23. MROO TUNING DATA
#
# For the tuning plot we deliberately use ALL stored MROO
# configurations, not only the one selected above.
# ============================================================

all_mroo_df = (

    all_results_df[
        all_results_df[
            "algorithm"
        ]
        ==
        "MROO"
    ]

    .copy()
)


# ------------------------------------------------------------
# Fallback to raw MROO storage if history does not contain
# all configs.
# ------------------------------------------------------------

if (
    len(
        all_mroo_df
    ) == 0
    and
    MROO_RESULT_FILE.exists()
):

    all_mroo_df = (
        pd.read_csv(
            MROO_RESULT_FILE,
            low_memory=False,
        )
    )


# ============================================================
# 24. BUILD MROO CONFIGURATION IDS IF NEEDED
# ============================================================

def build_mroo_config_key(
    row,
    kappa_columns,
):

    values = []


    # --------------------------------------------------------
    # lambda_1
    # --------------------------------------------------------

    if (
        "lambda_1"
        in row.index
        and
        not pd.isna(
            row[
                "lambda_1"
            ]
        )
    ):

        values.append(
            (
                "lambda_1",
                float(
                    row[
                        "lambda_1"
                    ]
                ),
            )
        )


    # --------------------------------------------------------
    # lambda_2 if present
    # --------------------------------------------------------

    if (
        "lambda_2"
        in row.index
        and
        not pd.isna(
            row[
                "lambda_2"
            ]
        )
    ):

        values.append(
            (
                "lambda_2",
                float(
                    row[
                        "lambda_2"
                    ]
                ),
            )
        )


    # --------------------------------------------------------
    # eta
    # --------------------------------------------------------

    if (
        "eta"
        in row.index
        and
        not pd.isna(
            row[
                "eta"
            ]
        )
    ):

        values.append(
            (
                "eta",
                float(
                    row[
                        "eta"
                    ]
                ),
            )
        )


    # --------------------------------------------------------
    # kappa
    # --------------------------------------------------------

    if len(
        kappa_columns
    ) > 0:

        for column in kappa_columns:

            values.append(
                (
                    column,
                    float(
                        row[
                            column
                        ]
                    ),
                )
            )


    elif (
        "kappa_init"
        in row.index
        and
        not pd.isna(
            row[
                "kappa_init"
            ]
        )
    ):

        kappa_values = (
            parse_kappa(
                row[
                    "kappa_init"
                ]
            )
        )


        for (
            index,
            value,
        ) in enumerate(
            kappa_values
        ):

            values.append(
                (
                    f"kappa_init_{index}",
                    float(
                        value
                    ),
                )
            )


    return tuple(
        values
    )


if len(
    all_mroo_df
) > 0:

    MROO_KAPPA_COLUMNS = (
        get_kappa_columns(
            all_mroo_df
        )
    )


    all_mroo_df[
        "_configuration_key"
    ] = [

        build_mroo_config_key(

            row,

            MROO_KAPPA_COLUMNS,
        )

        for _, row
        in all_mroo_df.iterrows()
    ]


    # ========================================================
    # 25. MROO CONFIGURATION SUMMARY
    # ========================================================

    mroo_config_summary = (

        all_mroo_df

        .groupby(
            "_configuration_key",
            as_index=False,
            dropna=False,
        )

        .agg(

            n_windows=(
                "window_id",
                "nunique",
            ),

            mean_hitting_cost=(
                "hitting_cost",
                "mean",
            ),

            mean_long_term_cost=(
                "long_term_cost",
                "mean",
            ),

            mean_total_cost=(
                "total_cost",
                "mean",
            ),
        )

        .sort_values(
            "mean_total_cost"
        )

        .reset_index(
            drop=True
        )
    )


    complete_mroo_configs = (

        mroo_config_summary[
            mroo_config_summary[
                "n_windows"
            ]
            ==
            N_WINDOWS
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    print(
        "\n========================================"
    )

    print(
        "ALL COMPLETE MROO CONFIGURATIONS"
    )

    print(
        "========================================"
    )


    if len(
        complete_mroo_configs
    ) == 0:

        print(
            "No complete MROO configurations."
        )


    else:

        print(
            complete_mroo_configs[
                [
                    "n_windows",
                    "mean_hitting_cost",
                    "mean_long_term_cost",
                    "mean_total_cost",
                ]
            ]
            .to_string(
                index=False
            )
        )


        # ====================================================
        # 26. LABEL MROO CONFIGURATIONS
        # ====================================================

        mroo_plot_data = []

        mroo_labels = []


        for (
            config_index,
            config_row,
        ) in (
            complete_mroo_configs
            .iterrows()
        ):

            key = (
                config_row[
                    "_configuration_key"
                ]
            )


            config_df = (

                all_mroo_df[
                    all_mroo_df[
                        "_configuration_key"
                    ]
                    ==
                    key
                ]

                .copy()

                .drop_duplicates(

                    subset=[
                        "window_id",
                    ],

                    keep="last",
                )

                .sort_values(
                    "window_id"
                )

                .reset_index(
                    drop=True
                )
            )


            if len(
                config_df
            ) != N_WINDOWS:

                continue


            first_row = (
                config_df.iloc[
                    0
                ]
            )


            label_parts = []


            if (
                "lambda_1"
                in first_row.index
                and
                not pd.isna(
                    first_row[
                        "lambda_1"
                    ]
                )
            ):

                label_parts.append(
                    rf"$\lambda_1$="
                    +
                    f"{float(first_row['lambda_1']):.3g}"
                )


            if (
                "eta"
                in first_row.index
                and
                not pd.isna(
                    first_row[
                        "eta"
                    ]
                )
            ):

                label_parts.append(
                    rf"$\eta$="
                    +
                    f"{float(first_row['eta']):.3g}"
                )


            if len(
                MROO_KAPPA_COLUMNS
            ) > 0:

                kappa_values = np.asarray(

                    [

                        float(
                            first_row[
                                column
                            ]
                        )

                        for column
                        in MROO_KAPPA_COLUMNS
                    ],

                    dtype=float,
                )


            elif (
                "kappa_init"
                in first_row.index
            ):

                kappa_values = (
                    parse_kappa(
                        first_row[
                            "kappa_init"
                        ]
                    )
                )


            else:

                kappa_values = (
                    np.asarray(
                        [],
                        dtype=float,
                    )
                )


            if len(
                kappa_values
            ) > 0:

                label_parts.append(
                    rf"$\kappa$="
                    +
                    kappa_string(
                        kappa_values
                    )
                )


            if len(
                label_parts
            ) == 0:

                label = (
                    f"Config {config_index + 1}"
                )

            else:

                label = (
                    "\n".join(
                        label_parts
                    )
                )


            mroo_labels.append(
                label
            )


            mroo_plot_data.append(
                config_df
            )


        # ====================================================
        # 27. MROO TUNING BOXPLOTS
        # ====================================================

        if len(
            mroo_plot_data
        ) > 0:

            figure_width = max(

                15.5,

                2.2
                *
                len(
                    mroo_plot_data
                ),
            )


            fig_mroo_box, axes_mroo_box = (
                plt.subplots(

                    1,

                    3,

                    figsize=(
                        figure_width,
                        6.0,
                    ),
                )
            )


            for (
                ax,
                (
                    cost_column,
                    ylabel,
                ),
            ) in zip(
                axes_mroo_box,
                COSTS,
            ):

                data = [

                    config_df[
                        cost_column
                    ]

                    .dropna()

                    .to_numpy()

                    for config_df
                    in mroo_plot_data
                ]


                ax.boxplot(

                    data,

                    tick_labels=
                        mroo_labels,

                    patch_artist=True,

                    showmeans=False,

                    widths=0.58,
                )


                ax.set_ylabel(

                    ylabel,

                    fontsize=13,

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


                for label in (
                    ax.get_xticklabels()
                ):

                    label.set_rotation(
                        45
                    )

                    label.set_ha(
                        "right"
                    )


            fig_mroo_box.suptitle(

                (
                    "MROO Matrix-Memory "
                    "Hyperparameter Comparison "
                    f"(beta={BETA:g}, "
                    f"m={M:g}, "
                    f"T={WINDOW_SIZE})"
                ),

                fontsize=14,

                fontweight="bold",
            )


            fig_mroo_box.subplots_adjust(

                left=0.06,

                right=0.99,

                bottom=0.39,

                top=0.88,

                wspace=0.28,
            )


            MROO_BOXPLOT_FILE = (

                SETTING_DIR

                /
                (
                    "mroo_matrix_memory_tuning"
                    f"_beta_{BETA:g}"
                    f"_m_{M:g}"
                    f"_T_{WINDOW_SIZE}.png"
                )
            )


            fig_mroo_box.savefig(

                MROO_BOXPLOT_FILE,

                dpi=300,

                bbox_inches="tight",
            )


            # =================================================
            # 28. MROO TUNING VIOLIN PLOTS
            # =================================================

            fig_mroo_violin, axes_mroo_violin = (
                plt.subplots(

                    1,

                    3,

                    figsize=(
                        figure_width,
                        6.0,
                    ),
                )
            )


            for (
                ax,
                (
                    cost_column,
                    ylabel,
                ),
            ) in zip(
                axes_mroo_violin,
                COSTS,
            ):

                data = [

                    config_df[
                        cost_column
                    ]

                    .dropna()

                    .to_numpy()

                    for config_df
                    in mroo_plot_data
                ]


                positions = np.arange(

                    1,

                    len(
                        data
                    )
                    + 1,
                )


                ax.violinplot(

                    data,

                    positions=positions,

                    widths=0.75,

                    showmeans=False,

                    showmedians=True,

                    showextrema=True,
                )


                ax.set_xticks(
                    positions
                )


                ax.set_xticklabels(
                    mroo_labels
                )


                ax.set_ylabel(

                    ylabel,

                    fontsize=13,

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


                for label in (
                    ax.get_xticklabels()
                ):

                    label.set_rotation(
                        45
                    )

                    label.set_ha(
                        "right"
                    )


            fig_mroo_violin.suptitle(

                (
                    "MROO Matrix-Memory "
                    "Hyperparameter Distributions "
                    f"(beta={BETA:g}, "
                    f"m={M:g}, "
                    f"T={WINDOW_SIZE})"
                ),

                fontsize=14,

                fontweight="bold",
            )


            fig_mroo_violin.subplots_adjust(

                left=0.06,

                right=0.99,

                bottom=0.39,

                top=0.88,

                wspace=0.28,
            )


            MROO_VIOLIN_FILE = (

                SETTING_DIR

                /
                (
                    "mroo_matrix_memory_tuning_violin"
                    f"_beta_{BETA:g}"
                    f"_m_{M:g}"
                    f"_T_{WINDOW_SIZE}.png"
                )
            )


            fig_mroo_violin.savefig(

                MROO_VIOLIN_FILE,

                dpi=300,

                bbox_inches="tight",
            )


# ============================================================
# 29. FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "PLOTTING COMPLETE"
)

print(
    "========================================"
)


print(
    "\nSelected-config boxplot:"
)

print(
    ALGORITHM_BOXPLOT_FILE
)


print(
    "\nSelected-config violin plot:"
)

print(
    ALGORITHM_VIOLIN_FILE
)


print(
    "\nExact rows used for the plot:"
)

print(
    SELECTED_RESULT_FILE
)


print(
    "\nSelected configuration summary:"
)

print(
    SELECTED_SUMMARY_FILE
)


if (
    "MROO_BOXPLOT_FILE"
    in globals()
):

    print(
        "\nMROO tuning boxplot:"
    )

    print(
        MROO_BOXPLOT_FILE
    )


if (
    "MROO_VIOLIN_FILE"
    in globals()
):

    print(
        "\nMROO tuning violin plot:"
    )

    print(
        MROO_VIOLIN_FILE
    )


plt.show()