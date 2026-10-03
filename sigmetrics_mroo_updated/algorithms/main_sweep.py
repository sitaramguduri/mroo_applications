import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_DIR = (
    ALGORITHM_DIR.parent
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

EXPERIMENT_MODE = "window"

# 1-minute resolution:
# 24 hours * 60 minutes = 1440 slots
WINDOW_SIZE = 1440

N_WINDOWS = 100

WINDOW_SEED = 0


# ============================================================
# FIXED WEIGHT METADATA
#
# The actual fixed weight is defined in weights.py.
# ============================================================

CONST_W1 = 0.2
CONST_W2 = 0.3
CONST_W3 = 0.5


# ============================================================
# BETA / M GRID
# ============================================================

BETA_VALUES = [
    # 0.15,
    # 7.5,
    # 15,
    # 1.5,
    # 30.0,
    75,
    # 180.0,
    # 450.0,
    # 900.0,
    # 1800.0,
]

M_VALUES = [
    150.0,
]


# Remove accidental duplicates.
BETA_VALUES = list(
    dict.fromkeys(
        float(x)
        for x in BETA_VALUES
    )
)

M_VALUES = list(
    dict.fromkeys(
        float(x)
        for x in M_VALUES
    )
)


# ============================================================
# PARALLELISM
#
# Parallelizes beta/m settings.
#
# Each runner also parallelizes its own windows.
# ============================================================

MAX_SETTING_WORKERS = 3


# ============================================================
# HELPERS
# ============================================================

def safe_float_label(value):
    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
    )


# ============================================================
# OUTPUT DIRECTORY
#
# Separate folder for the 1-minute / 24-hour experiment.
# ============================================================

WEIGHT_LABEL = (
    "fixed_weights"
    f"_w1_{safe_float_label(CONST_W1)}"
    f"_w2_{safe_float_label(CONST_W2)}"
    f"_w3_{safe_float_label(CONST_W3)}"
)

SWEEP_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "rho_7"
    / "1min_24hour"
    / WEIGHT_LABEL
)

SWEEP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MASTER OUTPUT FILES
# ============================================================

MASTER_WINDOW_FILE = (
    SWEEP_DIR
    / "all_algorithms_beta_m_window_results.csv"
)

MASTER_SUMMARY_FILE = (
    SWEEP_DIR
    / "all_algorithms_beta_m_summary.csv"
)


# ============================================================
# ALGORITHMS
# ============================================================

RUNNERS = {
    "MROO":
        "mroo_run.py",

    # "S-MROO-MAX":
    #     "smroo_max_run.py",

    # "S-MROO-SUM":
    #     "smroo_sum_run.py",

    "DMD":
        "dmd_run.py",

    # "OFFLINE-OPT":
    #     "offline_opt_runner.py",

    # "GREEDY":
    #     "greedy_run.py",
}


# ============================================================
# RESULT FILES
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
# SUMMARY FILES
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
# DONE FILES
# ============================================================

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
# CHECK WHETHER ALGORITHM IS COMPLETE
# ============================================================

def algorithm_is_complete(
    algorithm,
    setting_dir,
):
    result_file = (
        setting_dir
        / RESULT_FILES[algorithm]
    )

    summary_file = (
        setting_dir
        / SUMMARY_FILES[algorithm]
    )

    done_file = (
        setting_dir
        / DONE_FILES[algorithm]
    )

    if (
        not result_file.exists()
        or not summary_file.exists()
        or not done_file.exists()
    ):
        return False

    try:
        result_df = pd.read_csv(
            result_file
        )

        summary_df = pd.read_csv(
            summary_file
        )

    except Exception:
        return False

    return (
        len(result_df) > 0
        and len(summary_df) > 0
    )


# ============================================================
# MARK ALGORITHM COMPLETE
# ============================================================

def mark_algorithm_complete(
    algorithm,
    setting_dir,
):
    done_file = (
        setting_dir
        / DONE_FILES[algorithm]
    )

    done_file.write_text(
        "completed\n",
        encoding="utf-8",
    )


# ============================================================
# RUN ONE ALGORITHM
# ============================================================

def run_algorithm(
    algorithm,
    beta,
    m,
    setting_dir,
):
    env = os.environ.copy()

    # --------------------------------------------------------
    # beta / m
    # --------------------------------------------------------

    env["MROO_BETA"] = str(beta)
    env["MROO_M"] = str(m)

    # --------------------------------------------------------
    # Window configuration
    # --------------------------------------------------------

    env["MROO_WINDOW_SIZE"] = str(
        WINDOW_SIZE
    )

    env["MROO_N_WINDOWS"] = str(
        N_WINDOWS
    )

    env["MROO_WINDOW_SEED"] = str(
        WINDOW_SEED
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    env["MROO_RESULT_DIR"] = str(
        setting_dir
    )

    # --------------------------------------------------------
    # Prevent BLAS thread oversubscription
    # --------------------------------------------------------

    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # --------------------------------------------------------
    # Runner path
    # --------------------------------------------------------

    runner_path = (
        ALGORITHM_DIR
        / RUNNERS[algorithm]
    )

    if not runner_path.exists():
        raise FileNotFoundError(
            f"Runner for {algorithm} does not exist:\n"
            f"{runner_path}"
        )

    print(
        f"\n"
        f"[beta={beta:g}, "
        f"m={m:g}, "
        f"beta/m={beta / m:g}] "
        f"Starting {algorithm}",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(runner_path),
        ],
        cwd=PROJECT_DIR,
        env=env,
        check=True,
    )

    print(
        f"[beta={beta:g}, "
        f"m={m:g}] "
        f"Finished {algorithm}",
        flush=True,
    )


# ============================================================
# LOAD OUTPUT FILES
# ============================================================

def load_algorithm_outputs(
    algorithm,
    setting_dir,
):
    result_file = (
        setting_dir
        / RESULT_FILES[algorithm]
    )

    summary_file = (
        setting_dir
        / SUMMARY_FILES[algorithm]
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"{algorithm} did not produce:\n"
            f"{result_file}"
        )

    if not summary_file.exists():
        raise FileNotFoundError(
            f"{algorithm} did not produce:\n"
            f"{summary_file}"
        )

    window_df = pd.read_csv(
        result_file
    )

    summary_df = pd.read_csv(
        summary_file
    )

    if len(window_df) == 0:
        raise RuntimeError(
            f"{algorithm} result file is empty:\n"
            f"{result_file}"
        )

    if len(summary_df) == 0:
        raise RuntimeError(
            f"{algorithm} summary file is empty:\n"
            f"{summary_file}"
        )

    return (
        window_df,
        summary_df,
    )


# ============================================================
# MROO KAPPA HELPERS
# ============================================================

def get_mroo_kappa_component_columns(df):
    prefix = "kappa_init_"

    columns = []

    for column in df.columns:
        column_str = str(column)

        if not column_str.startswith(
            prefix
        ):
            continue

        suffix = column_str[
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
# MROO CONFIGURATION SELECTION
# ============================================================

def select_best_mroo_configuration(
    summary_df,
):
    required = {
        "lambda_1",
        "eta",
        "mean_total_cost",
    }

    missing = (
        required
        - set(summary_df.columns)
    )

    if missing:
        raise RuntimeError(
            "MROO summary is missing columns: "
            f"{sorted(missing)}"
        )

    kappa_columns = (
        get_mroo_kappa_component_columns(
            summary_df
        )
    )

    if (
        len(kappa_columns) == 0
        and
        "kappa_init" not in summary_df.columns
    ):
        raise RuntimeError(
            "MROO summary contains neither "
            "scalar kappa_init nor "
            "vector kappa_init_i columns."
        )

    return (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


# ============================================================
# GET SELECTED MROO WINDOWS
# ============================================================

def get_selected_mroo_windows(
    window_df,
    best_config,
):
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
        missing_components = [
            column
            for column in kappa_columns
            if column not in best_config.index
        ]

        if missing_components:
            raise RuntimeError(
                "Best MROO configuration is missing "
                "kappa component columns: "
                f"{missing_components}"
            )

        for column in kappa_columns:
            target_value = float(
                best_config[
                    column
                ]
            )

            mask &= np.isclose(
                window_df[
                    column
                ].astype(float),
                target_value,
            )

    # --------------------------------------------------------
    # OLD SCALAR KAPPA FORMAT
    # --------------------------------------------------------

    else:
        if (
            "kappa_init"
            not in best_config.index
        ):
            raise RuntimeError(
                "Best MROO configuration does not "
                "contain kappa_init."
            )

        if (
            "kappa_init"
            not in window_df.columns
        ):
            raise RuntimeError(
                "MROO window results do not "
                "contain kappa_init."
            )

        kappa_init = float(
            best_config[
                "kappa_init"
            ]
        )

        mask &= np.isclose(
            window_df[
                "kappa_init"
            ].astype(float),
            kappa_init,
        )

    selected = (
        window_df[
            mask
        ]
        .copy()
    )

    if len(selected) != N_WINDOWS:
        print(
            "\nMROO configuration selection failed.",
            flush=True,
        )

        print(
            "lambda_1 =",
            lambda_1,
            flush=True,
        )

        print(
            "eta =",
            eta,
            flush=True,
        )

        if len(kappa_columns) > 0:
            print(
                "kappa_init =",
                [
                    float(
                        best_config[
                            column
                        ]
                    )
                    for column in kappa_columns
                ],
                flush=True,
            )
        else:
            print(
                "kappa_init =",
                kappa_init,
                flush=True,
            )

        print(
            "matching windows =",
            len(selected),
            flush=True,
        )

        raise RuntimeError(
            f"Expected {N_WINDOWS} MROO windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# S-MROO CONFIGURATION SELECTION
# ============================================================

def select_best_smroo_configuration(
    summary_df,
    algorithm,
):
    required = {
        "parameter_pair",
        "lambda_1",
        "lambda_2",
        "mean_total_cost",
    }

    missing = (
        required
        - set(summary_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{algorithm} summary is missing columns: "
            f"{sorted(missing)}"
        )

    return (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


# ============================================================
# GET SELECTED S-MROO WINDOWS
# ============================================================

def get_selected_smroo_windows(
    window_df,
    best_config,
    algorithm,
):
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
            f"Expected {N_WINDOWS} "
            f"{algorithm} windows for "
            f"parameter_pair={parameter_pair}, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# DMD CONFIGURATION SELECTION
# ============================================================

def select_dmd_configuration(
    summary_df,
):
    required = {
        "eta",
        "kappa_init",
        "mean_total_cost",
    }

    missing = (
        required
        - set(summary_df.columns)
    )

    if missing:
        raise RuntimeError(
            "DMD summary is missing columns: "
            f"{sorted(missing)}"
        )

    return (
        summary_df
        .sort_values(
            "mean_total_cost"
        )
        .iloc[0]
    )


# ============================================================
# GET SELECTED DMD WINDOWS
# ============================================================

def get_selected_dmd_windows(
    window_df,
    best_config,
):
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
            f"Expected {N_WINDOWS} DMD windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# OFFLINE OPT SELECTION
# ============================================================

def select_offline_opt_configuration(
    summary_df,
):
    required = {
        "mean_total_cost",
    }

    missing = (
        required
        - set(summary_df.columns)
    )

    if missing:
        raise RuntimeError(
            "OFFLINE-OPT summary is missing columns: "
            f"{sorted(missing)}"
        )

    if len(summary_df) != 1:
        raise RuntimeError(
            "Expected exactly one OFFLINE-OPT summary row, "
            f"found {len(summary_df)}."
        )

    return (
        summary_df
        .iloc[0]
    )


def get_selected_offline_opt_windows(
    window_df,
):
    selected = (
        window_df
        .copy()
    )

    if len(selected) != N_WINDOWS:
        raise RuntimeError(
            f"Expected {N_WINDOWS} OFFLINE-OPT windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# GREEDY SELECTION
# ============================================================

def select_greedy_configuration(
    summary_df,
):
    required = {
        "mean_total_cost",
    }

    missing = (
        required
        - set(summary_df.columns)
    )

    if missing:
        raise RuntimeError(
            "GREEDY summary is missing columns: "
            f"{sorted(missing)}"
        )

    if len(summary_df) != 1:
        raise RuntimeError(
            "Expected exactly one GREEDY summary row, "
            f"found {len(summary_df)}."
        )

    return (
        summary_df
        .iloc[0]
    )


def get_selected_greedy_windows(
    window_df,
):
    selected = (
        window_df
        .copy()
    )

    if len(selected) != N_WINDOWS:
        raise RuntimeError(
            f"Expected {N_WINDOWS} GREEDY windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# ADD EXPERIMENT METADATA
# ============================================================

def add_metadata(
    df,
    algorithm,
    beta,
    m,
):
    df = df.copy()

    metadata = {
        "algorithm":
            algorithm,

        "beta":
            float(beta),

        "m":
            float(m),

        "beta_over_m":
            float(
                beta / m
            ),

        "fixed_weight_w1":
            CONST_W1,

        "fixed_weight_w2":
            CONST_W2,

        "fixed_weight_w3":
            CONST_W3,
    }

    for (
        column,
        value,
    ) in metadata.items():
        df[
            column
        ] = value

    first = [
        "algorithm",
        "beta",
        "m",
        "beta_over_m",
    ]

    remaining = [
        column
        for column in df.columns
        if column not in first
    ]

    return df[
        first
        + remaining
    ]


# ============================================================
# BUILD SUMMARY ROW
# ============================================================

def build_summary_row(
    algorithm,
    beta,
    m,
    selected_windows,
    best_config,
):
    row = {
        "algorithm":
            algorithm,

        "beta":
            float(beta),

        "m":
            float(m),

        "beta_over_m":
            float(
                beta / m
            ),

        "fixed_weight_w1":
            CONST_W1,

        "fixed_weight_w2":
            CONST_W2,

        "fixed_weight_w3":
            CONST_W3,

        "n_windows":
            int(
                len(
                    selected_windows
                )
            ),

        "lambda_1":
            (
                float(
                    best_config[
                        "lambda_1"
                    ]
                )
                if (
                    "lambda_1"
                    in best_config.index
                    and
                    not pd.isna(
                        best_config[
                            "lambda_1"
                        ]
                    )
                )
                else np.nan
            ),

        "mean_hitting_cost":
            float(
                selected_windows[
                    "hitting_cost"
                ].mean()
            ),

        "std_hitting_cost":
            float(
                selected_windows[
                    "hitting_cost"
                ].std()
            ),

        "mean_long_term_cost":
            float(
                selected_windows[
                    "long_term_cost"
                ].mean()
            ),

        "std_long_term_cost":
            float(
                selected_windows[
                    "long_term_cost"
                ].std()
            ),

        "mean_total_cost":
            float(
                selected_windows[
                    "total_cost"
                ].mean()
            ),

        "std_total_cost":
            float(
                selected_windows[
                    "total_cost"
                ].std()
            ),
    }

    optional = [
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

    for column in optional:
        if column in best_config.index:
            value = (
                best_config[
                    column
                ]
            )

            if pd.isna(
                value
            ):
                row[
                    column
                ] = np.nan

            elif column in {
                "parameter_pair",
                "R",
            }:
                row[
                    column
                ] = int(
                    value
                )

            elif column == "solver_status":
                row[
                    column
                ] = str(
                    value
                )

            elif column == "kappa_init":
                # Scalar DMD / old MROO values stay numeric.
                # Vector MROO labels remain strings.
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

            else:
                row[
                    column
                ] = float(
                    value
                )

        else:
            row[
                column
            ] = np.nan

    # --------------------------------------------------------
    # Preserve vector MROO kappa components
    # --------------------------------------------------------

    kappa_component_columns = [
        column
        for column in best_config.index
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
        value = (
            best_config[
                column
            ]
        )

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
# VERIFY SAME WINDOWS ACROSS ALGORITHMS
# ============================================================

def verify_same_windows(
    combined_df,
):
    reference = None

    for algorithm in RUNNERS:
        current = (
            combined_df[
                combined_df[
                    "algorithm"
                ]
                == algorithm
            ][
                [
                    "window_id",
                    "start_index",
                    "end_index",
                ]
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
                f"{algorithm} has "
                f"{len(current)} windows; "
                f"expected {N_WINDOWS}."
            )

        if reference is None:
            reference = current

        elif not current.equals(
            reference
        ):
            raise RuntimeError(
                "Algorithms did not use "
                "the same window definitions."
            )


# ============================================================
# RUN ONE BETA / M SETTING
# ============================================================

def run_one_setting(
    beta,
    m,
):
    beta = float(
        beta
    )

    m = float(
        m
    )

    setting_dir = (
        SWEEP_DIR
        / (
            f"beta_{safe_float_label(beta)}"
            f"_m_{safe_float_label(m)}"
            f"_T_{WINDOW_SIZE}"
        )
    )

    setting_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        "========================================\n"
        f"SETTING\n"
        f"beta       = {beta:g}\n"
        f"m          = {m:g}\n"
        f"beta/m     = {beta / m:g}\n"
        f"windows    = {N_WINDOWS}\n"
        f"window T   = {WINDOW_SIZE}\n"
        f"algorithms = {list(RUNNERS.keys())}\n"
        "========================================",
        flush=True,
    )

    selected_frames = []
    summary_rows = []

    # ========================================================
    # RUN ALGORITHMS
    # ========================================================

    for algorithm in RUNNERS:

        # ====================================================
        # MROO
        #
        # Always run the MROO runner so that it can inspect
        # mroo_completed_configs.csv and only compute missing
        # eta / kappa combinations.
        # ====================================================

        if algorithm == "MROO":
            print(
                f"\n"
                f"[beta={beta:g}, m={m:g}] "
                f"Checking MROO tuning configurations...",
                flush=True,
            )

            run_algorithm(
                algorithm,
                beta,
                m,
                setting_dir,
            )

            load_algorithm_outputs(
                algorithm,
                setting_dir,
            )

            mark_algorithm_complete(
                algorithm,
                setting_dir,
            )

        # ====================================================
        # ALL OTHER ALGORITHMS
        # ====================================================

        elif algorithm_is_complete(
            algorithm,
            setting_dir,
        ):
            print(
                f"\n"
                f"[beta={beta:g}, m={m:g}] "
                f"{algorithm} already complete - loading.",
                flush=True,
            )

        else:
            run_algorithm(
                algorithm,
                beta,
                m,
                setting_dir,
            )

            load_algorithm_outputs(
                algorithm,
                setting_dir,
            )

            mark_algorithm_complete(
                algorithm,
                setting_dir,
            )

        (
            window_df,
            summary_df,
        ) = load_algorithm_outputs(
            algorithm,
            setting_dir,
        )

        # ====================================================
        # SELECT BEST CONFIGURATION
        # ====================================================

        if algorithm == "MROO":
            best_config = (
                select_best_mroo_configuration(
                    summary_df
                )
            )

            selected_windows = (
                get_selected_mroo_windows(
                    window_df,
                    best_config,
                )
            )

        elif algorithm == "S-MROO-MAX":
            best_config = (
                select_best_smroo_configuration(
                    summary_df,
                    algorithm,
                )
            )

            selected_windows = (
                get_selected_smroo_windows(
                    window_df,
                    best_config,
                    algorithm,
                )
            )

        elif algorithm == "S-MROO-SUM":
            best_config = (
                select_best_smroo_configuration(
                    summary_df,
                    algorithm,
                )
            )

            selected_windows = (
                get_selected_smroo_windows(
                    window_df,
                    best_config,
                    algorithm,
                )
            )

        elif algorithm == "DMD":
            best_config = (
                select_dmd_configuration(
                    summary_df
                )
            )

            selected_windows = (
                get_selected_dmd_windows(
                    window_df,
                    best_config,
                )
            )

        elif algorithm == "OFFLINE-OPT":
            best_config = (
                select_offline_opt_configuration(
                    summary_df
                )
            )

            selected_windows = (
                get_selected_offline_opt_windows(
                    window_df
                )
            )

        elif algorithm == "GREEDY":
            best_config = (
                select_greedy_configuration(
                    summary_df
                )
            )

            selected_windows = (
                get_selected_greedy_windows(
                    window_df
                )
            )

        else:
            raise RuntimeError(
                f"Unknown algorithm: "
                f"{algorithm}"
            )

        # ====================================================
        # ADD METADATA
        # ====================================================

        selected_windows = (
            add_metadata(
                selected_windows,
                algorithm,
                beta,
                m,
            )
        )

        selected_frames.append(
            selected_windows
        )

        # ====================================================
        # SUMMARY ROW
        # ====================================================

        summary_row = (
            build_summary_row(
                algorithm,
                beta,
                m,
                selected_windows,
                best_config,
            )
        )

        summary_rows.append(
            summary_row
        )

        # ====================================================
        # PRINT ALGORITHM SUMMARY
        # ====================================================

        print(
            f"\n"
            f"[beta={beta:g}, "
            f"m={m:g}] "
            f"{algorithm}:"
        )

        if not pd.isna(
            summary_row[
                "lambda_1"
            ]
        ):
            print(
                f"    lambda_1 = "
                f"{summary_row['lambda_1']}"
            )

        if not pd.isna(
            summary_row[
                "lambda_2"
            ]
        ):
            print(
                f"    lambda_2 = "
                f"{summary_row['lambda_2']}"
            )

        if not pd.isna(
            summary_row[
                "parameter_pair"
            ]
        ):
            print(
                f"    parameter_pair = "
                f"{summary_row['parameter_pair']}"
            )

        if not pd.isna(
            summary_row[
                "eta"
            ]
        ):
            print(
                f"    eta = "
                f"{summary_row['eta']}"
            )

        if not pd.isna(
            summary_row[
                "kappa_init"
            ]
        ):
            print(
                f"    kappa_init = "
                f"{summary_row['kappa_init']}"
            )

        print(
            f"    hitting   = "
            f"{summary_row['mean_hitting_cost']:.6f} "
            f"+/- "
            f"{summary_row['std_hitting_cost']:.6f}"
        )

        print(
            f"    long-term = "
            f"{summary_row['mean_long_term_cost']:.6f} "
            f"+/- "
            f"{summary_row['std_long_term_cost']:.6f}"
        )

        print(
            f"    total     = "
            f"{summary_row['mean_total_cost']:.6f} "
            f"+/- "
            f"{summary_row['std_total_cost']:.6f}"
        )

    # ========================================================
    # COMBINE SELECTED RESULTS FROM CURRENT RUN
    # ========================================================

    combined_windows = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )

    combined_windows = (
        combined_windows
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
        len(RUNNERS)
        * N_WINDOWS
    )

    if len(
        combined_windows
    ) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} "
            f"window rows for "
            f"beta={beta}, m={m}; "
            f"found {len(combined_windows)}."
        )

    verify_same_windows(
        combined_windows
    )

    # ========================================================
    # PRESERVE PREVIOUS COMBINED RESULTS
    #
    # IMPORTANT:
    #
    # Do NOT simply overwrite:
    #
    #   combined_algorithm_window_results.csv
    #   combined_algorithm_summary.csv
    #
    # Existing algorithms that are NOT currently being run are
    # preserved.
    #
    # Algorithms that ARE currently being run are replaced with
    # their newest results so duplicates are not created.
    # ========================================================

    combined_window_file = (
        setting_dir
        / "combined_algorithm_window_results.csv"
    )

    combined_summary_file = (
        setting_dir
        / "combined_algorithm_summary.csv"
    )

    current_algorithms = set(
        RUNNERS.keys()
    )

    # ========================================================
    # MERGE WINDOW RESULTS
    # ========================================================

    combined_windows_to_save = (
        combined_windows.copy()
    )

    if combined_window_file.exists():

        old_combined_windows = pd.read_csv(
            combined_window_file
        )

        if (
            len(old_combined_windows) > 0
            and
            "algorithm"
            in old_combined_windows.columns
        ):

            # Remove OLD rows only for algorithms
            # that are being updated in this run.
            #
            # Example:
            #
            # RUNNERS = {
            #     "MROO": ...,
            #     "DMD": ...
            # }
            #
            # Then old MROO and DMD rows are removed,
            # but GREEDY / S-MROO / OFFLINE-OPT stay.

            old_combined_windows = (
                old_combined_windows[
                    ~old_combined_windows[
                        "algorithm"
                    ].isin(
                        current_algorithms
                    )
                ]
                .copy()
            )

            # Add new current-run rows.
            combined_windows_to_save = pd.concat(
                [
                    old_combined_windows,
                    combined_windows,
                ],
                ignore_index=True,
                sort=False,
            )

    combined_windows_to_save = (
        combined_windows_to_save
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

    combined_windows_to_save.to_csv(
        combined_window_file,
        index=False,
    )

    print(
        "\nSaved combined window results:"
    )

    print(
        combined_window_file
    )

    print(
        "Algorithms stored:",
        sorted(
            combined_windows_to_save[
                "algorithm"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    print(
        "Total combined window rows:",
        len(
            combined_windows_to_save
        ),
    )

    # ========================================================
    # MERGE SUMMARY RESULTS
    # ========================================================

    current_summary_df = pd.DataFrame(
        summary_rows
    )

    combined_summary_to_save = (
        current_summary_df.copy()
    )

    if combined_summary_file.exists():

        old_combined_summary = pd.read_csv(
            combined_summary_file
        )

        if (
            len(old_combined_summary) > 0
            and
            "algorithm"
            in old_combined_summary.columns
        ):

            # Remove old summary rows only for algorithms
            # that were processed during this run.

            old_combined_summary = (
                old_combined_summary[
                    ~old_combined_summary[
                        "algorithm"
                    ].isin(
                        current_algorithms
                    )
                ]
                .copy()
            )

            # Add newest summary rows.
            combined_summary_to_save = pd.concat(
                [
                    old_combined_summary,
                    current_summary_df,
                ],
                ignore_index=True,
                sort=False,
            )

    combined_summary_to_save = (
        combined_summary_to_save
        .sort_values(
            "algorithm"
        )
        .reset_index(
            drop=True
        )
    )

    combined_summary_to_save.to_csv(
        combined_summary_file,
        index=False,
    )

    print(
        "\nSaved combined summary:"
    )

    print(
        combined_summary_file
    )

    print(
        "Algorithms stored:",
        sorted(
            combined_summary_to_save[
                "algorithm"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    # ========================================================
    # IMPORTANT
    #
    # Return ONLY the current-run results here.
    #
    # Do not return combined_windows_to_save because that also
    # contains algorithms from previous runs. Returning that
    # would interfere with the master-file logic in main().
    # ========================================================

    return (
        combined_windows,
        summary_rows,
    )


# ============================================================
# SAVE MASTER FILES
# ============================================================

def save_master_results(
    all_window_frames,
    all_summary_rows,
):
    if all_window_frames:
        all_windows_df = pd.concat(
            all_window_frames,
            ignore_index=True,
            sort=False,
        )

        all_windows_df = (
            all_windows_df
            .sort_values(
                [
                    "beta_over_m",
                    "algorithm",
                    "window_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        all_windows_df.to_csv(
            MASTER_WINDOW_FILE,
            index=False,
        )

    else:
        all_windows_df = (
            pd.DataFrame()
        )

    if all_summary_rows:
        summary_df = pd.DataFrame(
            all_summary_rows
        )

        summary_df = (
            summary_df
            .sort_values(
                [
                    "beta_over_m",
                    "algorithm",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        summary_df.to_csv(
            MASTER_SUMMARY_FILE,
            index=False,
        )

    else:
        summary_df = (
            pd.DataFrame()
        )

    return (
        all_windows_df,
        summary_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    settings = list(
        dict.fromkeys(
            (
                (
                    float(beta),
                    float(m),
                )
                for beta in BETA_VALUES
                for m in M_VALUES
            )
        )
    )

    total_settings = len(
        settings
    )

    if total_settings == 0:
        raise RuntimeError(
            "No beta/m settings are enabled."
        )

    worker_count = min(
        MAX_SETTING_WORKERS,
        total_settings,
    )

    print(
        "\n========================================"
    )

    print(
        "MROO + S-MROO-MAX + S-MROO-SUM + "
        "DMD + OFFLINE-OPT + GREEDY "
        "100-WINDOW BETA/M SWEEP"
    )

    print(
        "========================================"
    )

    print(
        "Experiment mode =",
        EXPERIMENT_MODE,
    )

    print(
        "Fixed raw weights =",
        [
            CONST_W1,
            CONST_W2,
            CONST_W3,
        ],
    )

    print(
        "Window size =",
        WINDOW_SIZE,
    )

    print(
        "Number of windows =",
        N_WINDOWS,
    )

    print(
        "Window seed =",
        WINDOW_SEED,
    )

    print(
        "Beta values =",
        BETA_VALUES,
    )

    print(
        "m values =",
        M_VALUES,
    )

    print(
        "Algorithms =",
        list(
            RUNNERS.keys()
        ),
    )

    print(
        "Total beta/m settings =",
        total_settings,
    )

    print(
        "Parallel setting workers =",
        worker_count,
    )

    print(
        "Output directory =",
        SWEEP_DIR,
    )

    all_window_frames = []

    all_summary_rows = []

    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:

        future_to_setting = {}

        for (
            beta,
            m,
        ) in settings:

            future = executor.submit(
                run_one_setting,
                beta,
                m,
            )

            future_to_setting[
                future
            ] = (
                beta,
                m,
            )

        completed = 0

        for future in as_completed(
            future_to_setting
        ):

            beta, m = (
                future_to_setting[
                    future
                ]
            )

            try:
                (
                    window_df,
                    summary_rows,
                ) = future.result()

            except Exception as error:

                print(
                    "\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                print(
                    "SETTING FAILED"
                )

                print(
                    "beta =",
                    beta,
                )

                print(
                    "m =",
                    m,
                )

                print(
                    "Error =",
                    error,
                )

                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                raise

            all_window_frames.append(
                window_df
            )

            all_summary_rows.extend(
                summary_rows
            )

            completed += 1

            print(
                f"\nCompleted beta/m settings: "
                f"{completed}/{total_settings}",
                flush=True,
            )

            save_master_results(
                all_window_frames,
                all_summary_rows,
            )

    (
        all_windows_df,
        summary_df,
    ) = save_master_results(
        all_window_frames,
        all_summary_rows,
    )

    print(
        "\n========================================"
    )

    print(
        "ALGORITHM WINDOW SWEEP COMPLETE"
    )

    print(
        "========================================"
    )

    display_columns = [
        "algorithm",
        "beta",
        "m",
        "beta_over_m",
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

    for column in display_columns:
        if column not in summary_df.columns:
            summary_df[
                column
            ] = np.nan

    print(
        "\nSUMMARY:\n"
    )

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nIndividual window rows =",
        len(
            all_windows_df
        ),
    )

    expected_window_rows = (
        total_settings
        * len(
            RUNNERS
        )
        * N_WINDOWS
    )

    print(
        "Expected window rows =",
        expected_window_rows,
    )

    if len(
        all_windows_df
    ) != expected_window_rows:

        print(
            "WARNING: actual number of master rows "
            "does not equal expected number."
        )

    print(
        "\nAll individual window results:"
    )

    print(
        MASTER_WINDOW_FILE
    )

    print(
        "\nSummary by algorithm and beta/m:"
    )

    print(
        MASTER_SUMMARY_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()