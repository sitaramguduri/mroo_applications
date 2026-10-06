import json
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
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_MODE = "window"

# 1-minute resolution:
# 24 hours * 60 minutes = 1440 slots
WINDOW_SIZE = 1440

N_WINDOWS = 100

WINDOW_SEED = 0


# ============================================================
# ALGORITHM HYPERPARAMETERS
# ============================================================


# ------------------------------------------------------------
# MROO / DMD
#
# IMPORTANT:
#
# MROO and DMD use the SAME eta/kappa grid.
#
# kappa is specified here as a scalar.
#
# Example:
#
#     kappa = 3.0
#
# DMD stores this scalar as:
#
#     kappa_init = 3.0
#
# but internally dmd_run.py expands it to:
#
#     [3.0, 3.0, 3.0]
#
# MROO also expands scalar kappa over all D dimensions.
#
# Therefore both algorithms use the same effective initial
# kappa vector.
#
# Change ONLY these two lists when tuning MROO and DMD.
# ------------------------------------------------------------

MROO_DMD_ETA_VALUES = [
    0.000000001,
]

MROO_DMD_KAPPA_VALUES = [
    3.0,
]


# MROO receives the shared values.
MROO_ETA_VALUES = list(
    MROO_DMD_ETA_VALUES
)

MROO_KAPPA_VALUES = list(
    MROO_DMD_KAPPA_VALUES
)


# DMD receives exactly the same values.
DMD_ETA_VALUES = list(
    MROO_DMD_ETA_VALUES
)

DMD_KAPPA_VALUES = list(
    MROO_DMD_KAPPA_VALUES
)


# ------------------------------------------------------------
# S-MROO-SUM
# ------------------------------------------------------------

SMROO_SUM_LAMBDA_1 = 1

SMROO_SUM_LAMBDA_2 = 7000


# ------------------------------------------------------------
# S-MROO-MAX
# ------------------------------------------------------------

SMROO_MAX_LAMBDA_1 = 0.02

SMROO_MAX_LAMBDA_2 = 0


# ============================================================
# BETA / M GRID
# ============================================================

BETA_VALUES = [
    600,
]

M_VALUES = [
    100,
]


BETA_VALUES = list(
    dict.fromkeys(
        float(value)
        for value in BETA_VALUES
    )
)

M_VALUES = list(
    dict.fromkeys(
        float(value)
        for value in M_VALUES
    )
)


# ============================================================
# PARALLEL SETTINGS
# ============================================================

MAX_SETTING_WORKERS = 4


# ============================================================
# NEW MATRIX-MEMORY RESULT DIRECTORY
# ============================================================

FORMULATION_LABEL = "matrix_memory"

SWEEP_DIR = (
    ALGORITHM_DIR
    / "matrix_memory_v1"
    / "1min_24hour"
    / FORMULATION_LABEL
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
# RUNNERS
# ============================================================

RUNNERS = {

    "MROO":
        "mroo_run.py",

    "S-MROO-MAX":
        "smroo_max_run.py",

    "S-MROO-SUM":
        "smroo_sum_run.py",

    "DMD":
        "dmd_run.py",

    "OFFLINE-OPT":
        "offline_opt_runner.py",

    "GREEDY":
        "greedy_run.py",
}


# ============================================================
# RESULT FILES
#
# DMD uses the ALL-results file because it may contain many
# eta/kappa configurations.
# ============================================================

RESULT_FILES = {

    "MROO":
        "mroo_window_results.csv",

    "S-MROO-MAX":
        "smroo_max_window_results.csv",

    "S-MROO-SUM":
        "smroo_sum_window_results.csv",

    "DMD":
        "dmd_all_window_results.csv",

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
        "dmd_all_window_summary.csv",

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
# BASIC HELPERS
# ============================================================

def safe_float_label(value):
    """
    Stable filename-safe float label with enough precision to
    distinguish nearby hyperparameter values.
    """

    value = float(value)

    return (
        f"{value:.12g}"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "")
    )


def config_isclose(
    values,
    target,
):

    return np.isclose(
        values,
        target,
        rtol=1e-12,
        atol=1e-15,
    )


# ============================================================
# PERSISTENT RESULT STORAGE
#
# Storage now has three levels:
#
# 1. Runner output
#
# 2. Permanent per-configuration snapshots
#
# 3. Persistent all-configuration history
#
# Current comparison files are kept separately.
# ============================================================

HISTORY_DIRNAME = "history"

CONFIGURATION_DIRNAME = "configurations"

CURRENT_COMPARISON_DIRNAME = (
    "current_comparison"
)

SETTING_HISTORY_WINDOW_FILENAME = (
    "all_algorithm_window_results.csv"
)

SETTING_HISTORY_SUMMARY_FILENAME = (
    "all_algorithm_summary.csv"
)


def _value_present(value):

    if value is None:

        return False

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
            dict,
        ),
    ):

        return True

    try:

        missing = pd.isna(
            value
        )

    except Exception:

        return True

    if isinstance(
        missing,
        (
            np.ndarray,
            list,
            tuple,
        ),
    ):

        return True

    return not bool(
        missing
    )


def _config_value_label(
    value,
):

    if not _value_present(
        value
    ):

        return "none"

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        arr = np.asarray(
            value,
            dtype=float,
        ).reshape(-1)

        return "x".join(
            safe_float_label(
                v
            )
            for v in arr
        )

    try:

        return safe_float_label(
            float(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        text = str(
            value
        )

        return (
            text
            .replace(
                " ",
                "",
            )
            .replace(
                "/",
                "_",
            )
            .replace(
                "\\",
                "_",
            )
            .replace(
                "[",
                "",
            )
            .replace(
                "]",
                "",
            )
            .replace(
                ",",
                "x",
            )
        )


# ============================================================
# S-MROO REQUESTED LAMBDAS
# ============================================================

def get_smroo_target_lambdas(
    algorithm,
):

    if algorithm == "S-MROO-SUM":

        return (
            float(
                SMROO_SUM_LAMBDA_1
            ),
            float(
                SMROO_SUM_LAMBDA_2
            ),
        )

    if algorithm == "S-MROO-MAX":

        return (
            float(
                SMROO_MAX_LAMBDA_1
            ),
            float(
                SMROO_MAX_LAMBDA_2
            ),
        )

    raise ValueError(
        "get_smroo_target_lambdas supports only "
        "S-MROO-SUM and S-MROO-MAX."
    )


# ============================================================
# CONFIGURATION IDENTIFIERS
# ============================================================

def build_config_id_from_row(
    algorithm,
    row,
):

    # --------------------------------------------------------
    # S-MROO
    # --------------------------------------------------------

    if algorithm in {
        "S-MROO-SUM",
        "S-MROO-MAX",
    }:

        lambda_1 = row.get(
            "lambda_1",
            np.nan,
        )

        lambda_2 = row.get(
            "lambda_2",
            np.nan,
        )

        return (
            f"lambda1_"
            f"{_config_value_label(lambda_1)}"
            f"_lambda2_"
            f"{_config_value_label(lambda_2)}"
        )

    # --------------------------------------------------------
    # DMD
    # --------------------------------------------------------

    if algorithm == "DMD":

        eta = row.get(
            "eta",
            np.nan,
        )

        kappa = row.get(
            "kappa_init",
            np.nan,
        )

        return (
            f"eta_"
            f"{_config_value_label(eta)}"
            f"_kappa_"
            f"{_config_value_label(kappa)}"
        )

    # --------------------------------------------------------
    # MROO
    # --------------------------------------------------------

    if algorithm == "MROO":

        parts = []

        for column in [
            "lambda_1",
            "lambda_2",
            "eta",
        ]:

            if (
                column in row.index
                and
                _value_present(
                    row[
                        column
                    ]
                )
            ):

                parts.append(
                    f"{column}_"
                    f"{_config_value_label(row[column])}"
                )

        kappa_columns = sorted(

            [
                column

                for column
                in row.index

                if (
                    str(
                        column
                    ).startswith(
                        "kappa_init_"
                    )
                    and
                    str(
                        column
                    )[
                        len(
                            "kappa_init_"
                        ):
                    ].isdigit()
                )
            ],

            key=
                lambda column:
                int(
                    str(
                        column
                    )[
                        len(
                            "kappa_init_"
                        ):
                    ]
                ),
        )

        if len(
            kappa_columns
        ) > 0:

            kappa_label = (
                "x".join(
                    _config_value_label(
                        row[
                            column
                        ]
                    )
                    for column
                    in kappa_columns
                )
            )

            parts.append(
                f"kappa_{kappa_label}"
            )

        elif (
            "kappa_init"
            in row.index
            and
            _value_present(
                row[
                    "kappa_init"
                ]
            )
        ):

            parts.append(
                "kappa_"
                +
                _config_value_label(
                    row[
                        "kappa_init"
                    ]
                )
            )

        if len(
            parts
        ) == 0:

            return "default"

        return "_".join(
            parts
        )

    # --------------------------------------------------------
    # GREEDY / OFFLINE-OPT
    # --------------------------------------------------------

    return "default"


def add_config_id_column(
    frame,
    algorithm,
):

    frame = (
        frame.copy()
    )

    if len(
        frame
    ) == 0:

        frame[
            "config_id"
        ] = pd.Series(
            dtype=str
        )

        return frame

    frame[
        "config_id"
    ] = [

        build_config_id_from_row(
            algorithm,
            row,
        )

        for _, row
        in frame.iterrows()
    ]

    return frame


# ============================================================
# S-MROO CONFIGURATION DIRECTORY
# ============================================================

def get_smroo_configuration_dir(
    algorithm,
    setting_dir,
):

    (
        lambda_1,
        lambda_2,
    ) = (
        get_smroo_target_lambdas(
            algorithm
        )
    )

    config_id = (
        f"lambda1_"
        f"{safe_float_label(lambda_1)}"
        f"_lambda2_"
        f"{safe_float_label(lambda_2)}"
    )

    return (
        setting_dir
        /
        CONFIGURATION_DIRNAME
        /
        algorithm
        /
        config_id
    )


def get_algorithm_result_dir(
    algorithm,
    setting_dir,
):

    # --------------------------------------------------------
    # S-MROO gets a unique output directory for every
    # lambda pair.
    #
    # This is what prevents later experiments from replacing
    # previous S-MROO files.
    # --------------------------------------------------------

    if algorithm in {
        "S-MROO-SUM",
        "S-MROO-MAX",
    }:

        result_dir = (
            get_smroo_configuration_dir(
                algorithm,
                setting_dir,
            )
        )

        result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return result_dir

    # --------------------------------------------------------
    # Other runners keep their existing location.
    #
    # MROO and DMD already maintain multiple configurations
    # internally.
    # --------------------------------------------------------

    return setting_dir


# ============================================================
# HISTORY METADATA
# ============================================================

def _add_history_metadata(
    frame,
    algorithm,
    beta,
    m,
):

    frame = (
        frame.copy()
    )

    frame[
        "algorithm"
    ] = algorithm

    frame[
        "beta"
    ] = float(
        beta
    )

    frame[
        "m"
    ] = float(
        m
    )

    frame[
        "beta_over_m"
    ] = float(
        beta
        /
        m
    )

    frame[
        "experiment_window_size"
    ] = int(
        WINDOW_SIZE
    )

    frame[
        "experiment_window_seed"
    ] = int(
        WINDOW_SEED
    )

    frame = (
        add_config_id_column(
            frame,
            algorithm,
        )
    )

    first = [

        "algorithm",

        "beta",

        "m",

        "beta_over_m",

        "experiment_window_size",

        "experiment_window_seed",

        "config_id",
    ]

    remaining = [

        column

        for column
        in frame.columns

        if column
        not in first
    ]

    return frame[
        first
        +
        remaining
    ]


# ============================================================
# PERSISTENT CSV MERGE
# ============================================================

def _merge_persistent_csv(
    new_df,
    output_file,
    dedupe_columns,
    sort_columns,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    # --------------------------------------------------------
    # Existing data
    # --------------------------------------------------------

    if output_file.exists():

        try:

            existing_df = (
                pd.read_csv(
                    output_file
                )
            )

            if len(
                existing_df
            ) > 0:

                frames.append(
                    existing_df
                )

        except Exception as error:

            raise RuntimeError(
                "Could not read persistent result file:\n"
                f"{output_file}\n"
                f"Original error: {error}"
            ) from error

    # --------------------------------------------------------
    # New data
    # --------------------------------------------------------

    if len(
        new_df
    ) > 0:

        frames.append(
            new_df.copy()
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    if len(
        frames
    ) == 0:

        merged_df = (
            pd.DataFrame()
        )

    else:

        merged_df = (
            pd.concat(
                frames,
                ignore_index=True,
                sort=False,
            )
        )

        actual_dedupe_columns = [

            column

            for column
            in dedupe_columns

            if column
            in merged_df.columns
        ]

        if len(
            actual_dedupe_columns
        ) > 0:

            merged_df = (

                merged_df

                .drop_duplicates(

                    subset=
                        actual_dedupe_columns,

                    keep="last",
                )
            )

        actual_sort_columns = [

            column

            for column
            in sort_columns

            if column
            in merged_df.columns
        ]

        if len(
            actual_sort_columns
        ) > 0:

            merged_df = (

                merged_df

                .sort_values(
                    actual_sort_columns
                )
            )

        merged_df = (

            merged_df

            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    merged_df.to_csv(
        output_file,
        index=False,
    )

    return merged_df


# ============================================================
# ARCHIVE ALL OUTPUTS FROM ONE ALGORITHM
# ============================================================

def archive_algorithm_outputs(
    algorithm,
    beta,
    m,
    setting_dir,
    window_df,
    summary_df,
):

    history_dir = (
        setting_dir
        /
        HISTORY_DIRNAME
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_window_file = (
        history_dir
        /
        SETTING_HISTORY_WINDOW_FILENAME
    )

    history_summary_file = (
        history_dir
        /
        SETTING_HISTORY_SUMMARY_FILENAME
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    history_windows = (
        _add_history_metadata(
            window_df,
            algorithm,
            beta,
            m,
        )
    )

    history_summary = (
        _add_history_metadata(
            summary_df,
            algorithm,
            beta,
            m,
        )
    )

    # --------------------------------------------------------
    # Persistent all-window history
    # --------------------------------------------------------

    all_history_windows = (
        _merge_persistent_csv(

            history_windows,

            history_window_file,

            dedupe_columns=[

                "algorithm",

                "beta",

                "m",

                "experiment_window_size",

                "experiment_window_seed",

                "config_id",

                "window_id",
            ],

            sort_columns=[

                "algorithm",

                "config_id",

                "window_id",
            ],
        )
    )

    # --------------------------------------------------------
    # Persistent all-summary history
    # --------------------------------------------------------

    all_history_summary = (
        _merge_persistent_csv(

            history_summary,

            history_summary_file,

            dedupe_columns=[

                "algorithm",

                "beta",

                "m",

                "experiment_window_size",

                "experiment_window_seed",

                "config_id",
            ],

            sort_columns=[

                "algorithm",

                "config_id",
            ],
        )
    )

    # ========================================================
    # WRITE PER-CONFIGURATION SNAPSHOTS
    # ========================================================

    for (
        config_id,
        config_windows,
    ) in history_windows.groupby(
        "config_id",
        dropna=False,
    ):

        config_id = str(
            config_id
        )

        config_dir = (
            setting_dir
            /
            CONFIGURATION_DIRNAME
            /
            algorithm
            /
            config_id
        )

        config_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_windows = (

            config_windows

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

        config_windows.to_csv(

            config_dir
            /
            "window_results.csv",

            index=False,
        )

        # ----------------------------------------------------
        # Matching summary
        # ----------------------------------------------------

        matching_summary = (

            history_summary[

                history_summary[
                    "config_id"
                ].astype(
                    str
                )
                ==
                config_id
            ]

            .copy()

            .reset_index(
                drop=True
            )
        )

        if len(
            matching_summary
        ) > 0:

            matching_summary.to_csv(

                config_dir
                /
                "summary.csv",

                index=False,
            )

        # ----------------------------------------------------
        # DONE marker only when the configuration has all
        # requested windows.
        # ----------------------------------------------------

        if (

            "window_id"
            in config_windows.columns

            and

            config_windows[
                "window_id"
            ]
            .dropna()
            .astype(
                int
            )
            .nunique()
            ==
            N_WINDOWS

            and

            len(
                matching_summary
            ) > 0
        ):

            (
                config_dir
                /
                "DONE.txt"
            ).write_text(

                "completed\n",

                encoding=
                    "utf-8",
            )

    print(
        "\nPersistent history updated:"
    )

    print(
        history_window_file
    )

    print(
        history_summary_file
    )

    return (
        all_history_windows,
        all_history_summary,
    )


# ============================================================
# MIGRATE OLD S-MROO FILES
#
# This means existing results from before this storage change
# are not automatically lost.
# ============================================================

def migrate_legacy_smroo_configuration(
    algorithm,
    setting_dir,
):

    target_dir = (
        get_smroo_configuration_dir(
            algorithm,
            setting_dir,
        )
    )

    target_result_file = (
        target_dir
        /
        RESULT_FILES[
            algorithm
        ]
    )

    target_summary_file = (
        target_dir
        /
        SUMMARY_FILES[
            algorithm
        ]
    )

    # Already migrated.
    if (
        target_result_file.exists()
        and
        target_summary_file.exists()
    ):

        return

    legacy_result_file = (
        setting_dir
        /
        RESULT_FILES[
            algorithm
        ]
    )

    legacy_summary_file = (
        setting_dir
        /
        SUMMARY_FILES[
            algorithm
        ]
    )

    if (
        not legacy_result_file.exists()
        or
        not legacy_summary_file.exists()
    ):

        return

    try:

        legacy_results = (
            pd.read_csv(
                legacy_result_file
            )
        )

        legacy_summary = (
            pd.read_csv(
                legacy_summary_file
            )
        )

    except Exception:

        return

    required = {
        "lambda_1",
        "lambda_2",
    }

    if (

        not required.issubset(
            legacy_results.columns
        )

        or

        not required.issubset(
            legacy_summary.columns
        )
    ):

        return

    (
        lambda_1,
        lambda_2,
    ) = (
        get_smroo_target_lambdas(
            algorithm
        )
    )

    result_mask = (

        config_isclose(

            legacy_results[
                "lambda_1"
            ].astype(
                float
            ),

            lambda_1,
        )

        &

        config_isclose(

            legacy_results[
                "lambda_2"
            ].astype(
                float
            ),

            lambda_2,
        )
    )

    summary_mask = (

        config_isclose(

            legacy_summary[
                "lambda_1"
            ].astype(
                float
            ),

            lambda_1,
        )

        &

        config_isclose(

            legacy_summary[
                "lambda_2"
            ].astype(
                float
            ),

            lambda_2,
        )
    )

    matching_results = (

        legacy_results[
            result_mask
        ]

        .copy()
    )

    matching_summary = (

        legacy_summary[
            summary_mask
        ]

        .copy()
    )

    if (

        len(
            matching_results
        ) == 0

        or

        len(
            matching_summary
        ) == 0
    ):

        return

    if (
        "window_id"
        not in matching_results.columns
    ):

        return

    matching_results = (

        matching_results

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

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching_results.to_csv(
        target_result_file,
        index=False,
    )

    matching_summary.to_csv(
        target_summary_file,
        index=False,
    )

    if (

        matching_results[
            "window_id"
        ]

        .dropna()

        .astype(
            int
        )

        .nunique()

        ==

        N_WINDOWS
    ):

        (
            target_dir
            /
            DONE_FILES[
                algorithm
            ]
        ).write_text(

            "completed\n",

            encoding=
                "utf-8",
        )

    print(
        "\nMigrated legacy "
        f"{algorithm} configuration to:"
    )

    print(
        target_dir
    )


# ============================================================
# GENERIC COMPLETION CHECK
# ============================================================

def algorithm_is_complete(
    algorithm,
    setting_dir,
):

    result_dir = (
        get_algorithm_result_dir(
            algorithm,
            setting_dir,
        )
    )

    result_file = (
        result_dir
        /
        RESULT_FILES[
            algorithm
        ]
    )

    summary_file = (
        result_dir
        /
        SUMMARY_FILES[
            algorithm
        ]
    )

    done_file = (
        result_dir
        /
        DONE_FILES[
            algorithm
        ]
    )

    if (

        not result_file.exists()

        or

        not summary_file.exists()

        or

        not done_file.exists()
    ):

        return False

    try:

        result_df = (
            pd.read_csv(
                result_file
            )
        )

        summary_df = (
            pd.read_csv(
                summary_file
            )
        )

    except Exception:

        return False

    return (

        len(
            result_df
        ) > 0

        and

        len(
            summary_df
        ) > 0
    )


# ============================================================
# MARK COMPLETE
# ============================================================

def mark_algorithm_complete(
    algorithm,
    setting_dir,
):

    result_dir = (
        get_algorithm_result_dir(
            algorithm,
            setting_dir,
        )
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    done_file = (
        result_dir
        /
        DONE_FILES[
            algorithm
        ]
    )

    done_file.write_text(
        "completed\n",
        encoding="utf-8",
    )


# ============================================================
# CHECK EXACT S-MROO CONFIGURATION
# ============================================================

def smroo_requested_configuration_complete(
    algorithm,
    setting_dir,
):

    # --------------------------------------------------------
    # First migrate an old root-level result if possible.
    # --------------------------------------------------------

    migrate_legacy_smroo_configuration(
        algorithm,
        setting_dir,
    )

    result_dir = (
        get_smroo_configuration_dir(
            algorithm,
            setting_dir,
        )
    )

    result_file = (
        result_dir
        /
        RESULT_FILES[
            algorithm
        ]
    )

    summary_file = (
        result_dir
        /
        SUMMARY_FILES[
            algorithm
        ]
    )

    if (

        not result_file.exists()

        or

        not summary_file.exists()
    ):

        return False

    try:

        result_df = (
            pd.read_csv(
                result_file
            )
        )

        summary_df = (
            pd.read_csv(
                summary_file
            )
        )

    except Exception:

        return False

    required = {
        "lambda_1",
        "lambda_2",
    }

    if (

        not required.issubset(
            result_df.columns
        )

        or

        not required.issubset(
            summary_df.columns
        )
    ):

        return False

    (
        target_lambda_1,
        target_lambda_2,
    ) = (
        get_smroo_target_lambdas(
            algorithm
        )
    )

    result_mask = (

        config_isclose(

            result_df[
                "lambda_1"
            ].astype(
                float
            ),

            target_lambda_1,
        )

        &

        config_isclose(

            result_df[
                "lambda_2"
            ].astype(
                float
            ),

            target_lambda_2,
        )
    )

    matching_results = (

        result_df[
            result_mask
        ]

        .copy()
    )

    if (
        "window_id"
        not in matching_results.columns
    ):

        return False

    n_complete_windows = (

        matching_results[
            "window_id"
        ]

        .dropna()

        .astype(
            int
        )

        .nunique()
    )

    if (
        n_complete_windows
        !=
        N_WINDOWS
    ):

        return False

    summary_mask = (

        config_isclose(

            summary_df[
                "lambda_1"
            ].astype(
                float
            ),

            target_lambda_1,
        )

        &

        config_isclose(

            summary_df[
                "lambda_2"
            ].astype(
                float
            ),

            target_lambda_2,
        )
    )

    if not bool(
        np.any(
            summary_mask
        )
    ):

        return False

    (
        result_dir
        /
        DONE_FILES[
            algorithm
        ]
    ).write_text(

        "completed\n",

        encoding=
            "utf-8",
    )

    return True


# ============================================================
# RUN ONE ALGORITHM
# ============================================================

def run_algorithm(
    algorithm,
    beta,
    m,
    setting_dir,
):

    env = (
        os.environ.copy()
    )

    # --------------------------------------------------------
    # beta / m
    # --------------------------------------------------------

    env[
        "MROO_BETA"
    ] = str(
        beta
    )

    env[
        "MROO_M"
    ] = str(
        m
    )

    # --------------------------------------------------------
    # windows
    # --------------------------------------------------------

    env[
        "MROO_WINDOW_SIZE"
    ] = str(
        WINDOW_SIZE
    )

    env[
        "MROO_N_WINDOWS"
    ] = str(
        N_WINDOWS
    )

    env[
        "MROO_WINDOW_SEED"
    ] = str(
        WINDOW_SEED
    )

    # --------------------------------------------------------
    # Output directory
    #
    # For S-MROO, this is lambda-specific.
    # --------------------------------------------------------

    result_dir = (
        get_algorithm_result_dir(
            algorithm,
            setting_dir,
        )
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env[
        "MROO_RESULT_DIR"
    ] = str(
        result_dir
    )

    # ========================================================
    # MROO PARAMETERS
    # ========================================================

    if algorithm == "MROO":

        env[
            "MROO_ETA_VALUES_JSON"
        ] = json.dumps(
            MROO_ETA_VALUES
        )

        env[
            "MROO_KAPPA_VALUES_JSON"
        ] = json.dumps(
            MROO_KAPPA_VALUES
        )

    # ========================================================
    # DMD PARAMETERS
    # ========================================================

    elif algorithm == "DMD":

        env[
            "DMD_ETA_VALUES"
        ] = ",".join(

            str(
                value
            )

            for value
            in DMD_ETA_VALUES
        )

        env[
            "DMD_KAPPA_VALUES"
        ] = ",".join(

            str(
                value
            )

            for value
            in DMD_KAPPA_VALUES
        )

    # ========================================================
    # S-MROO-SUM PARAMETERS
    # ========================================================

    elif algorithm == "S-MROO-SUM":

        env[
            "SMROO_SUM_LAMBDA_1"
        ] = str(
            SMROO_SUM_LAMBDA_1
        )

        env[
            "SMROO_SUM_LAMBDA_2"
        ] = str(
            SMROO_SUM_LAMBDA_2
        )

    # ========================================================
    # S-MROO-MAX PARAMETERS
    # ========================================================

    elif algorithm == "S-MROO-MAX":

        env[
            "SMROO_MAX_LAMBDA_1"
        ] = str(
            SMROO_MAX_LAMBDA_1
        )

        env[
            "SMROO_MAX_LAMBDA_2"
        ] = str(
            SMROO_MAX_LAMBDA_2
        )

    # --------------------------------------------------------
    # Prevent BLAS oversubscription
    # --------------------------------------------------------

    env[
        "OMP_NUM_THREADS"
    ] = "1"

    env[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    env[
        "MKL_NUM_THREADS"
    ] = "1"

    env[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    runner_path = (
        ALGORITHM_DIR
        /
        RUNNERS[
            algorithm
        ]
    )

    if not runner_path.exists():

        raise FileNotFoundError(
            f"Runner for {algorithm} does not exist:\n"
            f"{runner_path}"
        )

    print(
        "\n"
        f"[beta={beta:g}, "
        f"m={m:g}, "
        f"beta/m={beta / m:g}] "
        f"Starting {algorithm}\n"
        f"Result directory: {result_dir}",
        flush=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(
                runner_path
            ),
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
# LOAD OUTPUTS
# ============================================================

def load_algorithm_outputs(
    algorithm,
    setting_dir,
):

    result_dir = (
        get_algorithm_result_dir(
            algorithm,
            setting_dir,
        )
    )

    result_file = (
        result_dir
        /
        RESULT_FILES[
            algorithm
        ]
    )

    summary_file = (
        result_dir
        /
        SUMMARY_FILES[
            algorithm
        ]
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

    window_df = (
        pd.read_csv(
            result_file
        )
    )

    summary_df = (
        pd.read_csv(
            summary_file
        )
    )

    if len(
        window_df
    ) == 0:

        raise RuntimeError(
            f"{algorithm} result file is empty:\n"
            f"{result_file}"
        )

    if len(
        summary_df
    ) == 0:

        raise RuntimeError(
            f"{algorithm} summary file is empty:\n"
            f"{summary_file}"
        )

    return (
        window_df,
        summary_df,
    )


# ============================================================
# MROO KAPPA COLUMN HELPER
# ============================================================

def get_mroo_kappa_component_columns(
    df,
):

    prefix = "kappa_init_"

    columns = []

    for column in df.columns:

        name = str(
            column
        )

        if not name.startswith(
            prefix
        ):

            continue

        suffix = name[
            len(
                prefix
            ):
        ]

        if suffix.isdigit():

            columns.append(
                column
            )

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


# ============================================================
# SELECT BEST MROO CONFIGURATION
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
        -
        set(
            summary_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "MROO summary missing columns: "
            f"{sorted(missing)}"
        )

    kappa_columns = (
        get_mroo_kappa_component_columns(
            summary_df
        )
    )

    if len(
        kappa_columns
    ) == 0:

        raise RuntimeError(
            "MROO summary must contain "
            "kappa_init_0, kappa_init_1, ..."
        )

    requested_mask = np.zeros(
        len(
            summary_df
        ),
        dtype=bool,
    )

    for eta in MROO_ETA_VALUES:

        for kappa_value in MROO_KAPPA_VALUES:

            kappa_array = np.asarray(
                kappa_value,
                dtype=float,
            )

            if kappa_array.ndim == 0:

                kappa_array = np.full(
                    len(
                        kappa_columns
                    ),
                    float(
                        kappa_array
                    ),
                )

            if kappa_array.shape != (
                len(
                    kappa_columns
                ),
            ):

                raise ValueError(
                    "MROO kappa has shape "
                    f"{kappa_array.shape}, expected "
                    f"({len(kappa_columns)},)."
                )

            mask = config_isclose(

                summary_df[
                    "eta"
                ].astype(
                    float
                ),

                float(
                    eta
                ),
            )

            for (
                index,
                column,
            ) in enumerate(
                kappa_columns
            ):

                mask &= config_isclose(

                    summary_df[
                        column
                    ].astype(
                        float
                    ),

                    float(
                        kappa_array[
                            index
                        ]
                    ),
                )

            requested_mask |= (
                mask
            )

    candidate_df = (

        summary_df[
            requested_mask
        ]

        .copy()
    )

    if len(
        candidate_df
    ) == 0:

        raise RuntimeError(
            "No MROO summary row matches "
            "the requested eta/kappa values."
        )

    return (

        candidate_df

        .sort_values(
            "mean_total_cost"
        )

        .iloc[
            0
        ]
    )


# ============================================================
# SELECT MROO WINDOWS
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

        config_isclose(

            window_df[
                "lambda_1"
            ].astype(
                float
            ),

            lambda_1,
        )

        &

        config_isclose(

            window_df[
                "eta"
            ].astype(
                float
            ),

            eta,
        )
    )

    kappa_columns = (
        get_mroo_kappa_component_columns(
            window_df
        )
    )

    if len(
        kappa_columns
    ) == 0:

        raise RuntimeError(
            "MROO window results do not contain "
            "kappa_init_i columns."
        )

    for column in kappa_columns:

        target = float(
            best_config[
                column
            ]
        )

        mask &= config_isclose(

            window_df[
                column
            ].astype(
                float
            ),

            target,
        )

    selected = (

        window_df[
            mask
        ]

        .copy()
    )

    if (

        selected[
            "window_id"
        ]

        .nunique()

        !=

        N_WINDOWS
    ):

        raise RuntimeError(
            f"Expected {N_WINDOWS} MROO windows "
            f"but found "
            f"{selected['window_id'].nunique()}."
        )

    return selected


# ============================================================
# SELECT S-MROO CONFIGURATION
# ============================================================

def select_smroo_configuration(
    summary_df,
    algorithm,
):

    if algorithm == "S-MROO-SUM":

        target_lambda_1 = (
            SMROO_SUM_LAMBDA_1
        )

        target_lambda_2 = (
            SMROO_SUM_LAMBDA_2
        )

    elif algorithm == "S-MROO-MAX":

        target_lambda_1 = (
            SMROO_MAX_LAMBDA_1
        )

        target_lambda_2 = (
            SMROO_MAX_LAMBDA_2
        )

    else:

        raise ValueError(
            f"Unsupported algorithm: "
            f"{algorithm}"
        )

    required = {
        "lambda_1",
        "lambda_2",
        "mean_total_cost",
    }

    missing = (
        required
        -
        set(
            summary_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{algorithm} summary missing columns: "
            f"{sorted(missing)}"
        )

    mask = (

        config_isclose(

            summary_df[
                "lambda_1"
            ].astype(
                float
            ),

            target_lambda_1,
        )

        &

        config_isclose(

            summary_df[
                "lambda_2"
            ].astype(
                float
            ),

            target_lambda_2,
        )
    )

    selected = (

        summary_df[
            mask
        ]

        .copy()
    )

    if len(
        selected
    ) == 0:

        raise RuntimeError(
            f"{algorithm}: no summary row found for "
            f"lambda_1={target_lambda_1}, "
            f"lambda_2={target_lambda_2}."
        )

    if len(
        selected
    ) > 1:

        selected = (

            selected

            .sort_values(
                "mean_total_cost"
            )

            .head(
                1
            )
        )

    return (
        selected.iloc[
            0
        ]
    )


# ============================================================
# SELECT S-MROO WINDOWS
# ============================================================

def get_selected_smroo_windows(
    window_df,
    best_config,
    algorithm,
):

    lambda_1 = float(
        best_config[
            "lambda_1"
        ]
    )

    lambda_2 = float(
        best_config[
            "lambda_2"
        ]
    )

    mask = (

        config_isclose(

            window_df[
                "lambda_1"
            ].astype(
                float
            ),

            lambda_1,
        )

        &

        config_isclose(

            window_df[
                "lambda_2"
            ].astype(
                float
            ),

            lambda_2,
        )
    )

    selected = (

        window_df[
            mask
        ]

        .copy()
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

        raise RuntimeError(
            f"Expected {N_WINDOWS} {algorithm} windows "
            f"for lambda_1={lambda_1}, "
            f"lambda_2={lambda_2}; "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# SELECT DMD CONFIGURATION
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
        -
        set(
            summary_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "DMD summary missing columns: "
            f"{sorted(missing)}"
        )

    requested_mask = np.zeros(
        len(
            summary_df
        ),
        dtype=bool,
    )

    for eta in DMD_ETA_VALUES:

        for kappa in DMD_KAPPA_VALUES:

            requested_mask |= (

                config_isclose(

                    summary_df[
                        "eta"
                    ].astype(
                        float
                    ),

                    float(
                        eta
                    ),
                )

                &

                config_isclose(

                    summary_df[
                        "kappa_init"
                    ].astype(
                        float
                    ),

                    float(
                        kappa
                    ),
                )
            )

    candidate_df = (

        summary_df[
            requested_mask
        ]

        .copy()
    )

    if len(
        candidate_df
    ) == 0:

        available = (

            summary_df[
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
            "No DMD configuration matches the requested "
            "eta/kappa values.\n"
            f"Requested eta values: {DMD_ETA_VALUES}\n"
            f"Requested kappa values: {DMD_KAPPA_VALUES}\n"
            "Available DMD configurations:\n"
            f"{available.to_string(index=False)}"
        )

    return (

        candidate_df

        .sort_values(
            "mean_total_cost"
        )

        .iloc[
            0
        ]
    )


# ============================================================
# SELECT DMD WINDOWS
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

    kappa = float(
        best_config[
            "kappa_init"
        ]
    )

    mask = (

        config_isclose(

            window_df[
                "eta"
            ].astype(
                float
            ),

            eta,
        )

        &

        config_isclose(

            window_df[
                "kappa_init"
            ].astype(
                float
            ),

            kappa,
        )
    )

    selected = (

        window_df[
            mask
        ]

        .copy()
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

        raise RuntimeError(
            f"Expected {N_WINDOWS} DMD windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# OFFLINE OPT
# ============================================================

def select_offline_opt_configuration(
    summary_df,
):

    if len(
        summary_df
    ) == 0:

        raise RuntimeError(
            "OFFLINE-OPT summary is empty."
        )

    return (
        summary_df.iloc[
            0
        ]
    )


def get_selected_offline_opt_windows(
    window_df,
):

    selected = (

        window_df

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

        raise RuntimeError(
            f"Expected {N_WINDOWS} OFFLINE-OPT windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# GREEDY
# ============================================================

def select_greedy_configuration(
    summary_df,
):

    if len(
        summary_df
    ) == 0:

        raise RuntimeError(
            "GREEDY summary is empty."
        )

    return (
        summary_df.iloc[
            0
        ]
    )


def get_selected_greedy_windows(
    window_df,
):

    selected = (

        window_df

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

        raise RuntimeError(
            f"Expected {N_WINDOWS} GREEDY windows, "
            f"found {len(selected)}."
        )

    return selected


# ============================================================
# ADD METADATA
# ============================================================

def add_metadata(
    frame,
    algorithm,
    beta,
    m,
):

    frame = (
        frame.copy()
    )

    frame[
        "algorithm"
    ] = algorithm

    frame[
        "beta"
    ] = float(
        beta
    )

    frame[
        "m"
    ] = float(
        m
    )

    frame[
        "beta_over_m"
    ] = float(
        beta
        /
        m
    )

    frame[
        "experiment_window_size"
    ] = int(
        WINDOW_SIZE
    )

    frame[
        "experiment_window_seed"
    ] = int(
        WINDOW_SEED
    )

    frame = (
        add_config_id_column(
            frame,
            algorithm,
        )
    )

    first = [

        "algorithm",

        "beta",

        "m",

        "beta_over_m",

        "experiment_window_size",

        "experiment_window_seed",

        "config_id",
    ]

    remaining = [

        column

        for column
        in frame.columns

        if column
        not in first
    ]

    return frame[
        first
        +
        remaining
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
            float(
                beta
            ),

        "m":
            float(
                m
            ),

        "beta_over_m":
            float(
                beta
                /
                m
            ),

        "n_windows":
            int(
                len(
                    selected_windows
                )
            ),

        "lambda_1":
            np.nan,

        "lambda_2":
            np.nan,

        "eta":
            np.nan,

        "kappa_init":
            np.nan,

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

    for column in [

        "lambda_1",

        "lambda_2",

        "eta",

        "gamma_R_q",

        "L_q",
    ]:

        if (

            column
            in best_config.index

            and

            not pd.isna(
                best_config[
                    column
                ]
            )
        ):

            row[
                column
            ] = float(
                best_config[
                    column
                ]
            )

    # --------------------------------------------------------
    # Scalar/string kappa
    # --------------------------------------------------------

    if (

        "kappa_init"
        in best_config.index

        and

        not pd.isna(
            best_config[
                "kappa_init"
            ]
        )
    ):

        value = (
            best_config[
                "kappa_init"
            ]
        )

        try:

            row[
                "kappa_init"
            ] = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            row[
                "kappa_init"
            ] = str(
                value
            )

    # --------------------------------------------------------
    # MROO vector-kappa columns
    # --------------------------------------------------------

    for column in (

        get_mroo_kappa_component_columns(

            pd.DataFrame(
                columns=
                    best_config.index
            )
        )
    ):

        value = (
            best_config[
                column
            ]
        )

        if not pd.isna(
            value
        ):

            row[
                column
            ] = float(
                value
            )

    return row


# ============================================================
# VERIFY SAME WINDOWS
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
                ==
                algorithm

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

        if len(
            current
        ) != N_WINDOWS:

            raise RuntimeError(
                f"{algorithm} has "
                f"{len(current)} rows; "
                f"expected {N_WINDOWS}."
            )

        if reference is None:

            reference = (
                current
            )

        elif not current.equals(
            reference
        ):

            raise RuntimeError(
                "Algorithms did not use "
                "the same windows."
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
        /
        (
            f"beta_"
            f"{safe_float_label(beta)}"
            f"_m_"
            f"{safe_float_label(m)}"
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
        "SETTING\n"
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

        # ----------------------------------------------------
        # MROO
        # ----------------------------------------------------

        if algorithm == "MROO":

            run_algorithm(
                algorithm,
                beta,
                m,
                setting_dir,
            )

            mark_algorithm_complete(
                algorithm,
                setting_dir,
            )

        # ----------------------------------------------------
        # DMD
        # ----------------------------------------------------

        elif algorithm == "DMD":

            run_algorithm(
                algorithm,
                beta,
                m,
                setting_dir,
            )

            mark_algorithm_complete(
                algorithm,
                setting_dir,
            )

        # ----------------------------------------------------
        # S-MROO-SUM / S-MROO-MAX
        #
        # Exact lambda pair gets its own permanent directory.
        # ----------------------------------------------------

        elif algorithm in {

            "S-MROO-SUM",

            "S-MROO-MAX",
        }:

            if (
                smroo_requested_configuration_complete(
                    algorithm,
                    setting_dir,
                )
            ):

                (
                    lambda_1,
                    lambda_2,
                ) = (
                    get_smroo_target_lambdas(
                        algorithm
                    )
                )

                print(
                    "\n"
                    f"[beta={beta:g}, m={m:g}] "
                    f"{algorithm} already complete "
                    f"for lambda_1={lambda_1}, "
                    f"lambda_2={lambda_2} "
                    "- loading permanent configuration results.",
                    flush=True,
                )

            else:

                run_algorithm(
                    algorithm,
                    beta,
                    m,
                    setting_dir,
                )

                (
                    produced_windows,
                    produced_summary,
                ) = (
                    load_algorithm_outputs(
                        algorithm,
                        setting_dir,
                    )
                )

                (
                    target_lambda_1,
                    target_lambda_2,
                ) = (
                    get_smroo_target_lambdas(
                        algorithm
                    )
                )

                produced_mask = (

                    config_isclose(

                        produced_windows[
                            "lambda_1"
                        ].astype(
                            float
                        ),

                        target_lambda_1,
                    )

                    &

                    config_isclose(

                        produced_windows[
                            "lambda_2"
                        ].astype(
                            float
                        ),

                        target_lambda_2,
                    )
                )

                produced_target = (

                    produced_windows[
                        produced_mask
                    ]

                    .copy()
                )

                if (

                    "window_id"
                    not in produced_target.columns

                    or

                    produced_target[
                        "window_id"
                    ]

                    .dropna()

                    .astype(
                        int
                    )

                    .nunique()

                    !=

                    N_WINDOWS
                ):

                    raise RuntimeError(
                        f"{algorithm} finished but did not produce "
                        f"{N_WINDOWS} windows for "
                        f"lambda_1={target_lambda_1}, "
                        f"lambda_2={target_lambda_2}."
                    )

                mark_algorithm_complete(
                    algorithm,
                    setting_dir,
                )

        # ----------------------------------------------------
        # OFFLINE-OPT / GREEDY
        # ----------------------------------------------------

        elif algorithm_is_complete(
            algorithm,
            setting_dir,
        ):

            print(
                "\n"
                f"[beta={beta:g}, m={m:g}] "
                f"{algorithm} already complete "
                "- loading existing results.",
                flush=True,
            )

        else:

            run_algorithm(
                algorithm,
                beta,
                m,
                setting_dir,
            )

            mark_algorithm_complete(
                algorithm,
                setting_dir,
            )

        # ====================================================
        # LOAD OUTPUTS
        # ====================================================

        (
            window_df,
            summary_df,
        ) = (
            load_algorithm_outputs(
                algorithm,
                setting_dir,
            )
        )

        # ====================================================
        # ARCHIVE ALL CONFIGURATIONS BEFORE SELECTING ONE
        # ====================================================

        archive_algorithm_outputs(
            algorithm,
            beta,
            m,
            setting_dir,
            window_df,
            summary_df,
        )

        # ====================================================
        # SELECT CONFIGURATION FOR CURRENT COMPARISON
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

        elif algorithm in {

            "S-MROO-SUM",

            "S-MROO-MAX",
        }:

            best_config = (
                select_smroo_configuration(
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
        # CURRENT SUMMARY ROW
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

        summary_row[
            "experiment_window_size"
        ] = int(
            WINDOW_SIZE
        )

        summary_row[
            "experiment_window_seed"
        ] = int(
            WINDOW_SEED
        )

        summary_row[
            "config_id"
        ] = (
            build_config_id_from_row(

                algorithm,

                pd.Series(
                    summary_row
                ),
            )
        )

        summary_rows.append(
            summary_row
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            "\n"
            f"[beta={beta:g}, m={m:g}] "
            f"{algorithm}:"
        )

        if not pd.isna(
            summary_row[
                "lambda_1"
            ]
        ):

            print(
                "    lambda_1 =",
                summary_row[
                    "lambda_1"
                ],
            )

        if not pd.isna(
            summary_row[
                "lambda_2"
            ]
        ):

            print(
                "    lambda_2 =",
                summary_row[
                    "lambda_2"
                ],
            )

        if not pd.isna(
            summary_row[
                "eta"
            ]
        ):

            print(
                "    eta =",
                summary_row[
                    "eta"
                ],
            )

        if not pd.isna(
            summary_row[
                "kappa_init"
            ]
        ):

            print(
                "    kappa_init =",
                summary_row[
                    "kappa_init"
                ],
            )

        print(
            "    config_id  = "
            f"{summary_row['config_id']}"
        )

        print(
            "    hitting    = "
            f"{summary_row['mean_hitting_cost']:.6f} "
            "+/- "
            f"{summary_row['std_hitting_cost']:.6f}"
        )

        print(
            "    long-term  = "
            f"{summary_row['mean_long_term_cost']:.6f} "
            "+/- "
            f"{summary_row['std_long_term_cost']:.6f}"
        )

        print(
            "    total      = "
            f"{summary_row['mean_total_cost']:.6f} "
            "+/- "
            f"{summary_row['std_total_cost']:.6f}"
        )

    # ========================================================
    # CURRENT COMPARISON
    #
    # Exactly one selected configuration for each algorithm.
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

        len(
            RUNNERS
        )

        *

        N_WINDOWS
    )

    if len(
        combined_windows
    ) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} rows "
            f"for beta={beta}, m={m}; "
            f"found {len(combined_windows)}."
        )

    verify_same_windows(
        combined_windows
    )

    current_summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    # ========================================================
    # EXPLICIT CURRENT-COMPARISON DIRECTORY
    # ========================================================

    current_comparison_dir = (
        setting_dir
        /
        CURRENT_COMPARISON_DIRNAME
    )

    current_comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_window_file = (
        current_comparison_dir
        /
        "combined_algorithm_window_results.csv"
    )

    current_summary_file = (
        current_comparison_dir
        /
        "combined_algorithm_summary.csv"
    )

    combined_windows.to_csv(
        current_window_file,
        index=False,
    )

    current_summary_df.to_csv(
        current_summary_file,
        index=False,
    )

    # ========================================================
    # BACKWARD-COMPATIBLE ROOT FILES
    #
    # plot_beta_m_costs.py can continue reading the old paths.
    # These represent CURRENT comparison only.
    # ========================================================

    legacy_combined_window_file = (
        setting_dir
        /
        "combined_algorithm_window_results.csv"
    )

    legacy_combined_summary_file = (
        setting_dir
        /
        "combined_algorithm_summary.csv"
    )

    combined_windows.to_csv(
        legacy_combined_window_file,
        index=False,
    )

    current_summary_df.to_csv(
        legacy_combined_summary_file,
        index=False,
    )

    print(
        "\nSaved CURRENT comparison window results:"
    )

    print(
        current_window_file
    )

    print(
        "\nSaved CURRENT comparison summary:"
    )

    print(
        current_summary_file
    )

    print(
        "\nBackward-compatible current comparison files:"
    )

    print(
        legacy_combined_window_file
    )

    print(
        legacy_combined_summary_file
    )

    print(
        "\nPermanent ALL-CONFIGURATION history:"
    )

    print(
        setting_dir
        /
        HISTORY_DIRNAME
        /
        SETTING_HISTORY_WINDOW_FILENAME
    )

    print(
        setting_dir
        /
        HISTORY_DIRNAME
        /
        SETTING_HISTORY_SUMMARY_FILENAME
    )

    return (
        combined_windows,
        summary_rows,
    )


# ============================================================
# SAVE MASTER RESULTS
#
# Unlike the previous implementation, these files are merged
# with previous runs instead of being recreated from only the
# current invocation.
# ============================================================

def save_master_results(
    all_window_frames,
    all_summary_rows,
):

    # ========================================================
    # WINDOW MASTER
    # ========================================================

    if len(
        all_window_frames
    ) > 0:

        new_windows_df = (
            pd.concat(
                all_window_frames,
                ignore_index=True,
                sort=False,
            )
        )

        if (
            "config_id"
            not in new_windows_df.columns
        ):

            new_windows_df[
                "config_id"
            ] = np.nan

        for (
            index,
            row,
        ) in new_windows_df.iterrows():

            if not _value_present(
                row.get(
                    "config_id",
                    np.nan,
                )
            ):

                new_windows_df.at[
                    index,
                    "config_id",
                ] = (
                    build_config_id_from_row(
                        row[
                            "algorithm"
                        ],
                        row,
                    )
                )

        # ----------------------------------------------------
        # Existing historical master
        # ----------------------------------------------------

        if MASTER_WINDOW_FILE.exists():

            try:

                old_windows_df = (
                    pd.read_csv(
                        MASTER_WINDOW_FILE
                    )
                )

            except Exception as error:

                raise RuntimeError(
                    "Could not read existing master window file:\n"
                    f"{MASTER_WINDOW_FILE}\n"
                    f"Original error: {error}"
                ) from error

            if len(
                old_windows_df
            ) > 0:

                if (
                    "config_id"
                    not in old_windows_df.columns
                ):

                    old_windows_df[
                        "config_id"
                    ] = np.nan

                if (
                    "algorithm"
                    in old_windows_df.columns
                ):

                    for (
                        index,
                        row,
                    ) in old_windows_df.iterrows():

                        if not _value_present(
                            row.get(
                                "config_id",
                                np.nan,
                            )
                        ):

                            old_windows_df.at[
                                index,
                                "config_id",
                            ] = (
                                build_config_id_from_row(
                                    row[
                                        "algorithm"
                                    ],
                                    row,
                                )
                            )

                new_windows_df = (
                    pd.concat(
                        [
                            old_windows_df,
                            new_windows_df,
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                )

        window_dedupe_columns = [

            column

            for column in [

                "algorithm",

                "beta",

                "m",

                "experiment_window_size",

                "experiment_window_seed",

                "config_id",

                "window_id",
            ]

            if column
            in new_windows_df.columns
        ]

        if len(
            window_dedupe_columns
        ) > 0:

            new_windows_df = (

                new_windows_df

                .drop_duplicates(

                    subset=
                        window_dedupe_columns,

                    keep="last",
                )
            )

        window_sort_columns = [

            column

            for column in [

                "beta_over_m",

                "beta",

                "m",

                "algorithm",

                "config_id",

                "window_id",
            ]

            if column
            in new_windows_df.columns
        ]

        if len(
            window_sort_columns
        ) > 0:

            new_windows_df = (

                new_windows_df

                .sort_values(
                    window_sort_columns
                )
            )

        all_windows_df = (

            new_windows_df

            .reset_index(
                drop=True
            )
        )

        all_windows_df.to_csv(
            MASTER_WINDOW_FILE,
            index=False,
        )

    else:

        if MASTER_WINDOW_FILE.exists():

            all_windows_df = (
                pd.read_csv(
                    MASTER_WINDOW_FILE
                )
            )

        else:

            all_windows_df = (
                pd.DataFrame()
            )

    # ========================================================
    # SUMMARY MASTER
    # ========================================================

    if len(
        all_summary_rows
    ) > 0:

        new_summary_df = (
            pd.DataFrame(
                all_summary_rows
            )
        )

        if (
            "config_id"
            not in new_summary_df.columns
        ):

            new_summary_df[
                "config_id"
            ] = np.nan

        for (
            index,
            row,
        ) in new_summary_df.iterrows():

            if not _value_present(
                row.get(
                    "config_id",
                    np.nan,
                )
            ):

                new_summary_df.at[
                    index,
                    "config_id",
                ] = (
                    build_config_id_from_row(
                        row[
                            "algorithm"
                        ],
                        row,
                    )
                )

        # ----------------------------------------------------
        # Existing historical master
        # ----------------------------------------------------

        if MASTER_SUMMARY_FILE.exists():

            try:

                old_summary_df = (
                    pd.read_csv(
                        MASTER_SUMMARY_FILE
                    )
                )

            except Exception as error:

                raise RuntimeError(
                    "Could not read existing master summary file:\n"
                    f"{MASTER_SUMMARY_FILE}\n"
                    f"Original error: {error}"
                ) from error

            if len(
                old_summary_df
            ) > 0:

                if (
                    "config_id"
                    not in old_summary_df.columns
                ):

                    old_summary_df[
                        "config_id"
                    ] = np.nan

                if (
                    "algorithm"
                    in old_summary_df.columns
                ):

                    for (
                        index,
                        row,
                    ) in old_summary_df.iterrows():

                        if not _value_present(
                            row.get(
                                "config_id",
                                np.nan,
                            )
                        ):

                            old_summary_df.at[
                                index,
                                "config_id",
                            ] = (
                                build_config_id_from_row(
                                    row[
                                        "algorithm"
                                    ],
                                    row,
                                )
                            )

                new_summary_df = (
                    pd.concat(
                        [
                            old_summary_df,
                            new_summary_df,
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                )

        summary_dedupe_columns = [

            column

            for column in [

                "algorithm",

                "beta",

                "m",

                "experiment_window_size",

                "experiment_window_seed",

                "config_id",
            ]

            if column
            in new_summary_df.columns
        ]

        if len(
            summary_dedupe_columns
        ) > 0:

            new_summary_df = (

                new_summary_df

                .drop_duplicates(

                    subset=
                        summary_dedupe_columns,

                    keep="last",
                )
            )

        summary_sort_columns = [

            column

            for column in [

                "beta_over_m",

                "beta",

                "m",

                "algorithm",

                "config_id",
            ]

            if column
            in new_summary_df.columns
        ]

        if len(
            summary_sort_columns
        ) > 0:

            new_summary_df = (

                new_summary_df

                .sort_values(
                    summary_sort_columns
                )
            )

        summary_df = (

            new_summary_df

            .reset_index(
                drop=True
            )
        )

        summary_df.to_csv(
            MASTER_SUMMARY_FILE,
            index=False,
        )

    else:

        if MASTER_SUMMARY_FILE.exists():

            summary_df = (
                pd.read_csv(
                    MASTER_SUMMARY_FILE
                )
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
                    float(
                        beta
                    ),

                    float(
                        m
                    ),
                )

                for beta
                in BETA_VALUES

                for m
                in M_VALUES
            )
        )
    )

    total_settings = len(
        settings
    )

    if total_settings == 0:

        raise RuntimeError(
            "No beta/m settings configured."
        )

    worker_count = min(
        MAX_SETTING_WORKERS,
        total_settings,
    )

    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "MATRIX-MEMORY ALGORITHM WINDOW SWEEP"
    )

    print(
        "========================================"
    )

    print(
        "Experiment mode =",
        EXPERIMENT_MODE,
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
        "Shared MROO/DMD eta values =",
        MROO_DMD_ETA_VALUES,
    )

    print(
        "Shared MROO/DMD kappa values =",
        MROO_DMD_KAPPA_VALUES,
    )

    print(
        "S-MROO-SUM lambdas =",
        (
            SMROO_SUM_LAMBDA_1,
            SMROO_SUM_LAMBDA_2,
        ),
    )

    print(
        "S-MROO-MAX lambdas =",
        (
            SMROO_MAX_LAMBDA_1,
            SMROO_MAX_LAMBDA_2,
        ),
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

    # ========================================================
    # CURRENT INVOCATION STORAGE
    # ========================================================

    all_window_frames = []

    all_summary_rows = []

    # ========================================================
    # RUN BETA/M SETTINGS
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=
            worker_count
    ) as executor:

        future_to_setting = {}

        for (
            beta,
            m,
        ) in settings:

            future = (
                executor.submit(
                    run_one_setting,
                    beta,
                    m,
                )
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

            (
                beta,
                m,
            ) = (
                future_to_setting[
                    future
                ]
            )

            try:

                (
                    window_df,
                    summary_rows,
                ) = (
                    future.result()
                )

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
                "\nCompleted beta/m settings: "
                f"{completed}/{total_settings}",
                flush=True,
            )

            # ------------------------------------------------
            # Persist after every completed setting.
            # ------------------------------------------------

            save_master_results(
                all_window_frames,
                all_summary_rows,
            )

    # ========================================================
    # FINAL MASTER SAVE
    # ========================================================

    (
        all_windows_df,
        summary_df,
    ) = (
        save_master_results(
            all_window_frames,
            all_summary_rows,
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

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

        "config_id",

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
        "\nHISTORICAL SUMMARY:\n"
    )

    print(

        summary_df[
            display_columns
        ]

        .to_string(
            index=False
        )
    )

    print(
        "\nHistorical individual window rows =",
        len(
            all_windows_df
        ),
    )

    expected_current_window_rows = (

        total_settings

        *

        len(
            RUNNERS
        )

        *

        N_WINDOWS
    )

    print(
        "Current invocation comparison rows =",
        expected_current_window_rows,
    )

    print(
        "Historical master rows retained =",
        len(
            all_windows_df
        ),
    )

    print(
        "\nHistorical selected-configuration window results:"
    )

    print(
        MASTER_WINDOW_FILE
    )

    print(
        "\nHistorical selected-configuration summaries:"
    )

    print(
        MASTER_SUMMARY_FILE
    )

    print(
        "\nFor ALL configurations of a specific beta/m setting, "
        "use:"
    )

    print(
        "<setting_dir>/history/"
        "all_algorithm_window_results.csv"
    )

    print(
        "<setting_dir>/history/"
        "all_algorithm_summary.csv"
    )

    print(
        "\nEach exact configuration is also stored under:"
    )

    print(
        "<setting_dir>/configurations/"
        "<algorithm>/<config_id>/"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()