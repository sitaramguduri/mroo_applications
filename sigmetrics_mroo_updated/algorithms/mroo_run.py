import ast
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
# ============================================================
# PROJECT PATHS
# ============================================================
ALGORITHM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ALGORITHM_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))
from mroo import run_mroo
from config import (
    LAMBDA_1_VALUES,
    A_MATRICES,
)
# ============================================================
# DIMENSION / A-MATRIX VALIDATION
#
# There are exactly D matrices:
#     A_1, ..., A_D
#
# and each A_i is D x D.
#
# Therefore:
#     A_MATRICES.shape = (D, D, D)
# ============================================================
A_MATRICES = np.asarray(
    A_MATRICES,
    dtype=float,
)
if A_MATRICES.ndim != 3:
    raise ValueError(
        "A_MATRICES must be a 3-dimensional array. "
        f"Found shape {A_MATRICES.shape}."
    )
D = int(
    A_MATRICES.shape[0]
)
if A_MATRICES.shape != (
    D,
    D,
    D,
):
    raise ValueError(
        "A_MATRICES must have shape "
        f"({D}, {D}, {D}), "
        f"but got {A_MATRICES.shape}."
    )
# ============================================================
# DATA
# ============================================================
DATA_FILE = (
    PROJECT_DIR
    / "new_results"
    / "mroo_demand_trace_norm_weekday_aligned_1min.csv"
)
DEMAND_COLUMNS = [
    "code",
    "chat",
    "non-interactive",
]
# ============================================================
# RESULT DIRECTORY
# main_sweep.py can override this with MROO_RESULT_DIR.
# ============================================================
DEFAULT_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "window_experiment"
)
RESULT_DIR = Path(
    os.environ.get(
        "MROO_RESULT_DIR",
        str(DEFAULT_RESULT_DIR),
    )
)
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
print(
    "MROO result directory:",
    RESULT_DIR,
)
# ============================================================
# WINDOW CONFIGURATION
# ============================================================
WINDOW_SIZE = int(
    os.environ.get(
        "MROO_WINDOW_SIZE",
        "1440",
    )
)
N_WINDOWS = int(
    os.environ.get(
        "MROO_N_WINDOWS",
        "100",
    )
)
RANDOM_SEED = int(
    os.environ.get(
        "MROO_WINDOW_SEED",
        "0",
    )
)
# ============================================================
# SHARED WINDOW FILE
# ============================================================
WINDOW_FILE = (
    ALGORITHM_DIR
    / "new_results"
    / (
        f"paper_window_starts"
        f"_size_{WINDOW_SIZE}"
        f"_n_{N_WINDOWS}"
        f"_seed_{RANDOM_SEED}.csv"
    )
)
# ============================================================
# OUTPUT FILES
# ============================================================
RESULT_FILE = (
    RESULT_DIR
    / "mroo_window_results.csv"
)
SUMMARY_FILE = (
    RESULT_DIR
    / "mroo_window_summary.csv"
)
COMPLETED_CONFIG_FILE = (
    RESULT_DIR
    / "mroo_completed_configs.csv"
)
CURRENT_RUN_CONFIG_FILE = (
    RESULT_DIR
    / "mroo_current_run_configs.csv"
)
# ============================================================
# PARALLEL SETTINGS
# ============================================================
MAX_WORKERS = int(
    os.environ.get(
        "MROO_WINDOW_WORKERS",
        "8",
    )
)
# ============================================================
# KAPPA HELPERS
# ============================================================
def _parse_kappa_value(
    kappa_value,
):
    if isinstance(
        kappa_value,
        str,
    ):
        text = (
            kappa_value.strip()
        )
        try:
            return ast.literal_eval(
                text
            )
        except (
            ValueError,
            SyntaxError,
        ):
            return float(
                text
            )
    return kappa_value
def normalize_kappa_init(
    kappa_value,
    D,
):
    """
    Scalar:
        1.2 -> [1.2, 1.2, 1.2]
    Vector:
        [1.5, 1.0, 0.5] -> used directly
    """
    kappa_value = (
        _parse_kappa_value(
            kappa_value
        )
    )
    arr = np.asarray(
        kappa_value,
        dtype=float,
    )
    # --------------------------------------------------------
    # Scalar
    # --------------------------------------------------------
    if arr.ndim == 0:
        arr = np.full(
            D,
            float(arr),
            dtype=float,
        )
    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------
    elif arr.shape != (
        D,
    ):
        raise ValueError(
            "kappa value must be either a scalar "
            f"or a vector of shape ({D},), "
            f"but got shape {arr.shape}."
        )
    if np.any(
        ~np.isfinite(
            arr
        )
    ):
        raise ValueError(
            "All kappa_init values must be finite."
        )
    if np.any(
        arr < 0.0
    ):
        raise ValueError(
            "All kappa_init values must be nonnegative."
        )
    return arr
def kappa_tuple(
    kappa_value,
    D,
):
    arr = (
        normalize_kappa_init(
            kappa_value,
            D,
        )
    )
    return tuple(
        round(
            float(x),
            12,
        )
        for x in arr
    )
def kappa_label(
    kappa_value,
    D,
):
    arr = (
        normalize_kappa_init(
            kappa_value,
            D,
        )
    )
    if np.allclose(
        arr,
        arr[0],
        rtol=0.0,
        atol=1e-12,
    ):
        return (
            f"{arr[0]:.8g}"
        )
    return (
        "["
        +
        ",".join(
            f"{x:.8g}"
            for x in arr
        )
        +
        "]"
    )
def kappa_component_columns(
    D,
):
    return [
        f"kappa_init_{i}"
        for i in range(
            D
        )
    ]
def ensure_kappa_columns(
    df,
    D,
):
    """
    Makes old scalar result / summary / completion files
    compatible with the vector kappa format.
    """
    if df is None:
        return df
    df = df.copy()
    component_columns = (
        kappa_component_columns(
            D
        )
    )
    # --------------------------------------------------------
    # Empty dataframe
    # --------------------------------------------------------
    if len(
        df
    ) == 0:
        for column in component_columns:
            if (
                column
                not in df.columns
            ):
                df[
                    column
                ] = pd.Series(
                    dtype=float
                )
        return df
    # --------------------------------------------------------
    # Already vector format
    # --------------------------------------------------------
    if all(
        column in df.columns
        for column in component_columns
    ):
        labels = []
        for _, row in df.iterrows():
            vector = np.array(
                [
                    row[
                        column
                    ]
                    for column
                    in component_columns
                ],
                dtype=float,
            )
            labels.append(
                kappa_label(
                    vector,
                    D,
                )
            )
        df[
            "kappa_init"
        ] = labels
        return df
    # --------------------------------------------------------
    # Old scalar format
    # --------------------------------------------------------
    if (
        "kappa_init"
        not in df.columns
    ):
        raise RuntimeError(
            "Existing MROO file has neither "
            "kappa_init nor kappa_init_i columns."
        )
    recovered = np.asarray(
        [
            normalize_kappa_init(
                value,
                D,
            )
            for value
            in df[
                "kappa_init"
            ]
        ],
        dtype=float,
    )
    for (
        i,
        column,
    ) in enumerate(
        component_columns
    ):
        df[
            column
        ] = (
            recovered[
                :,
                i,
            ]
        )
    df[
        "kappa_init"
    ] = [
        kappa_label(
            vector,
            D,
        )
        for vector
        in recovered
    ]
    return df
def config_key(
    lambda_1,
    eta,
    kappa_value,
    D,
):
    return (
        round(
            float(
                lambda_1
            ),
            12,
        ),
        round(
            float(
                eta
            ),
            12,
        ),
        kappa_tuple(
            kappa_value,
            D,
        ),
    )
# ============================================================
# LOAD DATA
# ============================================================
def load_demand():
    data = pd.read_csv(
        DATA_FILE
    )
    return (
        data[
            DEMAND_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )
# ============================================================
# CREATE / LOAD PAPER-STYLE WINDOWS
# ============================================================
def get_window_starts(
    T_full,
):
    # --------------------------------------------------------
    # Reuse existing window definitions
    # --------------------------------------------------------
    if WINDOW_FILE.exists():
        window_df = pd.read_csv(
            WINDOW_FILE
        )
        required_columns = {
            "window_id",
            "start_index",
            "end_index",
        }
        missing = (
            required_columns
            -
            set(
                window_df.columns
            )
        )
        if missing:
            raise RuntimeError(
                "Existing window file is missing columns: "
                f"{sorted(missing)}"
            )
        starts = (
            window_df[
                "start_index"
            ]
            .to_numpy(
                dtype=int
            )
        )
        if np.any(
            starts
            +
            WINDOW_SIZE
            >
            T_full
        ):
            raise RuntimeError(
                "Existing window file contains windows "
                "outside the current demand trace."
            )
        if len(
            starts
        ) != N_WINDOWS:
            raise RuntimeError(
                f"Existing window file contains "
                f"{len(starts)} windows, "
                f"but N_WINDOWS={N_WINDOWS}."
            )
        print(
            "\nLoaded existing shared windows from:"
        )
        print(
            WINDOW_FILE
        )
        return starts
    # --------------------------------------------------------
    # Create new windows
    # --------------------------------------------------------
    if WINDOW_SIZE > T_full:
        raise ValueError(
            f"WINDOW_SIZE={WINDOW_SIZE} exceeds "
            f"trace length={T_full}."
        )
    total_valid_windows = (
        T_full
        -
        WINDOW_SIZE
        +
        1
    )
    if N_WINDOWS > total_valid_windows:
        raise ValueError(
            f"Requested {N_WINDOWS} windows, "
            f"but only {total_valid_windows} "
            "valid windows exist."
        )
    number_of_full_days = (
        T_full
        //
        WINDOW_SIZE
    )
    exact_day_starts = (
        np.arange(
            number_of_full_days,
            dtype=int,
        )
        *
        WINDOW_SIZE
    )
    exact_day_starts = (
        exact_day_starts[
            :min(
                len(
                    exact_day_starts
                ),
                N_WINDOWS,
            )
        ]
    )
    selected = list(
        exact_day_starts.astype(
            int
        )
    )
    all_valid_starts = np.arange(
        0,
        T_full
        -
        WINDOW_SIZE
        +
        1,
        dtype=int,
    )
    candidate_starts = (
        np.setdiff1d(
            all_valid_starts,
            exact_day_starts,
            assume_unique=False,
        )
    )
    rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )
    remaining_needed = (
        N_WINDOWS
        -
        len(
            selected
        )
    )
    if remaining_needed > 0:
        sampled_starts = rng.choice(
            candidate_starts,
            size=remaining_needed,
            replace=False,
        )
        selected.extend(
            sampled_starts.astype(
                int
            )
        )
    starts = np.asarray(
        selected,
        dtype=int,
    )
    window_types = (
        [
            "exact_day"
        ]
        *
        len(
            exact_day_starts
        )
        +
        [
            "sampled"
        ]
        *
        remaining_needed
    )
    window_df = pd.DataFrame(
        {
            "window_id":
                np.arange(
                    1,
                    N_WINDOWS + 1,
                ),
            "start_index":
                starts,
            "end_index":
                starts
                +
                WINDOW_SIZE,
            "window_type":
                window_types,
        }
    )
    WINDOW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    window_df.to_csv(
        WINDOW_FILE,
        index=False,
    )
    print(
        "\nCreated shared paper-style windows:"
    )
    print(
        WINDOW_FILE
    )
    print(
        "Exact-day windows =",
        len(
            exact_day_starts
        ),
    )
    print(
        "Sampled windows =",
        remaining_needed,
    )
    return starts
# ============================================================
# RUN ONE WINDOW / PARAMETER CONFIGURATION
# ============================================================
def run_single_job(
    demand,
    D,
    window_id,
    start,
    eta,
    kappa_value,
    lambda_1,
):
    start = int(
        start
    )
    end = (
        start
        +
        WINDOW_SIZE
    )
    window_demand = (
        demand[
            start:end
        ]
    )
    if len(
        window_demand
    ) != WINDOW_SIZE:
        raise RuntimeError(
            f"Window {window_id} has "
            f"{len(window_demand)} rows "
            f"instead of {WINDOW_SIZE}."
        )
    kappa_init = (
        normalize_kappa_init(
            kappa_value,
            D,
        )
    )
    result = run_mroo(
        demand=window_demand,
        eta=eta,
        kappa_init=kappa_init,
        lambda_1=lambda_1,
    )
    row = {
        "window_id":
            int(
                window_id
            ),
        "start_index":
            start,
        "end_index":
            end,
        "window_size":
            WINDOW_SIZE,
        "eta":
            float(
                eta
            ),
        "kappa_init":
            kappa_label(
                kappa_init,
                D,
            ),
        "lambda_1":
            float(
                result[
                    "lambda_1"
                ]
            ),
        "lambda_2":
            float(
                result[
                    "lambda_2"
                ]
            ),
        "hitting_cost":
            float(
                result[
                    "hitting_cost"
                ]
            ),
        "long_term_cost":
            float(
                result[
                    "long_term_cost"
                ]
            ),
        "total_cost":
            float(
                result[
                    "total_cost"
                ]
            ),
    }
    for i in range(
        D
    ):
        row[
            f"kappa_init_{i}"
        ] = float(
            kappa_init[
                i
            ]
        )
    return row
# ============================================================
# COMPLETED CONFIG HELPERS
# ============================================================
def save_completed_configs(
    completed_configs,
    D,
):
    rows = []
    for (
        lambda_1,
        eta,
        kappa_components,
    ) in sorted(
        completed_configs
    ):
        kappa_arr = np.asarray(
            kappa_components,
            dtype=float,
        )
        row = {
            "lambda_1":
                float(
                    lambda_1
                ),
            "eta":
                float(
                    eta
                ),
            "kappa_init":
                kappa_label(
                    kappa_arr,
                    D,
                ),
        }
        for i in range(
            D
        ):
            row[
                f"kappa_init_{i}"
            ] = float(
                kappa_arr[
                    i
                ]
            )
        rows.append(
            row
        )
    columns = [
        "lambda_1",
        "eta",
        "kappa_init",
        *kappa_component_columns(
            D
        ),
    ]
    pd.DataFrame(
        rows,
        columns=columns,
    ).to_csv(
        COMPLETED_CONFIG_FILE,
        index=False,
    )
def load_completed_configs(
    D,
):
    # --------------------------------------------------------
    # Normal completion file
    # --------------------------------------------------------
    if COMPLETED_CONFIG_FILE.exists():
        completed_df = pd.read_csv(
            COMPLETED_CONFIG_FILE
        )
        required_base = {
            "lambda_1",
            "eta",
        }
        missing = (
            required_base
            -
            set(
                completed_df.columns
            )
        )
        if missing:
            raise RuntimeError(
                "Completed-config file is missing columns: "
                f"{sorted(missing)}"
            )
        completed_df = (
            ensure_kappa_columns(
                completed_df,
                D,
            )
        )
        component_columns = (
            kappa_component_columns(
                D
            )
        )
        completed = set()
        for _, row in completed_df.iterrows():
            vector = [
                row[
                    column
                ]
                for column
                in component_columns
            ]
            completed.add(
                config_key(
                    row[
                        "lambda_1"
                    ],
                    row[
                        "eta"
                    ],
                    vector,
                    D,
                )
            )
        save_completed_configs(
            completed,
            D,
        )
        return completed
    # --------------------------------------------------------
    # Migration from existing summary if no completion file
    # --------------------------------------------------------
    completed = set()
    if SUMMARY_FILE.exists():
        summary_df = pd.read_csv(
            SUMMARY_FILE
        )
        required_columns = {
            "lambda_1",
            "eta",
            "n_windows",
        }
        if required_columns.issubset(
            summary_df.columns
        ):
            summary_df = (
                ensure_kappa_columns(
                    summary_df,
                    D,
                )
            )
            component_columns = (
                kappa_component_columns(
                    D
                )
            )
            complete_rows = (
                summary_df[
                    summary_df[
                        "n_windows"
                    ].astype(
                        int
                    )
                    ==
                    N_WINDOWS
                ]
            )
            for _, row in complete_rows.iterrows():
                vector = [
                    row[
                        column
                    ]
                    for column
                    in component_columns
                ]
                completed.add(
                    config_key(
                        row[
                            "lambda_1"
                        ],
                        row[
                            "eta"
                        ],
                        vector,
                        D,
                    )
                )
    if completed:
        save_completed_configs(
            completed,
            D,
        )
        print(
            "\nCreated completed-config file from "
            "existing MROO summary:"
        )
        print(
            COMPLETED_CONFIG_FILE
        )
    return completed
def mark_config_complete(
    completed_configs,
    lambda_1,
    eta,
    kappa_value,
    D,
):
    completed_configs.add(
        config_key(
            lambda_1,
            eta,
            kappa_value,
            D,
        )
    )
    save_completed_configs(
        completed_configs,
        D,
    )
# ============================================================
# RESULT / RESUME HELPERS
# ============================================================
def result_identity_columns(
    D,
):
    return [
        "lambda_1",
        "eta",
        *kappa_component_columns(
            D
        ),
        "window_id",
    ]
def normalize_result_dataframe(
    df,
    D,
):
    df = (
        ensure_kappa_columns(
            df,
            D,
        )
    )
    if len(
        df
    ) == 0:
        return df
    return (
        df
        .drop_duplicates(
            subset=
                result_identity_columns(
                    D
                ),
            keep="last",
        )
        .sort_values(
            by=
                result_identity_columns(
                    D
                ),
        )
        .reset_index(
            drop=True
        )
    )
def load_existing_results(
    D,
):
    """
    Existing per-window results are the restart state.
    If the process stops, all rows already written to
    mroo_window_results.csv are skipped next time.
    """
    if not RESULT_FILE.exists():
        return pd.DataFrame()
    existing_results_df = pd.read_csv(
        RESULT_FILE,
        low_memory=False,
    )
    if len(
        existing_results_df
    ) == 0:
        return existing_results_df
    return (
        normalize_result_dataframe(
            existing_results_df,
            D,
        )
    )
def get_completed_window_jobs(
    existing_results_df,
    D,
):
    """
    Returns:
        {
            (config_key, window_id),
            ...
        }
    for every window already stored on disk.
    """
    completed_window_jobs = set()
    if len(
        existing_results_df
    ) == 0:
        return completed_window_jobs
    component_columns = (
        kappa_component_columns(
            D
        )
    )
    for _, row in existing_results_df.iterrows():
        kappa_values = [
            float(
                row[
                    column
                ]
            )
            for column
            in component_columns
        ]
        key = config_key(
            row[
                "lambda_1"
            ],
            row[
                "eta"
            ],
            kappa_values,
            D,
        )
        completed_window_jobs.add(
            (
                key,
                int(
                    row[
                        "window_id"
                    ]
                ),
            )
        )
    return completed_window_jobs
def count_completed_windows_for_config(
    completed_window_jobs,
    key,
):
    return sum(
        1
        for (
            config_id,
            _window_id,
        ) in completed_window_jobs
        if config_id
        ==
        key
    )
# ============================================================
# MAIN
# ============================================================
def main():
    # ========================================================
    # LOAD DATA
    # ========================================================
    demand = load_demand()
    (
        T_FULL,
        demand_D,
    ) = (
        demand.shape
    )
    # ========================================================
    # VALIDATE DIMENSION
    # ========================================================
    if demand_D != D:
        raise ValueError(
            "Demand dimension does not match "
            "the configured dimension. "
            f"Demand dimension = {demand_D}, "
            f"D = {D}."
        )
    print(
        "\n========================================"
    )
    print(
        "PAPER-STYLE MROO WINDOW EXPERIMENT"
    )
    print(
        "========================================"
    )
    print(
        "Full trace length =",
        T_FULL,
    )
    print(
        "Dimension D =",
        D,
    )
    print(
        "A_MATRICES shape =",
        A_MATRICES.shape,
    )
    print(
        "Window size =",
        WINDOW_SIZE,
    )
    print(
        "Number of windows =",
        N_WINDOWS,
    )
    # ========================================================
    # SHARED WINDOWS
    # ========================================================
    window_starts = (
        get_window_starts(
            T_full=T_FULL
        )
    )
    # ========================================================
    # MROO TUNING VALUES
    # ========================================================
    # ETA_VALUES = [
    #     1,
    #     1e-1,
    #     1e-2,
    #     1e-3,
    #     1e-4,
    #     1e-5,
    #     1e-6,
    #     10,
    #     100,
    # ]
    # for beta 300 and m 150
    # ETA_VALUES = [
    #     0.005,
    # ]
    # KAPPA_VALUES = [
    #     [4,6,15],
    #     [5,   7,   17.5],   # pred ≈ 811, beats DMD by 44
    #     [6,   8,   20  ],
    # ]
    eta_json = os.environ.get(
        "MROO_ETA_VALUES_JSON"
    )
    if eta_json:
        ETA_VALUES = [
            float(value)
            for value in json.loads(
                eta_json
            )
        ]
    else:
        ETA_VALUES = [
            1.55e-4,
        ]

    kappa_json = os.environ.get(
        "MROO_KAPPA_VALUES_JSON"
    )
    if kappa_json:
        KAPPA_VALUES = json.loads(
            kappa_json
        )
    else:
        KAPPA_VALUES = [
            [4.0, 6.0, 78.8],
        ]

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================
    ETA_VALUES = list(
        dict.fromkeys(
            float(
                x
            )
            for x in ETA_VALUES
        )
    )
    unique_kappas = []
    seen_kappas = set()
    for value in KAPPA_VALUES:
        key = (
            kappa_tuple(
                value,
                D,
            )
        )
        if (
            key
            not in seen_kappas
        ):
            seen_kappas.add(
                key
            )
            unique_kappas.append(
                value
            )
    KAPPA_VALUES = (
        unique_kappas
    )
    # ========================================================
    # SAVE CONFIGURATIONS REQUESTED IN THIS EXECUTION
    # ========================================================
    current_run_config_rows = []
    for eta in ETA_VALUES:
        for kappa_value in KAPPA_VALUES:
            kappa_arr = (
                normalize_kappa_init(
                    kappa_value,
                    D,
                )
            )
            for lambda_1 in LAMBDA_1_VALUES:
                row = {
                    "lambda_1":
                        float(
                            lambda_1
                        ),
                    "eta":
                        float(
                            eta
                        ),
                    "kappa_init":
                        kappa_label(
                            kappa_arr,
                            D,
                        ),
                }
                for i in range(
                    D
                ):
                    row[
                        f"kappa_init_{i}"
                    ] = float(
                        kappa_arr[
                            i
                        ]
                    )
                current_run_config_rows.append(
                    row
                )
    current_run_config_df = pd.DataFrame(
        current_run_config_rows
    )
    current_run_config_df = (
        current_run_config_df
        .drop_duplicates(
            subset=[
                "lambda_1",
                "eta",
                *kappa_component_columns(
                    D
                ),
            ]
        )
        .reset_index(
            drop=True
        )
    )
    current_run_config_df.to_csv(
        CURRENT_RUN_CONFIG_FILE,
        index=False,
    )
    print(
        "\n========================================"
    )
    print(
        "CURRENT MROO RUN CONFIGURATIONS"
    )
    print(
        "========================================"
    )
    print(
        current_run_config_df.to_string(
            index=False
        )
    )
    print(
        "\nCurrent-run configuration file:"
    )
    print(
        CURRENT_RUN_CONFIG_FILE
    )
    print(
        "\nLambda_1 values:"
    )
    for lambda_1 in LAMBDA_1_VALUES:
        print(
            " ",
            lambda_1,
        )
    print(
        "\nEta values:"
    )
    for eta in ETA_VALUES:
        print(
            " ",
            eta,
        )
    print(
        "\nKappa initial values:"
    )
    for kappa_value in KAPPA_VALUES:
        print(
            " ",
            kappa_label(
                kappa_value,
                D,
            ),
        )
    # ========================================================
    # LOAD COMPLETED CONFIGS + EXISTING WINDOW RESULTS
    # ========================================================
    completed_configs = (
        load_completed_configs(
            D
        )
    )
    existing_results_df = (
        load_existing_results(
            D
        )
    )
    completed_window_jobs = (
        get_completed_window_jobs(
            existing_results_df,
            D,
        )
    )
    # --------------------------------------------------------
    # Completion file cannot be trusted without results.
    # --------------------------------------------------------
    if (
        len(
            completed_configs
        ) > 0
        and
        not RESULT_FILE.exists()
    ):
        print(
            "\nWARNING:"
        )
        print(
            "Completed-config entries exist but "
            "mroo_window_results.csv does not exist."
        )
        print(
            "Ignoring completion entries and rerunning "
            "the requested configurations."
        )
        completed_configs = set()
        save_completed_configs(
            completed_configs,
            D,
        )
    print(
        "\nCompleted parameter combinations =",
        len(
            completed_configs
        ),
    )
    print(
        "Existing completed window jobs =",
        len(
            completed_window_jobs
        ),
    )
    # ========================================================
    # BUILD ONLY MISSING JOBS
    # ========================================================
    jobs = []
    skipped_configs = 0
    skipped_windows = 0
    for eta in ETA_VALUES:
        for kappa_value in KAPPA_VALUES:
            canonical_kappa = (
                kappa_tuple(
                    kappa_value,
                    D,
                )
            )
            for lambda_1 in LAMBDA_1_VALUES:
                key = (
                    config_key(
                        lambda_1,
                        eta,
                        canonical_kappa,
                        D,
                    )
                )
                # ------------------------------------------------
                # Entire configuration already completed
                # ------------------------------------------------
                if (
                    key
                    in completed_configs
                ):
                    skipped_configs += 1
                    print(
                        "\nSkipping completed configuration:"
                    )
                    print(
                        "  lambda_1 =",
                        lambda_1,
                    )
                    print(
                        "  eta      =",
                        eta,
                    )
                    print(
                        "  kappa    =",
                        kappa_label(
                            canonical_kappa,
                            D,
                        ),
                    )
                    continue
                existing_for_config = 0
                missing_for_config = 0
                # ------------------------------------------------
                # Check each of the 100 windows independently.
                # ------------------------------------------------
                for (
                    window_id,
                    start,
                ) in enumerate(
                    window_starts,
                    start=1,
                ):
                    window_job_key = (
                        key,
                        int(
                            window_id
                        ),
                    )
                    # --------------------------------------------
                    # Already saved -> do not rerun
                    # --------------------------------------------
                    if (
                        window_job_key
                        in completed_window_jobs
                    ):
                        skipped_windows += 1
                        existing_for_config += 1
                        continue
                    # --------------------------------------------
                    # Missing -> schedule
                    # --------------------------------------------
                    jobs.append(
                        (
                            int(
                                window_id
                            ),
                            int(
                                start
                            ),
                            float(
                                eta
                            ),
                            canonical_kappa,
                            float(
                                lambda_1
                            ),
                        )
                    )
                    missing_for_config += 1
                if existing_for_config > 0:
                    print(
                        "\nResuming configuration:"
                    )
                    print(
                        "  lambda_1 =",
                        lambda_1,
                    )
                    print(
                        "  eta      =",
                        eta,
                    )
                    print(
                        "  kappa    =",
                        kappa_label(
                            canonical_kappa,
                            D,
                        ),
                    )
                    print(
                        "  existing windows =",
                        existing_for_config,
                    )
                    print(
                        "  missing windows  =",
                        missing_for_config,
                    )
                # ------------------------------------------------
                # This handles a crash where all 100 result rows
                # were saved but completed-config file had not yet
                # been updated.
                # ------------------------------------------------
                if (
                    existing_for_config
                    ==
                    N_WINDOWS
                    and
                    key
                    not in completed_configs
                ):
                    mark_config_complete(
                        completed_configs,
                        lambda_1,
                        eta,
                        canonical_kappa,
                        D,
                    )
                    skipped_configs += 1
                    print(
                        "  -> All windows already exist; "
                        "marked configuration complete."
                    )
    total_runs = len(
        jobs
    )
    worker_count = (
        min(
            MAX_WORKERS,
            total_runs,
        )
        if total_runs > 0
        else 0
    )
    # ========================================================
    # EXPERIMENT SUMMARY
    # ========================================================
    print(
        "\n========================================"
    )
    print(
        "EXPERIMENT SUMMARY"
    )
    print(
        "========================================"
    )
    print(
        "MROO runs needed =",
        total_runs,
    )
    print(
        "Completed configurations skipped =",
        skipped_configs,
    )
    print(
        "Previously completed windows skipped =",
        skipped_windows,
    )
    print(
        "Workers =",
        worker_count,
    )
    print(
        "Shared window file =",
        WINDOW_FILE,
    )
    print(
        "Result file =",
        RESULT_FILE,
    )
    print(
        "Completed-config file =",
        COMPLETED_CONFIG_FILE,
    )
    # ========================================================
    # EXISTING RESULTS WORKING LIST
    # ========================================================
    if len(
        existing_results_df
    ) > 0:
        results = (
            existing_results_df
            .to_dict(
                orient="records"
            )
        )
    else:
        results = []
    # ========================================================
    # RUN ONLY MISSING JOBS
    # ========================================================
    if total_runs > 0:
        with ProcessPoolExecutor(
            max_workers=
                worker_count
        ) as executor:
            future_to_job = {}
            for (
                window_id,
                start,
                eta,
                kappa_value,
                lambda_1,
            ) in jobs:
                future = (
                    executor.submit(
                        run_single_job,
                        demand,
                        D,
                        window_id,
                        start,
                        eta,
                        kappa_value,
                        lambda_1,
                    )
                )
                future_to_job[
                    future
                ] = (
                    window_id,
                    start,
                    eta,
                    kappa_value,
                    lambda_1,
                )
            completed = 0
            for future in as_completed(
                future_to_job
            ):
                (
                    window_id,
                    start,
                    eta,
                    kappa_value,
                    lambda_1,
                ) = (
                    future_to_job[
                        future
                    ]
                )
                try:
                    row = (
                        future.result()
                    )
                except Exception as error:
                    print(
                        "\n========================================"
                    )
                    print(
                        "WINDOW RUN FAILED"
                    )
                    print(
                        "========================================"
                    )
                    print(
                        "window_id =",
                        window_id,
                    )
                    print(
                        "start =",
                        start,
                    )
                    print(
                        "lambda_1 =",
                        lambda_1,
                    )
                    print(
                        "eta =",
                        eta,
                    )
                    print(
                        "kappa_init =",
                        kappa_label(
                            kappa_value,
                            D,
                        ),
                    )
                    print(
                        "error =",
                        error,
                    )
                    raise
                results.append(
                    row
                )
                completed += 1
                print(
                    f"Completed "
                    f"{completed}/{total_runs} | "
                    f"window={window_id} | "
                    f"start={start} | "
                    f"lambda={lambda_1:.6g} | "
                    f"eta={eta:.6g} | "
                    f"kappa="
                    f"{kappa_label(kappa_value, D)} | "
                    f"hit="
                    f"{row['hitting_cost']:.4f} | "
                    f"long="
                    f"{row['long_term_cost']:.4f} | "
                    f"total="
                    f"{row['total_cost']:.4f}",
                    flush=True,
                )
                # ============================================
                # SAVE AFTER EVERY COMPLETED WINDOW
                # ============================================
                current_df = pd.DataFrame(
                    results
                )
                current_df = (
                    normalize_result_dataframe(
                        current_df,
                        D,
                    )
                )
                current_df.to_csv(
                    RESULT_FILE,
                    index=False,
                )
                # ============================================
                # RECORD WINDOW AS COMPLETE IN MEMORY
                # ============================================
                current_key = (
                    config_key(
                        lambda_1,
                        eta,
                        kappa_value,
                        D,
                    )
                )
                completed_window_jobs.add(
                    (
                        current_key,
                        int(
                            window_id
                        ),
                    )
                )
                # ============================================
                # MARK CONFIG COMPLETE IMMEDIATELY ON WINDOW 100
                # ============================================
                completed_for_config = (
                    count_completed_windows_for_config(
                        completed_window_jobs,
                        current_key,
                    )
                )
                if (
                    completed_for_config
                    ==
                    N_WINDOWS
                    and
                    current_key
                    not in completed_configs
                ):
                    mark_config_complete(
                        completed_configs,
                        lambda_1,
                        eta,
                        kappa_value,
                        D,
                    )
                    print(
                        "\nMarked configuration complete immediately:"
                    )
                    print(
                        "  lambda_1 =",
                        lambda_1,
                    )
                    print(
                        "  eta      =",
                        eta,
                    )
                    print(
                        "  kappa    =",
                        kappa_label(
                            kappa_value,
                            D,
                        ),
                    )
    else:
        print(
            "\nNo new MROO runs are needed."
        )
    # ========================================================
    # FINAL RESULTS
    # ========================================================
    results_df = pd.DataFrame(
        results
    )
    if len(
        results_df
    ) == 0:
        raise RuntimeError(
            "No MROO result rows are available."
        )
    results_df = (
        normalize_result_dataframe(
            results_df,
            D,
        )
    )
    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )
    # ========================================================
    # FINAL COMPLETION CHECK
    #
    # Safety check only. Normally completed configs are now
    # written immediately during execution.
    # ========================================================
    component_columns = (
        kappa_component_columns(
            D
        )
    )
    for eta in ETA_VALUES:
        for kappa_value in KAPPA_VALUES:
            kappa_arr = (
                normalize_kappa_init(
                    kappa_value,
                    D,
                )
            )
            for lambda_1 in LAMBDA_1_VALUES:
                mask = np.isclose(
                    results_df[
                        "lambda_1"
                    ].astype(
                        float
                    ),
                    float(
                        lambda_1
                    ),
                )
                mask &= np.isclose(
                    results_df[
                        "eta"
                    ].astype(
                        float
                    ),
                    float(
                        eta
                    ),
                )
                for (
                    i,
                    column,
                ) in enumerate(
                    component_columns
                ):
                    mask &= np.isclose(
                        results_df[
                            column
                        ].astype(
                            float
                        ),
                        float(
                            kappa_arr[
                                i
                            ]
                        ),
                    )
                config_count = int(
                    results_df.loc[
                        mask,
                        "window_id",
                    ]
                    .nunique()
                )
                if (
                    config_count
                    ==
                    N_WINDOWS
                ):
                    key = config_key(
                        lambda_1,
                        eta,
                        kappa_arr,
                        D,
                    )
                    if (
                        key
                        not in completed_configs
                    ):
                        mark_config_complete(
                            completed_configs,
                            lambda_1,
                            eta,
                            kappa_arr,
                            D,
                        )
                        print(
                            "\nMarked configuration complete:"
                        )
                        print(
                            "  lambda_1 =",
                            lambda_1,
                        )
                        print(
                            "  eta      =",
                            eta,
                        )
                        print(
                            "  kappa    =",
                            kappa_label(
                                kappa_arr,
                                D,
                            ),
                        )
                elif (
                    config_count
                    >
                    N_WINDOWS
                ):
                    raise RuntimeError(
                        "Too many distinct windows found for "
                        "MROO configuration: "
                        f"lambda_1={lambda_1}, "
                        f"eta={eta}, "
                        f"kappa_init="
                        f"{kappa_label(kappa_arr, D)}. "
                        f"Expected {N_WINDOWS}, "
                        f"found {config_count}."
                    )
    # ========================================================
    # SUMMARY ACROSS WINDOWS
    # ========================================================
    group_columns = [
        "lambda_1",
        "eta",
        *component_columns,
    ]
    summary_df = (
        results_df
        .groupby(
            group_columns,
            as_index=False,
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
            median_total_cost=(
                "total_cost",
                "median",
            ),
            min_total_cost=(
                "total_cost",
                "min",
            ),
            max_total_cost=(
                "total_cost",
                "max",
            ),
        )
    )
    # --------------------------------------------------------
    # Human-readable kappa label
    # --------------------------------------------------------
    summary_df.insert(
        2,
        "kappa_init",
        [
            kappa_label(
                [
                    row[
                        column
                    ]
                    for column
                    in component_columns
                ],
                D,
            )
            for _, row
            in summary_df.iterrows()
        ],
    )
    summary_df = (
        summary_df
        .sort_values(
            by=[
                "mean_total_cost",
            ]
        )
        .reset_index(
            drop=True
        )
    )
    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )
    # ========================================================
    # PRINT RESULTS
    # ========================================================
    print(
        "\n========================================"
    )
    print(
        "WINDOW EXPERIMENT COMPLETE"
    )
    print(
        "========================================"
    )
    print(
        "\nSummary across windows:\n"
    )
    print(
        summary_df.to_string(
            index=False
        )
    )
    print(
        "\n========================================"
    )
    print(
        "FILES SAVED"
    )
    print(
        "========================================"
    )
    print(
        "\nWindow definitions:"
    )
    print(
        WINDOW_FILE
    )
    print(
        "\nPer-window MROO results:"
    )
    print(
        RESULT_FILE
    )
    print(
        "\nSummary:"
    )
    print(
        SUMMARY_FILE
    )
    print(
        "\nCompleted parameter combinations:"
    )
    print(
        COMPLETED_CONFIG_FILE
    )
    print(
        "\nConfigurations requested in this run:"
    )
    print(
        CURRENT_RUN_CONFIG_FILE
    )
# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
