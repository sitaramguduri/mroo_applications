import os
import sys

from pathlib import Path

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_DIR = (
    ALGORITHM_DIR.parent
)

if str(PROJECT_DIR) not in sys.path:

    sys.path.append(
        str(PROJECT_DIR)
    )


# ============================================================
# IMPORT DMD
# ============================================================

from dmd import run_dmd

from config import (
    A_MATRICES,
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
#
# main_sweep.py passes MROO_RESULT_DIR separately for every
# beta/m setting.
#
# DMD must write DIRECTLY into that directory.
#
# Do NOT append another "matrix_memory" folder here.
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
    "DMD result directory:",
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

ALL_RESULT_FILE = (
    RESULT_DIR
    / "dmd_all_window_results.csv"
)

ALL_SUMMARY_FILE = (
    RESULT_DIR
    / "dmd_all_window_summary.csv"
)

ALL_KAPPA_FILE = (
    RESULT_DIR
    / "dmd_all_window_kappa.csv"
)


# Best configuration only.
# These filenames are what main_sweep.py expects.

RESULT_FILE = (
    RESULT_DIR
    / "dmd_window_results.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "dmd_window_summary.csv"
)

COMBINED_KAPPA_FILE = (
    RESULT_DIR
    / "dmd_window_kappa.csv"
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
# LOAD DEMAND
# ============================================================

def load_demand():

    data = pd.read_csv(
        DATA_FILE
    )

    demand = (
        data[
            DEMAND_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    return demand


# ============================================================
# CREATE / LOAD SHARED WINDOW STARTS
# ============================================================

def get_window_starts(
    T_full,
):

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
                "Existing shared window file is missing "
                f"columns: {sorted(missing)}"
            )

        starts = (
            window_df[
                "start_index"
            ]
            .to_numpy(
                dtype=int
            )
        )

        if len(
            starts
        ) != N_WINDOWS:

            raise RuntimeError(
                f"Expected {N_WINDOWS} windows, "
                f"but shared window file contains "
                f"{len(starts)}."
            )

        if np.any(
            starts
            + WINDOW_SIZE
            > T_full
        ):

            raise RuntimeError(
                "Shared window file contains a window "
                "outside the demand trace."
            )

        print(
            "\nLoaded existing shared windows from:"
        )

        print(
            WINDOW_FILE
        )

        return starts


    if WINDOW_SIZE > T_full:

        raise ValueError(
            f"WINDOW_SIZE={WINDOW_SIZE} exceeds "
            f"trace length={T_full}."
        )


    total_valid_windows = (
        T_full
        - WINDOW_SIZE
        + 1
    )


    if N_WINDOWS > total_valid_windows:

        raise ValueError(
            f"Requested {N_WINDOWS} windows, "
            f"but only {total_valid_windows} "
            "valid window starts exist."
        )


    number_of_full_windows = (
        T_full
        // WINDOW_SIZE
    )


    exact_starts = (
        np.arange(
            number_of_full_windows,
            dtype=int,
        )
        * WINDOW_SIZE
    )


    exact_starts = (
        exact_starts[
            :min(
                len(
                    exact_starts
                ),
                N_WINDOWS,
            )
        ]
    )


    selected = list(
        exact_starts.astype(
            int
        )
    )


    all_valid_starts = np.arange(
        0,
        T_full
        - WINDOW_SIZE
        + 1,
        dtype=int,
    )


    candidate_starts = (
        np.setdiff1d(
            all_valid_starts,
            exact_starts,
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

        sampled_starts = (
            rng.choice(
                candidate_starts,
                size=remaining_needed,
                replace=False,
            )
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
        ["exact"]
        * len(
            exact_starts
        )
        +
        ["sampled"]
        * remaining_needed
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
                + WINDOW_SIZE,

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
        "\nCreated shared windows:"
    )

    print(
        WINDOW_FILE
    )

    print(
        "Exact windows =",
        len(
            exact_starts
        ),
    )

    print(
        "Sampled windows =",
        remaining_needed,
    )


    return starts


# ============================================================
# RUN ONE WINDOW
# ============================================================

def run_single_job(
    demand,
    D,
    window_id,
    start,
    eta,
    kappa_scalar,
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
            f"{len(window_demand)} rows instead of "
            f"{WINDOW_SIZE}."
        )


    kappa_init = np.full(
        D,
        kappa_scalar,
        dtype=float,
    )


    result = run_dmd(
        demand=window_demand,
        eta=eta,
        kappa_init=kappa_init,
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
            float(
                kappa_scalar
            ),

        "lambda_1":
            np.nan,

        "lambda_2":
            np.nan,

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


    kappa_history = np.asarray(
        result[
            "kappa_history"
        ],
        dtype=float,
    )


    kappa_df = pd.DataFrame(
        kappa_history,
        columns=[
            f"kappa_{i}"
            for i in range(
                D
            )
        ],
    )


    kappa_df.insert(
        0,
        "t",
        np.arange(
            len(
                kappa_df
            )
        ),
    )


    kappa_df.insert(
        0,
        "global_t",
        start
        +
        np.arange(
            len(
                kappa_df
            )
        ),
    )


    kappa_df.insert(
        0,
        "window_id",
        int(
            window_id
        ),
    )


    kappa_df.insert(
        1,
        "start_index",
        start,
    )


    kappa_df.insert(
        2,
        "eta",
        float(
            eta
        ),
    )


    kappa_df.insert(
        3,
        "kappa_init",
        float(
            kappa_scalar
        ),
    )


    return (
        row,
        kappa_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    demand = (
        load_demand()
    )


    (
        T_FULL,
        D,
    ) = (
        demand.shape
    )


    # ========================================================
    # VALIDATE MEMORY MATRICES
    # ========================================================

    A_matrices = np.asarray(
        A_MATRICES,
        dtype=float,
    )


    if A_matrices.shape != (
        D,
        D,
        D,
    ):

        raise ValueError(
            "A_MATRICES must have shape "
            f"({D}, {D}, {D}), "
            f"but got {A_matrices.shape}."
        )


    print(
        "\n========================================"
    )

    print(
        "DMD WINDOW EXPERIMENT"
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
        A_matrices.shape,
    )

    print(
        "Window size =",
        WINDOW_SIZE,
    )

    print(
        "Number of windows =",
        N_WINDOWS,
    )


    window_starts = (
        get_window_starts(
            T_full=T_FULL
        )
    )


    # ========================================================
    # DMD PARAMETERS
    # ========================================================

    def parse_float_list_env(
        variable_name,
        default_values,
    ):

        raw_value = (
            os.environ.get(
                variable_name
            )
        )


        if raw_value is None:

            return [
                float(
                    value
                )
                for value
                in default_values
            ]


        raw_value = (
            raw_value.strip()
        )


        if not raw_value:

            return [
                float(
                    value
                )
                for value
                in default_values
            ]


        values = [

            float(
                value.strip()
            )

            for value
            in raw_value.split(
                ","
            )

            if value.strip()
        ]


        if len(
            values
        ) == 0:

            raise ValueError(
                f"{variable_name} was provided but "
                "contained no valid values."
            )


        return list(
            dict.fromkeys(
                values
            )
        )


    ETA_VALUES = (
        parse_float_list_env(
            "DMD_ETA_VALUES",
            [
                WINDOW_SIZE
                ** (
                    -1.0
                    / 3.0
                )
            ],
        )
    )


    KAPPA_VALUES = (
        parse_float_list_env(
            "DMD_KAPPA_VALUES",
            [
                1.0
                / WINDOW_SIZE
            ],
        )
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

    for kappa_scalar in KAPPA_VALUES:

        print(
            " ",
            kappa_scalar,
        )


    # ========================================================
    # LOAD PREVIOUS RESULTS
    # ========================================================

    required_result_columns = [

        "window_id",

        "start_index",

        "end_index",

        "window_size",

        "eta",

        "kappa_init",

        "hitting_cost",

        "long_term_cost",

        "total_cost",
    ]


    if ALL_RESULT_FILE.exists():

        existing_results_df = (
            pd.read_csv(
                ALL_RESULT_FILE
            )
        )


    elif RESULT_FILE.exists():

        existing_results_df = (
            pd.read_csv(
                RESULT_FILE
            )
        )


        if len(
            existing_results_df
        ) > 0:

            print(
                "\nMigrating existing DMD results into:"
            )

            print(
                ALL_RESULT_FILE
            )


            existing_results_df.to_csv(
                ALL_RESULT_FILE,
                index=False,
            )


    else:

        existing_results_df = (
            pd.DataFrame(
                columns=
                    required_result_columns
            )
        )


    if len(
        existing_results_df
    ) > 0:

        missing_columns = [

            column

            for column
            in required_result_columns

            if column
            not in existing_results_df.columns
        ]


        if missing_columns:

            raise RuntimeError(
                "Existing DMD all-results file is missing "
                f"columns: {missing_columns}"
            )


        existing_results_df = (

            existing_results_df

            .drop_duplicates(
                subset=[
                    "eta",
                    "kappa_init",
                    "window_id",
                ],
                keep="last",
            )

            .sort_values(
                by=[
                    "eta",
                    "kappa_init",
                    "window_id",
                ]
            )

            .reset_index(
                drop=True
            )
        )


    # ========================================================
    # BUILD ONLY MISSING JOBS
    # ========================================================

    jobs = []

    skipped_configs = 0

    skipped_windows = 0


    for eta in ETA_VALUES:

        eta = float(
            eta
        )


        for kappa_scalar in KAPPA_VALUES:

            kappa_scalar = float(
                kappa_scalar
            )


            if len(
                existing_results_df
            ) > 0:

                config_mask = (

                    np.isclose(
                        existing_results_df[
                            "eta"
                        ].astype(
                            float
                        ),
                        eta,
                        rtol=1e-12,
                        atol=1e-15,
                    )

                    &

                    np.isclose(
                        existing_results_df[
                            "kappa_init"
                        ].astype(
                            float
                        ),
                        kappa_scalar,
                        rtol=1e-12,
                        atol=1e-15,
                    )
                )


                config_rows = (
                    existing_results_df[
                        config_mask
                    ]
                    .copy()
                )


                completed_window_ids = set(

                    config_rows[
                        "window_id"
                    ]

                    .astype(
                        int
                    )

                    .tolist()
                )


            else:

                completed_window_ids = (
                    set()
                )


            if len(
                completed_window_ids
            ) >= N_WINDOWS:

                skipped_configs += 1

                print(
                    "\nSkipping completed DMD configuration:"
                )

                print(
                    "  eta   =",
                    eta,
                )

                print(
                    "  kappa =",
                    kappa_scalar,
                )

                continue


            missing_for_config = 0


            for (
                window_id,
                start,
            ) in enumerate(
                window_starts,
                start=1,
            ):

                if (
                    int(
                        window_id
                    )
                    in completed_window_ids
                ):

                    skipped_windows += 1

                    continue


                jobs.append(
                    (
                        int(
                            window_id
                        ),

                        int(
                            start
                        ),

                        eta,

                        kappa_scalar,
                    )
                )


                missing_for_config += 1


            if (
                len(
                    completed_window_ids
                ) > 0

                and

                missing_for_config > 0
            ):

                print(
                    "\nResuming partial DMD configuration:"
                )

                print(
                    "  eta =",
                    eta,
                )

                print(
                    "  kappa =",
                    kappa_scalar,
                )

                print(
                    "  completed windows =",
                    len(
                        completed_window_ids
                    ),
                )

                print(
                    "  missing windows =",
                    missing_for_config,
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
        "DMD runs needed =",
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
        "All-result file =",
        ALL_RESULT_FILE,
    )


    # ========================================================
    # WORKING RESULT TABLE
    # ========================================================

    if len(
        existing_results_df
    ) > 0:

        working_results_df = (
            existing_results_df.copy()
        )


    else:

        working_results_df = (
            pd.DataFrame(
                columns=
                    required_result_columns
            )
        )


    new_kappa_results = []


    # ========================================================
    # RUN MISSING JOBS
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
                kappa_scalar,
            ) in jobs:

                future = (
                    executor.submit(
                        run_single_job,
                        demand,
                        D,
                        window_id,
                        start,
                        eta,
                        kappa_scalar,
                    )
                )


                future_to_job[
                    future
                ] = (
                    window_id,
                    start,
                    eta,
                    kappa_scalar,
                )


            completed = 0


            for future in as_completed(
                future_to_job
            ):

                (
                    window_id,
                    start,
                    eta,
                    kappa_scalar,
                ) = (
                    future_to_job[
                        future
                    ]
                )


                try:

                    (
                        row,
                        kappa_df,
                    ) = (
                        future.result()
                    )


                except Exception as error:

                    print(
                        "\n========================================"
                    )

                    print(
                        "DMD WINDOW RUN FAILED"
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
                        "eta =",
                        eta,
                    )

                    print(
                        "kappa_init =",
                        kappa_scalar,
                    )

                    print(
                        "error =",
                        error,
                    )

                    raise


                row_df = (
                    pd.DataFrame(
                        [
                            row
                        ]
                    )
                )


                working_results_df = (
                    pd.concat(
                        [
                            working_results_df,
                            row_df,
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                )


                working_results_df = (

                    working_results_df

                    .drop_duplicates(
                        subset=[
                            "eta",
                            "kappa_init",
                            "window_id",
                        ],
                        keep="last",
                    )

                    .sort_values(
                        by=[
                            "eta",
                            "kappa_init",
                            "window_id",
                        ]
                    )

                    .reset_index(
                        drop=True
                    )
                )


                working_results_df.to_csv(
                    ALL_RESULT_FILE,
                    index=False,
                )


                new_kappa_results.append(
                    kappa_df
                )


                completed += 1


                print(
                    f"Completed "
                    f"{completed}/{total_runs} | "
                    f"window={window_id} | "
                    f"start={start} | "
                    f"eta={eta:.6g} | "
                    f"kappa={kappa_scalar:.6g} | "
                    f"hit={row['hitting_cost']:.4f} | "
                    f"long={row['long_term_cost']:.4f} | "
                    f"total={row['total_cost']:.4f}",
                    flush=True,
                )


    else:

        print(
            "\nAll requested DMD configurations are "
            "already complete. No DMD windows were rerun."
        )


    # ========================================================
    # CANONICAL RESULT TABLE
    # ========================================================

    if ALL_RESULT_FILE.exists():

        all_results_df = (
            pd.read_csv(
                ALL_RESULT_FILE
            )
        )


    else:

        all_results_df = (
            working_results_df.copy()
        )


    if len(
        all_results_df
    ) == 0:

        raise RuntimeError(
            "No DMD window results are available."
        )


    all_results_df = (

        all_results_df

        .drop_duplicates(
            subset=[
                "eta",
                "kappa_init",
                "window_id",
            ],
            keep="last",
        )

        .sort_values(
            by=[
                "eta",
                "kappa_init",
                "window_id",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    all_results_df.to_csv(
        ALL_RESULT_FILE,
        index=False,
    )


    # ========================================================
    # KAPPA TRAJECTORIES
    # ========================================================

    if ALL_KAPPA_FILE.exists():

        old_kappa_df = pd.read_csv(
            ALL_KAPPA_FILE
        )


    else:

        old_kappa_df = (
            pd.DataFrame()
        )


    if len(
        new_kappa_results
    ) > 0:

        current_kappa_df = (
            pd.concat(
                new_kappa_results,
                ignore_index=True,
            )
        )


        if len(
            old_kappa_df
        ) > 0:

            all_kappa_df = (
                pd.concat(
                    [
                        old_kappa_df,
                        current_kappa_df,
                    ],
                    ignore_index=True,
                    sort=False,
                )
            )


        else:

            all_kappa_df = (
                current_kappa_df.copy()
            )


        all_kappa_df = (

            all_kappa_df

            .drop_duplicates(
                subset=[
                    "eta",
                    "kappa_init",
                    "window_id",
                    "t",
                ],
                keep="last",
            )

            .sort_values(
                by=[
                    "eta",
                    "kappa_init",
                    "window_id",
                    "t",
                ]
            )

            .reset_index(
                drop=True
            )
        )


        all_kappa_df.to_csv(
            ALL_KAPPA_FILE,
            index=False,
        )


    else:

        all_kappa_df = (
            old_kappa_df.copy()
        )


    # ========================================================
    # SUMMARY FOR ALL STORED CONFIGURATIONS
    # ========================================================

    all_summary_df = (

        all_results_df

        .groupby(
            [
                "eta",
                "kappa_init",
            ],
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


    complete_summary_df = (

        all_summary_df[

            all_summary_df[
                "n_windows"
            ]
            ==
            N_WINDOWS
        ]

        .copy()

        .sort_values(
            by=
                "mean_total_cost"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        complete_summary_df
    ) == 0:

        raise RuntimeError(
            "No complete DMD configuration has "
            f"{N_WINDOWS} windows yet."
        )


    complete_summary_df.to_csv(
        ALL_SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best_config = (
        complete_summary_df.iloc[
            0
        ]
    )


    best_eta = float(
        best_config[
            "eta"
        ]
    )


    best_kappa = float(
        best_config[
            "kappa_init"
        ]
    )


    best_result_mask = (

        np.isclose(
            all_results_df[
                "eta"
            ].astype(
                float
            ),
            best_eta,
            rtol=1e-12,
            atol=1e-15,
        )

        &

        np.isclose(
            all_results_df[
                "kappa_init"
            ].astype(
                float
            ),
            best_kappa,
            rtol=1e-12,
            atol=1e-15,
        )
    )


    best_results_df = (

        all_results_df[
            best_result_mask
        ]

        .copy()

        .sort_values(
            by=
                "window_id"
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        best_results_df
    ) != N_WINDOWS:

        raise RuntimeError(
            "Best DMD configuration does not have "
            f"{N_WINDOWS} windows. "
            f"Found {len(best_results_df)}."
        )


    best_results_df.to_csv(
        RESULT_FILE,
        index=False,
    )


    best_summary_df = (

        complete_summary_df

        .iloc[
            [
                0
            ]
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    best_summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # BEST KAPPA TRAJECTORY
    # ========================================================

    if len(
        all_kappa_df
    ) > 0:

        best_kappa_mask = (

            np.isclose(
                all_kappa_df[
                    "eta"
                ].astype(
                    float
                ),
                best_eta,
                rtol=1e-12,
                atol=1e-15,
            )

            &

            np.isclose(
                all_kappa_df[
                    "kappa_init"
                ].astype(
                    float
                ),
                best_kappa,
                rtol=1e-12,
                atol=1e-15,
            )
        )


        best_kappa_df = (

            all_kappa_df[
                best_kappa_mask
            ]

            .copy()

            .sort_values(
                by=[
                    "window_id",
                    "t",
                ]
            )

            .reset_index(
                drop=True
            )
        )


        if len(
            best_kappa_df
        ) > 0:

            best_kappa_df.to_csv(
                COMBINED_KAPPA_FILE,
                index=False,
            )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "DMD WINDOW EXPERIMENT COMPLETE"
    )

    print(
        "========================================"
    )


    print(
        "\nAll complete DMD configurations:\n"
    )

    print(
        complete_summary_df.to_string(
            index=False
        )
    )


    print(
        "\nBest DMD configuration:"
    )

    print(
        "  eta   =",
        best_eta,
    )

    print(
        "  kappa =",
        best_kappa,
    )

    print(
        "  mean total cost =",
        float(
            best_config[
                "mean_total_cost"
            ]
        ),
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
        "\nAll DMD window results:"
    )

    print(
        ALL_RESULT_FILE
    )


    print(
        "\nAll DMD summaries:"
    )

    print(
        ALL_SUMMARY_FILE
    )


    print(
        "\nBest DMD window results:"
    )

    print(
        RESULT_FILE
    )


    print(
        "\nBest DMD summary:"
    )

    print(
        SUMMARY_FILE
    )


    print(
        "\nAll DMD kappa trajectories:"
    )

    print(
        ALL_KAPPA_FILE
    )


    print(
        "\nBest DMD kappa trajectory:"
    )

    print(
        COMBINED_KAPPA_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()