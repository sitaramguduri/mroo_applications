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
# IMPORTS
# ============================================================

from greedy import run_greedy

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
# main_sweep.py supplies MROO_RESULT_DIR separately for each
# beta / m setting.
#
# Greedy must write directly into that directory.
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
    "Greedy result directory:",
    RESULT_DIR,
)


# ============================================================
# WINDOW CONFIGURATION
#
# 1-minute slots:
#
# 24 hours * 60 = 1440
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
#
# Same window definitions used by all algorithms.
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
    / "greedy_window_results.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "greedy_window_summary.csv"
)


# ============================================================
# PARALLEL CONFIGURATION
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
# CREATE / LOAD SHARED WINDOWS
# ============================================================

def get_window_starts(
    T_full,
):

    # --------------------------------------------------------
    # Reuse existing shared windows
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
                "Shared window file is missing columns: "
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

        if len(
            starts
        ) != N_WINDOWS:

            raise RuntimeError(
                f"Expected {N_WINDOWS} windows, "
                f"but shared file contains "
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


    # --------------------------------------------------------
    # Validate requested window configuration
    # --------------------------------------------------------

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
            f"but only {total_valid_windows} valid "
            "starts exist."
        )


    # --------------------------------------------------------
    # Exact non-overlapping windows first
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Remaining random continuous windows
    # --------------------------------------------------------

    all_valid_starts = np.arange(
        0,
        T_full
        - WINDOW_SIZE
        + 1,
        dtype=int,
    )

    candidate_starts = np.setdiff1d(
        all_valid_starts,
        exact_starts,
        assume_unique=False,
    )

    remaining_needed = (
        N_WINDOWS
        - len(
            selected
        )
    )

    rng = np.random.default_rng(
        RANDOM_SEED
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


    # --------------------------------------------------------
    # Save shared definitions
    # --------------------------------------------------------

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
# RUN ONE GREEDY WINDOW
# ============================================================

def run_single_job(
    demand,
    window_id,
    start,
):

    start = int(
        start
    )

    end = (
        start
        + WINDOW_SIZE
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
            f"Window {window_id} contains "
            f"{len(window_demand)} rows instead of "
            f"{WINDOW_SIZE}."
        )


    # --------------------------------------------------------
    # Run Greedy
    #
    # greedy.py now evaluates memory as:
    #
    # d_{t,i}
    # =
    # beta / 2
    # *
    # ||A_i(u_t-u_{t-1})||_2^2
    #
    # Greedy's action itself still minimizes hitting cost only.
    # --------------------------------------------------------

    result = run_greedy(
        demand=window_demand
    )


    # --------------------------------------------------------
    # Comparable result row
    # --------------------------------------------------------

    return {

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


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD FULL TRACE
    # ========================================================

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
    # VALIDATE MATRIX-BASED MEMORY CONFIGURATION
    #
    # Exactly D matrices are required:
    #
    #     A_1, ..., A_D
    #
    # and every A_i is D x D.
    #
    # Therefore:
    #
    #     A_MATRICES.shape = (D, D, D)
    # ========================================================

    a_matrices = np.asarray(
        A_MATRICES,
        dtype=float,
    )

    if a_matrices.shape != (
        D,
        D,
        D,
    ):

        raise ValueError(
            "A_MATRICES must have shape "
            f"({D}, {D}, {D}), "
            f"but got {a_matrices.shape}."
        )


    print(
        "\n========================================"
    )

    print(
        "GREEDY MATRIX-MEMORY WINDOW EXPERIMENT"
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
        a_matrices.shape,
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
    # SAME WINDOWS AS ALL OTHER ALGORITHMS
    # ========================================================

    window_starts = (
        get_window_starts(
            T_full=T_FULL
        )
    )


    # ========================================================
    # BUILD JOBS
    # ========================================================

    jobs = []

    for (
        window_id,
        start,
    ) in enumerate(
        window_starts,
        start=1,
    ):

        jobs.append(
            (
                int(
                    window_id
                ),
                int(
                    start
                ),
            )
        )


    total_runs = len(
        jobs
    )

    worker_count = min(
        MAX_WORKERS,
        total_runs,
    )


    print(
        "\n========================================"
    )

    print(
        "GREEDY CONFIGURATION"
    )

    print(
        "========================================"
    )

    print(
        "Workers =",
        worker_count,
    )

    print(
        "Total runs =",
        total_runs,
    )

    print(
        "Shared window file =",
        WINDOW_FILE,
    )

    print(
        "Result file =",
        RESULT_FILE,
    )


    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    results = []


    with ProcessPoolExecutor(
        max_workers=
            worker_count
    ) as executor:

        future_to_job = {}


        for (
            window_id,
            start,
        ) in jobs:

            future = executor.submit(
                run_single_job,
                demand,
                window_id,
                start,
            )

            future_to_job[
                future
            ] = (
                window_id,
                start,
            )


        completed = 0


        for future in as_completed(
            future_to_job
        ):

            (
                window_id,
                start,
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
                    "GREEDY WINDOW FAILED"
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
                f"hit={row['hitting_cost']:.4f} | "
                f"long={row['long_term_cost']:.4f} | "
                f"total={row['total_cost']:.4f}",
                flush=True,
            )


            # ------------------------------------------------
            # Save partial progress
            # ------------------------------------------------

            partial_df = pd.DataFrame(
                results
            )

            partial_df = (
                partial_df
                .sort_values(
                    "window_id"
                )
                .reset_index(
                    drop=True
                )
            )

            partial_df.to_csv(
                RESULT_FILE,
                index=False,
            )


    # ========================================================
    # FINAL RESULT FILE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df

        .sort_values(
            by=[
                "window_id",
            ]
        )

        .reset_index(
            drop=True
        )
    )


    if len(
        results_df
    ) != N_WINDOWS:

        raise RuntimeError(
            f"Expected {N_WINDOWS} Greedy windows, "
            f"found {len(results_df)}."
        )


    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )


    # ========================================================
    # SUMMARY ACROSS WINDOWS
    # ========================================================

    summary_df = pd.DataFrame(
        {

            "n_windows": [
                len(
                    results_df
                )
            ],


            "mean_hitting_cost": [
                results_df[
                    "hitting_cost"
                ].mean()
            ],


            "std_hitting_cost": [
                results_df[
                    "hitting_cost"
                ].std()
            ],


            "mean_long_term_cost": [
                results_df[
                    "long_term_cost"
                ].mean()
            ],


            "std_long_term_cost": [
                results_df[
                    "long_term_cost"
                ].std()
            ],


            "mean_total_cost": [
                results_df[
                    "total_cost"
                ].mean()
            ],


            "std_total_cost": [
                results_df[
                    "total_cost"
                ].std()
            ],


            "median_total_cost": [
                results_df[
                    "total_cost"
                ].median()
            ],


            "min_total_cost": [
                results_df[
                    "total_cost"
                ].min()
            ],


            "max_total_cost": [
                results_df[
                    "total_cost"
                ].max()
            ],
        }
    )


    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "GREEDY WINDOW EXPERIMENT COMPLETE"
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
        "\nShared window definitions:"
    )

    print(
        WINDOW_FILE
    )


    print(
        "\nPer-window Greedy results:"
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()