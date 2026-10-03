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
# IMPORT S-MROO-SUM + CONFIG
# ============================================================

from smroo_sum import run_smroo_sum

from config import (
    M,
    BETA,
    H,
    RHO,
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
# main_sweep.py sets MROO_RESULT_DIR independently for every
# beta/m setting.
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
    "S-MROO-SUM result directory:",
    RESULT_DIR,
)


# ============================================================
# WINDOW CONFIGURATION
#
# Same 100-window configuration as MROO and S-MROO-MAX.
# ============================================================

WINDOW_SIZE = int(
    os.environ.get(
        "MROO_WINDOW_SIZE",
        "8640",
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
# EXACTLY the same file used by MROO and S-MROO-MAX.
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
    / "smroo_sum_window_results.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "smroo_sum_window_summary.csv"
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
# S-MROO-SUM PARAMETER PAIRS
#
# Keep both theoretical parameter pairs for now.
#
# main_sweep.py can later select the pair with the lowest mean
# total cost across the same 100 windows.
# ============================================================

PARAMETER_PAIRS = [
    1,
    2,
]


# ============================================================
# LOAD DEMAND
# ============================================================

def load_demand():

    data = pd.read_csv(
        DATA_FILE
    )

    demand = data[
        DEMAND_COLUMNS
    ].to_numpy(
        dtype=float
    )

    return demand


# ============================================================
# CREATE / LOAD PAPER-STYLE WINDOW STARTS
#
# Same logic as the updated MROO/S-MROO-MAX runners:
#
#   7 exact daily windows
#   + remaining randomly sampled 24-hour windows
#
# There is NO artificial one-hour minimum separation.
# ============================================================

def get_window_starts(
    T_full,
):

    # --------------------------------------------------------
    # Reuse existing shared window definitions
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
            - set(window_df.columns)
        )

        if missing:

            raise RuntimeError(
                "Existing shared window file is missing "
                f"columns: {sorted(missing)}"
            )

        starts = window_df[
            "start_index"
        ].to_numpy(
            dtype=int
        )

        if len(starts) != N_WINDOWS:

            raise RuntimeError(
                f"Expected {N_WINDOWS} windows, "
                f"but shared window file contains "
                f"{len(starts)}."
            )

        if np.any(
            starts + WINDOW_SIZE > T_full
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
            f"window starts exist."
        )


    # --------------------------------------------------------
    # Exact 24-hour day windows
    # --------------------------------------------------------

    number_of_full_days = (
        T_full
        // WINDOW_SIZE
    )

    exact_day_starts = (
        np.arange(
            number_of_full_days,
            dtype=int,
        )
        * WINDOW_SIZE
    )

    exact_day_starts = (
        exact_day_starts[
            :min(
                len(exact_day_starts),
                N_WINDOWS,
            )
        ]
    )

    selected = list(
        exact_day_starts.astype(int)
    )


    # --------------------------------------------------------
    # All remaining valid continuous 24-hour starts
    # --------------------------------------------------------

    all_valid_starts = np.arange(
        0,
        T_full - WINDOW_SIZE + 1,
        dtype=int,
    )

    candidate_starts = np.setdiff1d(
        all_valid_starts,
        exact_day_starts,
        assume_unique=False,
    )


    # --------------------------------------------------------
    # Reproducible random sampling
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    remaining_needed = (
        N_WINDOWS
        - len(selected)
    )

    if remaining_needed > 0:

        sampled_starts = rng.choice(
            candidate_starts,
            size=remaining_needed,
            replace=False,
        )

        selected.extend(
            sampled_starts.astype(int)
        )


    starts = np.asarray(
        selected,
        dtype=int,
    )


    # --------------------------------------------------------
    # Save shared window file
    # --------------------------------------------------------

    window_types = (
        ["exact_day"]
        * len(exact_day_starts)
        +
        ["sampled"]
        * (
            N_WINDOWS
            - len(exact_day_starts)
        )
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
                starts + WINDOW_SIZE,

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
        len(exact_day_starts),
    )

    print(
        "Sampled windows =",
        remaining_needed,
    )

    return starts


# ============================================================
# RUN ONE WINDOW / PARAMETER PAIR
# ============================================================

def run_single_job(
    demand,
    parameter_pair,
    window_id,
    start,
):

    start = int(start)

    end = (
        start
        + WINDOW_SIZE
    )

    window_demand = demand[
        start:end
    ]

    if len(window_demand) != WINDOW_SIZE:

        raise RuntimeError(
            f"Window {window_id} contains "
            f"{len(window_demand)} rows instead of "
            f"{WINDOW_SIZE}."
        )


    # --------------------------------------------------------
    # Run S-MROO-SUM
    # --------------------------------------------------------

    result = run_smroo_sum(
        demand=window_demand,
        parameter_pair=parameter_pair,
    )


    # --------------------------------------------------------
    # Return one result row
    # --------------------------------------------------------

    return {

        "window_id":
            int(window_id),

        "start_index":
            start,

        "end_index":
            end,

        "window_size":
            WINDOW_SIZE,

        "parameter_pair":
            int(parameter_pair),

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

        "gamma_R_q":
            float(
                result[
                    "gamma_R_q"
                ]
            ),

        "L_q":
            float(
                result[
                    "L_q"
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


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD FULL TRACE
    # ========================================================

    demand = load_demand()

    T_FULL, D = demand.shape


    print(
        "\n========================================"
    )

    print(
        "PAPER-STYLE S-MROO-SUM WINDOW EXPERIMENT"
    )

    print(
        "========================================"
    )

    print(
        "Full trace length =",
        T_FULL,
    )

    print(
        "Dimension =",
        D,
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
        "m =",
        M,
    )

    print(
        "beta =",
        BETA,
    )

    print(
        "beta/m =",
        BETA / M,
    )

    print(
        "H =",
        H,
    )

    print(
        "rho =",
        RHO,
    )

    print(
        "L_q =",
        RHO * np.sqrt(D),
    )


    # ========================================================
    # LOAD EXACT SAME WINDOWS AS MROO / S-MROO-MAX
    # ========================================================

    window_starts = get_window_starts(
        T_full=T_FULL
    )


    # ========================================================
    # BUILD JOBS
    #
    # With 2 parameter pairs and 100 windows:
    #
    # total jobs = 200
    # ========================================================

    jobs = []

    for parameter_pair in PARAMETER_PAIRS:

        for (
            window_id,
            start,
        ) in enumerate(
            window_starts,
            start=1,
        ):

            jobs.append(
                (
                    int(parameter_pair),
                    int(window_id),
                    int(start),
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
        "EXPERIMENT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Parameter pairs =",
        PARAMETER_PAIRS,
    )

    print(
        "Total runs =",
        total_runs,
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


    # ========================================================
    # RUN ALL WINDOWS IN PARALLEL
    # ========================================================

    results = []

    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:


        future_to_job = {}


        for (
            parameter_pair,
            window_id,
            start,
        ) in jobs:


            future = executor.submit(
                run_single_job,
                demand,
                parameter_pair,
                window_id,
                start,
            )


            future_to_job[
                future
            ] = (
                parameter_pair,
                window_id,
                start,
            )


        completed = 0


        for future in as_completed(
            future_to_job
        ):


            (
                parameter_pair,
                window_id,
                start,
            ) = future_to_job[
                future
            ]


            try:

                row = future.result()


            except Exception as error:

                print(
                    "\n========================================"
                )

                print(
                    "S-MROO-SUM WINDOW FAILED"
                )

                print(
                    "========================================"
                )

                print(
                    "parameter_pair =",
                    parameter_pair,
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
                f"pair={parameter_pair} | "
                f"lambda1={row['lambda_1']:.6g} | "
                f"lambda2={row['lambda_2']:.6g} | "
                f"hit={row['hitting_cost']:.4f} | "
                f"long={row['long_term_cost']:.4f} | "
                f"total={row['total_cost']:.4f}",
                flush=True,
            )


            # ------------------------------------------------
            # Save partial progress
            # ------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
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
                "parameter_pair",
                "window_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )


    # ========================================================
    # SUMMARY ACROSS 100 WINDOWS FOR EACH PARAMETER PAIR
    # ========================================================

    summary_df = (
        results_df
        .groupby(
            [
                "parameter_pair",
                "lambda_1",
                "lambda_2",
                "gamma_R_q",
                "L_q",
            ],
            as_index=False,
        )
        .agg(

            n_windows=(
                "window_id",
                "count",
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


    summary_df = (
        summary_df
        .sort_values(
            by="mean_total_cost"
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
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "S-MROO-SUM WINDOW EXPERIMENT COMPLETE"
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
        "\nPer-window S-MROO-SUM results:"
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
