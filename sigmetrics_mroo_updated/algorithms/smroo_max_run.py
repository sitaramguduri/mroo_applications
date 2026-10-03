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
# IMPORT S-MROO-MAX + CONFIG
# ============================================================

from smroo_max import run_smroo_max

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
# beta/m combination.
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
    "S-MROO-MAX result directory:",
    RESULT_DIR,
)


# ============================================================
# WINDOW CONFIGURATION
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
# Exactly the same file used by MROO.
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
    / "smroo_max_window_results.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "smroo_max_window_summary.csv"
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
# S-MROO-MAX PARAMETER PAIRS
#
# Pair 1:
#
#   lambda_1 =
#       m / (m gamma + beta L_q H^2)
#
#   lambda_2 = 0
#
# Pair 2:
#
#   lambda_1 = 1
#
#   lambda_2 =
#       m(gamma - 1) + beta L_q H^2
#
# Existing pair/window results are skipped.
# ============================================================

PARAMETER_PAIRS = [
    1,
    2,
]


# ============================================================
# LOAD DATA
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
# CREATE / LOAD SHARED WINDOW STARTS
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
            - set(
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

        if len(starts) != N_WINDOWS:

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
                "Shared window file contains windows "
                "outside the current demand trace."
            )

        print(
            "\nLoaded existing shared windows from:"
        )

        print(
            WINDOW_FILE
        )

        return starts


    # --------------------------------------------------------
    # Validate requested configuration
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
            f"but only {total_valid_windows} "
            f"valid windows exist."
        )


    # --------------------------------------------------------
    # Exact non-overlapping windows
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
                len(exact_starts),
                N_WINDOWS,
            )
        ]
    )

    selected = list(
        exact_starts.astype(int)
    )


    # --------------------------------------------------------
    # All possible continuous windows
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
    # Save shared windows
    # --------------------------------------------------------

    window_types = (
        ["exact"]
        * len(exact_starts)
        +
        ["sampled"]
        * (
            N_WINDOWS
            - len(exact_starts)
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
        len(exact_starts),
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
    parameter_pair,
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


    # --------------------------------------------------------
    # Extract one continuous window
    # --------------------------------------------------------

    window_demand = demand[
        start:end
    ]

    if len(
        window_demand
    ) != WINDOW_SIZE:

        raise RuntimeError(
            f"Window {window_id} contains "
            f"{len(window_demand)} rows instead of "
            f"{WINDOW_SIZE}."
        )


    # --------------------------------------------------------
    # Run S-MROO-MAX
    #
    # One complete window is treated as one frame.
    # --------------------------------------------------------

    result = run_smroo_max(
        demand=window_demand,
        parameter_pair=parameter_pair,
        R=WINDOW_SIZE,
    )


    # --------------------------------------------------------
    # Store one result per window
    # --------------------------------------------------------

    row = {

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

        "R":
            int(
                result[
                    "R"
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

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD FULL TRACE
    # ========================================================

    demand = load_demand()

    T_FULL, D = (
        demand.shape
    )


    print(
        "\n========================================"
    )

    print(
        "S-MROO-MAX WINDOW EXPERIMENT"
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


    # ========================================================
    # LOAD EXACT SAME WINDOWS AS OTHER ALGORITHMS
    # ========================================================

    window_starts = get_window_starts(
        T_full=T_FULL
    )


    # ========================================================
    # LOAD EXISTING S-MROO-MAX RESULTS
    #
    # Important:
    #
    # Existing Pair 1 results are preserved.
    #
    # Only pair/window combinations that are missing will
    # actually be run.
    # ========================================================

    if RESULT_FILE.exists():

        existing_df = pd.read_csv(
            RESULT_FILE
        )


        if len(
            existing_df
        ) > 0:

            required_existing_columns = {
                "parameter_pair",
                "window_id",
            }

            missing_existing_columns = (
                required_existing_columns
                - set(
                    existing_df.columns
                )
            )

            if missing_existing_columns:

                raise RuntimeError(
                    "Existing S-MROO-MAX result file is missing "
                    f"columns: "
                    f"{sorted(missing_existing_columns)}"
                )


        print(
            "\nLoaded existing S-MROO-MAX results:"
        )

        print(
            RESULT_FILE
        )

        print(
            "Existing rows =",
            len(existing_df),
        )


        if (
            "parameter_pair"
            in existing_df.columns
            and len(existing_df) > 0
        ):

            print(
                "\nExisting rows by parameter pair:"
            )

            print(
                existing_df[
                    "parameter_pair"
                ]
                .value_counts()
                .sort_index()
            )


    else:

        existing_df = (
            pd.DataFrame()
        )

        print(
            "\nNo existing S-MROO-MAX results found."
        )


    # ========================================================
    # BUILD ONLY MISSING JOBS
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


            window_id = int(
                window_id
            )

            start = int(
                start
            )


            # ------------------------------------------------
            # Check if this pair/window already exists
            # ------------------------------------------------

            already_done = False


            if len(
                existing_df
            ) > 0:

                existing_match = (
                    existing_df[

                        (
                            existing_df[
                                "parameter_pair"
                            ]
                            .astype(int)
                            == int(
                                parameter_pair
                            )
                        )

                        &

                        (
                            existing_df[
                                "window_id"
                            ]
                            .astype(int)
                            == window_id
                        )

                    ]
                )


                already_done = (
                    len(
                        existing_match
                    )
                    > 0
                )


            # ------------------------------------------------
            # Skip completed Pair 1 / Pair 2 windows
            # ------------------------------------------------

            if already_done:

                continue


            # ------------------------------------------------
            # Add missing result as a job
            # ------------------------------------------------

            jobs.append(
                (
                    int(parameter_pair),
                    window_id,
                    start,
                )
            )


    total_runs = len(
        jobs
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
        "Existing rows =",
        len(existing_df),
    )

    print(
        "Missing runs =",
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


    # --------------------------------------------------------
    # Useful check
    # --------------------------------------------------------

    if total_runs > 0:

        missing_pairs = pd.Series(
            [
                job[0]
                for job in jobs
            ]
        ).value_counts().sort_index()

        print(
            "\nMissing runs by parameter pair:"
        )

        print(
            missing_pairs
        )


    # ========================================================
    # START WITH EXISTING RESULTS
    #
    # This is the important part that prevents Pair 1 from
    # being overwritten.
    # ========================================================

    if len(
        existing_df
    ) > 0:

        results = (
            existing_df
            .to_dict(
                orient="records"
            )
        )

    else:

        results = []


    # ========================================================
    # RUN ONLY MISSING WINDOWS
    # ========================================================

    if total_runs > 0:

        worker_count = min(
            MAX_WORKERS,
            total_runs,
        )


        print(
            "Workers =",
            worker_count,
        )


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
                        "S-MROO-MAX WINDOW FAILED"
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


                # ------------------------------------------------
                # Append the newly completed window
                # ------------------------------------------------

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

                    f"lambda1="
                    f"{row['lambda_1']:.6g} | "

                    f"lambda2="
                    f"{row['lambda_2']:.6g} | "

                    f"hit="
                    f"{row['hitting_cost']:.4f} | "

                    f"long="
                    f"{row['long_term_cost']:.4f} | "

                    f"total="
                    f"{row['total_cost']:.4f}",

                    flush=True,

                )


                # =================================================
                # SAVE PROGRESS AFTER EVERY COMPLETED WINDOW
                #
                # This contains:
                #
                # old Pair 1 rows
                # +
                # completed Pair 2 rows
                #
                # so progress is safe if execution stops.
                # =================================================

                current_df = (
                    pd.DataFrame(
                        results
                    )
                )


                current_df = (

                    current_df

                    .drop_duplicates(
                        subset=[
                            "parameter_pair",
                            "window_id",
                        ],
                        keep="last",
                    )

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


                current_df.to_csv(
                    RESULT_FILE,
                    index=False,
                )


    else:

        print(
            "\nNo missing S-MROO-MAX runs. "
            "All requested pair/window results already exist."
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
            "No S-MROO-MAX results are available."
        )


    # --------------------------------------------------------
    # Remove any accidental duplicates
    # --------------------------------------------------------

    results_df = (

        results_df

        .drop_duplicates(
            subset=[
                "parameter_pair",
                "window_id",
            ],
            keep="last",
        )

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


    # --------------------------------------------------------
    # Verify number of rows for each requested pair
    # --------------------------------------------------------

    print(
        "\nFinal rows by parameter pair:"
    )

    print(
        results_df[
            "parameter_pair"
        ]
        .value_counts()
        .sort_index()
    )


    for parameter_pair in PARAMETER_PAIRS:

        pair_count = len(

            results_df[

                results_df[
                    "parameter_pair"
                ]
                .astype(int)

                == int(
                    parameter_pair
                )

            ]

        )


        if pair_count != N_WINDOWS:

            print(
                f"WARNING: parameter pair "
                f"{parameter_pair} has "
                f"{pair_count} rows; "
                f"expected {N_WINDOWS}."
            )


    # --------------------------------------------------------
    # Save complete Pair 1 + Pair 2 CSV
    # --------------------------------------------------------

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )


    # ========================================================
    # SUMMARY ACROSS WINDOWS
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
                "R",
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
            by=[
                "parameter_pair",
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
    # PRINT FINAL SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "S-MROO-MAX WINDOW EXPERIMENT COMPLETE"
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
        "\nPer-window S-MROO-MAX results:"
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