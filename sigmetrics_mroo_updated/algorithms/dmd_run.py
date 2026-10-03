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
# IMPORTANT:
#
# We use MROO_RESULT_DIR here on purpose.
#
# main_sweep.py already sets this environment variable
# independently for every beta/m setting.
#
# This means DMD results will automatically go into the same
# beta/m/T result folder as all the other algorithms.
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
#
# Uses exactly the same environment variables as MROO.
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
# EXACTLY the same file used by:
#
# MROO
# S-MROO-SUM
# S-MROO-MAX
# GREEDY
# OFFLINE-OPT
#
# DMD must use the same windows as well.
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


    demand = data[

        DEMAND_COLUMNS

    ].to_numpy(

        dtype=float

    )


    return demand


# ============================================================
# CREATE / LOAD SHARED WINDOW STARTS
#
# This is intentionally the same logic as mroo_run.py.
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

            - set(
                window_df.columns
            )

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


        # ----------------------------------------------------
        # Verify number of windows
        # ----------------------------------------------------

        if len(starts) != N_WINDOWS:

            raise RuntimeError(

                f"Expected {N_WINDOWS} windows, "

                f"but shared window file contains "

                f"{len(starts)}."

            )


        # ----------------------------------------------------
        # Verify windows fit in trace
        # ----------------------------------------------------

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

            f"but only {total_valid_windows} "

            f"valid window starts exist."

        )


    # --------------------------------------------------------
    # Exact non-overlapping full windows
    #
    # For WINDOW_SIZE=8640 this gives 7 day windows.
    #
    # For WINDOW_SIZE=17280 this gives 3 full 48h windows.
    #
    # For WINDOW_SIZE=25920 this gives 2 full 72h windows.
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
    # Every other valid continuous starting point
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
    # Save shared window file
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

        + WINDOW_SIZE

    )


    # --------------------------------------------------------
    # Extract one continuous window
    # --------------------------------------------------------

    window_demand = demand[

        start:end

    ]


    if len(window_demand) != WINDOW_SIZE:

        raise RuntimeError(

            f"Window {window_id} has "

            f"{len(window_demand)} rows instead of "

            f"{WINDOW_SIZE}."

        )


    # --------------------------------------------------------
    # Initial dual variable
    #
    # Each window is an independent experiment.
    # --------------------------------------------------------

    kappa_init = np.full(

        D,

        kappa_scalar,

        dtype=float,

    )


    # --------------------------------------------------------
    # Run DMD
    # --------------------------------------------------------

    result = run_dmd(

        demand=window_demand,

        eta=eta,

        kappa_init=kappa_init,

    )


    # ========================================================
    # RESULT ROW
    # ========================================================

    row = {


        "window_id":

            int(window_id),


        "start_index":

            start,


        "end_index":

            end,


        "window_size":

            WINDOW_SIZE,


        "eta":

            float(eta),


        "kappa_init":

            float(kappa_scalar),


        # ----------------------------------------------------
        # DMD has no lambda_1 / lambda_2.
        #
        # We store NaN so it is obvious that these parameters
        # are not part of the DMD algorithm.
        # ----------------------------------------------------

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


    # ========================================================
    # KAPPA TRAJECTORY
    # ========================================================

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

            for i in range(D)

        ],

    )


    # --------------------------------------------------------
    # Local t
    # --------------------------------------------------------

    kappa_df.insert(

        0,

        "t",

        np.arange(

            len(kappa_df)

        ),

    )


    # --------------------------------------------------------
    # Global trace position
    # --------------------------------------------------------

    kappa_df.insert(

        0,

        "global_t",

        start

        + np.arange(

            len(kappa_df)

        ),

    )


    # --------------------------------------------------------
    # Window ID
    # --------------------------------------------------------

    kappa_df.insert(

        0,

        "window_id",

        int(window_id),

    )


    # --------------------------------------------------------
    # Start index
    # --------------------------------------------------------

    kappa_df.insert(

        1,

        "start_index",

        start,

    )


    # --------------------------------------------------------
    # Eta
    # --------------------------------------------------------

    kappa_df.insert(

        2,

        "eta",

        float(eta),

    )


    # --------------------------------------------------------
    # Kappa initialization
    # --------------------------------------------------------

    kappa_df.insert(

        3,

        "kappa_init",

        float(kappa_scalar),

    )


    return (

        row,

        kappa_df,

    )


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


    # ========================================================
    # LOAD EXACT SAME WINDOWS
    # ========================================================

    window_starts = get_window_starts(

        T_full=T_FULL

    )


    # ========================================================
    # DMD PARAMETERS
    #
    # Use the same theoretical scaling as current MROO:
    #
    # eta     = T_window^(-1/3)
    #
    # kappa_1 = 1 / T_window
    #
    # No lambda_1.
    # No lambda_2.
    # ========================================================

    ETA_VALUES = [

        WINDOW_SIZE

        ** (-1.0 / 3.0)

    ]


    KAPPA_VALUES = [

        1.0

        / WINDOW_SIZE

    ]


    # ========================================================
    # PRINT PARAMETERS
    # ========================================================

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
    # BUILD JOBS
    #
    # One job =
    #
    # one window
    # x one eta
    # x one kappa initialization
    #
    # With the current configuration:
    #
    # 100 windows x 1 eta x 1 kappa = 100 runs
    # ========================================================

    jobs = []


    for eta in ETA_VALUES:


        for kappa_scalar in KAPPA_VALUES:


            for (

                window_id,

                start,

            ) in enumerate(

                window_starts,

                start=1,

            ):


                jobs.append(

                    (

                        int(window_id),

                        int(start),

                        float(eta),

                        float(kappa_scalar),

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
        "Total DMD runs =",
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
    # RUN JOBS
    # ========================================================

    results = []

    all_kappa_results = []


    with ProcessPoolExecutor(

        max_workers=worker_count

    ) as executor:


        future_to_job = {}


        for (

            window_id,

            start,

            eta,

            kappa_scalar,

        ) in jobs:


            future = executor.submit(


                run_single_job,


                demand,


                D,


                window_id,


                start,


                eta,


                kappa_scalar,


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

            ) = future_to_job[

                future

            ]


            try:


                (

                    row,

                    kappa_df,

                ) = future.result()


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


            results.append(
                row
            )


            all_kappa_results.append(
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


            # ------------------------------------------------
            # Save partial progress
            # ------------------------------------------------

            current_df = pd.DataFrame(
                results
            )


            current_df.to_csv(

                RESULT_FILE,

                index=False,

            )


    # ========================================================
    # FINAL RESULT DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df = (

        results_df

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


    results_df.to_csv(

        RESULT_FILE,

        index=False,

    )


    # ========================================================
    # SAVE KAPPA HISTORIES
    # ========================================================

    if len(
        all_kappa_results
    ) > 0:


        combined_kappa_df = pd.concat(

            all_kappa_results,

            ignore_index=True,

        )


        combined_kappa_df = (

            combined_kappa_df

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


        combined_kappa_df.to_csv(

            COMBINED_KAPPA_FILE,

            index=False,

        )


    # ========================================================
    # SUMMARY ACROSS WINDOWS
    # ========================================================

    summary_df = (

        results_df

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
        "\nPer-window DMD results:"
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
        "\nKappa trajectories:"
    )

    print(
        COMBINED_KAPPA_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()