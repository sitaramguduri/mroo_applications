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

# PROJECT PATH

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

# IMPORT

# ============================================================



from offline_opt import run_offline_opt

from config import A_MATRICES





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

# main_sweep.py supplies MROO_RESULT_DIR for each beta/m.

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

    "Offline OPT result directory:",

    RESULT_DIR,

)





# ============================================================

# WINDOW CONFIGURATION

#

# Same settings as:

#

# MROO

# S-MROO-MAX

# S-MROO-SUM

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

# EXACT SAME FILE USED BY ALL ALGORITHMS

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

    / "offline_opt_window_results.csv"

)





SUMMARY_FILE = (

    RESULT_DIR

    / "offline_opt_window_summary.csv"

)





# ============================================================

# PARALLEL CONFIGURATION

#

# Offline optimization is much heavier.

# ============================================================



MAX_WORKERS = int(



    os.environ.get(

        "OFFLINE_OPT_WINDOW_WORKERS",

        "2",

    )

)





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

# GET SHARED WINDOW STARTS

# ============================================================



def get_window_starts(

    T_full,

):



    # --------------------------------------------------------

    # Prefer existing shared window file.

    #

    # Usually MROO will already have created this.

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



                "Shared window file is missing columns: "



                f"{sorted(missing)}"

            )





        starts = window_df[

            "start_index"

        ].to_numpy(

            dtype=int

        )





        if len(starts) != N_WINDOWS:



            raise RuntimeError(



                f"Expected {N_WINDOWS} windows, "



                f"but shared file has "

                f"{len(starts)}."

            )





        if np.any(



            starts

            + WINDOW_SIZE

            > T_full



        ):



            raise RuntimeError(



                "Shared window file contains "



                "windows outside the trace."

            )





        print(

            "\nLoaded shared windows:"

        )





        print(

            WINDOW_FILE

        )





        return starts





    # ========================================================

    # FALLBACK:

    #

    # Create same paper-style windows if the shared file

    # has not been created yet.

    # ========================================================



    if WINDOW_SIZE > T_full:



        raise ValueError(



            f"WINDOW_SIZE={WINDOW_SIZE} "



            f"is larger than trace length={T_full}."

        )





    valid_starts = np.arange(



        0,



        T_full

        - WINDOW_SIZE

        + 1,



        dtype=int,

    )





    # --------------------------------------------------------

    # Exact full-day windows

    # --------------------------------------------------------



    number_of_full_days = (



        T_full

        // WINDOW_SIZE

    )





    day_starts = (



        np.arange(

            number_of_full_days,

            dtype=int,

        )



        * WINDOW_SIZE

    )





    day_starts = (



        day_starts[

            :min(

                N_WINDOWS,

                len(day_starts),

            )

        ]

    )





    selected = list(

        day_starts

    )





    # --------------------------------------------------------

    # Remaining randomly sampled continuous windows

    # --------------------------------------------------------



    extra_pool = np.setdiff1d(



        valid_starts,



        day_starts,

    )





    remaining = (



        N_WINDOWS

        - len(selected)

    )





    rng = np.random.default_rng(

        RANDOM_SEED

    )





    if remaining > 0:



        sampled = rng.choice(



            extra_pool,



            size=remaining,



            replace=False,

        )





        selected.extend(

            sampled.astype(int)

        )





    starts = np.asarray(

        selected,

        dtype=int,

    )





    # --------------------------------------------------------

    # Save shared definition

    # --------------------------------------------------------



    window_types = (



        ["exact_day"]

        * len(day_starts)



        +



        ["sampled"]

        * remaining

    )





    window_df = pd.DataFrame({



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

    })





    WINDOW_FILE.parent.mkdir(

        parents=True,

        exist_ok=True,

    )





    window_df.to_csv(

        WINDOW_FILE,

        index=False,

    )





    print(

        "\nCreated shared window file:"

    )





    print(

        WINDOW_FILE

    )





    return starts





# ============================================================

# RUN ONE WINDOW

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





    # --------------------------------------------------------

    # Exact same demand window as all other algorithms

    # --------------------------------------------------------



    window_demand = demand[

        start:end

    ]





    if len(

        window_demand

    ) != WINDOW_SIZE:



        raise RuntimeError(



            f"Window {window_id} has "



            f"{len(window_demand)} rows, "



            f"expected {WINDOW_SIZE}."

        )





    # --------------------------------------------------------

    # OFFLINE OPT

    # --------------------------------------------------------



    result = run_offline_opt(



        demand=window_demand

    )





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



        "solver_status":

            result[

                "solver_status"

            ],



        "solver_objective":

            float(

                result[

                    "solver_objective"

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

    # LOAD TRACE

    # ========================================================



    demand = load_demand()





    T_FULL, D = demand.shape


    # ========================================================
    # VALIDATE MATRIX-BASED MEMORY CONFIGURATION
    #
    # There must be exactly D matrices:
    #
    #     A_1, ..., A_D
    #
    # and each A_i must be D x D.
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

        "PAPER-STYLE OFFLINE OPT WINDOW EXPERIMENT"

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

    # SAME WINDOWS

    # ========================================================



    window_starts = get_window_starts(



        T_full=T_FULL

    )





    # ========================================================

    # JOB LIST

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

        "OFFLINE OPT CONFIGURATION"

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

        "Shared windows =",

        WINDOW_FILE,

    )





    # ========================================================

    # PARALLEL EXECUTION

    # ========================================================



    results = []





    with ProcessPoolExecutor(



        max_workers=worker_count



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

                    "OFFLINE OPT WINDOW FAILED"

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



                f"status={row['solver_status']} | "



                f"hit={row['hitting_cost']:.4f} | "



                f"long={row['long_term_cost']:.4f} | "



                f"total={row['total_cost']:.4f}",



                flush=True,

            )





            # ------------------------------------------------

            # Save progress continuously

            # ------------------------------------------------



            pd.DataFrame(



                results



            ).to_csv(



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

            "window_id"

        )



        .reset_index(

            drop=True

        )

    )





    # ========================================================

    # CHECK SOLVER OBJECTIVE

    # ========================================================



    results_df[

        "objective_difference"

    ] = np.abs(



        results_df[

            "solver_objective"

        ]



        -



        results_df[

            "total_cost"

        ]

    )





    results_df.to_csv(



        RESULT_FILE,



        index=False,

    )





    print(



        "\nMaximum |solver objective - total cost| =",



        results_df[

            "objective_difference"

        ].max(),

    )





    # ========================================================

    # SUMMARY ACROSS 100 WINDOWS

    # ========================================================



    summary_df = pd.DataFrame({



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





        "max_objective_difference": [



            results_df[

                "objective_difference"

            ].max()

        ],

    })





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

        "OFFLINE OPT WINDOW EXPERIMENT COMPLETE"

    )





    print(

        "========================================"

    )





    print(

        "\nSummary:\n"

    )





    print(



        summary_df.to_string(

            index=False

        )

    )





    print(

        "\nDetailed results:"

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

        "\nShared window file:"

    )





    print(

        WINDOW_FILE

    )





# ============================================================

# RUN

# ============================================================



if __name__ == "__main__":



    main()