import sys
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

import os

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
# IMPORT S-MROO-SUM
# ============================================================

from smroo_sum import (
    run_smroo_sum,
)

from config import (
    SMROO_SUM_LAMBDA_1_VALUES,
    SMROO_SUM_LAMBDA_2,
)


# ============================================================
# FILES
# ============================================================

DATA_FILE = (
    PROJECT_DIR
    / "new_results"
    / "mroo_demand_trace_norm_weekday_aligned.csv"
)


# ============================================================
# RESULT DIRECTORY
# ============================================================

DEFAULT_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
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


RESULT_FILE = (
    RESULT_DIR
    / "smroo_sum_full_horizon_results.csv"
)


# ============================================================
# DEMAND COLUMNS
# ============================================================

DEMAND_COLUMNS = [
    "code",
    "chat",
    "non-interactive",
]


# ============================================================
# PARAMETER PAIR
# ============================================================

PARAMETER_PAIR = 1


# ============================================================
# PARALLEL SETTINGS
# ============================================================

MAX_WORKERS = 3


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results():

    if (
        RESULT_FILE.exists()
        and RESULT_FILE.stat().st_size > 0
    ):

        try:

            existing_df = pd.read_csv(
                RESULT_FILE
            )

        except pd.errors.EmptyDataError:

            existing_df = pd.DataFrame()

    else:

        existing_df = pd.DataFrame()


    return existing_df


# ============================================================
# CHECK WHETHER A LAMBDA HAS ALREADY BEEN RUN
# ============================================================

def lambda_already_completed(
    lambda_1,
    completed_lambdas,
):

    return any(

        np.isclose(
            float(lambda_1),
            float(existing_lambda),
            rtol=1e-10,
            atol=1e-12,
        )

        for existing_lambda
        in completed_lambdas
    )


# ============================================================
# COMBINE OLD + NEW RESULTS
# ============================================================

def combine_results(
    existing_df,
    new_results,
):

    new_df = pd.DataFrame(
        new_results
    )


    # --------------------------------------------------------
    # Nothing old
    # --------------------------------------------------------

    if len(existing_df) == 0:

        combined_df = (
            new_df.copy()
        )


    # --------------------------------------------------------
    # Nothing new
    # --------------------------------------------------------

    elif len(new_df) == 0:

        combined_df = (
            existing_df.copy()
        )


    # --------------------------------------------------------
    # Old + new
    # --------------------------------------------------------

    else:

        combined_df = pd.concat(
            [
                existing_df,
                new_df,
            ],
            ignore_index=True,
        )


    if len(combined_df) == 0:

        return combined_df


    # --------------------------------------------------------
    # If the same lambda somehow appears twice, keep the
    # newest version.
    # --------------------------------------------------------

    if (
        "lambda_1"
        in combined_df.columns
    ):

        combined_df = (
            combined_df
            .drop_duplicates(
                subset=[
                    "lambda_1",
                ],
                keep="last",
            )
            .sort_values(
                by="lambda_1"
            )
            .reset_index(
                drop=True
            )
        )


    return combined_df


# ============================================================
# RUN ONE LAMBDA CONFIGURATION
# ============================================================

def run_one_configuration(
    lambda_1,
    demand,
    T,
):

    print(
        f"Starting S-MROO-SUM "
        f"lambda_1={lambda_1:.8g}",
        flush=True,
    )


    result = run_smroo_sum(

        demand=demand,

        parameter_pair=PARAMETER_PAIR,

        lambda_1=lambda_1,

        lambda_2=SMROO_SUM_LAMBDA_2,
    )


    row = {

        "T":
            T,

        "parameter_pair":
            PARAMETER_PAIR,

        "lambda_1":
            result[
                "lambda_1"
            ],

        "lambda_2":
            result[
                "lambda_2"
            ],

        "theoretical_lambda_1":
            result[
                "theoretical_lambda_1"
            ],

        "gamma_R_q":
            result[
                "gamma_R_q"
            ],

        "L_q":
            result[
                "L_q"
            ],

        "hitting_cost":
            result[
                "hitting_cost"
            ],

        "long_term_cost":
            result[
                "long_term_cost"
            ],

        "total_cost":
            result[
                "total_cost"
            ],
    }


    print(
        f"Finished S-MROO-SUM "
        f"lambda_1={lambda_1:.8g}, "
        f"total_cost={row['total_cost']:.8f}",
        flush=True,
    )


    return row


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = pd.read_csv(
        DATA_FILE
    )


    demand = data[
        DEMAND_COLUMNS
    ].to_numpy(
        dtype=float
    )


    T, D = (
        demand.shape
    )


    print(
        "\n========================================"
    )

    print(
        "FULL-HORIZON S-MROO-SUM LAMBDA SWEEP"
    )

    print(
        "========================================"
    )

    print(
        "Horizon length T =",
        T,
    )

    print(
        "Dimension D =",
        D,
    )


    # ========================================================
    # SANITY CHECK
    # ========================================================

    expected_T = (
        7
        * 24
        * 60
        * 6
    )


    if T != expected_T:

        print(
            f"\nWARNING: expected "
            f"{expected_T} time steps "
            f"for 7 days, found {T}."
        )


    # ========================================================
    # CONFIGURED LAMBDA VALUES
    # ========================================================

    configured_lambdas = [

        float(lambda_1)

        for lambda_1
        in SMROO_SUM_LAMBDA_1_VALUES
    ]


    print(
        "\nConfigured lambda_1 values:"
    )


    for lambda_1 in configured_lambdas:

        print(
            "  ",
            lambda_1,
        )


    # ========================================================
    # LOAD EXISTING RESULTS
    # ========================================================

    existing_df = (
        load_existing_results()
    )


    if (
        len(existing_df) > 0
        and "lambda_1"
        in existing_df.columns
    ):

        completed_lambdas = (

            existing_df[
                "lambda_1"
            ]
            .dropna()
            .astype(float)
            .tolist()
        )

    else:

        completed_lambdas = []


    print(
        "\nAlready completed lambda_1 values:"
    )


    if len(
        completed_lambdas
    ) == 0:

        print(
            "  None"
        )

    else:

        for lambda_1 in sorted(
            completed_lambdas
        ):

            print(
                "  ",
                lambda_1,
            )


    # ========================================================
    # FIND ONLY MISSING LAMBDAS
    # ========================================================

    lambda_values_to_run = [

        lambda_1

        for lambda_1
        in configured_lambdas

        if not lambda_already_completed(
            lambda_1=lambda_1,
            completed_lambdas=
                completed_lambdas,
        )
    ]


    print(
        "\nNew lambda_1 values to run:"
    )


    if len(
        lambda_values_to_run
    ) == 0:

        print(
            "  None"
        )

    else:

        for lambda_1 in (
            lambda_values_to_run
        ):

            print(
                "  ",
                lambda_1,
            )


    # ========================================================
    # NOTHING NEW TO RUN
    # ========================================================

    if len(
        lambda_values_to_run
    ) == 0:

        print(
            "\nAll configured lambda_1 values "
            "are already completed."
        )

        print(
            "No new S-MROO-SUM runs required."
        )


        if len(
            existing_df
        ) > 0:

            result_df = (
                existing_df
                .sort_values(
                    by="lambda_1"
                )
                .reset_index(
                    drop=True
                )
            )


            best = (
                result_df
                .sort_values(
                    by="total_cost"
                )
                .iloc[0]
            )


            print(
                "\n========================================"
            )

            print(
                "EXISTING S-MROO-SUM RESULTS"
            )

            print(
                "========================================"
            )

            print(
                result_df.to_string(
                    index=False
                )
            )


            print(
                "\n========================================"
            )

            print(
                "BEST S-MROO-SUM CONFIGURATION"
            )

            print(
                "========================================"
            )

            print(
                "lambda_1 =",
                best[
                    "lambda_1"
                ],
            )

            print(
                "lambda_2 =",
                best[
                    "lambda_2"
                ],
            )

            print(
                "hitting_cost =",
                best[
                    "hitting_cost"
                ],
            )

            print(
                "long_term_cost =",
                best[
                    "long_term_cost"
                ],
            )

            print(
                "total_cost =",
                best[
                    "total_cost"
                ],
            )


        return


    # ========================================================
    # RUN ONLY MISSING LAMBDA VALUES IN PARALLEL
    # ========================================================

    results = []


    worker_count = min(
        MAX_WORKERS,
        len(
            lambda_values_to_run
        ),
    )


    print(
        "\nNumber of new configurations =",
        len(
            lambda_values_to_run
        ),
    )

    print(
        "Number of workers =",
        worker_count,
    )


    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:


        future_to_lambda = {}


        for lambda_1 in (
            lambda_values_to_run
        ):


            future = executor.submit(

                run_one_configuration,

                float(
                    lambda_1
                ),

                demand,

                T,
            )


            future_to_lambda[
                future
            ] = lambda_1


        completed = 0


        for future in as_completed(
            future_to_lambda
        ):


            lambda_1 = (
                future_to_lambda[
                    future
                ]
            )


            try:

                row = (
                    future.result()
                )


            except Exception as error:

                print(
                    "\nS-MROO-SUM RUN FAILED"
                )

                print(
                    "lambda_1 =",
                    lambda_1,
                )

                print(
                    "Error =",
                    error,
                )

                raise


            results.append(
                row
            )


            completed += 1


            print(
                f"\nCompleted "
                f"{completed}/"
                f"{len(lambda_values_to_run)} "
                f"new lambda runs.",
                flush=True,
            )


            # =================================================
            # SAVE AFTER EVERY COMPLETED NEW LAMBDA
            #
            # Existing data is preserved.
            # =================================================

            current_df = combine_results(
                existing_df=
                    existing_df,

                new_results=
                    results,
            )


            current_df.to_csv(
                RESULT_FILE,
                index=False,
            )


            print(
                "Saved progress to:",
                RESULT_FILE,
                flush=True,
            )


    # ========================================================
    # FINAL COMBINED RESULTS
    #
    # OLD + NEW
    # ========================================================

    result_df = combine_results(
        existing_df=
            existing_df,

        new_results=
            results,
    )


    result_df.to_csv(
        RESULT_FILE,
        index=False,
    )


    # ========================================================
    # BEST CONFIGURATION ACROSS ALL OLD + NEW RESULTS
    # ========================================================

    best = (
        result_df
        .sort_values(
            by="total_cost"
        )
        .iloc[0]
    )


    # ========================================================
    # PRINT ALL RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "S-MROO-SUM RESULTS"
    )

    print(
        "========================================"
    )


    print(
        result_df.to_string(
            index=False
        )
    )


    # ========================================================
    # PRINT BEST CONFIGURATION
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "BEST S-MROO-SUM CONFIGURATION"
    )

    print(
        "========================================"
    )


    print(
        "lambda_1 =",
        best[
            "lambda_1"
        ],
    )

    print(
        "lambda_2 =",
        best[
            "lambda_2"
        ],
    )

    print(
        "hitting_cost =",
        best[
            "hitting_cost"
        ],
    )

    print(
        "long_term_cost =",
        best[
            "long_term_cost"
        ],
    )

    print(
        "total_cost =",
        best[
            "total_cost"
        ],
    )


    print(
        "\nNew lambda runs completed =",
        len(
            results
        ),
    )

    print(
        "Total stored configurations =",
        len(
            result_df
        ),
    )


    print(
        "\nSaved to:",
        RESULT_FILE,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()