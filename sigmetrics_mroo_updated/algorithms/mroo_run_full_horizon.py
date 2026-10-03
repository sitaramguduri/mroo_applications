import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# IMPORT MROO + CONFIG
# ============================================================

from mroo import run_mroo

from config import (
    LAMBDA_1_VALUES,
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
#
# Normal run:
#   algorithms/new_results/
#
# Beta/m sweep:
#   master script provides MROO_RESULT_DIR
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
    "MROO result directory:",
    RESULT_DIR,
)


RESULT_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_results.csv"
)

COMBINED_KAPPA_FILE = (
    RESULT_DIR
    / "mroo_full_horizon_kappa_all_lambda.csv"
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
# PARALLEL SETTINGS
#
# Here the independent MROO parameter configurations are
# parallelized.
#
# Time steps inside each MROO run remain sequential.
# ============================================================

MAX_WORKERS = 8


# ============================================================
# HELPER FOR SAFE FILENAME LABELS
# ============================================================

def make_label(
    value,
):

    return (
        f"{value:.8g}"
        .replace(
            ".",
            "p",
        )
        .replace(
            "-",
            "m",
        )
        .replace(
            "+",
            "",
        )
    )


# ============================================================
# WORKER FUNCTION
#
# Each worker runs ONE independent:
#
#   (eta, kappa_init, lambda_1)
#
# configuration.
# ============================================================

def run_one_configuration(
    eta,
    kappa_scalar,
    lambda_1,
    demand,
    D,
    T,
):

    # --------------------------------------------------------
    # Initial dual vector
    # --------------------------------------------------------

    kappa_init = np.full(
        D,
        kappa_scalar,
        dtype=float,
    )


    print(
        f"Starting "
        f"lambda_1={lambda_1:.8g}, "
        f"eta={eta:.8g}, "
        f"kappa_init={kappa_scalar:.8g}",
        flush=True,
    )


    # ========================================================
    # RUN MROO
    # ========================================================

    result = run_mroo(

        demand=demand,

        eta=eta,

        kappa_init=kappa_init,

        lambda_1=lambda_1,
    )


    # ========================================================
    # COST RESULTS
    # ========================================================

    row = {

        "T":
            T,

        "eta":
            eta,

        "kappa_init":
            kappa_scalar,

        "lambda_1":
            result[
                "lambda_1"
            ],

        "lambda_2":
            result[
                "lambda_2"
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


    # ========================================================
    # KAPPA HISTORY
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


    kappa_df.insert(
        0,
        "t",
        np.arange(
            len(kappa_df)
        ),
    )


    kappa_df.insert(
        1,
        "eta",
        eta,
    )


    kappa_df.insert(
        2,
        "kappa_init",
        kappa_scalar,
    )


    kappa_df.insert(
        3,
        "lambda_1",
        lambda_1,
    )


    print(
        f"Finished "
        f"lambda_1={lambda_1:.8g}, "
        f"eta={eta:.8g}, "
        f"total_cost={row['total_cost']:.8f}",
        flush=True,
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
    # LOAD FULL 7-DAY TRACE
    # ========================================================

    data = pd.read_csv(
        DATA_FILE
    )


    demand = data[
        DEMAND_COLUMNS
    ].to_numpy(
        dtype=float
    )


    T, D = demand.shape


    print(
        "Full horizon length:",
        T,
    )

    print(
        "Dimension:",
        D,
    )


    # ========================================================
    # PARAMETER VALUES
    #
    # Keep eta and kappa fixed.
    #
    # Sweep only lambda_1.
    # ========================================================

    ETA_VALUES = [
        T ** (-1.0 / 3.0)
    ]


    KAPPA_VALUES = [
        1.0 / T
    ]


    print(
        "\nLambda_1 values:"
    )

    for lambda_1 in LAMBDA_1_VALUES:

        print(
            "  ",
            lambda_1,
        )


    print(
        "\nEta values:"
    )

    for eta in ETA_VALUES:

        print(
            "  ",
            eta,
        )


    print(
        "\nKappa initial values:"
    )

    for kappa_scalar in KAPPA_VALUES:

        print(
            "  ",
            kappa_scalar,
        )


    # ========================================================
    # BUILD CONFIGURATIONS
    # ========================================================

    configs = []


    for eta in ETA_VALUES:

        for kappa_scalar in KAPPA_VALUES:

            for lambda_1 in LAMBDA_1_VALUES:

                configs.append(
                    (
                        float(eta),
                        float(kappa_scalar),
                        float(lambda_1),
                    )
                )


    total_runs = len(
        configs
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FULL-HORIZON MROO LAMBDA SWEEP"
    )

    print(
        "========================================"
    )

    print(
        "T =",
        T,
    )

    print(
        "Total parameter configurations =",
        total_runs,
    )

    print(
        "Maximum workers =",
        min(
            MAX_WORKERS,
            total_runs,
        ),
    )


    # ========================================================
    # STORAGE
    # ========================================================

    results = []

    all_kappa_results = []


    # ========================================================
    # RUN CONFIGURATIONS IN PARALLEL
    # ========================================================

    worker_count = min(
        MAX_WORKERS,
        total_runs,
    )


    with ProcessPoolExecutor(
        max_workers=worker_count
    ) as executor:


        future_to_config = {}


        for (
            eta,
            kappa_scalar,
            lambda_1,
        ) in configs:


            future = executor.submit(

                run_one_configuration,

                eta,

                kappa_scalar,

                lambda_1,

                demand,

                D,

                T,
            )


            future_to_config[
                future
            ] = (
                eta,
                kappa_scalar,
                lambda_1,
            )


        completed = 0


        for future in as_completed(
            future_to_config
        ):


            (
                eta,
                kappa_scalar,
                lambda_1,
            ) = (
                future_to_config[
                    future
                ]
            )


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
                    "RUN FAILED"
                )

                print(
                    "========================================"
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
                    kappa_scalar,
                )

                print(
                    "Error:",
                    error,
                )

                raise


            completed += 1


            # ====================================================
            # STORE RESULTS
            # ====================================================

            results.append(
                row
            )

            all_kappa_results.append(
                kappa_df
            )


            print(
                f"\nCompleted "
                f"{completed}/{total_runs}"
            )

            print(
                "lambda_1 =",
                row[
                    "lambda_1"
                ],
            )

            print(
                "eta =",
                row[
                    "eta"
                ],
            )

            print(
                "kappa_init =",
                row[
                    "kappa_init"
                ],
            )

            print(
                "Hitting cost   =",
                row[
                    "hitting_cost"
                ],
            )

            print(
                "Long-term cost =",
                row[
                    "long_term_cost"
                ],
            )

            print(
                "Total cost     =",
                row[
                    "total_cost"
                ],
            )


            # ====================================================
            # SAVE COST RESULTS AFTER EACH COMPLETED RUN
            # ====================================================

            current_results_df = (
                pd.DataFrame(
                    results
                )
                .sort_values(
                    by=[
                        "lambda_1",
                        "eta",
                        "kappa_init",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


            current_results_df.to_csv(
                RESULT_FILE,
                index=False,
            )


            # ====================================================
            # SAVE COMBINED KAPPA HISTORY
            # ====================================================

            current_kappa_df = pd.concat(
                all_kappa_results,
                ignore_index=True,
            )


            current_kappa_df = (
                current_kappa_df
                .sort_values(
                    by=[
                        "lambda_1",
                        "eta",
                        "kappa_init",
                        "t",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


            current_kappa_df.to_csv(
                COMBINED_KAPPA_FILE,
                index=False,
            )


            # ====================================================
            # SAVE SEPARATE KAPPA FILE FOR THIS CONFIGURATION
            # ====================================================

            eta_label = make_label(
                row[
                    "eta"
                ]
            )

            kappa_label = make_label(
                row[
                    "kappa_init"
                ]
            )

            lambda_label = make_label(
                row[
                    "lambda_1"
                ]
            )


            individual_kappa_file = (
                RESULT_DIR
                / (
                    "mroo_full_horizon_"
                    f"lambda1_{lambda_label}_"
                    f"eta_{eta_label}_"
                    f"init_{kappa_label}.csv"
                )
            )


            kappa_df.to_csv(
                individual_kappa_file,
                index=False,
            )


            print(
                "Saved kappa:",
                individual_kappa_file,
            )


    # ========================================================
    # FINAL COST RESULTS
    # ========================================================

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            by=[
                "lambda_1",
                "eta",
                "kappa_init",
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
    # FINAL COMBINED KAPPA RESULTS
    # ========================================================

    combined_kappa_df = (
        pd.concat(
            all_kappa_results,
            ignore_index=True,
        )
        .sort_values(
            by=[
                "lambda_1",
                "eta",
                "kappa_init",
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
    # PRINT FINAL TABLE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "FULL-HORIZON MROO RESULTS"
    )

    print(
        "========================================"
    )


    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    best = (
        results_df
        .sort_values(
            by="total_cost"
        )
        .iloc[0]
    )


    print(
        "\n========================================"
    )

    print(
        "BEST MROO CONFIGURATION"
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
        "eta =",
        best[
            "eta"
        ],
    )

    print(
        "kappa_init =",
        best[
            "kappa_init"
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


    # ========================================================
    # FILES SAVED
    # ========================================================

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
        "Cost results:"
    )

    print(
        RESULT_FILE
    )


    print(
        "\nCombined kappa history:"
    )

    print(
        COMBINED_KAPPA_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()