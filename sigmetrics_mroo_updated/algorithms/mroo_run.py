import ast

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





# ============================================================

# IMPORT MROO + CONFIG

# ============================================================



from mroo import run_mroo

from config import LAMBDA_1_VALUES





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

#

# We no longer save the full kappa trajectory.

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






# ------------------------------------------------------------
# Configurations requested in the CURRENT execution.
# This is overwritten each time mroo_run.py is executed.
# ------------------------------------------------------------

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



def _parse_kappa_value(kappa_value):



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

    # Scalar input

    # --------------------------------------------------------



    if arr.ndim == 0:



        arr = np.full(

            D,

            float(arr),

            dtype=float,

        )



    # --------------------------------------------------------

    # Vector input

    # --------------------------------------------------------



    elif arr.shape != (D,):



        raise ValueError(

            "kappa value must be either a scalar "

            f"or a vector of shape ({D},), "

            f"but got shape {arr.shape}."

        )



    if np.any(

        ~np.isfinite(arr)

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



    arr = normalize_kappa_init(

        kappa_value,

        D,

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



    arr = normalize_kappa_init(

        kappa_value,

        D,

    )



    # --------------------------------------------------------

    # Preserve old scalar-style display when all components

    # are identical.

    # --------------------------------------------------------



    if np.allclose(

        arr,

        arr[0],

        rtol=0.0,

        atol=1e-12,

    ):



        return f"{arr[0]:.8g}"



    return (

        "["

        + ",".join(

            f"{x:.8g}"

            for x in arr

        )

        + "]"

    )





def kappa_component_columns(

    D,

):



    return [



        f"kappa_init_{i}"



        for i in range(D)



    ]





def ensure_kappa_columns(

    df,

    D,

):



    """

    Makes old scalar result / summary / completion files

    compatible with the new vector kappa format.

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



    if len(df) == 0:



        for column in component_columns:



            if column not in df.columns:



                df[column] = pd.Series(

                    dtype=float

                )



        return df



    # --------------------------------------------------------

    # Already new format

    # --------------------------------------------------------



    if all(



        column in df.columns



        for column in component_columns



    ):



        labels = []



        for _, row in df.iterrows():



            vector = np.array(



                [

                    row[column]

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



    if "kappa_init" not in df.columns:



        raise RuntimeError(

            "Existing MROO file has neither "

            "kappa_init nor kappa_init_i columns."

        )



    recovered = []



    for value in df[

        "kappa_init"

    ]:



        recovered.append(



            normalize_kappa_init(

                value,

                D,

            )



        )



    recovered = np.asarray(

        recovered,

        dtype=float,

    )



    for i, column in enumerate(

        component_columns

    ):



        df[column] = (

            recovered[:, i]

        )



    df[

        "kappa_init"

    ] = [



        kappa_label(

            vector,

            D,

        )



        for vector in recovered



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

            float(lambda_1),

            12,

        ),



        round(

            float(eta),

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



    return data[

        DEMAND_COLUMNS

    ].to_numpy(

        dtype=float

    )





# ============================================================

# CREATE / LOAD PAPER-STYLE WINDOWS

# ============================================================



def get_window_starts(

    T_full,

):



    # --------------------------------------------------------

    # Reuse existing common windows if already created.

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



        starts = window_df[

            "start_index"

        ].to_numpy(

            dtype=int

        )



        if np.any(



            starts

            + WINDOW_SIZE

            > T_full



        ):



            raise RuntimeError(

                "Existing window file contains windows "

                "outside the current demand trace."

            )



        if len(starts) != N_WINDOWS:



            raise RuntimeError(

                f"Existing window file contains "

                f"{len(starts)} windows, but "

                f"N_WINDOWS={N_WINDOWS}."

            )



        print(

            "\nLoaded existing shared windows from:"

        )



        print(

            WINDOW_FILE

        )



        return starts



    # --------------------------------------------------------

    # Sanity checks

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

            f"Requested {N_WINDOWS} windows, but only "

            f"{total_valid_windows} valid windows exist."

        )



    # --------------------------------------------------------

    # Exact-day starts

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



    exact_day_starts = exact_day_starts[



        :min(

            len(exact_day_starts),

            N_WINDOWS,

        )



    ]



    selected = list(

        exact_day_starts.astype(

            int

        )

    )



    # --------------------------------------------------------

    # All valid continuous windows

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



        exact_day_starts,



        assume_unique=False,



    )



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



            sampled_starts.astype(

                int

            )



        )



    starts = np.asarray(

        selected,

        dtype=int,

    )



    window_types = (



        ["exact_day"]

        * len(exact_day_starts)



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



        + WINDOW_SIZE



    )



    window_demand = demand[

        start:end

    ]



    if len(

        window_demand

    ) != WINDOW_SIZE:



        raise RuntimeError(



            f"Window {window_id} has "

            f"{len(window_demand)} rows instead of "

            f"{WINDOW_SIZE}."



        )



    # --------------------------------------------------------

    # Scalar or vector kappa initialization

    # --------------------------------------------------------



    kappa_init = normalize_kappa_init(



        kappa_value,



        D,



    )



    # --------------------------------------------------------

    # Run MROO

    # --------------------------------------------------------



    result = run_mroo(



        demand=window_demand,



        eta=eta,



        kappa_init=kappa_init,



        lambda_1=lambda_1,



    )



    # --------------------------------------------------------

    # Only save aggregate costs for this window.

    #

    # We intentionally do NOT save result["kappa_history"].

    # --------------------------------------------------------



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



    # --------------------------------------------------------

    # Save the initial kappa vector itself.

    #

    # These are tiny metadata columns, not trajectories.

    # --------------------------------------------------------



    for i in range(

        D

    ):



        row[

            f"kappa_init_{i}"

        ] = float(

            kappa_init[i]

        )



    return row





# ============================================================

# COMPLETED-CONFIG HELPERS

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

                kappa_arr[i]

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

    # Normal case

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



            - set(

                completed_df.columns

            )



        )



        if missing:



            raise RuntimeError(

                "Completed-config file is missing columns: "

                f"{sorted(missing)}"

            )



        completed_df = ensure_kappa_columns(



            completed_df,



            D,



        )



        component_columns = (



            kappa_component_columns(

                D

            )



        )



        completed = set()



        for _, row in completed_df.iterrows():



            vector = [



                row[column]



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



        # ----------------------------------------------------

        # Rewrite old scalar completion files into the new

        # vector-compatible form.

        # ----------------------------------------------------



        save_completed_configs(



            completed,



            D,



        )



        return completed



    # --------------------------------------------------------

    # One-time migration from existing summary

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



            summary_df = ensure_kappa_columns(



                summary_df,



                D,



            )



            component_columns = (



                kappa_component_columns(

                    D

                )



            )



            complete_rows = summary_df[



                summary_df[

                    "n_windows"

                ].astype(

                    int

                )



                == N_WINDOWS



            ]



            for _, row in complete_rows.iterrows():



                vector = [



                    row[column]



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

# RESULT HELPERS

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



    df = ensure_kappa_columns(



        df,



        D,



    )



    if len(df) == 0:



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





# ============================================================

# MAIN

# ============================================================



def main():



    # ========================================================

    # LOAD DATA

    # ========================================================



    demand = load_demand()



    T_FULL, D = (

        demand.shape

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

    # SHARED WINDOWS

    # ========================================================



    window_starts = get_window_starts(



        T_full=T_FULL



    )



    # ========================================================

    # MROO TUNING VALUES

    #

    # Scalar values still work:

    #

    #     1.2

    #

    # means:

    #

    #     [1.2, 1.2, 1.2]

    #

    # Vector values:

    #

    #     [1.2, 0.8, 0.6]

    #

    # are used directly.

    # ========================================================



    ETA_VALUES = [

        # 0.9,
        # 0.8,1.0,1.1
        0.7,
        0.75



    ]



    KAPPA_VALUES = [
        # [0.8,1.0,1.0],
        [0.8,0.85,0.9],
        # [0.9,1.0,1.0],
        # [0.75,0.85,0.85],
        # [0.85,0.85,0.85]

        # [

        #     0.8,

        #     0.8,

        #     0.7,

        # ],



    ]



    # ========================================================

    # REMOVE DUPLICATES

    # ========================================================



    ETA_VALUES = list(



        dict.fromkeys(



            float(x)



            for x in ETA_VALUES



        )



    )



    unique_kappas = []



    seen_kappas = set()



    for value in KAPPA_VALUES:



        key = kappa_tuple(



            value,



            D,



        )



        if key not in seen_kappas:



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

            kappa_arr = normalize_kappa_init(
                kappa_value,
                D,
            )

            for lambda_1 in LAMBDA_1_VALUES:

                row = {
                    "lambda_1": float(lambda_1),
                    "eta": float(eta),
                    "kappa_init": kappa_label(
                        kappa_arr,
                        D,
                    ),
                }

                for i in range(D):

                    row[
                        f"kappa_init_{i}"
                    ] = float(
                        kappa_arr[i]
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

    # ========================================================

    # PRINT GRID

    # ========================================================



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

    # LOAD COMPLETED CONFIGS

    # ========================================================



    completed_configs = (



        load_completed_configs(

            D

        )



    )



    # --------------------------------------------------------

    # Safety:

    #

    # If completed-config file exists but results were deleted,

    # do not skip configurations.

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



    # ========================================================

    # BUILD JOBS

    # ========================================================



    jobs = []



    skipped_configs = 0



    for eta in ETA_VALUES:



        for kappa_value in KAPPA_VALUES:



            canonical_kappa = kappa_tuple(



                kappa_value,



                D,



            )



            for lambda_1 in LAMBDA_1_VALUES:



                key = config_key(



                    lambda_1,



                    eta,



                    canonical_kappa,



                    D,



                )



                if key in completed_configs:



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



                            float(

                                eta

                            ),



                            canonical_kappa,



                            float(

                                lambda_1

                            ),



                        )



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

    # LOAD EXISTING RESULTS

    # ========================================================



    if RESULT_FILE.exists():



        existing_results_df = pd.read_csv(



            RESULT_FILE,



            low_memory=False,



        )



        existing_results_df = (



            normalize_result_dataframe(



                existing_results_df,



                D,



            )



        )



        results = (



            existing_results_df



            .to_dict(

                orient="records"

            )



        )



    else:



        results = []



    # ========================================================

    # RUN ONLY NEW JOBS

    #

    # No kappa history is stored.

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



                future = executor.submit(



                    run_single_job,



                    demand,



                    D,



                    window_id,



                    start,



                    eta,



                    kappa_value,



                    lambda_1,



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



                # --------------------------------------------

                # Save after every completed window.

                # --------------------------------------------



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

    # MARK FULL CONFIGURATIONS COMPLETE

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



                for i, column in enumerate(



                    component_columns



                ):



                    mask &= np.isclose(



                        results_df[

                            column

                        ].astype(

                            float

                        ),



                        float(

                            kappa_arr[i]

                        ),



                    )



                config_count = int(



                    mask.sum()



                )



                if config_count == N_WINDOWS:



                    key = config_key(



                        lambda_1,



                        eta,



                        kappa_arr,



                        D,



                    )



                    if key not in completed_configs:



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



                elif config_count > N_WINDOWS:



                    raise RuntimeError(



                        "Too many rows found for MROO "

                        "configuration: "



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



    # --------------------------------------------------------

    # Human-readable combined kappa label

    # --------------------------------------------------------



    summary_df.insert(



        2,



        "kappa_init",



        [



            kappa_label(



                [



                    row[column]



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