from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





# ============================================================

# 1. BASE PATH

# ============================================================



ALGORITHM_DIR = (

    Path(__file__)

    .resolve()

    .parent

)



BASE_RESULT_DIR = (

    ALGORITHM_DIR

    / "new_results"

    / "window_beta_m_sweep"

    / "rho_7"

    / "1min_24hour"

    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"

)





# ============================================================

# 2. CHOOSE EXPERIMENT

# ============================================================



SETTING_FOLDER = "beta_75_m_150_T_1440"





# ============================================================

# 3. CHOOSE S-MROO-SUM VERSION

#

# "selected"

#     -> use the S-MROO-SUM configuration selected by

#        main_sweep.py

#

# "lambda2_zero"

#     -> use only lambda_2 = 0 from the raw

#        smroo_sum_window_results.csv

# ============================================================



SMROO_SUM_MODE = "selected"



# ============================================================

# 4. CONSTANTS / FILES

# ============================================================



N_WINDOWS = 100



SETTING_DIR = (

    BASE_RESULT_DIR

    / SETTING_FOLDER

)



COMBINED_RESULT_FILE = (

    SETTING_DIR

    / "combined_algorithm_window_results.csv"

)



SMROO_SUM_RESULT_FILE = (

    SETTING_DIR

    / "smroo_sum_window_results.csv"

)



MROO_RESULT_FILE = (

    SETTING_DIR

    / "mroo_window_results.csv"

)



MROO_SUMMARY_FILE = (

    SETTING_DIR

    / "mroo_window_summary.csv"

)


MROO_CURRENT_RUN_CONFIG_FILE = (

    SETTING_DIR

    / "mroo_current_run_configs.csv"

)





# ============================================================

# 5. CHECK FILES

# ============================================================



if not COMBINED_RESULT_FILE.exists():



    raise FileNotFoundError(

        f"\nCould not find combined result file:\n"

        f"{COMBINED_RESULT_FILE}"

    )





if not MROO_RESULT_FILE.exists():



    raise FileNotFoundError(

        f"\nCould not find raw MROO result file:\n"

        f"{MROO_RESULT_FILE}"

    )


if not MROO_CURRENT_RUN_CONFIG_FILE.exists():



    raise FileNotFoundError(

        f"\nCould not find current-run MROO configuration file:\n"

        f"{MROO_CURRENT_RUN_CONFIG_FILE}\n"

        "Run main_sweep.py with the updated mroo_run.py first."

    )





print(

    "\n========================================"

)



print(

    "READING RESULTS"

)



print(

    "========================================"

)



print(

    COMBINED_RESULT_FILE

)





# ============================================================

# 6. LOAD COMBINED RESULTS

# ============================================================



df = pd.read_csv(

    COMBINED_RESULT_FILE

)





# ============================================================

# 7. EXTRACT BETA, M, WINDOW SIZE

# ============================================================



if "beta" not in df.columns:



    raise RuntimeError(

        "'beta' column not found."

    )





if "m" not in df.columns:



    raise RuntimeError(

        "'m' column not found."

    )





beta_values = (

    df[

        "beta"

    ]

    .astype(float)

    .unique()

)





m_values = (

    df[

        "m"

    ]

    .astype(float)

    .unique()

)





if len(beta_values) != 1:



    raise RuntimeError(

        f"Expected exactly one beta value, "

        f"found {beta_values}"

    )





if len(m_values) != 1:



    raise RuntimeError(

        f"Expected exactly one m value, "

        f"found {m_values}"

    )





BETA = float(

    beta_values[0]

)





M = float(

    m_values[0]

)





# ------------------------------------------------------------

# Window size

# ------------------------------------------------------------



if "window_size" in df.columns:



    window_values = (

        df[

            "window_size"

        ]

        .dropna()

        .astype(int)

        .unique()

    )



    if len(window_values) != 1:



        raise RuntimeError(

            f"Expected one window size, "

            f"found {window_values}"

        )



    WINDOW_SIZE = int(

        window_values[0]

    )



else:



    try:



        WINDOW_SIZE = int(

            SETTING_FOLDER

            .split(

                "_T_"

            )[-1]

        )



    except Exception:



        raise RuntimeError(

            "Could not determine window size."

        )





# ============================================================

# 1-minute resolution:

#

# 60 slots = 1 hour

# 1440 slots = 24 hours

# ============================================================



window_hours = (

    WINDOW_SIZE

    / 60.0

)





# ============================================================

# 8. OPTIONAL:

# REPLACE S-MROO-SUM WITH lambda_2 = 0

# ============================================================



if SMROO_SUM_MODE == "lambda2_zero":



    if not SMROO_SUM_RESULT_FILE.exists():



        raise FileNotFoundError(

            "\nCould not find raw S-MROO-SUM file:\n"

            f"{SMROO_SUM_RESULT_FILE}"

        )





    print(

        "\n========================================"

    )



    print(

        "FILTERING S-MROO-SUM"

    )



    print(

        "========================================"

    )





    smroo_df = pd.read_csv(

        SMROO_SUM_RESULT_FILE

    )





    if "lambda_2" not in smroo_df.columns:



        raise RuntimeError(

            "'lambda_2' column not found in "

            "smroo_sum_window_results.csv"

        )





    print(

        "\nAvailable lambda_2 values:"

    )



    print(

        sorted(

            smroo_df[

                "lambda_2"

            ]

            .dropna()

            .astype(float)

            .unique()

        )

    )





    smroo_zero_df = smroo_df[

        np.isclose(

            smroo_df[

                "lambda_2"

            ].astype(float),

            0.0,

        )

    ].copy()





    print(

        "\nOriginal S-MROO-SUM rows:",

        len(smroo_df)

    )





    print(

        "lambda_2 = 0 rows:",

        len(smroo_zero_df)

    )





    if len(smroo_zero_df) == 0:



        raise RuntimeError(

            "No S-MROO-SUM rows with "

            "lambda_2 = 0 were found."

        )





    smroo_zero_df[

        "beta"

    ] = BETA





    smroo_zero_df[

        "m"

    ] = M





    smroo_zero_df[

        "beta_over_m"

    ] = (

        BETA / M

    )





    smroo_zero_df[

        "algorithm"

    ] = "S-MROO-SUM"





    if "window_size" not in smroo_zero_df.columns:



        smroo_zero_df[

            "window_size"

        ] = WINDOW_SIZE





    # Remove currently selected S-MROO-SUM.

    df = df[

        df[

            "algorithm"

        ]

        != "S-MROO-SUM"

    ].copy()





    # Add lambda_2 = 0 version.

    df = pd.concat(

        [

            df,

            smroo_zero_df,

        ],

        ignore_index=True,

        sort=False,

    )





elif SMROO_SUM_MODE == "selected":



    print(

        "\nUsing S-MROO-SUM already stored in "

        "combined_algorithm_window_results.csv"

    )





else:



    raise ValueError(

        "SMROO_SUM_MODE must be either "

        "'selected' or 'lambda2_zero'."

    )





# ============================================================

# 9. PRINT EXPERIMENT INFORMATION

# ============================================================



print(

    "\n========================================"

)



print(

    "SELECTED EXPERIMENT"

)



print(

    "========================================"

)





print(

    f"folder          = {SETTING_FOLDER}"

)



print(

    f"beta            = {BETA:g}"

)



print(

    f"m               = {M:g}"

)



print(

    f"beta/m          = {BETA / M:g}"

)



print(

    f"window size     = {WINDOW_SIZE}"

)



print(

    f"window          = {window_hours:g} hours"

)



print(

    f"S-MROO-SUM mode = {SMROO_SUM_MODE}"

)





print(

    "\nRows by algorithm:"

)



print(

    df[

        "algorithm"

    ]

    .value_counts()

)





# ============================================================

# 10. ALGORITHM ORDER

# ============================================================



PLOT_ORDER = [

    "OFFLINE-OPT",

    "GREEDY",

    "S-MROO-SUM",

    "S-MROO-MAX",

    "DMD",

    "MROO",

]





# ============================================================

# 11. ALGORITHM LABELS

# ============================================================



if SMROO_SUM_MODE == "lambda2_zero":



    smroo_sum_label = (

        "S-MROO\nSUM\n"

        r"$\lambda_2=0$"

    )



else:



    smroo_sum_label = (

        "S-MROO\nSUM"

    )





ALGORITHM_LABELS = {



    "OFFLINE-OPT":

        "OPT",



    "GREEDY":

        "GRD",



    "S-MROO-SUM":

        smroo_sum_label,



    "S-MROO-MAX":

        "S-MROO\nMAX",



    "DMD":

        "DMD",



    "MROO":

        "MROO",



}





# ------------------------------------------------------------

# Only keep algorithms that actually exist.

# ------------------------------------------------------------



available_algorithms = set(

    df[

        "algorithm"

    ]

    .unique()

)





PLOT_ORDER = [

    algorithm

    for algorithm

    in PLOT_ORDER

    if algorithm

    in available_algorithms

]





if len(

    PLOT_ORDER

) == 0:



    raise RuntimeError(

        "No expected algorithms found."

    )





print(

    "\nPlotting algorithms:"

)



print(

    PLOT_ORDER

)





# ============================================================

# 12. VERIFY NUMBER OF WINDOWS

# ============================================================



for algorithm in PLOT_ORDER:



    algorithm_df = df[

        df[

            "algorithm"

        ]

        == algorithm

    ]





    count = len(

        algorithm_df

    )





    print(

        f"{algorithm}: {count} rows"

    )





    if count != N_WINDOWS:



        print(

            f"WARNING: "

            f"{algorithm} has {count} rows, "

            f"expected {N_WINDOWS}."

        )





# ============================================================

# 13. VERIFY SAME WINDOWS

# ============================================================



required_window_columns = [

    "window_id",

    "start_index",

    "end_index",

]





if all(

    column in df.columns

    for column

    in required_window_columns

):



    reference_windows = None





    for algorithm in PLOT_ORDER:



        algorithm_windows = (

            df[

                df[

                    "algorithm"

                ]

                == algorithm

            ][

                required_window_columns

            ]

            .sort_values(

                "window_id"

            )

            .reset_index(

                drop=True

            )

        )





        if reference_windows is None:



            reference_windows = (

                algorithm_windows

            )



        elif not algorithm_windows.equals(

            reference_windows

        ):



            raise RuntimeError(

                "Algorithms did not use "

                "the same windows."

            )





    print(

        "\nVerified: all algorithms "

        "use the same windows."

    )





# ============================================================

# 14. ALGORITHM COST SUMMARY

# ============================================================



summary = (

    df[

        df[

            "algorithm"

        ]

        .isin(

            PLOT_ORDER

        )

    ]



    .groupby(

        "algorithm"

    )



    .agg(



        mean_hitting=(

            "hitting_cost",

            "mean",

        ),



        std_hitting=(

            "hitting_cost",

            "std",

        ),



        mean_long_term=(

            "long_term_cost",

            "mean",

        ),



        std_long_term=(

            "long_term_cost",

            "std",

        ),



        mean_total=(

            "total_cost",

            "mean",

        ),



        std_total=(

            "total_cost",

            "std",

        ),



    )



    .reindex(

        PLOT_ORDER

    )

)





print(

    "\n========================================"

)



print(

    "ALGORITHM COST SUMMARY"

)



print(

    "========================================"

)



print(

    summary.to_string()

)





# ============================================================

# 15. COSTS TO PLOT

# ============================================================



COSTS = [



    (

        "hitting_cost",

        "Hitting Cost",

    ),



    (

        "long_term_cost",

        "Long-Term Cost",

    ),



    (

        "total_cost",

        "Total Cost",

    ),



]





# ============================================================

# 16. ALGORITHM COMPARISON FIGURE

# ============================================================



fig, axes = plt.subplots(

    1,

    3,

    figsize=(

        15.5,

        4.8,

    ),

)





# ============================================================

# 17. DRAW ALGORITHM BOXPLOTS

# ============================================================



for ax, (

    cost_column,

    ylabel,

) in zip(

    axes,

    COSTS,

):



    box_data = [



        df[

            df[

                "algorithm"

            ]

            == algorithm

        ][

            cost_column

        ]

        .dropna()

        .to_numpy()



        for algorithm

        in PLOT_ORDER



    ]





    bp = ax.boxplot(

        box_data,



        tick_labels=[

            ALGORITHM_LABELS[

                algorithm

            ]

            for algorithm

            in PLOT_ORDER

        ],



        patch_artist=True,



        showmeans=False,



        widths=0.58,



        medianprops={

            "linewidth": 1.6,

            "color": "black",

        },



        whiskerprops={

            "linewidth": 1.1,

            "color": "black",

        },



        capprops={

            "linewidth": 1.1,

            "color": "black",

        },



        boxprops={

            "linewidth": 1.1,

            "color": "black",

        },



        flierprops={

            "marker": "o",

            "markersize": 2.5,

            "markerfacecolor": "none",

            "markeredgecolor": "black",

            "markeredgewidth": 0.6,

        },

    )





    colors = (

        plt.rcParams[

            "axes.prop_cycle"

        ]

        .by_key()[

            "color"

        ]

    )





    for index, box in enumerate(

        bp[

            "boxes"

        ]

    ):



        box.set_facecolor(

            colors[

                index

                % len(colors)

            ]

        )



        box.set_alpha(

            0.42

        )





    ax.set_ylabel(

        ylabel,

        fontsize=14,

        fontweight="bold",

    )





    ax.tick_params(

        axis="x",

        labelsize=10.5,

        width=1.2,

        length=5,

    )





    for label in ax.get_xticklabels():



        label.set_fontweight(

            "bold"

        )



        label.set_rotation(

            0

        )





    ax.tick_params(

        axis="y",

        labelsize=10,

        width=1.2,

        length=4,

    )





    for label in ax.get_yticklabels():



        label.set_fontweight(

            "bold"

        )





    ax.grid(

        axis="y",

        linestyle="--",

        linewidth=0.7,

        alpha=0.45,

    )





    ax.set_axisbelow(

        True

    )





    for spine in ax.spines.values():



        spine.set_linewidth(

            1.25

        )





# ============================================================

# 18. ALGORITHM PLOT SPACING

# ============================================================



fig.subplots_adjust(

    left=0.065,

    right=0.99,

    bottom=0.22,

    top=0.97,

    wspace=0.28,

)





# ============================================================

# 19. ALGORITHM PLOT OUTPUT FILE

# ============================================================



if SMROO_SUM_MODE == "lambda2_zero":



    smroo_output_label = (

        "smroo_sum_lambda2_zero"

    )



else:



    smroo_output_label = (

        "smroo_sum_selected"

    )





ALGORITHM_OUTPUT_FILE = (

    SETTING_DIR

    / (

        f"boxplot_all_costs"

        f"_beta_{BETA:g}"

        f"_m_{M:g}"

        f"_T_{WINDOW_SIZE}"

        f"_{smroo_output_label}.png"

    )

)





# ============================================================

# 20. SAVE ALGORITHM COMPARISON

# ============================================================



fig.savefig(

    ALGORITHM_OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight",

)





print(

    "\n========================================"

)



print(

    "ALGORITHM COMPARISON PLOT SAVED"

)



print(

    "========================================"

)



print(

    ALGORITHM_OUTPUT_FILE

)





# ============================================================

# 21. LOAD ALL RAW MROO TUNING RESULTS

# ============================================================



mroo_df = pd.read_csv(

    MROO_RESULT_FILE

)





required_mroo_columns = {



    "window_id",



    "lambda_1",



    "eta",



    "kappa_init",



    "hitting_cost",



    "long_term_cost",



    "total_cost",



}





missing = (

    required_mroo_columns

    - set(

        mroo_df.columns

    )

)





if missing:



    raise RuntimeError(

        "MROO result file is missing columns: "

        f"{sorted(missing)}"

    )





# ============================================================

# 22. AVAILABLE MROO PARAMETERS

# ============================================================



lambda_values = sorted(

    mroo_df[

        "lambda_1"

    ]

    .dropna()

    .astype(float)

    .unique()

)





eta_values = sorted(

    mroo_df[

        "eta"

    ]

    .dropna()

    .astype(float)

    .unique()

)





# ============================================================

# IMPORTANT CHANGE:

#

# kappa_init is now allowed to be an array-like string:

#

#     [1.2,0.8,0.6]

#

# Therefore it must NOT be converted to float.

# ============================================================



kappa_values = sorted(

    mroo_df[

        "kappa_init"

    ]

    .dropna()

    .astype(str)

    .unique()

)





print(

    "\n========================================"

)



print(

    "MROO TUNING VALUES"

)



print(

    "========================================"

)





print(

    "lambda_1 values =",

    lambda_values,

)





print(

    "eta values =",

    eta_values,

)





print(

    "kappa_init values =",

    kappa_values,

)





# ============================================================

# 23. BUILD UNIQUE MROO CONFIGURATIONS

# ============================================================



configurations = (

    mroo_df[

        [

            "lambda_1",

            "eta",

            "kappa_init",

        ]

    ]



    .drop_duplicates()



    .sort_values(

        [

            "lambda_1",

            "eta",

            "kappa_init",

        ]

    )



    .reset_index(

        drop=True

    )

)





print(

    "\nNumber of MROO configurations =",

    len(configurations),

)





# ============================================================

# 24. CHECK EACH CONFIGURATION HAS 100 WINDOWS

# ============================================================



for _, config in configurations.iterrows():



    lambda_1 = float(

        config[

            "lambda_1"

        ]

    )





    eta = float(

        config[

            "eta"

        ]

    )





    # --------------------------------------------------------

    # IMPORTANT CHANGE:

    #

    # kappa_init is now a string representation of a vector.

    # --------------------------------------------------------



    kappa = str(

        config[

            "kappa_init"

        ]

    )





    mask = (



        np.isclose(

            mroo_df[

                "lambda_1"

            ].astype(float),



            lambda_1,

        )



        &



        np.isclose(

            mroo_df[

                "eta"

            ].astype(float),



            eta,

        )



        &



        (

            mroo_df[

                "kappa_init"

            ].astype(str)



            == kappa

        )



    )





    count = int(

        mask.sum()

    )





    print(

        f"lambda_1={lambda_1:.6g}, "

        f"eta={eta:.6g}, "

        f"kappa={kappa}: "

        f"{count} windows"

    )





    if count != N_WINDOWS:



        print(

            "WARNING: expected "

            f"{N_WINDOWS} windows."

        )





# ============================================================

# 25. MROO TUNING SUMMARY

# ============================================================



mroo_summary = (

    mroo_df



    .groupby(

        [

            "lambda_1",

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



    )



    .sort_values(

        "mean_total_cost"

    )



    .reset_index(

        drop=True

    )

)





print(

    "\n========================================"

)



print(

    "MROO TUNING SUMMARY"

)



print(

    "========================================"

)





print(

    mroo_summary.to_string(

        index=False

    )

)





# ============================================================

# 26. PRINT BEST MROO CONFIGURATION

# ============================================================



best_mroo = (

    mroo_summary

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

    best_mroo[

        "lambda_1"

    ],

)





print(

    "eta =",

    best_mroo[

        "eta"

    ],

)





print(

    "kappa_init =",

    best_mroo[

        "kappa_init"

    ],

)





print(

    "mean hitting cost =",

    best_mroo[

        "mean_hitting_cost"

    ],

)





print(

    "mean long-term cost =",

    best_mroo[

        "mean_long_term_cost"

    ],

)





print(

    "mean total cost =",

    best_mroo[

        "mean_total_cost"

    ],

)





# ============================================================

# COMPARE CURRENT-RUN CONFIGURATIONS WITH BEST PREVIOUS CONFIG

# ============================================================



def kappa_components(value):

    """Convert scalar/vector kappa labels to numeric vectors."""

    import ast



    parsed = ast.literal_eval(str(value))

    return np.asarray(

        parsed,

        dtype=float,

    ).reshape(-1)





def same_kappa_values(

    value_a,

    value_b,

):

    a = kappa_components(

        value_a

    )



    b = kappa_components(

        value_b

    )



    # Scalar kappa means the same initial value in every dimension.

    if a.size == 1 and b.size > 1:

        a = np.full(

            b.size,

            a[0],

        )



    elif b.size == 1 and a.size > 1:

        b = np.full(

            a.size,

            b[0],

        )



    return (

        a.shape == b.shape

        and

        bool(

            np.allclose(

                a,

                b,

                rtol=1e-9,

                atol=1e-12,

            )

        )

    )





def configuration_mask(

    frame,

    config,

):

    return (

        np.isclose(

            frame[

                "lambda_1"

            ].astype(float),

            float(

                config[

                    "lambda_1"

                ]

            ),

            rtol=1e-9,

            atol=1e-12,

        )

        &

        np.isclose(

            frame[

                "eta"

            ].astype(float),

            float(

                config[

                    "eta"

                ]

            ),

            rtol=1e-9,

            atol=1e-12,

        )

        &

        frame[

            "kappa_init"

        ].map(

            lambda value:

                same_kappa_values(

                    value,

                    config[

                        "kappa_init"

                    ],

                )

        )

    )





# ------------------------------------------------------------

# Read configurations requested in the latest MROO run.

# ------------------------------------------------------------



current_run_configs = pd.read_csv(

    MROO_CURRENT_RUN_CONFIG_FILE,

    low_memory=False,

)



required_current_config_columns = {

    "lambda_1",

    "eta",

    "kappa_init",

}



missing_current_columns = (

    required_current_config_columns

    - set(

        current_run_configs.columns

    )

)



if missing_current_columns:



    raise RuntimeError(

        "Current-run MROO configuration file is missing columns: "

        f"{sorted(missing_current_columns)}"

    )





if len(current_run_configs) == 0:



    raise RuntimeError(

        "Current-run MROO configuration file is empty."

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

    current_run_configs.to_string(

        index=False

    )

)





# ------------------------------------------------------------

# Match every current-run configuration to its saved summary.

# ------------------------------------------------------------



current_summary_rows = []



for _, requested_config in current_run_configs.iterrows():



    matches = mroo_summary[

        configuration_mask(

            mroo_summary,

            requested_config,

        )

    ]



    if len(matches) == 0:



        raise RuntimeError(

            "\nA configuration from mroo_current_run_configs.csv "

            "has no saved MROO summary result:\n"

            f"{requested_config.to_dict()}"

        )





    if len(matches) > 1:



        raise RuntimeError(

            "\nA current-run configuration matched multiple "

            "MROO summary rows:\n"

            f"{requested_config.to_dict()}"

        )





    matched_row = matches.iloc[0]



    if int(

        matched_row[

            "n_windows"

        ]

    ) != N_WINDOWS:



        raise RuntimeError(

            "\nCurrent-run MROO configuration is incomplete:\n"

            f"{requested_config.to_dict()}\n"

            f"Found {int(matched_row['n_windows'])} windows "

            f"instead of {N_WINDOWS}."

        )





    current_summary_rows.append(

        matched_row

    )





current_configs = pd.DataFrame(

    current_summary_rows

).reset_index(

    drop=True

)





# ------------------------------------------------------------

# Find configurations that were NOT requested in this run.

# Since mroo_summary is sorted by mean_total_cost, the first

# remaining row is the best previous saved configuration.

# ------------------------------------------------------------



is_current_run = np.zeros(

    len(mroo_summary),

    dtype=bool,

)



for _, config in current_configs.iterrows():



    is_current_run |= (

        configuration_mask(

            mroo_summary,

            config,

        )

        .to_numpy()

    )





previous_configs = (

    mroo_summary[

        ~is_current_run

    ]

    .copy()

    .sort_values(

        "mean_total_cost"

    )

    .reset_index(

        drop=True

    )

)





# ------------------------------------------------------------

# Plot:

#   best previous saved configuration

#   + every configuration from the latest MROO run

# ------------------------------------------------------------



if len(previous_configs) > 0:



    best_previous = (

        previous_configs

        .iloc[[0]]

        .copy()

    )



    configurations = pd.concat(

        [

            best_previous,

            current_configs,

        ],

        ignore_index=True,

    )



    comparison_roles = [

        "Best previous"

    ] + [

        f"Current run {i + 1}"

        for i in range(

            len(current_configs)

        )

    ]



else:



    configurations = (

        current_configs.copy()

    )



    comparison_roles = [

        f"Current run {i + 1}"

        for i in range(

            len(current_configs)

        )

    ]





# ------------------------------------------------------------

# Verify all compared configurations are complete and use

# exactly the same windows.

# ------------------------------------------------------------



reference_windows = None



for _, config in configurations.iterrows():



    selected = mroo_df[

        configuration_mask(

            mroo_df,

            config,

        )

    ]



    window_columns = [

        column

        for column

        in required_window_columns

        if column in selected.columns

    ]



    windows = (

        selected[

            window_columns

        ]

        .sort_values(

            "window_id"

        )

        .reset_index(

            drop=True

        )

    )



    if (

        selected[

            "window_id"

        ].duplicated().any()

        or

        len(selected) != N_WINDOWS

    ):



        raise RuntimeError(

            "MROO comparison requires one result per window "

            "and all expected windows."

        )





    if (

        reference_windows is not None

        and

        not windows.equals(

            reference_windows

        )

    ):



        raise RuntimeError(

            "Compared MROO configurations did not use "

            "the same windows."

        )





    reference_windows = windows





print(

    "\n========================================"

)



print(

    "MROO CONFIGURATIONS BEING COMPARED"

)



print(

    "========================================"

)



print(

    configurations.to_string(

        index=False

    )

)





# ============================================================

# 27. LABEL EACH TUNING CONFIGURATION

# ============================================================



def tuning_label(

    lambda_1,

    eta,

    kappa,

):



    return (

        rf"$\lambda_1$={lambda_1:.3g}"

        "\n"

        rf"$\eta$={eta:.3g}"

        "\n"

        rf"$\kappa$={kappa}"

    )





config_labels = []





for role, (_, config) in zip(comparison_roles, configurations.iterrows()):



    config_labels.append(



        role + "\n" + tuning_label(



            float(

                config[

                    "lambda_1"

                ]

            ),



            float(

                config[

                    "eta"

                ]

            ),



            str(

                config[

                    "kappa_init"

                ]

            ),



        )



    )





# ============================================================

# 28. MROO COSTS

# ============================================================



MROO_COSTS = [



    (

        "hitting_cost",

        "Hitting Cost",

    ),



    (

        "long_term_cost",

        "Long-Term Cost",

    ),



    (

        "total_cost",

        "Total Cost",

    ),



]





# ============================================================

# 29. MROO TUNING FIGURE

# ============================================================



figure_width = max(

    15.5,

    1.4

    * len(

        configurations

    ),

)





fig_tune, axes_tune = plt.subplots(

    1,

    3,

    figsize=(

        figure_width,

        6.0,

    ),

)





# ============================================================

# 30. DRAW THE MROO CONFIGURATION COMPARISON

# ============================================================



for ax, (

    cost_column,

    ylabel,

) in zip(

    axes_tune,

    MROO_COSTS,

):



    box_data = []





    for _, config in configurations.iterrows():



        lambda_1 = float(

            config[

                "lambda_1"

            ]

        )





        eta = float(

            config[

                "eta"

            ]

        )





        mask = configuration_mask(

            mroo_df,

            config,

        )





        config_data = (

            mroo_df[

                mask

            ][

                cost_column

            ]



            .dropna()



            .to_numpy()

        )





        box_data.append(

            config_data

        )





    bp = ax.boxplot(



        box_data,



        tick_labels=config_labels,



        patch_artist=True,



        showmeans=False,



        widths=0.58,



        medianprops={

            "linewidth": 1.6,

            "color": "black",

        },



        whiskerprops={

            "linewidth": 1.1,

            "color": "black",

        },



        capprops={

            "linewidth": 1.1,

            "color": "black",

        },



        boxprops={

            "linewidth": 1.1,

            "color": "black",

        },



        flierprops={

            "marker": "o",

            "markersize": 2.5,

            "markerfacecolor": "none",

            "markeredgecolor": "black",

            "markeredgewidth": 0.6,

        },



    )





    colors = (

        plt.rcParams[

            "axes.prop_cycle"

        ]

        .by_key()[

            "color"

        ]

    )





    for index, box in enumerate(

        bp[

            "boxes"

        ]

    ):



        box.set_facecolor(

            colors[

                index

                % len(colors)

            ]

        )





        box.set_alpha(

            0.42

        )





    ax.set_ylabel(

        ylabel,

        fontsize=14,

        fontweight="bold",

    )





    ax.tick_params(

        axis="x",

        labelsize=8,

        width=1.2,

        length=5,

    )





    for label in ax.get_xticklabels():



        label.set_rotation(

            45

        )



        label.set_ha(

            "right"

        )





    ax.tick_params(

        axis="y",

        labelsize=10,

        width=1.2,

        length=4,

    )





    for label in ax.get_yticklabels():



        label.set_fontweight(

            "bold"

        )





    ax.grid(

        axis="y",

        linestyle="--",

        linewidth=0.7,

        alpha=0.45,

    )





    ax.set_axisbelow(

        True

    )





    for spine in ax.spines.values():



        spine.set_linewidth(

            1.25

        )





# ============================================================

# 31. MROO TUNING FIGURE TITLE / SPACING

# ============================================================



fig_tune.suptitle(

    (

        "MROO Hyperparameter Tuning "

        f"(current run vs. best previous configuration; "

        f"beta={BETA:g}, "

        f"m={M:g}, "

        f"T={WINDOW_SIZE})"

    ),

    fontsize=15,

    fontweight="bold",

)





fig_tune.subplots_adjust(

    left=0.06,

    right=0.99,

    bottom=0.36,

    top=0.88,

    wspace=0.28,

)





# ============================================================

# 32. SAVE MROO TUNING PLOT

# ============================================================



MROO_TUNING_OUTPUT_FILE = (

    SETTING_DIR

    / (

        f"mroo_tuning_current_run_vs_best_eta_kappa"

        f"_beta_{BETA:g}"

        f"_m_{M:g}"

        f"_T_{WINDOW_SIZE}.png"

    )

)





fig_tune.savefig(

    MROO_TUNING_OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight",

)





print(

    "\n========================================"

)



print(

    "MROO TUNING PLOT SAVED"

)



print(

    "========================================"

)



print(

    MROO_TUNING_OUTPUT_FILE

)





# ============================================================

# 33. SHOW BOTH FIGURES

# ============================================================



plt.show()
