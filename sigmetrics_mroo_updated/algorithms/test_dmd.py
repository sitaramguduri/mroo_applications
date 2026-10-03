from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
#
# CHANGE ONLY THIS FOLDER IF NEEDED
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
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)

SETTING_FOLDER = "beta_180_m_15_T_17280"

SETTING_DIR = (
    BASE_RESULT_DIR
    / SETTING_FOLDER
)


COMBINED_FILE = (
    SETTING_DIR
    / "combined_algorithm_window_results.csv"
)

DMD_KAPPA_FILE = (
    SETTING_DIR
    / "dmd_window_kappa.csv"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

if not COMBINED_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n{COMBINED_FILE}"
    )


if not DMD_KAPPA_FILE.exists():

    raise FileNotFoundError(
        f"Could not find:\n{DMD_KAPPA_FILE}"
    )


# ============================================================
# 3. LOAD RESULTS
# ============================================================

combined = pd.read_csv(
    COMBINED_FILE
)

kappa_df = pd.read_csv(
    DMD_KAPPA_FILE
)


print("\n========================================")
print("FILES")
print("========================================")

print(
    "Combined results:",
    COMBINED_FILE,
)

print(
    "DMD kappa:",
    DMD_KAPPA_FILE,
)


# ============================================================
# 4. CHECK ALGORITHMS PRESENT
# ============================================================

print("\n========================================")
print("ALGORITHMS")
print("========================================")

print(
    combined["algorithm"]
    .value_counts()
)


required_algorithms = {
    "DMD",
    "OFFLINE-OPT",
}


missing = (
    required_algorithms
    - set(
        combined[
            "algorithm"
        ].unique()
    )
)


if missing:

    raise RuntimeError(
        f"Missing algorithms: {missing}"
    )


# ============================================================
# 5. GENERAL COST SUMMARY
# ============================================================

summary = (
    combined
    .groupby(
        "algorithm"
    )
    .agg(

        n_windows=(
            "window_id",
            "count",
        ),

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
)


print("\n========================================")
print("COST SUMMARY")
print("========================================")

print(
    summary.to_string()
)


# ============================================================
# 6. DMD VS OPT WINDOW-BY-WINDOW
# ============================================================

opt = (
    combined[
        combined[
            "algorithm"
        ]
        == "OFFLINE-OPT"
    ]
    .sort_values(
        "window_id"
    )
    .reset_index(
        drop=True
    )
)


dmd = (
    combined[
        combined[
            "algorithm"
        ]
        == "DMD"
    ]
    .sort_values(
        "window_id"
    )
    .reset_index(
        drop=True
    )
)


# Verify exact same windows

window_columns = [
    "window_id",
    "start_index",
    "end_index",
]


if not opt[
    window_columns
].equals(
    dmd[
        window_columns
    ]
):

    raise RuntimeError(
        "DMD and OPT did not use "
        "the same windows."
    )


print("\nVerified: DMD and OPT use the same windows.")


# ============================================================
# 7. DMD - OPT TOTAL COST GAP
# ============================================================

total_difference = (
    dmd[
        "total_cost"
    ].to_numpy()

    -

    opt[
        "total_cost"
    ].to_numpy()
)


print("\n========================================")
print("DMD VS OPT")
print("========================================")


print(
    "Minimum DMD - OPT total cost:",
    total_difference.min(),
)


print(
    "Maximum DMD - OPT total cost:",
    total_difference.max(),
)


print(
    "Mean DMD - OPT total cost:",
    total_difference.mean(),
)


print(
    "Median DMD - OPT total cost:",
    np.median(
        total_difference
    ),
)


n_dmd_better = int(
    np.sum(
        total_difference
        < -1e-6
    )
)


print(
    "Number of windows where "
    "DMD total < OPT total:",
    n_dmd_better,
)


if n_dmd_better > 0:

    bad_indices = np.where(
        total_difference
        < -1e-6
    )[0]


    print(
        "\nWARNING: DMD beats OPT on "
        f"{n_dmd_better} windows."
    )


    print(
        "Corresponding window IDs:"
    )


    print(
        dmd.iloc[
            bad_indices
        ][
            "window_id"
        ].to_list()
    )


else:

    print(
        "\nPASS: DMD never beats OPT."
    )


# ============================================================
# 8. BUILD PER-WINDOW DMD VS OPT TABLE
# ============================================================

comparison_df = pd.DataFrame({

    "window_id":
        dmd[
            "window_id"
        ],

    "start_index":
        dmd[
            "start_index"
        ],

    "end_index":
        dmd[
            "end_index"
        ],

    "opt_hitting":
        opt[
            "hitting_cost"
        ],

    "dmd_hitting":
        dmd[
            "hitting_cost"
        ],

    "opt_long_term":
        opt[
            "long_term_cost"
        ],

    "dmd_long_term":
        dmd[
            "long_term_cost"
        ],

    "opt_total":
        opt[
            "total_cost"
        ],

    "dmd_total":
        dmd[
            "total_cost"
        ],

})


comparison_df[
    "dmd_minus_opt"
] = (

    comparison_df[
        "dmd_total"
    ]

    -

    comparison_df[
        "opt_total"
    ]

)


comparison_df[
    "dmd_over_opt"
] = (

    comparison_df[
        "dmd_total"
    ]

    /

    comparison_df[
        "opt_total"
    ]

)


print("\n========================================")
print("DMD / OPT RATIO")
print("========================================")


print(
    comparison_df[
        "dmd_over_opt"
    ].describe()
)


# ============================================================
# 9. DMD LONG-TERM COST SANITY CHECK
# ============================================================

print("\n========================================")
print("DMD LONG-TERM COST")
print("========================================")


print(
    dmd[
        "long_term_cost"
    ].describe()
)


print(
    "\nMean DMD long-term / "
    "mean DMD hitting =",
    dmd[
        "long_term_cost"
    ].mean()
    /
    dmd[
        "hitting_cost"
    ].mean(),
)


# ============================================================
# 10. KAPPA COLUMNS
# ============================================================

kappa_columns = [

    column

    for column
    in kappa_df.columns

    if column.startswith(
        "kappa_"
    )

    and column
    != "kappa_init"

]


if len(
    kappa_columns
) == 0:

    raise RuntimeError(
        "No kappa trajectory columns found."
    )


print("\n========================================")
print("KAPPA COLUMNS")
print("========================================")

print(
    kappa_columns
)


# ============================================================
# 11. KAPPA SUMMARY OVER ALL WINDOWS
# ============================================================

kappa_values = (
    kappa_df[
        kappa_columns
    ]
    .to_numpy(
        dtype=float
    )
)


print("\n========================================")
print("GLOBAL KAPPA SUMMARY")
print("========================================")


print(
    "Mean kappa by dimension:"
)

print(
    np.mean(
        kappa_values,
        axis=0,
    )
)


print(
    "\nMax kappa by dimension:"
)

print(
    np.max(
        kappa_values,
        axis=0,
    )
)


print(
    "\nMin kappa by dimension:"
)

print(
    np.min(
        kappa_values,
        axis=0,
    )
)


# ============================================================
# 12. FINAL KAPPA PER WINDOW
# ============================================================

final_kappa_rows = (
    kappa_df
    .sort_values(
        [
            "window_id",
            "t",
        ]
    )
    .groupby(
        "window_id"
    )
    .tail(1)
    .sort_values(
        "window_id"
    )
)


print("\n========================================")
print("FINAL KAPPA ACROSS WINDOWS")
print("========================================")


print(
    final_kappa_rows[
        kappa_columns
    ]
    .describe()
)


print(
    "\nMean final kappa:"
)

print(
    final_kappa_rows[
        kappa_columns
    ]
    .mean()
    .to_numpy()
)


# ============================================================
# 13. KAPPA GROWTH
#
# Compare initial and final dual variables.
# ============================================================

first_kappa_rows = (
    kappa_df
    .sort_values(
        [
            "window_id",
            "t",
        ]
    )
    .groupby(
        "window_id"
    )
    .head(1)
    .sort_values(
        "window_id"
    )
)


initial_kappa = (
    first_kappa_rows[
        kappa_columns
    ]
    .to_numpy(
        dtype=float
    )
)


final_kappa = (
    final_kappa_rows[
        kappa_columns
    ]
    .to_numpy(
        dtype=float
    )
)


kappa_growth = (
    final_kappa
    - initial_kappa
)


print("\n========================================")
print("KAPPA GROWTH")
print("========================================")


print(
    "Mean growth:"
)

print(
    np.mean(
        kappa_growth,
        axis=0,
    )
)


print(
    "\nMaximum growth:"
)

print(
    np.max(
        kappa_growth,
        axis=0,
    )
)


# ============================================================
# 14. CHECK WHETHER KAPPA MOSTLY INCREASES
# ============================================================

diff_count = 0

increase_count = 0

decrease_count = 0


for window_id, group in (
    kappa_df
    .sort_values(
        [
            "window_id",
            "t",
        ]
    )
    .groupby(
        "window_id"
    )
):


    values = (
        group[
            kappa_columns
        ]
        .to_numpy(
            dtype=float
        )
    )


    differences = np.diff(
        values,
        axis=0,
    )


    diff_count += (
        differences.size
    )


    increase_count += int(
        np.sum(
            differences > 1e-12
        )
    )


    decrease_count += int(
        np.sum(
            differences < -1e-12
        )
    )


print("\n========================================")
print("KAPPA DIRECTION")
print("========================================")


print(
    "Number of increasing entries:",
    increase_count,
)


print(
    "Number of decreasing entries:",
    decrease_count,
)


print(
    "Fraction increasing:",
    increase_count
    / diff_count,
)


print(
    "Fraction decreasing:",
    decrease_count
    / diff_count,
)


# ============================================================
# 15. SAVE DMD VS OPT COMPARISON
# ============================================================

OUTPUT_FILE = (
    SETTING_DIR
    / "dmd_vs_opt_diagnostic.csv"
)


comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n========================================")
print("SAVED")
print("========================================")

print(
    OUTPUT_FILE
)