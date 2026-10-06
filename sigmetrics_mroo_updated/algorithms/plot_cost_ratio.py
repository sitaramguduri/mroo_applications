from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHANGE ONLY THIS
# ============================================================

SAVE_NAME = "beta400_m100_good_v1"


# ------------------------------------------------------------
# Cost used for the ratio.
#
# Recommended:
#     "total_cost"
#
# You can also use:
#     "hitting_cost"
#     "long_term_cost"
# ------------------------------------------------------------

COST_COLUMN = "total_cost"


# ============================================================
# 2. PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


BASE_RESULT_DIR = (
    ALGORITHM_DIR
    / "matrix_memory_v1"
    / "1min_24hour"
    / "matrix_memory"
)


SAVED_CONFIG_DIR = (
    BASE_RESULT_DIR
    / "saved_plot_configurations"
)


SNAPSHOT_DIR = (
    SAVED_CONFIG_DIR
    / "snapshots"
    / SAVE_NAME
)


WINDOW_RESULT_FILE = (
    SNAPSHOT_DIR
    / "window_results.csv"
)


OUTPUT_DIR = (
    SNAPSHOT_DIR
    / "plots"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. ONLINE ALGORITHMS
#
# IMPORTANT:
#
# These are INTERNAL names stored in the CSV.
# Do not replace these with ROBD / S-MROO / DMROO here.
# ============================================================

ONLINE_ALGORITHM_ORDER = [

    "GREEDY",

    "S-MROO-SUM",

    "S-MROO-MAX",

    "DMD",

    "MROO",
]


# ============================================================
# 4. DISPLAY NAMES
# ============================================================

ALGORITHM_LABELS = {

    "GREEDY":
        "GREEDY",

    "S-MROO-SUM":
        "ROBD",

    "S-MROO-MAX":
        "S-MROO",

    "DMD":
        "DMD",

    "MROO":
        "DMROO",
}


# ============================================================
# 5. LOAD DATA
# ============================================================

if not WINDOW_RESULT_FILE.exists():

    raise FileNotFoundError(
        "\nCould not find:\n"
        f"{WINDOW_RESULT_FILE}\n\n"
        "Check SAVE_NAME."
    )


df = pd.read_csv(
    WINDOW_RESULT_FILE,
    low_memory=False,
)


print(
    "\n========================================"
)

print(
    "ONLINE / OFFLINE-OPT COST RATIO"
)

print(
    "========================================"
)


print(
    "\nSAVE_NAME:"
)

print(
    SAVE_NAME
)


print(
    "\nReading:"
)

print(
    WINDOW_RESULT_FILE
)


# ============================================================
# 6. VALIDATION
# ============================================================

required_columns = {

    "algorithm",

    "window_id",

    "beta",

    "m",

    COST_COLUMN,
}


missing_columns = (
    required_columns
    -
    set(
        df.columns
    )
)


if missing_columns:

    raise RuntimeError(
        "\nMissing required columns:\n"
        f"{sorted(missing_columns)}"
    )


# ============================================================
# 7. REMOVE DUPLICATE WINDOWS
# ============================================================

df = (

    df

    .drop_duplicates(

        subset=[
            "algorithm",
            "window_id",
        ],

        keep="last",
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 8. EXPERIMENT INFORMATION
# ============================================================

beta_values = (

    df[
        "beta"
    ]

    .dropna()

    .astype(float)

    .unique()
)


m_values = (

    df[
        "m"
    ]

    .dropna()

    .astype(float)

    .unique()
)


if len(beta_values) != 1:

    raise RuntimeError(
        "Expected exactly one beta value, "
        f"found {beta_values}."
    )


if len(m_values) != 1:

    raise RuntimeError(
        "Expected exactly one m value, "
        f"found {m_values}."
    )


BETA = float(
    beta_values[0]
)


M = float(
    m_values[0]
)


BETA_M_RATIO = (
    BETA
    /
    M
)


print(
    "\nbeta   =",
    BETA,
)


print(
    "m      =",
    M,
)


print(
    "beta/m =",
    BETA_M_RATIO,
)


print(
    "cost   =",
    COST_COLUMN,
)


# ============================================================
# 9. VERIFY REQUIRED ALGORITHMS EXIST
# ============================================================

required_algorithms = (
    set(
        ONLINE_ALGORITHM_ORDER
    )
    |
    {
        "OFFLINE-OPT"
    }
)


found_algorithms = set(

    df[
        "algorithm"
    ]

    .dropna()

    .unique()
)


missing_algorithms = (
    required_algorithms
    -
    found_algorithms
)


if missing_algorithms:

    raise RuntimeError(
        "\nMissing algorithms:\n"
        f"{sorted(missing_algorithms)}"
    )


# ============================================================
# 10. GET OFFLINE OPTIMUM
#
# There should be one OPT cost for every window.
# ============================================================

opt_df = (

    df[
        df[
            "algorithm"
        ]
        ==
        "OFFLINE-OPT"
    ]

    [
        [
            "window_id",
            COST_COLUMN,
        ]
    ]

    .copy()

    .rename(
        columns={
            COST_COLUMN:
                "offline_opt_cost"
        }
    )

    .sort_values(
        "window_id"
    )

    .reset_index(
        drop=True
    )
)


if (
    opt_df[
        "window_id"
    ].nunique()
    !=
    100
):

    raise RuntimeError(
        "\nOFFLINE-OPT does not contain exactly "
        "100 unique windows."
    )


# ============================================================
# 11. OPT COST MUST BE POSITIVE
#
# Necessary because it is the denominator.
# ============================================================

if np.any(

    opt_df[
        "offline_opt_cost"
    ]
    <=
    0.0

):

    bad_rows = (

        opt_df[
            opt_df[
                "offline_opt_cost"
            ]
            <=
            0.0
        ]
    )


    raise RuntimeError(
        "\nOFFLINE-OPT contains non-positive costs. "
        "Cannot form cost ratios:\n\n"
        f"{bad_rows.to_string(index=False)}"
    )


# ============================================================
# 12. COMPUTE WINDOW-BY-WINDOW RATIOS
#
# ratio =
#
#     online algorithm cost
#     ---------------------
#        OPT cost
#
# for THE SAME window.
# ============================================================

ratio_frames = []


for algorithm in ONLINE_ALGORITHM_ORDER:


    algorithm_df = (

        df[
            df[
                "algorithm"
            ]
            ==
            algorithm
        ]

        [
            [
                "window_id",
                COST_COLUMN,
            ]
        ]

        .copy()

        .rename(
            columns={
                COST_COLUMN:
                    "algorithm_cost"
            }
        )
    )


    # --------------------------------------------------------
    # Pair algorithm and OPT using window_id.
    # --------------------------------------------------------

    merged = algorithm_df.merge(

        opt_df,

        on=
            "window_id",

        how=
            "inner",

        validate=
            "one_to_one",
    )


    if len(
        merged
    ) != 100:

        raise RuntimeError(
            f"\n{algorithm}: expected 100 matched "
            f"algorithm/OPT windows but found "
            f"{len(merged)}."
        )


    merged[
        "cost_ratio"
    ] = (

        merged[
            "algorithm_cost"
        ]

        /

        merged[
            "offline_opt_cost"
        ]
    )


    merged[
        "algorithm"
    ] = (
        algorithm
    )


    merged[
        "algorithm_label"
    ] = (
        ALGORITHM_LABELS[
            algorithm
        ]
    )


    merged[
        "beta"
    ] = (
        BETA
    )


    merged[
        "m"
    ] = (
        M
    )


    merged[
        "beta_m_ratio"
    ] = (
        BETA_M_RATIO
    )


    ratio_frames.append(
        merged
    )


ratio_df = pd.concat(

    ratio_frames,

    ignore_index=True,
)


# ============================================================
# 13. VERIFY BASIC RATIO PROPERTIES
# ============================================================

if not np.all(

    np.isfinite(
        ratio_df[
            "cost_ratio"
        ]
    )

):

    raise RuntimeError(
        "\nNon-finite cost ratios were produced."
    )


# ------------------------------------------------------------
# Because OFFLINE-OPT should be the minimum offline objective,
# online total-cost ratios should normally be >= 1.
#
# We print a warning instead of crashing, because tiny solver
# tolerances can occasionally give values slightly below 1.
# ------------------------------------------------------------

below_one = (

    ratio_df[
        ratio_df[
            "cost_ratio"
        ]
        <
        1.0 - 1e-6
    ]
)


if (
    COST_COLUMN
    ==
    "total_cost"
    and
    len(
        below_one
    )
    >
    0
):

    print(
        "\nWARNING:"
    )

    print(
        len(
            below_one
        ),
        "window ratios are meaningfully below 1."
    )

    print(
        "Check OFFLINE-OPT convergence for those windows."
    )


# ============================================================
# 14. BUILD PLOT DATA
# ============================================================

plot_data = []


for algorithm in ONLINE_ALGORITHM_ORDER:

    values = (

        ratio_df[
            ratio_df[
                "algorithm"
            ]
            ==
            algorithm
        ]

        [
            "cost_ratio"
        ]

        .astype(float)

        .to_numpy()
    )


    if len(
        values
    ) != 100:

        raise RuntimeError(
            f"{algorithm}: expected 100 ratios, "
            f"found {len(values)}."
        )


    plot_data.append(
        values
    )


positions = np.arange(

    1,

    len(
        ONLINE_ALGORITHM_ORDER
    )
    +
    1,
)


labels = [

    ALGORITHM_LABELS[
        algorithm
    ]

    for algorithm
    in ONLINE_ALGORITHM_ORDER
]


# ============================================================
# 15. BOXPLOT
# ============================================================

fig_box, ax_box = plt.subplots(

    figsize=(
        9,
        6,
    )
)


ax_box.boxplot(

    plot_data,

    positions=
        positions,

    widths=
        0.60,

    patch_artist=
        True,

    showmeans=
        False,

    showfliers=
        True,
)


# ------------------------------------------------------------
# Ratio = 1 reference line
# ------------------------------------------------------------

ax_box.axhline(

    y=1.0,

    linestyle="--",

    linewidth=1.2,

    label="Offline optimum",
)


ax_box.set_xticks(
    positions
)


ax_box.set_xticklabels(

    labels,

    fontsize=11,
)


ax_box.set_ylabel(

    r"Cost Ratio $C_{\mathrm{ALG}} / C_{\mathrm{OPT}}$",

    fontsize=13,

    fontweight="bold",
)


ax_box.set_xlabel(

    "Online Algorithm",

    fontsize=13,

    fontweight="bold",
)


ax_box.set_title(

    (
        "Online-to-Offline Cost Ratio\n"
        f"$\\beta/m={BETA_M_RATIO:g}$ "
        f"($\\beta={BETA:g}$, $m={M:g}$)"
    ),

    fontsize=14,

    fontweight="bold",
)


ax_box.grid(

    axis="y",

    linestyle="--",

    linewidth=0.7,

    alpha=0.45,
)


ax_box.set_axisbelow(
    True
)


fig_box.tight_layout()


# ============================================================
# 16. SAVE BOXPLOT
# ============================================================

cost_name = (
    COST_COLUMN
    .replace(
        "_cost",
        "",
    )
)


BOX_PNG_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_boxplot.png"
    )
)


BOX_PDF_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_boxplot.pdf"
    )
)


fig_box.savefig(

    BOX_PNG_FILE,

    dpi=300,

    bbox_inches="tight",
)


fig_box.savefig(

    BOX_PDF_FILE,

    bbox_inches="tight",
)


# ============================================================
# 17. VIOLIN PLOT
# ============================================================

fig_violin, ax_violin = plt.subplots(

    figsize=(
        9,
        6,
    )
)


ax_violin.violinplot(

    plot_data,

    positions=
        positions,

    widths=
        0.80,

    showmeans=
        False,

    showmedians=
        True,

    showextrema=
        True,
)


# ------------------------------------------------------------
# Ratio = 1 reference line
# ------------------------------------------------------------

ax_violin.axhline(

    y=1.0,

    linestyle="--",

    linewidth=1.2,

    label="Offline optimum",
)


ax_violin.set_xticks(
    positions
)


ax_violin.set_xticklabels(

    labels,

    fontsize=11,
)


ax_violin.set_ylabel(

    r"Cost Ratio $C_{\mathrm{ALG}} / C_{\mathrm{OPT}}$",

    fontsize=13,

    fontweight="bold",
)


ax_violin.set_xlabel(

    "Online Algorithm",

    fontsize=13,

    fontweight="bold",
)


ax_violin.set_title(

    (
        "Online-to-Offline Cost Ratio Distribution\n"
        f"$\\beta/m={BETA_M_RATIO:g}$ "
        f"($\\beta={BETA:g}$, $m={M:g}$)"
    ),

    fontsize=14,

    fontweight="bold",
)


ax_violin.grid(

    axis="y",

    linestyle="--",

    linewidth=0.7,

    alpha=0.45,
)


ax_violin.set_axisbelow(
    True
)


fig_violin.tight_layout()


# ============================================================
# 18. SAVE VIOLIN PLOT
# ============================================================

VIOLIN_PNG_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_violin.png"
    )
)


VIOLIN_PDF_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_violin.pdf"
    )
)


fig_violin.savefig(

    VIOLIN_PNG_FILE,

    dpi=300,

    bbox_inches="tight",
)


fig_violin.savefig(

    VIOLIN_PDF_FILE,

    bbox_inches="tight",
)


# ============================================================
# 19. RATIO SUMMARY
# ============================================================

summary_rows = []


for algorithm in ONLINE_ALGORITHM_ORDER:

    values = (

        ratio_df[
            ratio_df[
                "algorithm"
            ]
            ==
            algorithm
        ]

        [
            "cost_ratio"
        ]

        .astype(float)
    )


    summary_rows.append(
        {
            "algorithm":
                ALGORITHM_LABELS[
                    algorithm
                ],

            "internal_algorithm":
                algorithm,

            "beta":
                BETA,

            "m":
                M,

            "beta_m_ratio":
                BETA_M_RATIO,

            "n_windows":
                len(
                    values
                ),

            "mean_cost_ratio":
                values.mean(),

            "std_cost_ratio":
                values.std(),

            "median_cost_ratio":
                values.median(),

            "min_cost_ratio":
                values.min(),

            "max_cost_ratio":
                values.max(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


print(
    "\n========================================"
)

print(
    "COST RATIO SUMMARY"
)

print(
    "========================================"
)


print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE RATIO DATA
#
# Save BOTH:
#
# 1. all 500 window-level ratios
# 2. algorithm summary
#
# This makes it easy to later create cross-beta/m figures.
# ============================================================

RATIO_WINDOW_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_window_results.csv"
    )
)


RATIO_SUMMARY_FILE = (

    OUTPUT_DIR

    /

    (
        f"{SAVE_NAME}"
        f"_{cost_name}"
        "_cost_ratio_summary.csv"
    )
)


ratio_df.to_csv(

    RATIO_WINDOW_FILE,

    index=False,
)


summary_df.to_csv(

    RATIO_SUMMARY_FILE,

    index=False,
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "COST RATIO PLOTS COMPLETE"
)

print(
    "========================================"
)


print(
    "\nBoxplot:"
)

print(
    BOX_PNG_FILE
)

print(
    BOX_PDF_FILE
)


print(
    "\nViolin plot:"
)

print(
    VIOLIN_PNG_FILE
)

print(
    VIOLIN_PDF_FILE
)


print(
    "\nWindow-level ratios:"
)

print(
    RATIO_WINDOW_FILE
)


print(
    "\nRatio summary:"
)

print(
    RATIO_SUMMARY_FILE
)


plt.show()