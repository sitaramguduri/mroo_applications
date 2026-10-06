from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CHANGE ONLY THIS
# ============================================================

SAVE_NAME = "beta400_m100_good_v1"


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

SUMMARY_FILE = (
    SNAPSHOT_DIR
    / "summary.csv"
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
# 3. ALGORITHM ORDER
# ============================================================

ALGORITHM_ORDER = [
    "OFFLINE-OPT",
    "GREEDY",
    "S-MROO-SUM",
    "S-MROO-MAX",
    "DMD",
    "MROO",
]

ALGORITHM_LABELS = {
    "OFFLINE-OPT": "OPT",
    "GREEDY": "GREEDY",
    "S-MROO-SUM": "ROBD",
    "S-MROO-MAX": "S-MROO",
    "DMD": "DMD",
    "MROO": "DMROO",
}


# ============================================================
# 4. COSTS TO PLOT
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
# 5. LOAD DATA
# ============================================================

if not WINDOW_RESULT_FILE.exists():

    raise FileNotFoundError(
        "\nCould not find saved configuration:\n"
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
    "SAVED CONFIGURATION VIOLIN PLOTS"
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
# 6. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "algorithm",
    "window_id",
    "hitting_cost",
    "long_term_cost",
    "total_cost",
    "beta",
    "m",
}

missing_columns = (
    required_columns
    -
    set(df.columns)
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
# 8. GET EXPERIMENT INFORMATION
# ============================================================

beta_values = (
    df["beta"]
    .dropna()
    .astype(float)
    .unique()
)

m_values = (
    df["m"]
    .dropna()
    .astype(float)
    .unique()
)

if len(beta_values) != 1:

    raise RuntimeError(
        f"Expected one beta value, found {beta_values}"
    )

if len(m_values) != 1:

    raise RuntimeError(
        f"Expected one m value, found {m_values}"
    )

BETA = float(
    beta_values[0]
)

M = float(
    m_values[0]
)

BETA_M_RATIO = (
    BETA / M
)


print(
    "\n========================================"
)

print(
    "EXPERIMENT"
)

print(
    "========================================"
)

print(
    "beta   =",
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


# ============================================================
# 9. VERIFY ALGORITHMS
# ============================================================

found_algorithms = set(
    df["algorithm"]
    .dropna()
    .unique()
)

missing_algorithms = (
    set(ALGORITHM_ORDER)
    -
    found_algorithms
)

if missing_algorithms:

    raise RuntimeError(
        "\nThese algorithms are missing from the saved config:\n"
        f"{sorted(missing_algorithms)}"
    )


# ============================================================
# 10. VERIFY WINDOW COUNTS
# ============================================================

window_counts = (
    df
    .groupby(
        "algorithm"
    )[
        "window_id"
    ]
    .nunique()
)

print(
    "\n========================================"
)

print(
    "WINDOW COUNTS"
)

print(
    "========================================"
)

print(
    window_counts
)

bad_counts = (
    window_counts[
        window_counts != 100
    ]
)

if len(bad_counts) > 0:

    raise RuntimeError(
        "\nSome algorithms do not contain 100 windows:\n"
        f"{bad_counts}"
    )


# ============================================================
# 11. BUILD DATA FOR PLOTS
# ============================================================

plot_data = {}

for algorithm in ALGORITHM_ORDER:

    algorithm_df = (
        df[
            df["algorithm"]
            ==
            algorithm
        ]
        .copy()
        .sort_values(
            "window_id"
        )
        .reset_index(
            drop=True
        )
    )

    plot_data[
        algorithm
    ] = algorithm_df


# ============================================================
# 12. THREE-PANEL VIOLIN PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        16,
        5.5,
    ),
)


for (
    ax,
    (
        cost_column,
        ylabel,
    ),
) in zip(
    axes,
    COSTS,
):


    data = [

        plot_data[
            algorithm
        ][
            cost_column
        ]
        .dropna()
        .astype(float)
        .to_numpy()

        for algorithm
        in ALGORITHM_ORDER
    ]


    positions = np.arange(
        1,
        len(
            ALGORITHM_ORDER
        )
        + 1,
    )


    violin_parts = ax.violinplot(

        data,

        positions=
            positions,

        widths=
            0.8,

        showmeans=
            False,

        showmedians=
            True,

        showextrema=
            True,
    )


    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(

        [
            ALGORITHM_LABELS[
                algorithm
            ]

            for algorithm
            in ALGORITHM_ORDER
        ],

        rotation=25,

        ha="right",

        fontsize=10,
    )


    ax.set_ylabel(
        ylabel,
        fontsize=13,
        fontweight="bold",
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


fig.suptitle(

    (
        f"{SAVE_NAME}\n"
        f"beta={BETA:g}, "
        f"m={M:g}, "
        f"beta/m={BETA_M_RATIO:g}"
    ),

    fontsize=14,

    fontweight="bold",
)


fig.subplots_adjust(

    left=0.06,

    right=0.99,

    bottom=0.23,

    top=0.82,

    wspace=0.27,
)


# ============================================================
# 13. SAVE THREE-PANEL FIGURE
# ============================================================

COMBINED_PNG = (
    OUTPUT_DIR
    / (
        f"{SAVE_NAME}"
        "_violin_all_costs.png"
    )
)

COMBINED_PDF = (
    OUTPUT_DIR
    / (
        f"{SAVE_NAME}"
        "_violin_all_costs.pdf"
    )
)


fig.savefig(
    COMBINED_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    COMBINED_PDF,
    bbox_inches="tight",
)


# ============================================================
# 14. INDIVIDUAL VIOLIN PLOT FOR EACH COST
# ============================================================

individual_files = []


for (
    cost_column,
    ylabel,
) in COSTS:


    fig_single, ax = plt.subplots(

        figsize=(
            9,
            6,
        )
    )


    data = [

        plot_data[
            algorithm
        ][
            cost_column
        ]
        .dropna()
        .astype(float)
        .to_numpy()

        for algorithm
        in ALGORITHM_ORDER
    ]


    positions = np.arange(
        1,
        len(
            ALGORITHM_ORDER
        )
        + 1,
    )


    violin_parts = ax.violinplot(

        data,

        positions=
            positions,

        widths=
            0.85,

        showmeans=
            False,

        showmedians=
            True,

        showextrema=
            True,
    )


    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(

        [
            ALGORITHM_LABELS[
                algorithm
            ]

            for algorithm
            in ALGORITHM_ORDER
        ],

        rotation=25,

        ha="right",

        fontsize=11,
    )


    ax.set_ylabel(
        ylabel,
        fontsize=14,
        fontweight="bold",
    )


    ax.set_title(

        (
            f"{ylabel}\n"
            f"beta={BETA:g}, "
            f"m={M:g}, "
            f"beta/m={BETA_M_RATIO:g}"
        ),

        fontsize=14,
        fontweight="bold",
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


    fig_single.tight_layout()


    short_name = (
        cost_column
        .replace(
            "_cost",
            "",
        )
    )


    png_file = (
        OUTPUT_DIR
        / (
            f"{SAVE_NAME}"
            f"_violin_{short_name}.png"
        )
    )


    pdf_file = (
        OUTPUT_DIR
        / (
            f"{SAVE_NAME}"
            f"_violin_{short_name}.pdf"
        )
    )


    fig_single.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
    )


    fig_single.savefig(
        pdf_file,
        bbox_inches="tight",
    )


    individual_files.append(
        (
            png_file,
            pdf_file,
        )
    )


# ============================================================
# 15. PRINT SUMMARY
# ============================================================

summary = (
    df
    .groupby(
        "algorithm",
        as_index=False,
    )
    .agg(
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
)


summary[
    "_order"
] = (
    summary[
        "algorithm"
    ]
    .map(
        {
            algorithm:
                index

            for index, algorithm
            in enumerate(
                ALGORITHM_ORDER
            )
        }
    )
)


summary = (
    summary
    .sort_values(
        "_order"
    )
    .drop(
        columns=[
            "_order",
        ]
    )
    .reset_index(
        drop=True
    )
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / (
        f"{SAVE_NAME}"
        "_cost_summary.csv"
    )
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print(
    "\n========================================"
)

print(
    "COST SUMMARY"
)

print(
    "========================================"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "VIOLIN PLOTS COMPLETE"
)

print(
    "========================================"
)

print(
    "\nCombined plot:"
)

print(
    COMBINED_PNG
)

print(
    COMBINED_PDF
)

print(
    "\nIndividual plots:"
)

for (
    png_file,
    pdf_file,
) in individual_files:

    print()

    print(
        png_file
    )

    print(
        pdf_file
    )


print(
    "\nSummary:"
)

print(
    SUMMARY_OUTPUT
)


plt.show()