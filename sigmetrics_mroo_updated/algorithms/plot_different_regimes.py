from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

BASE_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "different_regime_weights"
)

PLOT_DIR = (
    BASE_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

SLOT_SECONDS = 10

ONLINE_ALGORITHMS = [
    "Greedy",
    "MROO",
    "S-MROO-MAX",
    "S-MROO-SUM",
]


# ============================================================
# FIND ALL REGIME-LENGTH FOLDERS
# ============================================================

regime_dirs = sorted(
    BASE_DIR.glob(
        "regime_length_*_eps_0p1"
    )
)


if len(regime_dirs) == 0:

    raise FileNotFoundError(
        f"No regime-length folders found in:\n{BASE_DIR}"
    )


print("\nFound regime folders:")

for folder in regime_dirs:
    print(" ", folder.name)


# ============================================================
# LOAD ALL RESULTS
# ============================================================

all_results = []


for folder in regime_dirs:

    # --------------------------------------------------------
    # Parse regime length from folder name:
    #
    # regime_length_540_eps_0p1
    # -> 540
    # --------------------------------------------------------

    parts = folder.name.split("_")

    regime_length = int(
        parts[2]
    )


    result_file = (
        folder
        / "beta_m_all_results.csv"
    )


    if not result_file.exists():

        print(
            f"Skipping {folder.name}: "
            f"no beta_m_all_results.csv"
        )

        continue


    df = pd.read_csv(
        result_file
    )


    # --------------------------------------------------------
    # Keep only selected/best configuration
    # --------------------------------------------------------

    if "selected_best" in df.columns:

        df = df[
            df["selected_best"] == True
        ].copy()


    # --------------------------------------------------------
    # Add regime information
    # --------------------------------------------------------

    df["regime_length"] = (
        regime_length
    )

    df["regime_hours"] = (
        regime_length
        * SLOT_SECONDS
        / 3600.0
    )


    all_results.append(
        df
    )


if len(all_results) == 0:

    raise RuntimeError(
        "No result files were loaded."
    )


results = pd.concat(
    all_results,
    ignore_index=True,
)


# ============================================================
# KEEP ONLINE ALGORITHMS
# ============================================================

results = results[
    results["algorithm"].isin(
        ONLINE_ALGORITHMS
    )
].copy()


results = results.sort_values(
    [
        "regime_hours",
        "algorithm",
    ]
)


# ============================================================
# PRINT DATA USED
# ============================================================

print(
    "\nResults used for plotting:\n"
)

print(
    results[
        [
            "regime_length",
            "regime_hours",
            "algorithm",
            "cost_ratio_to_opt",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FIGURE 1
#
# RAW COST/OPT VS REGIME LENGTH
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


for algorithm in ONLINE_ALGORITHMS:

    alg_df = (
        results[
            results["algorithm"]
            == algorithm
        ]
        .sort_values(
            "regime_hours"
        )
    )


    if len(alg_df) == 0:
        continue


    ax.plot(
        alg_df["regime_hours"],
        alg_df["cost_ratio_to_opt"],
        marker="o",
        linewidth=2,
        markersize=6,
        label=algorithm,
    )


# ------------------------------------------------------------
# Log scale makes short regimes easier to inspect
# ------------------------------------------------------------

ax.set_xscale(
    "log"
)


# ------------------------------------------------------------
# Use exactly the regime lengths present in the data
# ------------------------------------------------------------

regime_hours = sorted(
    results[
        "regime_hours"
    ].unique()
)


ax.set_xticks(
    regime_hours
)

ax.set_xticklabels(
    [
        f"{value:g} h"
        for value in regime_hours
    ]
)


ax.set_xlabel(
    "Memory-weight regime length"
)

ax.set_ylabel(
    "Cost / OPT"
)

ax.set_title(
    "Effect of Memory-Weight Switching Frequency on Performance"
)


ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()


fig.tight_layout()


OUTPUT_FILE = (
    PLOT_DIR
    / "cost_ratio_vs_regime_length_raw.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


plt.close(
    fig
)


# ============================================================
# FIGURE 2
#
# COST/OPT GAP RELATIVE TO MROO
# ============================================================

pivot = (
    results
    .pivot_table(
        index="regime_hours",
        columns="algorithm",
        values="cost_ratio_to_opt",
        aggfunc="first",
    )
    .sort_index()
)


if "MROO" not in pivot.columns:

    raise RuntimeError(
        "MROO results are missing."
    )


BASELINES = [
    "Greedy",
    "S-MROO-MAX",
    "S-MROO-SUM",
]


fig, ax = plt.subplots(
    figsize=(8, 5.5)
)


for algorithm in BASELINES:

    if algorithm not in pivot.columns:
        continue


    gap = (
        pivot[algorithm]
        - pivot["MROO"]
    )


    ax.plot(
        pivot.index,
        gap,
        marker="o",
        linewidth=2,
        markersize=6,
        label=algorithm,
    )


ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    color="black",
)


ax.set_xscale(
    "log"
)


ax.set_xticks(
    pivot.index
)

ax.set_xticklabels(
    [
        f"{value:g} h"
        for value in pivot.index
    ]
)


ax.set_xlabel(
    "Memory-weight regime length"
)

ax.set_ylabel(
    "Cost/OPT gap relative to MROO"
)

ax.set_title(
    "Performance Gap Relative to MROO"
)


ax.grid(
    which="both",
    alpha=0.25,
)

ax.legend()


fig.tight_layout()


OUTPUT_FILE = (
    PLOT_DIR
    / "cost_ratio_gap_to_mroo_vs_regime_length.png"
)


fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)


print(
    "Saved:"
)

print(
    OUTPUT_FILE
)


plt.close(
    fig
)


# ============================================================
# PRINT GAP TABLE
# ============================================================

print(
    "\n========================================"
)

print(
    "COST/OPT GAP RELATIVE TO MROO"
)

print(
    "========================================"
)


gap_table = pd.DataFrame(
    index=pivot.index
)


for algorithm in BASELINES:

    if algorithm in pivot.columns:

        gap_table[
            algorithm
        ] = (
            pivot[algorithm]
            - pivot["MROO"]
        )


gap_table.index.name = (
    "regime_hours"
)


print(
    gap_table.to_string()
)


print(
    "\nDone."
)