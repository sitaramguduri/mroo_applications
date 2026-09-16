"""Compare OPT, MROO, ROBD-SUM, ROBD-MAX, and GREEDY."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greedy import run_greedy
from mroo import load_exposures, run_mroo
from offline_opt import run_offline_opt
from robd_max import run_robd_max
from robd_sum import run_robd_sum


# ==================================================
# Files
# ==================================================

PROJECT_DIRECTORY = Path(__file__).resolve().parent

TRACE_FILE = (
    PROJECT_DIRECTORY
    / "processed_powerdata"
    / "cell_a_power_trace.csv"
)

MROO_PARAMETER_FILE = (
    PROJECT_DIRECTORY
    / "results"
    / "mroo_tuning"
    / "best_parameters.json"
)

ROBD_SUM_PARAMETER_FILE = (
    PROJECT_DIRECTORY
    / "results"
    / "robd_sum_tuning"
    / "best_parameters.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "algorithm_comparison_with_opt"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

COST_FILE = (
    OUTPUT_DIRECTORY
    / "window_costs.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIRECTORY
    / "cost_summary.csv"
)

PLOT_FILE = (
    OUTPUT_DIRECTORY
    / "five_algorithm_boxplots.png"
)


# ==================================================
# Experiment settings
# ==================================================

# Five-minute measurements:
# 24 hours * 12 measurements per hour.
WINDOW_SIZE = 24 * 12

NUMBER_OF_WINDOWS = 100
WINDOW_SEED = 25

# Final 20% of the trace provides test windows.
COMPARISON_FRACTION = 0.20

# Timestamp interval in microseconds.
EXPECTED_INTERVAL = 5 * 60 * 1_000_000

# NeurIPS paper parameters for ROBD-MAX.
ROBD_MAX_LAMBDA_1 = 0.3
ROBD_MAX_LAMBDA_2 = 0.8

# Set to False only to recreate plots from a
# previously completed window_costs.csv.
RUN_ALGORITHMS = True

# Numerical tolerance used to verify OPT.
OPTIMALITY_TOLERANCE = 1e-6


# ==================================================
# Validation helper
# ==================================================

def require_equal(
    name,
    left,
    right,
):
    left_array = np.asarray(
        left,
        dtype=float,
    )

    right_array = np.asarray(
        right,
        dtype=float,
    )

    if not np.allclose(
        left_array,
        right_array,
    ):
        raise ValueError(
            f"{name} differs between "
            "MROO and ROBD-SUM settings"
        )


# ==================================================
# Load and sort trace
# ==================================================

trace = (
    pd.read_csv(TRACE_FILE)
    .sort_values("time")
    .reset_index(drop=True)
)

if len(trace) < WINDOW_SIZE:
    raise ValueError(
        "The complete trace is shorter than "
        "one 24-hour window"
    )


# ==================================================
# Load MROO parameters
# ==================================================

with MROO_PARAMETER_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    mroo_parameters = json.load(file)

best_eta = float(
    mroo_parameters["eta"]
)

best_kappa = float(
    mroo_parameters[
        "kappa_initial_scalar"
    ]
)

mroo_fixed = mroo_parameters[
    "fixed_parameters"
]


# ==================================================
# Load ROBD-SUM parameters
# ==================================================

with ROBD_SUM_PARAMETER_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    robd_sum_parameters = json.load(file)

robd_sum_lambda_1 = float(
    robd_sum_parameters["lambda_1"]
)

robd_sum_lambda_2 = float(
    robd_sum_parameters["lambda_2"]
)

robd_sum_fixed = robd_sum_parameters[
    "fixed_parameters"
]


# ==================================================
# Common application parameters
# ==================================================

budget = float(
    mroo_fixed["budget"]
)

upper = np.asarray(
    mroo_fixed["upper_bounds"],
    dtype=float,
)

u_initial = np.asarray(
    mroo_fixed["u_initial"],
    dtype=float,
)

mroo_lambda_1 = float(
    mroo_fixed["lambda_1"]
)

mroo_lambda_2 = float(
    mroo_fixed["lambda_2"]
)

rho = float(
    mroo_fixed["rho"]
)

entity_count = int(
    mroo_fixed["entities"]
)

exposure_seed_value = mroo_fixed.get(
    "exposure_seed"
)

if exposure_seed_value is None:
    exposure_seed = 7
else:
    exposure_seed = int(
        exposure_seed_value
    )


# ==================================================
# Verify common settings
# ==================================================

require_equal(
    "budget",
    budget,
    robd_sum_fixed["budget"],
)

require_equal(
    "upper bounds",
    upper,
    robd_sum_fixed["upper_bounds"],
)

require_equal(
    "u_initial",
    u_initial,
    robd_sum_fixed["u_initial"],
)

require_equal(
    "rho",
    rho,
    robd_sum_fixed["rho"],
)

if entity_count != int(
    robd_sum_fixed["entities"]
):
    raise ValueError(
        "The number of affected entities "
        "differs between algorithms"
    )


# ==================================================
# Construct common exposure sequence
# ==================================================

exposure_file_value = mroo_fixed.get(
    "exposure_file"
)

if exposure_file_value is None:
    exposure_file = None
else:
    exposure_file = Path(
        exposure_file_value
    )

full_exposures = load_exposures(
    path=exposure_file,
    horizon=len(trace),
    dimension=entity_count,
    seed=exposure_seed,
)

if len(full_exposures) != len(trace):
    raise RuntimeError(
        "Exposure sequence length does not "
        "match the trace"
    )

# Your present experiment requires constant exposure.
if not np.allclose(
    full_exposures,
    full_exposures[0],
):
    raise RuntimeError(
        "Exposure is not constant over time"
    )


# ==================================================
# Select final 20%
# ==================================================

comparison_start = int(
    len(trace)
    * (1.0 - COMPARISON_FRACTION)
)

comparison_trace = (
    trace.iloc[comparison_start:]
    .reset_index(drop=True)
)

comparison_exposures = (
    full_exposures[comparison_start:]
)

timestamps = comparison_trace[
    "time"
].to_numpy(dtype=np.int64)

if len(timestamps) < WINDOW_SIZE:
    raise ValueError(
        "The comparison trace is shorter than "
        "one 24-hour window"
    )


# ==================================================
# Find all continuous 24-hour windows
# ==================================================

valid_edges = (
    np.diff(timestamps)
    == EXPECTED_INTERVAL
).astype(int)

required_edges = WINDOW_SIZE - 1

continuous_edge_counts = np.convolve(
    valid_edges,
    np.ones(
        required_edges,
        dtype=int,
    ),
    mode="valid",
)

valid_start_positions = np.where(
    continuous_edge_counts
    == required_edges
)[0]

if (
    len(valid_start_positions)
    < NUMBER_OF_WINDOWS
):
    raise ValueError(
        f"Only {len(valid_start_positions)} "
        "continuous windows are available; "
        f"{NUMBER_OF_WINDOWS} were requested"
    )

rng = np.random.default_rng(
    WINDOW_SEED
)

selected_starts = np.sort(
    rng.choice(
        valid_start_positions,
        size=NUMBER_OF_WINDOWS,
        replace=False,
    )
)


# ==================================================
# Run algorithms
# ==================================================

if RUN_ALGORITHMS:
    print("\nExperiment setup")
    print("----------------")

    print(
        "Constant exposure vector:",
        full_exposures[0],
    )

    print(
        f"Selected {len(selected_starts)} "
        "continuous 24-hour windows"
    )

    print(
        f"MROO: eta={best_eta:g}, "
        f"kappa_1={best_kappa:g}, "
        f"lambda_1={mroo_lambda_1:g}, "
        f"lambda_2={mroo_lambda_2:g}"
    )

    print(
        "ROBD-SUM: "
        f"lambda_1={robd_sum_lambda_1:g}, "
        f"lambda_2={robd_sum_lambda_2:g}"
    )

    print(
        "ROBD-MAX: "
        f"lambda_1={ROBD_MAX_LAMBDA_1:g}, "
        f"lambda_2={ROBD_MAX_LAMBDA_2:g}"
    )

    comparison_rows = []

    for window_number, start in enumerate(
        selected_starts,
        start=1,
    ):
        start = int(start)
        end = start + WINDOW_SIZE

        window_trace = (
            comparison_trace
            .iloc[start:end]
            .reset_index(drop=True)
        )

        window_exposures = (
            comparison_exposures[
                start:end
            ]
        )

        start_time = int(
            window_trace[
                "time"
            ].iloc[0]
        )

        end_time = int(
            window_trace[
                "time"
            ].iloc[-1]
        )

        # ------------------------------------------
        # Offline optimal
        # ------------------------------------------

        _, offline_summary = run_offline_opt(
            trace=window_trace,
            exposures=window_exposures,
            u_initial=u_initial,
            budget=budget,
            upper=upper,
            rho=rho,
        )

        # ------------------------------------------
        # MROO
        # ------------------------------------------

        _, mroo_summary = run_mroo(
            trace=window_trace,
            exposures=window_exposures,
            u_initial=u_initial,
            budget=budget,
            upper=upper,
            eta=best_eta,
            lambda_1=mroo_lambda_1,
            lambda_2=mroo_lambda_2,
            rho=rho,
            kappa_initial=np.full(
                entity_count,
                best_kappa,
                dtype=float,
            ),
        )

        # ------------------------------------------
        # ROBD-SUM
        # ------------------------------------------

        _, robd_sum_summary = (
            run_robd_sum(
                trace=window_trace,
                exposures=window_exposures,
                u_initial=u_initial,
                budget=budget,
                upper=upper,
                lambda_1=(
                    robd_sum_lambda_1
                ),
                lambda_2=(
                    robd_sum_lambda_2
                ),
                rho=rho,
            )
        )

        # ------------------------------------------
        # ROBD-MAX
        # ------------------------------------------

        _, robd_max_summary = (
            run_robd_max(
                trace=window_trace,
                exposures=window_exposures,
                u_initial=u_initial,
                budget=budget,
                upper=upper,
                lambda_1=(
                    ROBD_MAX_LAMBDA_1
                ),
                lambda_2=(
                    ROBD_MAX_LAMBDA_2
                ),
                rho=rho,
            )
        )

        # ------------------------------------------
        # GREEDY
        # ------------------------------------------

        _, greedy_summary = run_greedy(
            trace=window_trace,
            exposures=window_exposures,
            u_initial=u_initial,
            budget=budget,
            upper=upper,
            rho=rho,
        )

        # ------------------------------------------
        # Verify OPT lower bound
        # ------------------------------------------

        offline_cost = float(
            offline_summary[
                "total_objective"
            ]
        )

        online_summaries = {
            "MROO": mroo_summary,
            "ROBD-SUM": robd_sum_summary,
            "ROBD-MAX": robd_max_summary,
            "GREEDY": greedy_summary,
        }

        for algorithm, summary in (
            online_summaries.items()
        ):
            online_cost = float(
                summary[
                    "total_objective"
                ]
            )

            if (
                offline_cost
                > online_cost
                + OPTIMALITY_TOLERANCE
            ):
                raise RuntimeError(
                    f"Window {window_number}: "
                    f"OPT cost "
                    f"{offline_cost:.10f} exceeds "
                    f"{algorithm} cost "
                    f"{online_cost:.10f}"
                )

        # ------------------------------------------
        # Store this window
        # ------------------------------------------

        for algorithm, summary in (
            ("OPT", offline_summary),
            ("MROO", mroo_summary),
            ("ROBD-SUM", robd_sum_summary),
            ("ROBD-MAX", robd_max_summary),
            ("GREEDY", greedy_summary),
        ):
            comparison_rows.append(
                {
                    "window": window_number,
                    "start_index": start,
                    "start_time": start_time,
                    "end_time": end_time,
                    "algorithm": algorithm,
                    "hitting_cost": float(
                        summary[
                            "average_hitting_cost"
                        ]
                    ),
                    "long_term_cost": float(
                        summary[
                            "long_term_cost"
                        ]
                    ),
                    "total_cost": float(
                        summary[
                            "total_objective"
                        ]
                    ),
                }
            )

        # Save after every completed window.
        # An interruption loses at most one window.
        pd.DataFrame(
            comparison_rows
        ).to_csv(
            COST_FILE,
            index=False,
        )

        print(
            f"[{window_number:>3}/"
            f"{NUMBER_OF_WINDOWS}] "
            f"completed; "
            f"OPT={offline_cost:.8f}"
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

else:
    if not COST_FILE.exists():
        raise FileNotFoundError(
            f"Cannot plot because "
            f"{COST_FILE} does not exist"
        )

    comparison = pd.read_csv(
        COST_FILE
    )


# ==================================================
# Validate stored comparison
# ==================================================

expected_algorithms = {
    "OPT",
    "MROO",
    "ROBD-SUM",
    "ROBD-MAX",
    "GREEDY",
}

present_algorithms = set(
    comparison[
        "algorithm"
    ].unique()
)

if present_algorithms != expected_algorithms:
    raise ValueError(
        "window_costs.csv contains "
        f"{sorted(present_algorithms)}, "
        f"expected "
        f"{sorted(expected_algorithms)}. "
        "Delete or rename the old four-algorithm "
        "window_costs.csv and rerun with "
        "RUN_ALGORITHMS=True."
    )

algorithm_counts = (
    comparison[
        "algorithm"
    ].value_counts()
)

for algorithm in expected_algorithms:
    count = int(
        algorithm_counts.get(
            algorithm,
            0,
        )
    )

    if count != NUMBER_OF_WINDOWS:
        raise ValueError(
            f"{algorithm} has {count} window "
            f"results; expected "
            f"{NUMBER_OF_WINDOWS}"
        )


# ==================================================
# Cost summary
# ==================================================

summary_table = (
    comparison
    .groupby("algorithm")[
        [
            "hitting_cost",
            "long_term_cost",
            "total_cost",
        ]
    ]
    .agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    )
)

summary_table.to_csv(
    SUMMARY_FILE
)

print("\nCost summary")
print("------------")
print(summary_table)


# ==================================================
# Final lower-bound verification
# ==================================================

pivoted_costs = comparison.pivot(
    index="window",
    columns="algorithm",
    values="total_cost",
)

for algorithm in (
    "MROO",
    "ROBD-SUM",
    "ROBD-MAX",
    "GREEDY",
):
    violations = (
        pivoted_costs["OPT"]
        > pivoted_costs[algorithm]
        + OPTIMALITY_TOLERANCE
    )

    if violations.any():
        bad_windows = (
            pivoted_costs.index[
                violations
            ].tolist()
        )

        raise RuntimeError(
            f"OPT exceeds {algorithm} on "
            f"windows {bad_windows}"
        )


# ==================================================
# Box plots
# ==================================================

algorithm_order = [
    "OPT",
    "MROO",
    "ROBD-SUM",
    "ROBD-MAX",
    "GREEDY",
]

metrics = [
    (
        "hitting_cost",
        "Hitting Cost",
    ),
    (
        "long_term_cost",
        "Long-term Cost",
    ),
    (
        "total_cost",
        "Total Cost",
    ),
]

colors = [
    "#D9D9D9",
    "#8FB8DE",
    "#8FD19E",
    "#F2B36D",
    "#C7A6D8",
]

figure, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(17, 5),
)

for axis, (
    column,
    title,
) in zip(
    axes,
    metrics,
):
    values = [
        comparison.loc[
            comparison["algorithm"]
            == algorithm,
            column,
        ].to_numpy(dtype=float)
        for algorithm
        in algorithm_order
    ]

    boxplot = axis.boxplot(
        values,
        tick_labels=algorithm_order,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.4,
        },
        whiskerprops={
            "linewidth": 1.1,
        },
        capprops={
            "linewidth": 1.1,
        },
        flierprops={
            "marker": ".",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3,
            "alpha": 0.6,
        },
    )

    for box, color in zip(
        boxplot["boxes"],
        colors,
    ):
        box.set_facecolor(color)
        box.set_alpha(0.85)

    axis.set_title(
        title,
        fontweight="bold",
    )

    axis.set_ylabel(title)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    axis.set_axisbelow(True)

figure.suptitle(
    "Cost distributions over 100 continuous "
    "24-hour PDU-demand windows",
    fontweight="bold",
)

figure.tight_layout()

figure.savefig(
    PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ==================================================
# Saved files
# ==================================================

print("\nSaved:")
print(COST_FILE)
print(SUMMARY_FILE)
print(PLOT_FILE)