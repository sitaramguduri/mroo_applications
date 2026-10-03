from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROJECT PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ALGORITHM_DIR.parent


# ============================================================
# IMPORT THE SAME v_t CALCULATION USED BY MROO
# ============================================================

from mroo import compute_v_t


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
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "demand_variation_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_FILE)

demand = data[
    DEMAND_COLUMNS
].to_numpy(
    dtype=float
)

T, D = demand.shape

print("\n========================================")
print("DEMAND VARIATION ANALYSIS")
print("========================================")
print("Data file:", DATA_FILE)
print("T =", T)
print("D =", D)


# ============================================================
# 1. DEMAND VARIATION
#
# Delta_y(t) = || y_t - y_{t-1} ||_2
# ============================================================

delta_y_vectors = (
    demand[1:]
    - demand[:-1]
)

delta_y = np.linalg.norm(
    delta_y_vectors,
    axis=1,
)


# ============================================================
# DEMAND STATISTICS
# ============================================================

print("\n----------------------------------------")
print("DEMAND VARIATION")
print("Delta_y(t) = ||y_t - y_{t-1}||_2")
print("----------------------------------------")

print(
    "Mean   =",
    np.mean(delta_y),
)

print(
    "Median =",
    np.median(delta_y),
)

print(
    "Std    =",
    np.std(delta_y),
)

print(
    "Min    =",
    np.min(delta_y),
)

print(
    "Max    =",
    np.max(delta_y),
)

print(
    "25%    =",
    np.percentile(delta_y, 25),
)

print(
    "75%    =",
    np.percentile(delta_y, 75),
)

print(
    "90%    =",
    np.percentile(delta_y, 90),
)

print(
    "95%    =",
    np.percentile(delta_y, 95),
)

print(
    "99%    =",
    np.percentile(delta_y, 99),
)


# ============================================================
# 2. COMPUTE INSTANTANEOUS HITTING-COST MINIMIZERS
#
# v_t = argmin_u f_t(u)
# ============================================================

print("\nComputing v_t for all time steps...")

v_values = np.zeros_like(
    demand
)

for t in range(T):

    v_values[t] = compute_v_t(
        demand[t]
    )

    if (
        (t + 1) % 1000 == 0
        or t == T - 1
    ):

        print(
            f"Computed {t + 1}/{T}",
            flush=True,
        )


# ============================================================
# 3. MOVEMENT OF v_t
#
# Delta_v(t) = || v_t - v_{t-1} ||_2
# ============================================================

delta_v_vectors = (
    v_values[1:]
    - v_values[:-1]
)

delta_v = np.linalg.norm(
    delta_v_vectors,
    axis=1,
)


# ============================================================
# v_t VARIATION STATISTICS
# ============================================================

print("\n----------------------------------------")
print("INSTANTANEOUS OPTIMUM VARIATION")
print("Delta_v(t) = ||v_t - v_{t-1}||_2")
print("----------------------------------------")

print(
    "Mean   =",
    np.mean(delta_v),
)

print(
    "Median =",
    np.median(delta_v),
)

print(
    "Std    =",
    np.std(delta_v),
)

print(
    "Min    =",
    np.min(delta_v),
)

print(
    "Max    =",
    np.max(delta_v),
)

print(
    "25%    =",
    np.percentile(delta_v, 25),
)

print(
    "75%    =",
    np.percentile(delta_v, 75),
)

print(
    "90%    =",
    np.percentile(delta_v, 90),
)

print(
    "95%    =",
    np.percentile(delta_v, 95),
)

print(
    "99%    =",
    np.percentile(delta_v, 99),
)


# ============================================================
# 4. SAVE NUMERICAL RESULTS
# ============================================================

variation_df = pd.DataFrame(
    {
        "t": np.arange(
            1,
            T,
        ),

        "delta_y":
            delta_y,

        "delta_v":
            delta_v,
    }
)

VARIATION_FILE = (
    OUTPUT_DIR
    / "demand_and_optimum_variation.csv"
)

variation_df.to_csv(
    VARIATION_FILE,
    index=False,
)


# ------------------------------------------------------------
# Save v_t as well
# ------------------------------------------------------------

v_df = pd.DataFrame(
    v_values,
    columns=[
        f"v_{i}"
        for i in range(D)
    ],
)

v_df.insert(
    0,
    "t",
    np.arange(T),
)

V_FILE = (
    OUTPUT_DIR
    / "instantaneous_optimum_v.csv"
)

v_df.to_csv(
    V_FILE,
    index=False,
)


# ============================================================
# 5. HISTOGRAM OF DEMAND VARIATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    delta_y,
    bins=60,
    edgecolor="black",
)

plt.xlabel(
    r"$\|y_t-y_{t-1}\|_2$"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Consecutive Demand Variation"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_y_distribution.png",
    dpi=300,
)

plt.show()


# ============================================================
# 6. HISTOGRAM OF v_t VARIATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    delta_v,
    bins=60,
    edgecolor="black",
)

plt.xlabel(
    r"$\|v_t-v_{t-1}\|_2$"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Instantaneous-Optimum Movement"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_v_distribution.png",
    dpi=300,
)

plt.show()


# ============================================================
# 7. TIME SERIES OF DEMAND VARIATION
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    np.arange(
        1,
        T,
    ),
    delta_y,
    linewidth=0.8,
)

plt.xlabel(
    "Time step (minute)"
)

plt.ylabel(
    r"$\|y_t-y_{t-1}\|_2$"
)

plt.title(
    "Demand Variation Over Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_y_over_time.png",
    dpi=300,
)

plt.show()


# ============================================================
# 8. TIME SERIES OF v_t MOVEMENT
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    np.arange(
        1,
        T,
    ),
    delta_v,
    linewidth=0.8,
)

plt.xlabel(
    "Time step (minute)"
)

plt.ylabel(
    r"$\|v_t-v_{t-1}\|_2$"
)

plt.title(
    "Instantaneous-Optimum Movement Over Time"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_v_over_time.png",
    dpi=300,
)

plt.show()


# ============================================================
# 9. DIRECT COMPARISON
#
# Normalize each by its mean so that their temporal patterns
# can be compared even if their absolute scales differ.
# ============================================================

delta_y_mean = np.mean(
    delta_y
)

delta_v_mean = np.mean(
    delta_v
)

normalized_delta_y = (
    delta_y
    / delta_y_mean
    if delta_y_mean > 0
    else delta_y
)

normalized_delta_v = (
    delta_v
    / delta_v_mean
    if delta_v_mean > 0
    else delta_v
)


plt.figure(
    figsize=(12, 5)
)

plt.plot(
    np.arange(
        1,
        T,
    ),
    normalized_delta_y,
    label=r"$\|y_t-y_{t-1}\|_2$",
    linewidth=0.8,
)

plt.plot(
    np.arange(
        1,
        T,
    ),
    normalized_delta_v,
    label=r"$\|v_t-v_{t-1}\|_2$",
    linewidth=0.8,
)

plt.xlabel(
    "Time step (minute)"
)

plt.ylabel(
    "Normalized movement"
)

plt.title(
    "Demand Variation vs. Instantaneous-Optimum Movement"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "delta_y_vs_delta_v.png",
    dpi=300,
)

plt.show()


# ============================================================
# 10. SUMMARY FILE
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "quantity": "delta_y",
            "mean": np.mean(delta_y),
            "median": np.median(delta_y),
            "std": np.std(delta_y),
            "min": np.min(delta_y),
            "max": np.max(delta_y),
            "p25": np.percentile(delta_y, 25),
            "p75": np.percentile(delta_y, 75),
            "p90": np.percentile(delta_y, 90),
            "p95": np.percentile(delta_y, 95),
            "p99": np.percentile(delta_y, 99),
        },

        {
            "quantity": "delta_v",
            "mean": np.mean(delta_v),
            "median": np.median(delta_v),
            "std": np.std(delta_v),
            "min": np.min(delta_v),
            "max": np.max(delta_v),
            "p25": np.percentile(delta_v, 25),
            "p75": np.percentile(delta_v, 75),
            "p90": np.percentile(delta_v, 90),
            "p95": np.percentile(delta_v, 95),
            "p99": np.percentile(delta_v, 99),
        },
    ]
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "variation_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("ANALYSIS COMPLETE")
print("========================================")

print(
    "\nSaved variation data to:"
)

print(
    VARIATION_FILE
)

print(
    "\nSaved v_t values to:"
)

print(
    V_FILE
)

print(
    "\nSaved summary to:"
)

print(
    SUMMARY_FILE
)

print(
    "\nPlots saved in:"
)

print(
    OUTPUT_DIR
)