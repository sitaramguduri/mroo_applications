from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mroo import (
    run_mroo,
    compute_v_t,
)

from dmd import run_dmd

from config import (
    LAMBDA_1_THEORY,
)


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ALGORITHM_DIR.parent


DATA_FILE = (
    PROJECT_DIR
    / "new_results"
    / "mroo_demand_trace_norm_weekday_aligned_1min.csv"
)

WINDOW_FILE = (
    ALGORITHM_DIR
    / "new_results"
    / "paper_window_starts_size_1440_n_100_seed_0.csv"
)

OUTPUT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "dmd_mroo_movement_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA
# ============================================================

DEMAND_COLUMNS = [
    "code",
    "chat",
    "non-interactive",
]

data = pd.read_csv(DATA_FILE)

demand_full = data[
    DEMAND_COLUMNS
].to_numpy(
    dtype=float
)


# ============================================================
# CHOOSE ONE WINDOW
# ============================================================

window_df = pd.read_csv(
    WINDOW_FILE
)

# Use first shared window.
window_id = int(
    window_df.iloc[0]["window_id"]
)

start = int(
    window_df.iloc[0]["start_index"]
)

end = int(
    window_df.iloc[0]["end_index"]
)

demand = demand_full[
    start:end
]

T, D = demand.shape


print("\n========================================")
print("DMD vs MROO MOVEMENT ANALYSIS")
print("========================================")

print("Window ID =", window_id)
print("Start =", start)
print("End =", end)
print("T =", T)


# ============================================================
# PARAMETERS
#
# Replace these with the best values from your experiment.
# ============================================================

MROO_ETA = 0.08
MROO_KAPPA = 0.06

DMD_ETA = (
    T ** (-1.0 / 3.0)
)

DMD_KAPPA = (
    1.0 / T
)


mroo_kappa_init = np.full(
    D,
    MROO_KAPPA,
    dtype=float,
)

dmd_kappa_init = np.full(
    D,
    DMD_KAPPA,
    dtype=float,
)


# ============================================================
# COMPUTE v_t
# ============================================================

print("\nComputing v_t...")

v_history = np.zeros_like(
    demand
)

for t in range(T):

    v_history[t] = compute_v_t(
        demand[t]
    )


# ============================================================
# RUN MROO
# ============================================================

print("\nRunning MROO...")

mroo_result = run_mroo(
    demand=demand,
    eta=MROO_ETA,
    kappa_init=mroo_kappa_init,
    lambda_1=LAMBDA_1_THEORY,
)

mroo_actions = np.asarray(
    mroo_result["actions"],
    dtype=float,
)


# ============================================================
# RUN DMD
# ============================================================

print("\nRunning DMD...")

dmd_result = run_dmd(
    demand=demand,
    eta=DMD_ETA,
    kappa_init=dmd_kappa_init,
)

dmd_actions = np.asarray(
    dmd_result["actions"],
    dtype=float,
)


# ============================================================
# INITIAL ALLOCATION
# ============================================================

u0 = (
    np.ones(D)
    / D
)


# ============================================================
# MOVEMENT:
#
# ||u_t - u_{t-1}||_2
# ============================================================

mroo_previous = np.vstack(
    [
        u0,
        mroo_actions[:-1],
    ]
)

dmd_previous = np.vstack(
    [
        u0,
        dmd_actions[:-1],
    ]
)


mroo_movement = np.linalg.norm(
    mroo_actions
    - mroo_previous,
    axis=1,
)

dmd_movement = np.linalg.norm(
    dmd_actions
    - dmd_previous,
    axis=1,
)


# ============================================================
# DISTANCE FROM INSTANTANEOUS MINIMIZER:
#
# ||u_t - v_t||_2
# ============================================================

mroo_distance_to_v = np.linalg.norm(
    mroo_actions
    - v_history,
    axis=1,
)

dmd_distance_to_v = np.linalg.norm(
    dmd_actions
    - v_history,
    axis=1,
)


# ============================================================
# ALSO COMPARE v_t MOVEMENT
# ============================================================

v_previous = np.vstack(
    [
        u0,
        v_history[:-1],
    ]
)

v_movement = np.linalg.norm(
    v_history
    - v_previous,
    axis=1,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n========================================")
print("MOVEMENT SUMMARY")
print("========================================")

print("\nMean ||u_t - u_{t-1}||_2")

print(
    "MROO =",
    np.mean(mroo_movement),
)

print(
    "DMD  =",
    np.mean(dmd_movement),
)

print(
    "v_t  =",
    np.mean(v_movement),
)


print("\nMedian ||u_t - u_{t-1}||_2")

print(
    "MROO =",
    np.median(mroo_movement),
)

print(
    "DMD  =",
    np.median(dmd_movement),
)

print(
    "v_t  =",
    np.median(v_movement),
)


print("\n========================================")
print("DISTANCE TO v_t")
print("========================================")

print("\nMean ||u_t - v_t||_2")

print(
    "MROO =",
    np.mean(mroo_distance_to_v),
)

print(
    "DMD  =",
    np.mean(dmd_distance_to_v),
)


print("\nMedian ||u_t - v_t||_2")

print(
    "MROO =",
    np.median(mroo_distance_to_v),
)

print(
    "DMD  =",
    np.median(dmd_distance_to_v),
)


# ============================================================
# COSTS
# ============================================================

print("\n========================================")
print("COSTS")
print("========================================")

print("\nMROO")
print(
    "Hitting   =",
    mroo_result["hitting_cost"],
)
print(
    "Long-term =",
    mroo_result["long_term_cost"],
)
print(
    "Total     =",
    mroo_result["total_cost"],
)


print("\nDMD")
print(
    "Hitting   =",
    dmd_result["hitting_cost"],
)
print(
    "Long-term =",
    dmd_result["long_term_cost"],
)
print(
    "Total     =",
    dmd_result["total_cost"],
)


# ============================================================
# SAVE DATA
# ============================================================

comparison_df = pd.DataFrame(
    {
        "t":
            np.arange(T),

        "v_movement":
            v_movement,

        "mroo_movement":
            mroo_movement,

        "dmd_movement":
            dmd_movement,

        "mroo_distance_to_v":
            mroo_distance_to_v,

        "dmd_distance_to_v":
            dmd_distance_to_v,
    }
)

comparison_df.to_csv(
    OUTPUT_DIR
    / "movement_comparison.csv",
    index=False,
)


# ============================================================
# PLOT 1:
# MOVEMENT
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    mroo_movement,
    label="MROO",
    linewidth=0.8,
)

plt.plot(
    dmd_movement,
    label="DMD",
    linewidth=0.8,
)

plt.plot(
    v_movement,
    label=r"$v_t$",
    linewidth=0.8,
)

plt.xlabel(
    "Time step"
)

plt.ylabel(
    r"$\|u_t-u_{t-1}\|_2$"
)

plt.title(
    "Allocation Movement"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "movement_over_time.png",
    dpi=300,
)

plt.show()


# ============================================================
# PLOT 2:
# DISTANCE FROM v_t
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    mroo_distance_to_v,
    label="MROO",
    linewidth=0.8,
)

plt.plot(
    dmd_distance_to_v,
    label="DMD",
    linewidth=0.8,
)

plt.xlabel(
    "Time step"
)

plt.ylabel(
    r"$\|u_t-v_t\|_2$"
)

plt.title(
    "Distance From Instantaneous Hitting-Cost Minimizer"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "distance_to_v.png",
    dpi=300,
)

plt.show()


# ============================================================
# PLOT 3:
# HISTOGRAM OF MOVEMENT
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    mroo_movement,
    bins=50,
    alpha=0.6,
    label="MROO",
)

plt.hist(
    dmd_movement,
    bins=50,
    alpha=0.6,
    label="DMD",
)

plt.xlabel(
    r"$\|u_t-u_{t-1}\|_2$"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Allocation Movement"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "movement_distribution.png",
    dpi=300,
)

plt.show()


print("\nSaved results to:")
print(
    OUTPUT_DIR
)