from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
)

MEMORY_FILE = (
    RESULT_DIR
    / "offline_opt_full_horizon_memory.csv"
)

DELTA_RESULT_FILE = (
    RESULT_DIR
    / "offline_opt_delta_R100.csv"
)

DELTA_PLOT_FILE = (
    RESULT_DIR
    / "offline_opt_delta_R100.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

FRAME_SIZE = 100


# ============================================================
# LOAD OFFLINE OPTIMAL MEMORY HISTORY
# ============================================================

memory_df = pd.read_csv(
    MEMORY_FILE
)

memory_history = memory_df[
    [
        "d_0",
        "d_1",
        "d_2",
    ]
].to_numpy(
    dtype=float
)


T_original, D = memory_history.shape


print(
    "Original horizon T =",
    T_original
)

print(
    "Dimension D =",
    D
)

print(
    "Frame size R =",
    FRAME_SIZE
)


# ============================================================
# USE COMPLETE FRAMES ONLY
# ============================================================

K = (
    T_original
    // FRAME_SIZE
)

T_used = (
    K
    * FRAME_SIZE
)

excluded_steps = (
    T_original
    - T_used
)


memory_used = memory_history[
    :T_used
]


print(
    "Number of complete frames K =",
    K
)

print(
    "T used =",
    T_used
)

print(
    "Excluded trailing steps =",
    excluded_steps
)


# ============================================================
# GLOBAL AVERAGE MEMORY VECTOR
#
# d_bar = (1 / T) sum_t d_t*
# ============================================================

d_bar = np.mean(
    memory_used,
    axis=0
)


print(
    "\nAverage memory vector d_bar =",
    d_bar
)


# ============================================================
# PER-STEP DEVIATION
#
# delta_t = d_t* - d_bar
# ============================================================

deviation = (
    memory_used
    - d_bar
)


# ============================================================
# DIVIDE INTO FRAMES
#
# Shape:
#     K x R x D
# ============================================================

framed_deviation = deviation.reshape(
    K,
    FRAME_SIZE,
    D,
)


# ============================================================
# SUM DEVIATION INSIDE EACH FRAME
#
# s_k = sum_{t in frame k} delta_t
# ============================================================

frame_sums = np.sum(
    framed_deviation,
    axis=1,
)


# ============================================================
# FRAME CONTRIBUTION TO DELTA
#
# delta_k = ||s_k||_2
# ============================================================

frame_delta = np.linalg.norm(
    frame_sums,
    ord=2,
    axis=1,
)


# ============================================================
# TOTAL ADVERSARIAL BUDGET
#
# delta = sum_k delta_k
# ============================================================

delta_total = np.sum(
    frame_delta
)


print(
    "\n========================================"
)

print(
    "OFFLINE OPT ADVERSARIAL BUDGET"
)

print(
    "========================================"
)

print(
    "Frame size R =",
    FRAME_SIZE
)

print(
    "Number of frames K =",
    K
)

print(
    "Delta =",
    delta_total
)


# ============================================================
# SAVE FRAME-LEVEL RESULTS
# ============================================================

result_df = pd.DataFrame({

    "frame_id":
        np.arange(
            1,
            K + 1
        ),

    "start_t":
        np.arange(
            0,
            T_used,
            FRAME_SIZE
        ),

    "end_t":
        np.arange(
            FRAME_SIZE,
            T_used + 1,
            FRAME_SIZE
        ),

    "frame_delta":
        frame_delta,

    "frame_sum_d0":
        frame_sums[:, 0],

    "frame_sum_d1":
        frame_sums[:, 1],

    "frame_sum_d2":
        frame_sums[:, 2],
})


result_df.to_csv(
    DELTA_RESULT_FILE,
    index=False
)


print(
    "\nSaved frame-level results to:",
    DELTA_RESULT_FILE
)


# ============================================================
# PLOT FRAME DELTA
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    result_df["frame_id"],
    result_df["frame_delta"],
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    r"$\left\|\sum_{t \in \mathrm{frame}} "
    r"(d_t^*-\bar{d})\right\|_2$"
)

plt.title(
    "Offline Optimal Frame Deviation "
    f"(R = {FRAME_SIZE})"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    DELTA_PLOT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


print(
    "Frame plot saved to:",
    DELTA_PLOT_FILE
)