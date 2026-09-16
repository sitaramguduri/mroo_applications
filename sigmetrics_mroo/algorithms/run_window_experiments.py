import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import numpy as np
import pandas as pd

from mroo import mroo_step
from greedy import greedy_step
from robd_sum import robd_sum_step
from robd_max import robd_max_step

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost
)

from weights import get_w_t


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

WINDOW_SIZE = 8640      # 24 hours at 10-second slots
NUM_WINDOWS = 100

MROO_ETA = 1e-6
MROO_KAPPA_INIT = np.array(
    [0.05, 0.05, 0.05],
    dtype=float
)

U_INIT = np.array([
    1/3,
    1/3,
    1/3
], dtype=float)


# --------------------------------------------------
# Load full processed trace
# --------------------------------------------------

data = pd.read_csv("../azure_y_t.csv")

demand = data[
    ["y_code", "y_conv", "y_noninteractive"]
].to_numpy()

T = len(demand)

print("Total slots:", T)
print("24-hour window size:", WINDOW_SIZE)


# --------------------------------------------------
# Construct 100 continuous windows
# --------------------------------------------------

max_start = T - WINDOW_SIZE

if max_start < 0:
    raise ValueError(
        "Trace is shorter than one 24-hour window."
    )

# Reproducible random windows
rng = np.random.default_rng(42)

# Always include the seven day-aligned windows when possible
window_starts = []

for day in range(7):
    start = day * WINDOW_SIZE

    if start <= max_start:
        window_starts.append(start)

# Fill remaining windows with random continuous windows
remaining = NUM_WINDOWS - len(window_starts)

if remaining > 0:

    possible_starts = np.arange(
        0,
        max_start + 1
    )

    random_starts = rng.choice(
        possible_starts,
        size=remaining,
        replace=False
    )

    window_starts.extend(
        random_starts.tolist()
    )

print(
    "Number of windows:",
    len(window_starts)
)


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate_histories(
    hitting_history,
    memory_history
):
    avg_hitting = np.mean(
        hitting_history
    )

    q_cost = long_term_cost(
        memory_history
    )

    total = (
        avg_hitting
        + q_cost
    )

    return (
        avg_hitting,
        q_cost,
        total
    )


# --------------------------------------------------
# Greedy
# --------------------------------------------------

def run_greedy_window(window):

    u_prev = U_INIT.copy()

    hitting_history = []
    memory_history = []

    for t in range(len(window)):

        y_t = window[t]

        u_t = greedy_step(y_t)

        f_t = hitting_cost(
            u_t,
            y_t
        )

        w_t = get_w_t(t, y_t)

        d_t = memory_cost(
            u_t,
            u_prev,
            w_t
        )

        hitting_history.append(f_t)
        memory_history.append(d_t)

        u_prev = u_t

    return evaluate_histories(
        hitting_history,
        memory_history
    )


# --------------------------------------------------
# ROBD-SUM
# --------------------------------------------------

def run_robd_sum_window(window):

    u_prev = U_INIT.copy()

    hitting_history = []
    memory_history = []

    for t in range(len(window)):

        y_t = window[t]

        u_t = robd_sum_step(
            y_t,
            u_prev
        )

        f_t = hitting_cost(
            u_t,
            y_t
        )

        w_t = get_w_t(t, y_t)

        d_t = memory_cost(
            u_t,
            u_prev,
            w_t
        )

        hitting_history.append(f_t)
        memory_history.append(d_t)

        u_prev = u_t

    return evaluate_histories(
        hitting_history,
        memory_history
    )


# --------------------------------------------------
# ROBD-MAX
# --------------------------------------------------

def run_robd_max_window(window):

    u_prev = U_INIT.copy()

    hitting_history = []
    memory_history = []

    for t in range(len(window)):

        y_t = window[t]

        u_t = robd_max_step(
            y_t,
            u_prev
        )

        f_t = hitting_cost(
            u_t,
            y_t
        )

        w_t = get_w_t(t, y_t)

        d_t = memory_cost(
            u_t,
            u_prev,
            w_t
        )

        hitting_history.append(f_t)
        memory_history.append(d_t)

        u_prev = u_t

    return evaluate_histories(
        hitting_history,
        memory_history
    )


# --------------------------------------------------
# MROO
# --------------------------------------------------

def run_mroo_window(window):

    u_prev = U_INIT.copy()

    kappa_t = (
        MROO_KAPPA_INIT.copy()
    )

    hitting_history = []
    memory_history = []

    for t in range(len(window)):

        y_t = window[t]

        w_t = get_w_t(t, y_t)

        result = mroo_step(
            y_t=y_t,
            u_prev=u_prev,
            w_t=w_t,
            kappa_t=kappa_t,
            eta=MROO_ETA
        )

        u_t = result["u_t"]
        d_t = result["d_t"]

        f_t = hitting_cost(
            u_t,
            y_t
        )

        hitting_history.append(f_t)
        memory_history.append(d_t)

        u_prev = u_t
        kappa_t = result[
            "kappa_next"
        ]

    return evaluate_histories(
        hitting_history,
        memory_history
    )


# --------------------------------------------------
# Run all experiments
# --------------------------------------------------

results = []

algorithms = {
    "GRD": run_greedy_window,
    "ROBD-SUM": run_robd_sum_window,
    "ROBD-MAX": run_robd_max_window,
    "MROO": run_mroo_window,
}


for window_id, start in enumerate(
    window_starts
):

    end = start + WINDOW_SIZE

    window = demand[
        start:end
    ]

    print(
        f"\nWindow {window_id + 1}/"
        f"{len(window_starts)} "
        f"[{start}:{end}]"
    )

    for name, runner in algorithms.items():

        print(
            f"  Running {name}..."
        )

        hitting, long_term, total = (
            runner(window)
        )

        results.append({
            "window":
                window_id,

            "start_slot":
                start,

            "algorithm":
                name,

            "hitting_cost":
                hitting,

            "long_term_cost":
                long_term,

            "total_cost":
                total
        })


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "window_algorithm_results.csv",
    index=False
)

print(
    "\nSaved results to "
    "window_algorithm_results.csv"
)