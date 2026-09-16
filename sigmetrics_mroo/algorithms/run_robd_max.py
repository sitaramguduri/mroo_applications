import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import numpy as np
import pandas as pd

from robd_max import robd_max_step
from cost import hitting_cost, memory_cost, long_term_cost
from weights import get_w_t


# --------------------------------------------------
# Load test data
# --------------------------------------------------

test_data = pd.read_csv(
    "../azure_test.csv"
)

test_demand = test_data[
    ["y_code", "y_conv", "y_noninteractive"]
].to_numpy()


# --------------------------------------------------
# Initial allocation
# --------------------------------------------------

u_prev = np.array([
    1/3,
    1/3,
    1/3
], dtype=float)


hitting_history = []
memory_history = []


# --------------------------------------------------
# Run ROBD-MAX
# --------------------------------------------------

for t in range(len(test_demand)):

    y_t = test_demand[t]

    # ROBD-MAX decision
    u_t = robd_max_step(
        y_t=y_t,
        u_prev=u_prev
    )

    # Hitting cost
    f_t = hitting_cost(
        u_t,
        y_t
    )

    # Same environmental weight vector
    w_t = get_w_t(
        t,
        y_t
    )

    # Actual memory cost used for final evaluation
    d_t = memory_cost(
        u_t,
        u_prev,
        w_t
    )

    hitting_history.append(f_t)
    memory_history.append(d_t)

    u_prev = u_t


# --------------------------------------------------
# Final objective
# --------------------------------------------------

avg_hitting_cost = np.mean(
    hitting_history
)

q_cost = long_term_cost(
    memory_history
)

total_cost = (
    avg_hitting_cost
    + q_cost
)


# --------------------------------------------------
# Print
# --------------------------------------------------

print("\nROBD-MAX test result:")

print(
    "Average hitting cost =",
    avg_hitting_cost
)

print(
    "Long-term cost =",
    q_cost
)

print(
    "Total cost =",
    total_cost
)


# --------------------------------------------------
# Save
# --------------------------------------------------

robd_max_result = pd.DataFrame([{
    "algorithm": "ROBD-MAX",
    "eta": np.nan,
    "kappa_init": np.nan,
    "avg_hitting_cost": avg_hitting_cost,
    "long_term_cost": q_cost,
    "total_cost": total_cost
}])

robd_max_result.to_csv(
    "robd_max_result.csv",
    index=False
)

print(
    "\nSaved to robd_max_result.csv"
)