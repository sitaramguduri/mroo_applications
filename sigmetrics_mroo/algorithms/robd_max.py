import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import numpy as np
from scipy.optimize import minimize

from cost import hitting_cost, l
from greedy import greedy_step


# --------------------------------------------------
# ROBD-MAX parameters
# --------------------------------------------------

LAMBDA_1 = 0.3
LAMBDA_2 = 0.8


# --------------------------------------------------
# Simplex constraints
# --------------------------------------------------

def simplex_constraints():
    return [{
        "type": "eq",
        "fun": lambda u: np.sum(u) - 1.0
    }]


def simplex_bounds(dim):
    return [(0.0, 1.0)] * dim


# --------------------------------------------------
# Max switching penalty
#
# max_i l_i/2 * (u_i - u_prev_i)^2
# --------------------------------------------------

def max_switching_cost(u_t, u_prev):

    u_t = np.asarray(u_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)

    per_workload_cost = (
        0.5 *
        l *
        (u_t - u_prev) ** 2
    )

    return np.max(per_workload_cost)


# --------------------------------------------------
# ROBD-MAX step
# --------------------------------------------------

def robd_max_step(
    y_t,
    u_prev,
    lambda_1=LAMBDA_1,
    lambda_2=LAMBDA_2
):

    y_t = np.asarray(y_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)

    # Current hitting-cost minimizer
    v_t = greedy_step(y_t)

    def objective(u):

        # Instantaneous hitting cost
        f = hitting_cost(
            u,
            y_t
        )

        # Maximum switching cost from previous action
        switching = max_switching_cost(
            u,
            u_prev
        )

        # Regularization toward hitting-cost minimizer
        regularization = max_switching_cost(
            u,
            v_t
        )

        return (
            f
            + lambda_1 * switching
            + lambda_2 * regularization
        )

    result = minimize(
        fun=objective,
        x0=u_prev,
        method="SLSQP",
        bounds=simplex_bounds(len(y_t)),
        constraints=simplex_constraints()
    )

    if not result.success:
        raise RuntimeError(
            f"ROBD-MAX optimization failed: "
            f"{result.message}"
        )

    return result.x
