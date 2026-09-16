import numpy as np
from scipy.optimize import minimize

from cost import hitting_cost


# --------------------------------------------------
# Simplex constraints
# u_i >= 0
# sum_i u_i = 1
# --------------------------------------------------

def simplex_constraints():
    return [{
        "type": "eq",
        "fun": lambda u: np.sum(u) - 1.0
    }]


def simplex_bounds(dim):
    return [(0.0, 1.0)] * dim


# --------------------------------------------------
# Greedy algorithm
#
# u_t = argmin f_t(u)
# --------------------------------------------------

def greedy_step(y_t):

    y_t = np.asarray(y_t, dtype=float)

    result = minimize(
        fun=lambda u: hitting_cost(u, y_t),

        # y_t is already a feasible allocation,
        # so it is a good starting point
        x0=y_t.copy(),

        method="SLSQP",

        bounds=simplex_bounds(len(y_t)),
        constraints=simplex_constraints()
    )

    if not result.success:
        raise RuntimeError(
            f"Greedy optimization failed: {result.message}"
        )

    return result.x