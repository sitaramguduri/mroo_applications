import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


# ============================================================
# IMPORTS
# ============================================================

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost,
)

from weights import get_w_t


# ============================================================
# SIMPLEX CONSTRAINTS
# ============================================================

def simplex_constraints():

    return [{
        "type": "eq",
        "fun": lambda u: np.sum(u) - 1.0
    }]


def simplex_bounds(dim):

    return [(0.0, 1.0)] * dim


# ============================================================
# GREEDY STEP
#
# u_t = argmin f_t(u)
#
# subject to:
#
# u_i >= 0
# sum_i u_i = 1
# ============================================================

def greedy_step(y_t):

    y_t = np.asarray(
        y_t,
        dtype=float
    )


    result = minimize(
        fun=lambda u: hitting_cost(
            u,
            y_t
        ),
        x0=y_t,
        method="SLSQP",
        bounds=simplex_bounds(
            len(y_t)
        ),
        constraints=simplex_constraints(),
        options={
            "ftol": 1e-12,
            "maxiter": 500,
        }
    )


    if not result.success:

        raise RuntimeError(
            "Greedy optimization failed: "
            f"{result.message}"
        )


    return result.x


# ============================================================
# RUN GREEDY
# ============================================================

def run_greedy(demand):

    demand = np.asarray(
        demand,
        dtype=float
    )

    T, D = demand.shape


    # --------------------------------------------------------
    # Initial allocation
    # --------------------------------------------------------

    u_prev = (
        np.ones(D)
        / D
    )


    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    actions = []

    hitting_history = []

    memory_history = []

    switching_history = []


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(T):

        y_t = demand[t]


        # ----------------------------------------------------
        # Greedy decision
        #
        # Only minimizes current hitting cost.
        # ----------------------------------------------------

        u_t = greedy_step(
            y_t
        )


        # ----------------------------------------------------
        # Exposure vector
        #
        # Needed only to evaluate realized memory cost.
        # It does NOT influence greedy's decision.
        # ----------------------------------------------------

        w_t = get_w_t(
            t=t,
            y_t=y_t,
        )


        # ----------------------------------------------------
        # Realized vector memory cost
        # ----------------------------------------------------

        d_t = memory_cost(
            u_t,
            u_prev,
            w_t
        )


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        actions.append(
            u_t.copy()
        )


        hitting_history.append(
            hitting_cost(
                u_t,
                y_t
            )
        )


        memory_history.append(
            d_t.copy()
        )


        # Since ||w_t||_2 = 1:
        #
        # ||d_t||_2 = d_bar

        switching_history.append(
            np.linalg.norm(
                d_t,
                ord=2
            )
        )


        # ----------------------------------------------------
        # Next round
        # ----------------------------------------------------

        u_prev = u_t


    # ========================================================
    # CONVERT HISTORIES
    # ========================================================

    actions = np.asarray(
        actions
    )

    hitting_history = np.asarray(
        hitting_history
    )

    memory_history = np.asarray(
        memory_history
    )

    switching_history = np.asarray(
        switching_history
    )


    # ========================================================
    # FINAL CUMULATIVE COST
    #
    # Same evaluation convention as MROO / S-MROO.
    # ========================================================

    hitting_cost_total = np.sum(
        hitting_history
    )


    # cost.py should compute:
    #
    # rho || sum_t d_t ||_inf

    final_long_term_cost = long_term_cost(
        memory_history
    )


    total_cost = (
        hitting_cost_total
        + final_long_term_cost
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        "actions":
            actions,

        "hitting_history":
            hitting_history,

        "memory_history":
            memory_history,

        "switching_history":
            switching_history,

        "hitting_cost":
            hitting_cost_total,

        "long_term_cost":
            final_long_term_cost,

        "total_cost":
            total_cost,
    }