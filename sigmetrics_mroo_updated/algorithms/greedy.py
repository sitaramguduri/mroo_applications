import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(
        str(PROJECT_DIR)
    )


# ============================================================
# IMPORTS
# ============================================================

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost,
)

from config import (
    A_MATRICES,
)


# ============================================================
# VALIDATE MEMORY MATRICES
#
# There are exactly D matrices:
#
#     A_1, ..., A_D
#
# and each A_i is D x D.
#
# Therefore:
#
#     A_MATRICES.shape = (D, D, D)
# ============================================================

A_MATRICES = np.asarray(
    A_MATRICES,
    dtype=float,
)

if A_MATRICES.ndim != 3:

    raise ValueError(
        "A_MATRICES must be a 3-dimensional array. "
        f"Found shape {A_MATRICES.shape}."
    )

D_CONFIG = int(
    A_MATRICES.shape[0]
)

if A_MATRICES.shape != (
    D_CONFIG,
    D_CONFIG,
    D_CONFIG,
):

    raise ValueError(
        "A_MATRICES must have shape "
        f"({D_CONFIG}, {D_CONFIG}, {D_CONFIG}), "
        f"but got {A_MATRICES.shape}."
    )


# ============================================================
# SIMPLEX CONSTRAINTS
# ============================================================

def simplex_constraints():

    return [
        {
            "type": "eq",
            "fun":
                lambda u:
                np.sum(u) - 1.0,
        }
    ]


def simplex_bounds(
    dim,
):

    return [
        (0.0, 1.0)
    ] * dim


# ============================================================
# GREEDY STEP
#
# u_t = argmin f_t(u)
#
# subject to:
#
#     u_i >= 0
#
#     sum_i u_i = 1
#
#
# Greedy ignores memory cost when choosing u_t.
# ============================================================

def greedy_step(
    y_t,
):

    y_t = np.asarray(
        y_t,
        dtype=float,
    )


    result = minimize(

        fun=lambda u:
            hitting_cost(
                u,
                y_t,
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
        },
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

def run_greedy(
    demand,
):

    demand = np.asarray(
        demand,
        dtype=float,
    )


    # --------------------------------------------------------
    # Validate demand
    # --------------------------------------------------------

    if demand.ndim != 2:

        raise ValueError(
            "demand must be a 2-dimensional array. "
            f"Found shape {demand.shape}."
        )


    (
        T,
        D,
    ) = (
        demand.shape
    )


    if D != D_CONFIG:

        raise ValueError(
            "Demand dimension does not match "
            "the configured A matrices. "
            f"Demand dimension = {D}, "
            f"configured dimension = {D_CONFIG}."
        )


    # --------------------------------------------------------
    # Initial allocation
    #
    # u_0 = uniform simplex allocation
    # --------------------------------------------------------

    u_prev = (
        np.ones(
            D
        )
        / D
    )


    # ========================================================
    # HISTORIES
    # ========================================================

    actions = []

    hitting_history = []

    memory_history = []


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(
        T
    ):

        y_t = (
            demand[t]
        )


        # ----------------------------------------------------
        # Greedy decision
        #
        # Only minimizes current hitting cost.
        #
        # Memory cost does NOT influence the action.
        # ----------------------------------------------------

        u_t = greedy_step(
            y_t
        )


        # ----------------------------------------------------
        # Realized vector-valued memory cost
        #
        # For i = 1, ..., D:
        #
        # d_{t,i}
        #
        # =
        #
        # beta / 2
        #
        # *
        #
        # ||A_i(u_t-u_{t-1})||_2^2
        # ----------------------------------------------------

        d_t = memory_cost(

            u_t=u_t,

            u_prev=u_prev,

            A_matrices=A_MATRICES,
        )


        # ----------------------------------------------------
        # Store action
        # ----------------------------------------------------

        actions.append(
            u_t.copy()
        )


        # ----------------------------------------------------
        # Store hitting cost
        # ----------------------------------------------------

        hitting_history.append(

            hitting_cost(
                u_t,
                y_t,
            )
        )


        # ----------------------------------------------------
        # Store vector memory cost
        # ----------------------------------------------------

        memory_history.append(
            d_t.copy()
        )


        # ----------------------------------------------------
        # Next round
        # ----------------------------------------------------

        u_prev = (
            u_t.copy()
        )


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


    # ========================================================
    # FINAL CUMULATIVE COST
    #
    # Actual common evaluation objective:
    #
    #     sum_t f_t(u_t)
    #
    #     +
    #
    #     || sum_t d_t ||_inf
    #
    # There is NO rho.
    # ========================================================

    hitting_cost_total = float(

        np.sum(
            hitting_history
        )
    )


    final_long_term_cost = float(

        long_term_cost(
            memory_history
        )
    )


    total_cost = (

        hitting_cost_total

        +

        final_long_term_cost
    )


    # ========================================================
    # CUMULATIVE MEMORY VECTOR
    # ========================================================

    cumulative_memory = np.sum(

        memory_history,

        axis=0,
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        # ----------------------------------------------------
        # Trajectories
        # ----------------------------------------------------

        "actions":
            actions,


        "hitting_history":
            hitting_history,


        "memory_history":
            memory_history,


        "cumulative_memory":
            cumulative_memory,


        # ----------------------------------------------------
        # Final costs
        # ----------------------------------------------------

        "hitting_cost":
            hitting_cost_total,


        "long_term_cost":
            final_long_term_cost,


        "total_cost":
            total_cost,
    }