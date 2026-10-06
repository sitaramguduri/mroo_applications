import sys
from pathlib import Path

import numpy as np
import cvxpy as cp


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

from config import (
    P,
    C,
    BETA,
    A_MATRICES,
)

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost,
)


# ============================================================
# VALIDATE MEMORY MATRICES
#
# We require:
#
#     A_1, ..., A_D
#
# with:
#
#     A_i in R^{D x D}
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
# PRINT CONFIGURATION
# ============================================================

print(
    "\nOFFLINE OPT CONFIG"
)

print(
    "-------------------"
)

print(
    "P =",
    P,
)

print(
    "C =",
    C,
)

print(
    "BETA =",
    BETA,
)

print(
    "D =",
    D_CONFIG,
)

print(
    "A_MATRICES shape =",
    A_MATRICES.shape,
)


for i, A_i in enumerate(
    A_MATRICES,
    start=1,
):

    print(
        f"\nA_{i} ="
    )

    print(
        A_i
    )


# ============================================================
# OFFLINE OPTIMAL
#
# Objective:
#
# min_{u_1, ..., u_T}
#
#     sum_t f_t(u_t)
#
#     +
#
#     || sum_t d_t ||_infinity
#
#
# where:
#
# f_t(u_t)
#
# =
#
# p^T u_t
#
# +
#
# sum_j c_j (u_{t,j} - y_{t,j})^2
#
#
# and:
#
# d_{t,i}
#
# =
#
# beta / 2
#
# *
#
# ||A_i(u_t - u_{t-1})||_2^2
# ============================================================

def run_offline_opt(
    demand,
):

    demand = np.asarray(
        demand,
        dtype=float,
    )


    # ========================================================
    # VALIDATE DEMAND
    # ========================================================

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


    # ========================================================
    # INITIAL ACTION
    #
    # u_0 = uniform allocation
    # ========================================================

    u_initial = (

        np.ones(
            D,
            dtype=float,
        )

        / D
    )


    # ========================================================
    # DECISION VARIABLE
    #
    # U[t, :] = u_t
    # ========================================================

    U = cp.Variable(
        (
            T,
            D,
        )
    )


    # ========================================================
    # SIMPLEX CONSTRAINTS
    #
    # u_t >= 0
    #
    # u_t <= 1
    #
    # sum_i u_{t,i} = 1
    # ========================================================

    constraints = [

        U >= 0.0,

        U <= 1.0,

        cp.sum(
            U,
            axis=1,
        )
        == 1.0,

    ]


    # ========================================================
    # HITTING COST
    #
    # sum_t [
    #
    #     p^T u_t
    #
    #     +
    #
    #     sum_j
    #     c_j (u_{t,j} - y_{t,j})^2
    #
    # ]
    # ========================================================

    linear_hitting = cp.sum(

        cp.multiply(
            U,
            P,
        )
    )


    quadratic_hitting = cp.sum(

        cp.multiply(

            cp.square(
                U - demand
            ),

            C,
        )
    )


    total_hitting_cost = (

        linear_hitting

        +

        quadratic_hitting
    )


    # ========================================================
    # ACTION DIFFERENCES
    #
    # Delta u_1
    #
    # =
    #
    # u_1 - u_0
    #
    #
    # Delta u_t
    #
    # =
    #
    # u_t - u_{t-1}
    #
    # for t >= 2
    # ========================================================

    first_difference = (

        U[0:1, :]

        -

        u_initial.reshape(
            1,
            D,
        )
    )


    remaining_differences = (

        U[1:, :]

        -

        U[:-1, :]
    )


    differences = cp.vstack(
        [
            first_difference,
            remaining_differences,
        ]
    )


    # ========================================================
    # VECTOR-VALUED MEMORY COST
    #
    # For each i:
    #
    # d_{t,i}
    #
    # =
    #
    # beta / 2
    #
    # *
    #
    # ||A_i Delta u_t||_2^2
    #
    #
    # We directly construct:
    #
    # cumulative_memory[i]
    #
    # =
    #
    # sum_t d_{t,i}
    # ========================================================

    cumulative_memory_components = []


    for i in range(
        D
    ):

        A_i = (
            A_MATRICES[i]
        )


        # ----------------------------------------------------
        # differences[t, :]
        #
        # is Delta u_t^T.
        #
        # Therefore:
        #
        # differences @ A_i.T
        #
        # produces rows:
        #
        # (A_i Delta u_t)^T
        # ----------------------------------------------------

        transformed = (

            differences

            @ A_i.T
        )


        # ----------------------------------------------------
        # Memory cost for channel i at each t:
        #
        # d_{t,i}
        #
        # =
        #
        # beta / 2
        #
        # *
        #
        # ||A_i Delta u_t||_2^2
        # ----------------------------------------------------

        memory_component = (

            BETA

            / 2.0

            *

            cp.sum(
                cp.square(
                    transformed
                ),
                axis=1,
            )
        )


        # ----------------------------------------------------
        # Cumulative memory channel:
        #
        # sum_t d_{t,i}
        # ----------------------------------------------------

        cumulative_i = cp.sum(
            memory_component
        )


        cumulative_memory_components.append(
            cumulative_i
        )


    # ========================================================
    # CUMULATIVE MEMORY VECTOR
    #
    # [
    #
    #     sum_t d_{t,1},
    #
    #     ...,
    #
    #     sum_t d_{t,D}
    #
    # ]
    # ========================================================

    cumulative_memory = cp.hstack(
        cumulative_memory_components
    )


    # ========================================================
    # LONG-TERM COST
    #
    # || sum_t d_t ||_infinity
    #
    # =
    #
    # max_i sum_t d_{t,i}
    #
    #
    # No rho.
    # ========================================================

    total_long_term_cost = cp.max(
        cumulative_memory
    )


    # ========================================================
    # TOTAL OBJECTIVE
    # ========================================================

    objective = cp.Minimize(

        total_hitting_cost

        +

        total_long_term_cost
    )


    problem = cp.Problem(
        objective,
        constraints,
    )


    # ========================================================
    # SOLVE
    # ========================================================

    problem.solve(

        solver=cp.CLARABEL,

        verbose=False,
    )


    # ========================================================
    # CHECK STATUS
    # ========================================================

    if problem.status not in [

        cp.OPTIMAL,

        cp.OPTIMAL_INACCURATE,

    ]:

        raise RuntimeError(
            "Offline optimization failed. "
            f"Status: {problem.status}"
        )


    # ========================================================
    # EXTRACT ACTIONS
    # ========================================================

    actions = np.asarray(
        U.value,
        dtype=float,
    )


    # ========================================================
    # RECOMPUTE COSTS NUMERICALLY
    #
    # This uses the exact same functions as the online
    # algorithms so that evaluation is consistent.
    # ========================================================

    hitting_history = []

    memory_history = []


    u_prev = (
        u_initial.copy()
    )


    for t in range(
        T
    ):

        u_t = (
            actions[t]
        )


        y_t = (
            demand[t]
        )


        # ----------------------------------------------------
        # Hitting cost
        # ----------------------------------------------------

        hit_t = hitting_cost(

            u_t,

            y_t,
        )


        # ----------------------------------------------------
        # Vector memory cost
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


        hitting_history.append(
            hit_t
        )


        memory_history.append(
            d_t.copy()
        )


        u_prev = (
            u_t.copy()
        )


    # ========================================================
    # CONVERT HISTORIES TO ARRAYS
    # ========================================================

    hitting_history = np.asarray(
        hitting_history,
        dtype=float,
    )


    memory_history = np.asarray(
        memory_history,
        dtype=float,
    )


    # ========================================================
    # CUMULATIVE MEMORY
    #
    # sum_t d_t
    # ========================================================

    cumulative_memory_numeric = np.sum(

        memory_history,

        axis=0,
    )


    # ========================================================
    # FINAL COSTS
    # ========================================================

    hitting_cost_total = float(

        np.sum(
            hitting_history
        )
    )


    long_term_cost_total = float(

        long_term_cost(
            memory_history
        )
    )


    total_cost = (

        hitting_cost_total

        +

        long_term_cost_total
    )


    # ========================================================
    # SOLVER / NUMERICAL CONSISTENCY CHECK
    #
    # problem.value and total_cost should be almost identical.
    # ========================================================

    solver_objective = float(
        problem.value
    )


    objective_gap = abs(

        solver_objective

        -

        total_cost
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
            cumulative_memory_numeric,


        # ----------------------------------------------------
        # Final costs
        # ----------------------------------------------------

        "hitting_cost":
            hitting_cost_total,


        "long_term_cost":
            long_term_cost_total,


        "total_cost":
            total_cost,


        # ----------------------------------------------------
        # Solver information
        # ----------------------------------------------------

        "solver_objective":
            solver_objective,


        "objective_gap":
            objective_gap,


        "solver_status":
            problem.status,

    }