import sys
from pathlib import Path

import numpy as np
import cvxpy as cp


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


# ============================================================
# IMPORTS
# ============================================================

from config import (
    P,
    C,
    ELL,
    RHO,
)

from weights import get_w_t
print("\nOFFLINE OPT CONFIG")
print("-------------------")
print("P    =", P)
print("C    =", C)
print("ELL  =", ELL)
print("RHO  =", RHO)
print("m    =", 2.0 * np.min(C))
print("beta =", np.max(ELL))

# ============================================================
# OFFLINE OPTIMAL
# ============================================================

def run_offline_opt(demand):

    demand = np.asarray(
        demand,
        dtype=float
    )

    T, D = demand.shape


    # ========================================================
    # PRECOMPUTE EXPOSURE VECTORS
    # ========================================================

    W = np.asarray(
        [
            get_w_t(
                t=t,
                y_t=demand[t],
            )
            for t in range(T)
        ],
        dtype=float,
    )


    if W.shape != (T, D):

        raise ValueError(
            f"W has shape {W.shape}, "
            f"expected {(T, D)}"
        )


    # ========================================================
    # INITIAL ACTION
    # ========================================================

    u_initial = (
        np.ones(D)
        / D
    )


    # ========================================================
    # DECISION VARIABLE
    # ========================================================

    U = cp.Variable(
        (T, D)
    )


    # ========================================================
    # SIMPLEX CONSTRAINTS
    # ========================================================

    constraints = [

        U >= 0.0,

        U <= 1.0,

        cp.sum(
            U,
            axis=1
        ) == 1.0,
    ]


    # ========================================================
    # VECTORIZED HITTING COST
    #
    # f_t(u_t)
    # =
    # sum_i p_i u_{t,i}
    # +
    # sum_i c_i (u_{t,i} - y_{t,i})^2
    # ========================================================

    linear_hitting = cp.sum(
        cp.multiply(
            U,
            P
        )
    )


    quadratic_hitting = cp.sum(
        cp.multiply(
            cp.square(
                U - demand
            ),
            C
        )
    )


    total_hitting_cost = (
        linear_hitting
        +
        quadratic_hitting
    )


    # ========================================================
    # VECTORIZED ACTION DIFFERENCES
    #
    # diff[0] = U[0] - u_initial
    #
    # diff[t] = U[t] - U[t-1]
    # ========================================================

    first_difference = (
        U[0:1, :]
        -
        u_initial.reshape(
            1,
            D
        )
    )


    remaining_differences = (
        U[1:, :]
        -
        U[:-1, :]
    )


    differences = cp.vstack([
        first_difference,
        remaining_differences,
    ])


    # ========================================================
    # VECTORIZED SCALAR MEMORY COST
    #
    # d_bar[t]
    # =
    # 0.5 * sum_i ell_i * diff[t,i]^2
    #
    # Result shape: (T,)
    # ========================================================

    d_bar = (
        0.5
        *
        cp.sum(
            cp.multiply(
                cp.square(
                    differences
                ),
                ELL
            ),
            axis=1
        )
    )


    # ========================================================
    # CUMULATIVE VECTOR MEMORY
    #
    # d_t = d_bar[t] * w_t
    #
    # sum_t d_t
    #
    # =
    #
    # W.T @ d_bar
    #
    # shape: (D,)
    # ========================================================

    cumulative_memory = (
        W.T
        @ d_bar
    )


    # ========================================================
    # LONG-TERM COST
    #
    # rho || sum_t d_t ||_inf
    #
    # Since cumulative_memory >= 0,
    # max(...) is equivalent to inf norm.
    # ========================================================

    total_long_term_cost = (
        RHO
        *
        cp.max(
            cumulative_memory
        )
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
        constraints
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
        dtype=float
    )


    # ========================================================
    # RECOMPUTE COSTS NUMERICALLY
    #
    # Same convention as online algorithms.
    # ========================================================

    # --------------------------------------------------------
    # Hitting cost
    # --------------------------------------------------------

    hitting_history = (

        np.sum(
            P
            * actions,
            axis=1
        )

        +

        np.sum(
            C
            * (
                actions
                - demand
            ) ** 2,
            axis=1
        )
    )


    # --------------------------------------------------------
    # Action differences
    # --------------------------------------------------------

    numerical_differences = np.vstack([

        actions[0]
        - u_initial,

        actions[1:]
        - actions[:-1],

    ])


    # --------------------------------------------------------
    # Scalar memory costs
    # --------------------------------------------------------

    switching_history = (

        0.5
        *
        np.sum(
            ELL
            * numerical_differences ** 2,
            axis=1
        )
    )


    # --------------------------------------------------------
    # Vector memory costs
    #
    # memory_history[t,j]
    # =
    # switching_history[t] * W[t,j]
    # --------------------------------------------------------

    memory_history = (

        switching_history[:, None]

        *

        W
    )


    # ========================================================
    # FINAL COSTS
    # ========================================================

    hitting_cost_total = np.sum(
        hitting_history
    )


    cumulative_memory_numeric = np.sum(
        memory_history,
        axis=0
    )


    long_term_cost_total = (

        RHO

        *

        np.max(
            cumulative_memory_numeric
        )
    )


    total_cost = (

        hitting_cost_total

        +

        long_term_cost_total
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

        "cumulative_memory":
            cumulative_memory_numeric,

        "hitting_cost":
            hitting_cost_total,

        "long_term_cost":
            long_term_cost_total,

        "total_cost":
            total_cost,

        "solver_objective":
            problem.value,

        "solver_status":
            problem.status,
    }