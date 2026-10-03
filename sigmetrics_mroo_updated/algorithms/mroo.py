import numpy as np

from scipy.optimize import (
    minimize,
    linprog,
)

import sys

from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

print(
    "project dir:",
    PROJECT_DIR
)

if str(PROJECT_DIR) not in sys.path:

    sys.path.append(
        str(PROJECT_DIR)
    )


# ============================================================
# IMPORT COST FUNCTIONS
# ============================================================

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost,
)


# ============================================================
# IMPORT WEIGHTS
# ============================================================

from weights import get_w_t


# ============================================================
# IMPORT CONFIGURATION
#
# Important:
#
# LAMBDA_1_THEORY is only the DEFAULT lambda_1.
#
# Individual runs may provide another scalar lambda_1,
# such as 0.1 or 0.01.
# ============================================================

from config import (
    ELL,
    M,
    RHO,
    LAMBDA_1_THEORY,
    LAMBDA_2,
)


# ============================================================
# AUXILIARY SET Z
# ============================================================

# Maximum possible scalar switching magnitude
# on the simplex.
#
# Maximum occurs when moving between two
# simplex vertices.

sorted_l = np.sort(
    ELL
)

Z_MAX = (
    0.5
    * (
        sorted_l[-1]
        +
        sorted_l[-2]
    )
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


def simplex_bounds(dim):

    return [
        (0.0, 1.0)
    ] * dim


# ============================================================
# HITTING-COST MINIMIZER
#
# v_t = argmin_u f_t(u)
# ============================================================

def compute_v_t(
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
            "Failed to compute v_t: "
            f"{result.message}"
        )


    return result.x


# ============================================================
# PRIMAL UPDATE
#
# u_t = argmin [
#
#     f_t(u)
#
#     + lambda_1
#       <kappa_t, d_t(u, u_prev)>
#
#     + lambda_2 * m/2
#       ||u - v_t||^2
#
# ]
# ============================================================

def primal_update(
    y_t,
    u_prev,
    w_t,
    kappa_t,
    lambda_1,
):

    y_t = np.asarray(
        y_t,
        dtype=float,
    )

    u_prev = np.asarray(
        u_prev,
        dtype=float,
    )

    w_t = np.asarray(
        w_t,
        dtype=float,
    )

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )


    # --------------------------------------------------------
    # Compute v_t
    # --------------------------------------------------------

    v_t = compute_v_t(
        y_t
    )


    # --------------------------------------------------------
    # MROO primal objective
    # --------------------------------------------------------

    def objective(u):

        # Hitting cost

        f = hitting_cost(
            u,
            y_t,
        )


        # Memory cost vector

        d_t = memory_cost(
            u,
            u_prev,
            w_t,
        )


        # Dual-weighted memory term

        dual_memory = (
            lambda_1
            * np.dot(
                kappa_t,
                d_t,
            )
        )


        # Regularization toward v_t

        regularization = (
            LAMBDA_2
            * M
            / 2.0
            * np.sum(
                (u - v_t) ** 2
            )
        )


        return (
            f
            +
            dual_memory
            +
            regularization
        )


    # --------------------------------------------------------
    # Solve primal problem
    # --------------------------------------------------------

    result = minimize(

        fun=objective,

        x0=u_prev,

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
            "MROO primal update failed: "
            f"{result.message}"
        )


    return (
        result.x,
        v_t,
    )


# ============================================================
# AUXILIARY UPDATE
#
# z_t = argmin [
#
#     rho ||z||_inf
#
#     - <kappa_t, z>
#
# ]
#
# subject to
#
#     0 <= z_i <= Z_MAX
#
# ============================================================

def auxiliary_update(
    kappa_t,
):

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )


    D = len(
        kappa_t
    )


    # --------------------------------------------------------
    # LP variables:
    #
    # [z_1, ..., z_D, s]
    #
    # s >= z_i
    #
    # therefore
    #
    # s = ||z||_inf
    #
    # at optimum.
    # --------------------------------------------------------

    objective = np.concatenate(
        [
            -kappa_t,
            [RHO],
        ]
    )


    # --------------------------------------------------------
    # Constraints:
    #
    # z_i - s <= 0
    # --------------------------------------------------------

    A_ub = np.zeros(
        (
            D,
            D + 1,
        )
    )


    for i in range(D):

        A_ub[
            i,
            i,
        ] = 1.0

        A_ub[
            i,
            -1,
        ] = -1.0


    b_ub = np.zeros(
        D
    )


    bounds = (
        [
            (0.0, Z_MAX)
        ] * D

        +

        [
            (0.0, Z_MAX)
        ]
    )


    result = linprog(

        c=objective,

        A_ub=A_ub,

        b_ub=b_ub,

        bounds=bounds,

        method="highs",
    )


    if not result.success:

        raise RuntimeError(
            "Auxiliary update failed: "
            f"{result.message}"
        )


    return result.x[:D]


# ============================================================
# DUAL UPDATE
#
# g_t =
#
#     z_t
#     - lambda_1 d_t
#
#
# kappa_{t+1}
#
#     =
#
# [kappa_t - eta g_t]_+
#
# ============================================================

def dual_update(
    kappa_t,
    z_t,
    d_t,
    eta,
    lambda_1,
):

    # --------------------------------------------------------
    # Subgradient
    # --------------------------------------------------------

    g_t = (
        z_t
        -
        lambda_1
        * d_t
    )


    # --------------------------------------------------------
    # Gradient / mirror-descent update
    # --------------------------------------------------------

    kappa_next = (
        kappa_t
        -
        eta
        * g_t
    )


    # --------------------------------------------------------
    # Projection onto nonnegative orthant
    # --------------------------------------------------------

    kappa_next = np.maximum(
        kappa_next,
        0.0,
    )


    return kappa_next


# ============================================================
# RUN MROO
#
# eta:
#     step size
#
# kappa_init:
#     initial dual vector
#
# lambda_1:
#     scalar MROO lambda_1 for THIS run
#
# If lambda_1 is not supplied, use the theoretical value
# from config.py.
# ============================================================

def run_mroo(
    demand,
    eta,
    kappa_init,
    lambda_1=None,
):

    # --------------------------------------------------------
    # Default lambda_1
    # --------------------------------------------------------

    if lambda_1 is None:

        lambda_1 = (
            LAMBDA_1_THEORY
        )


    lambda_1 = float(
        lambda_1
    )


    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if lambda_1 <= 0.0:

        raise ValueError(
            "lambda_1 must be positive."
        )


    if eta <= 0.0:

        raise ValueError(
            "eta must be positive."
        )


    # --------------------------------------------------------
    # Demand
    # --------------------------------------------------------

    demand = np.asarray(
        demand,
        dtype=float,
    )


    T, D = (
        demand.shape
    )


    # --------------------------------------------------------
    # Initial action
    # --------------------------------------------------------

    u_prev = (
        np.ones(D)
        / D
    )


    # --------------------------------------------------------
    # Initial dual variable
    # --------------------------------------------------------

    kappa_t = np.asarray(
        kappa_init,
        dtype=float,
    ).copy()


    if kappa_t.shape != (D,):

        raise ValueError(
            "kappa_init must have "
            f"shape ({D},), "
            f"found {kappa_t.shape}."
        )


    # ========================================================
    # HISTORIES
    # ========================================================

    actions = []

    hitting_history = []

    memory_history = []

    switching_history = []

    kappa_history = [
        kappa_t.copy()
    ]


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(T):

        y_t = demand[t]


        # ----------------------------------------------------
        # Exposure vector
        # ----------------------------------------------------

        w_t = get_w_t(
            t=t,
            y_t=y_t,
        )


        # ----------------------------------------------------
        # Primal update
        # ----------------------------------------------------

        u_t, v_t = primal_update(

            y_t=y_t,

            u_prev=u_prev,

            w_t=w_t,

            kappa_t=kappa_t,

            lambda_1=lambda_1,
        )


        # ----------------------------------------------------
        # Realized memory cost
        # ----------------------------------------------------

        d_t = memory_cost(
            u_t,
            u_prev,
            w_t,
        )


        # ----------------------------------------------------
        # Auxiliary update
        # ----------------------------------------------------

        z_t = auxiliary_update(
            kappa_t
        )


        # ----------------------------------------------------
        # Dual update
        # ----------------------------------------------------

        kappa_next = dual_update(

            kappa_t=kappa_t,

            z_t=z_t,

            d_t=d_t,

            eta=eta,

            lambda_1=lambda_1,
        )


        # ----------------------------------------------------
        # Save action
        # ----------------------------------------------------

        actions.append(
            u_t.copy()
        )


        # ----------------------------------------------------
        # Save hitting cost
        # ----------------------------------------------------

        hitting_history.append(

            hitting_cost(
                u_t,
                y_t,
            )

        )


        # ----------------------------------------------------
        # Save memory vector
        # ----------------------------------------------------

        memory_history.append(
            d_t.copy()
        )


        # ----------------------------------------------------
        # Save updated dual variable
        # ----------------------------------------------------

        kappa_history.append(
            kappa_next.copy()
        )


        # ----------------------------------------------------
        # Scalar switching magnitude
        #
        # Since ||w_t||_2 = 1:
        #
        # ||d_t||_2 = d_bar
        # ----------------------------------------------------

        switching_history.append(

            np.linalg.norm(
                d_t,
                ord=2,
            )

        )


        # ----------------------------------------------------
        # Move to next round
        # ----------------------------------------------------

        u_prev = (
            u_t
        )

        kappa_t = (
            kappa_next
        )


    # ========================================================
    # CONVERT HISTORIES TO ARRAYS
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

    kappa_history = np.asarray(
        kappa_history
    )


    # ========================================================
    # FINAL CUMULATIVE COSTS
    # ========================================================

    hitting_cost_total = np.sum(
        hitting_history
    )


    # Current cumulative convention:
    #
    # rho * || sum_t d_t ||_inf

    final_long_term_cost = (
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

        "kappa_history":
            kappa_history,

        "hitting_cost":
            hitting_cost_total,

        "long_term_cost":
            final_long_term_cost,

        "total_cost":
            total_cost,

        "eta":
            eta,

        "kappa_init":
            np.asarray(
                kappa_init
            ).copy(),

        "lambda_1":
            lambda_1,

        "lambda_2":
            LAMBDA_2,
    }