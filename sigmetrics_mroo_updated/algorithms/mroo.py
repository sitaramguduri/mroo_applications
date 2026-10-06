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
    PROJECT_DIR,
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
# IMPORT CONFIGURATION
#
# BETA
#     memory-cost scaling
#
# A_MATRICES
#     A_1, ..., A_D
#
# M
#     strong-convexity / theory parameter
#
# LAMBDA_1_THEORY
#     default lambda_1
#
# LAMBDA_2
#     regularization parameter
# ============================================================

from config import (
    BETA,
    A_MATRICES,
    M,
    LAMBDA_1_THEORY,
    LAMBDA_2,
)


# ============================================================
# VALIDATE MEMORY MATRICES
#
# We require exactly D matrices:
#
#     A_1, ..., A_D
#
# and each:
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


D = (
    A_MATRICES.shape[0]
)


if A_MATRICES.shape != (
    D,
    D,
    D,
):

    raise ValueError(
        "A_MATRICES must have shape "
        f"({D}, {D}, {D}), "
        f"but got {A_MATRICES.shape}."
    )


# ============================================================
# AUXILIARY SET Z
#
# Memory cost:
#
# d_{t,i}
#
# =
#
# beta / 2
# *
# || A_i (u_t - u_{t-1}) ||_2^2
#
#
# Since both u_t and u_{t-1} lie on the simplex:
#
# ||u_t - u_{t-1}||_2^2 <= 2
#
#
# Therefore:
#
# d_{t,i}
#
# <=
#
# beta / 2
# *
# ||A_i||_2^2
# *
# 2
#
# =
#
# beta * ||A_i||_2^2
#
#
# A common safe upper bound is:
#
# Z_MAX
#
# =
#
# beta * max_i ||A_i||_2^2
# ============================================================

A_SPECTRAL_NORMS = np.asarray(
    [
        np.linalg.norm(
            A_i,
            ord=2,
        )

        for A_i in A_MATRICES
    ],
    dtype=float,
)


Z_MAX = (

    BETA

    * np.max(
        A_SPECTRAL_NORMS ** 2
    )
)


# ============================================================
# PRINT MEMORY CONFIGURATION
# ============================================================

print(
    "\nMROO MEMORY CONFIG:"
)

print(
    "  beta =",
    BETA,
)

print(
    "  D =",
    D,
)

print(
    "  number of A matrices =",
    len(
        A_MATRICES
    ),
)

print(
    "  A spectral norms =",
    A_SPECTRAL_NORMS,
)

print(
    "  Z_MAX =",
    Z_MAX,
)


for i, A_i in enumerate(
    A_MATRICES,
    start=1,
):

    print(
        f"\n  A_{i} ="
    )

    print(
        A_i
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
# HITTING-COST MINIMIZER
#
# v_t
#
# =
#
# argmin_u f_t(u)
#
# subject to:
#
# u >= 0
# sum_i u_i = 1
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
# u_t
#
# =
#
# argmin_u [
#
#     f_t(u)
#
#     +
#
#     lambda_1
#     <kappa_t, d_t(u, u_prev)>
#
#     +
#
#     lambda_2
#     * M/2
#     * ||u - v_t||_2^2
#
# ]
#
#
# where:
#
# d_{t,i}
#
# =
#
# beta / 2
# *
# ||A_i(u-u_prev)||_2^2
# ============================================================

def primal_update(
    y_t,
    u_prev,
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

    def objective(
        u,
    ):

        # ----------------------------------------------------
        # Hitting cost
        # ----------------------------------------------------

        f = hitting_cost(
            u,
            y_t,
        )


        # ----------------------------------------------------
        # Vector-valued memory cost
        #
        # d_{t,i}
        #
        # =
        #
        # beta / 2
        # *
        # ||A_i(u-u_prev)||_2^2
        # ----------------------------------------------------

        d_t = memory_cost(

            u_t=u,

            u_prev=u_prev,

            A_matrices=A_MATRICES,
        )


        # ----------------------------------------------------
        # Dual-weighted memory term
        #
        # lambda_1
        # *
        # <kappa_t, d_t>
        # ----------------------------------------------------

        dual_memory = (

            lambda_1

            * np.dot(
                kappa_t,
                d_t,
            )
        )


        # ----------------------------------------------------
        # Regularization toward v_t
        # ----------------------------------------------------

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
# Long-term cost:
#
# q(z)
#
# =
#
# ||z||_inf
#
#
# Therefore:
#
# z_t
#
# =
#
# argmin_z [
#
#     ||z||_inf
#
#     -
#
#     <kappa_t, z>
#
# ]
#
#
# subject to:
#
# 0 <= z_i <= Z_MAX
#
#
# There is NO rho term.
# ============================================================

def auxiliary_update(
    kappa_t,
):

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )


    if kappa_t.shape != (
        D,
    ):

        raise ValueError(
            "kappa_t must have shape "
            f"({D},), "
            f"but got {kappa_t.shape}."
        )


    # --------------------------------------------------------
    # LP variables:
    #
    # [z_1, ..., z_D, s]
    #
    # where:
    #
    # s >= z_i
    #
    # Therefore:
    #
    # s = ||z||_inf
    #
    # at optimum.
    # --------------------------------------------------------

    objective = np.concatenate(
        [
            -kappa_t,

            [1.0],
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


    for i in range(
        D
    ):

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


    # --------------------------------------------------------
    # Bounds
    #
    # 0 <= z_i <= Z_MAX
    #
    # 0 <= s <= Z_MAX
    # --------------------------------------------------------

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


    return result.x[
        :D
    ]


# ============================================================
# DUAL UPDATE
#
# g_t
#
# =
#
# z_t
#
# -
#
# lambda_1 d_t
#
#
# kappa_{t+1}
#
# =
#
# [kappa_t - eta g_t]_+
#
#
# equivalently:
#
# kappa_{t+1}
#
# =
#
# [
#     kappa_t
#
#     +
#
#     eta(
#         lambda_1 d_t
#         -
#         z_t
#     )
# ]_+
# ============================================================

def dual_update(
    kappa_t,
    z_t,
    d_t,
    eta,
    lambda_1,
):

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )

    z_t = np.asarray(
        z_t,
        dtype=float,
    )

    d_t = np.asarray(
        d_t,
        dtype=float,
    )


    if kappa_t.shape != (
        D,
    ):

        raise ValueError(
            "kappa_t must have shape "
            f"({D},), "
            f"but got {kappa_t.shape}."
        )


    if z_t.shape != (
        D,
    ):

        raise ValueError(
            "z_t must have shape "
            f"({D},), "
            f"but got {z_t.shape}."
        )


    if d_t.shape != (
        D,
    ):

        raise ValueError(
            "d_t must have shape "
            f"({D},), "
            f"but got {d_t.shape}."
        )


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
#     dual step size
#
# kappa_init:
#     initial D-dimensional dual vector
#
# lambda_1:
#     scalar lambda_1 for this run
#
#
# If lambda_1 is not supplied:
#
#     use LAMBDA_1_THEORY
#     from config.py
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


    if demand.ndim != 2:

        raise ValueError(
            "demand must be a 2-dimensional array. "
            f"Found shape {demand.shape}."
        )


    (
        T,
        demand_D,
    ) = (
        demand.shape
    )


    # --------------------------------------------------------
    # Validate demand dimension
    # --------------------------------------------------------

    if demand_D != D:

        raise ValueError(
            "Demand dimension does not match "
            "the configured dimension. "
            f"Demand dimension = {demand_D}, "
            f"D = {D}."
        )


    # --------------------------------------------------------
    # Initial action
    #
    # u_0 = uniform allocation
    # --------------------------------------------------------

    u_prev = (

        np.ones(
            D
        )

        / D
    )


    # --------------------------------------------------------
    # Initial dual variable
    # --------------------------------------------------------

    kappa_t = np.asarray(
        kappa_init,
        dtype=float,
    ).copy()


    if kappa_t.shape != (
        D,
    ):

        raise ValueError(
            "kappa_init must have "
            f"shape ({D},), "
            f"found {kappa_t.shape}."
        )


    if np.any(
        kappa_t < 0.0
    ):

        raise ValueError(
            "kappa_init must be nonnegative."
        )


    # ========================================================
    # HISTORIES
    # ========================================================

    actions = []

    hitting_history = []

    memory_history = []

    kappa_history = [
        kappa_t.copy()
    ]


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(
        T
    ):

        y_t = demand[
            t
        ]


        # ----------------------------------------------------
        # Primal update
        # ----------------------------------------------------

        (
            u_t,
            v_t,
        ) = primal_update(

            y_t=y_t,

            u_prev=u_prev,

            kappa_t=kappa_t,

            lambda_1=lambda_1,
        )


        # ----------------------------------------------------
        # Realized vector-valued memory cost
        #
        # d_{t,i}
        #
        # =
        #
        # beta / 2
        # *
        # ||A_i(u_t-u_{t-1})||_2^2
        # ----------------------------------------------------

        d_t = memory_cost(

            u_t=u_t,

            u_prev=u_prev,

            A_matrices=A_MATRICES,
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
        # Save vector-valued memory cost
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
        # Move to next round
        # ----------------------------------------------------

        u_prev = (
            u_t.copy()
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

    kappa_history = np.asarray(
        kappa_history
    )


    # ========================================================
    # FINAL CUMULATIVE COSTS
    # ========================================================

    hitting_cost_total = np.sum(
        hitting_history
    )


    # --------------------------------------------------------
    # Long-term cost:
    #
    # || sum_t d_t ||_inf
    #
    # =
    #
    # max_i
    # sum_t
    #
    # beta / 2
    # *
    # ||A_i(u_t-u_{t-1})||_2^2
    # --------------------------------------------------------

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
                kappa_init,
                dtype=float,
            ).copy(),

        "lambda_1":
            lambda_1,

        "lambda_2":
            LAMBDA_2,

    }