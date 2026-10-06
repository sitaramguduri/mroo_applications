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
    M,
    BETA,
    H,
    A_MATRICES,
)


# ============================================================
# VALIDATE MEMORY MATRICES
#
# There are exactly D matrices:
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
            "type":
                "eq",

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
# THEORETICAL S-MROO-SUM PARAMETERS
#
# For the SUM surrogate:
#
#     q(z) = ||z||_1
#
# There is NO rho.
#
# Its Lipschitz constant with respect to ||.||_2 is:
#
#     L_q = sqrt(D)
#
#
# For SUM / r = 1:
#
#     gamma = 1
#
#
# Pair 1:
#
# lambda_1
#
# =
#
# m /
# (
#     m * gamma
#     +
#     beta * L_q * H^2
# )
#
# lambda_2 = 0
#
#
# Pair 2:
#
# lambda_1 = 1
#
# lambda_2
#
# =
#
# m(gamma - 1)
# +
# beta * L_q * H^2
#
#
# NOTE:
# We retain the existing theoretical use of BETA here.
# If the theorem's beta is intended to represent the smoothness
# constant of the matrix-valued memory function, this may later
# need to incorporate max_i ||A_i||_2^2.
# ============================================================

def get_smroo_sum_parameters(
    D,
    parameter_pair=1,
):

    # --------------------------------------------------------
    # SUM norm:
    #
    # q(z) = ||z||_1
    #
    # Lipschitz constant with respect to l2:
    #
    # L_q = sqrt(D)
    # --------------------------------------------------------

    L_q = np.sqrt(
        D
    )


    # --------------------------------------------------------
    # SUM / r = 1:
    #
    # gamma = 1
    # --------------------------------------------------------

    gamma_R_q = 1.0


    # --------------------------------------------------------
    # Shared memory term
    # --------------------------------------------------------

    memory_constant = (

        BETA

        * L_q

        * H**2
    )


    # --------------------------------------------------------
    # Pair 1
    #
    # lambda_1
    #
    # =
    #
    # M /
    # (
    #     M * gamma
    #     +
    #     memory_constant
    # )
    #
    # lambda_2 = 0
    # --------------------------------------------------------

    if parameter_pair == 1:

        lambda_1 = (

            M

            /

            (
                M
                * gamma_R_q

                +

                memory_constant
            )
        )


        lambda_2 = 0.0


    # --------------------------------------------------------
    # Pair 2
    #
    # lambda_1 = 1
    #
    # lambda_2
    #
    # =
    #
    # M(gamma - 1)
    # +
    # memory_constant
    # --------------------------------------------------------

    elif parameter_pair == 2:

        lambda_1 = 1.0


        lambda_2 = (

            M

            * (
                gamma_R_q
                - 1.0
            )

            +

            memory_constant
        )


    else:

        raise ValueError(
            "parameter_pair must be 1 or 2."
        )


    return (

        lambda_1,

        lambda_2,

        gamma_R_q,

        L_q,
    )


# ============================================================
# ONE S-MROO-SUM STEP
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
#     lambda_1 q(d_t)
#
#     +
#
#     lambda_2 / 2
#     ||u-v_t||_2^2
#
# ]
#
#
# where:
#
# q(d_t)
#
# =
#
# ||d_t||_1
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
# ||A_i(u-u_prev)||_2^2
# ============================================================

def smroo_sum_step(
    y_t,
    u_prev,
    lambda_1,
    lambda_2,
):

    y_t = np.asarray(
        y_t,
        dtype=float,
    )


    u_prev = np.asarray(
        u_prev,
        dtype=float,
    )


    D = len(
        y_t
    )


    if D != D_CONFIG:

        raise ValueError(
            "y_t dimension does not match "
            "the configured A matrices. "
            f"y_t dimension = {D}, "
            f"configured D = {D_CONFIG}."
        )


    if u_prev.shape != (
        D,
    ):

        raise ValueError(
            "u_prev must have shape "
            f"({D},), "
            f"but got {u_prev.shape}."
        )


    # --------------------------------------------------------
    # Hitting-cost minimizer
    # --------------------------------------------------------

    v_t = compute_v_t(
        y_t
    )


    # --------------------------------------------------------
    # Objective
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
        #
        # *
        #
        # ||A_i(u-u_prev)||_2^2
        # ----------------------------------------------------

        d_t = memory_cost(

            u_t=u,

            u_prev=u_prev,

            A_matrices=A_MATRICES,
        )


        # ----------------------------------------------------
        # SUM surrogate
        #
        # q(d_t)
        #
        # =
        #
        # ||d_t||_1
        #
        # Since d_t >= 0:
        #
        # =
        #
        # sum_i d_{t,i}
        #
        # There is NO rho.
        # ----------------------------------------------------

        q_d = np.sum(
            d_t
        )


        static_memory = (

            lambda_1

            * q_d
        )


        # ----------------------------------------------------
        # Algorithm regularization
        #
        # lambda_2 / 2
        #
        # *
        #
        # ||u-v_t||_2^2
        # ----------------------------------------------------

        regularization = (

            lambda_2

            / 2.0

            * np.sum(
                (
                    u
                    - v_t
                ) ** 2
            )
        )


        return (

            f

            +

            static_memory

            +

            regularization
        )


    # --------------------------------------------------------
    # Solve primal update
    # --------------------------------------------------------

    result = minimize(

        fun=objective,

        x0=u_prev,

        method="SLSQP",

        bounds=simplex_bounds(
            D
        ),

        constraints=simplex_constraints(),

        options={
            "ftol": 1e-12,
            "maxiter": 500,
        },
    )


    if not result.success:

        raise RuntimeError(
            "S-MROO-SUM primal update failed: "
            f"{result.message}"
        )


    u_t = (
        result.x
    )


    # --------------------------------------------------------
    # Realized vector-valued memory cost
    # --------------------------------------------------------

    d_t = memory_cost(

        u_t=u_t,

        u_prev=u_prev,

        A_matrices=A_MATRICES,
    )


    return (

        u_t,

        v_t,

        d_t,
    )


# ============================================================
# RUN S-MROO-SUM
#
# If lambda_1 / lambda_2 are None:
#
#     use theoretical parameters.
#
# Otherwise:
#
#     use supplied values.
# ============================================================

def run_smroo_sum(
    demand,
    parameter_pair=1,
    lambda_1=None,
    lambda_2=None,
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
    # Compute theoretical parameters
    # --------------------------------------------------------

    (
        theoretical_lambda_1,

        theoretical_lambda_2,

        gamma_R_q,

        L_q,

    ) = get_smroo_sum_parameters(

        D=D,

        parameter_pair=parameter_pair,
    )


    # --------------------------------------------------------
    # Use theoretical values unless overridden
    # --------------------------------------------------------

    if lambda_1 is None:

        lambda_1 = (
            theoretical_lambda_1
        )


    if lambda_2 is None:

        lambda_2 = (
            theoretical_lambda_2
        )


    lambda_1 = float(
        lambda_1
    )


    lambda_2 = float(
        lambda_2
    )


    if lambda_1 <= 0.0:

        raise ValueError(
            "lambda_1 must be positive."
        )


    if lambda_2 < 0.0:

        raise ValueError(
            "lambda_2 must be nonnegative."
        )


    # --------------------------------------------------------
    # Initial allocation
    #
    # u_0 = uniform allocation
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
        # S-MROO-SUM step
        # ----------------------------------------------------

        (
            u_t,

            v_t,

            d_t,

        ) = smroo_sum_step(

            y_t=y_t,

            u_prev=u_prev,

            lambda_1=lambda_1,

            lambda_2=lambda_2,
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
        # Save vector memory cost
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
    # ARRAYS
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
    # ========================================================

    hitting_cost_total = float(

        np.sum(
            hitting_history
        )
    )


    # --------------------------------------------------------
    # Actual objective uses:
    #
    # || sum_t d_t ||_infinity
    #
    # NOT the SUM surrogate used by S-MROO-SUM internally.
    # --------------------------------------------------------

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
        # Final actual objective costs
        # ----------------------------------------------------

        "hitting_cost":
            hitting_cost_total,


        "long_term_cost":
            final_long_term_cost,


        "total_cost":
            total_cost,


        # ----------------------------------------------------
        # Algorithm parameters
        # ----------------------------------------------------

        "lambda_1":
            lambda_1,


        "lambda_2":
            lambda_2,


        "theoretical_lambda_1":
            theoretical_lambda_1,


        "theoretical_lambda_2":
            theoretical_lambda_2,


        "gamma_R_q":
            gamma_R_q,


        "L_q":
            L_q,


        "parameter_pair":
            parameter_pair,

    }