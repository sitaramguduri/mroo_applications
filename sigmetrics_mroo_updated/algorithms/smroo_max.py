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

from weights import get_w_t

from config import (
    M,
    BETA,
    H,
    RHO,
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
# v_t = argmin f_t(u)
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
# THEORETICAL S-MROO-MAX PARAMETERS
#
# q(z) = rho ||z||_inf
#
# L_q = rho
#
# gamma_{R,inf} = min(R,D)
#
# Pair 1:
#
# lambda_1 =
#
# m / (
#     m gamma
#     + beta L_q H^2
# )
#
# lambda_2 = 0
#
#
# Pair 2:
#
# lambda_1 = 1
#
# lambda_2 =
#
# m(gamma - 1)
# + beta L_q H^2
# ============================================================

def get_smroo_max_parameters(
    D,
    R=None,
    parameter_pair=1,
):

    # --------------------------------------------------------
    # MAX-norm Lipschitz constant
    # --------------------------------------------------------

    L_q = float(
        RHO
    )


    # --------------------------------------------------------
    # Decomposition factor
    #
    # gamma_{R,inf} = min(R,D)
    # --------------------------------------------------------

    if R is None:

        gamma_R_q = float(
            D
        )

    else:

        gamma_R_q = float(
            min(
                R,
                D,
            )
        )


    # --------------------------------------------------------
    # beta * L_q * H^2
    # --------------------------------------------------------

    memory_constant = (
        BETA
        * L_q
        * H**2
    )


    # --------------------------------------------------------
    # THEORETICAL PAIR 1
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
    # THEORETICAL PAIR 2
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
# ONE S-MROO-MAX STEP
#
# Algorithm 1:
#
# v_t = argmin f_t(u)
#
# u_t = argmin [
#
#     f_t(u)
#
#     + lambda_1 q(d_t)
#
#     + lambda_2 / 2 ||u-v_t||^2
#
# ]
#
# MAX:
#
# q(d_t) = rho ||d_t||_inf
# ============================================================

def smroo_max_step(
    y_t,
    u_prev,
    w_t,
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

    w_t = np.asarray(
        w_t,
        dtype=float,
    )


    # --------------------------------------------------------
    # Hitting-cost minimizer
    # --------------------------------------------------------

    v_t = compute_v_t(
        y_t
    )


    # --------------------------------------------------------
    # S-MROO-MAX objective
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
        # ----------------------------------------------------

        d_t = memory_cost(
            u,
            u_prev,
            w_t,
        )


        # ----------------------------------------------------
        # q(d_t) = rho ||d_t||_inf
        # ----------------------------------------------------

        q_d = (
            RHO
            * np.max(
                np.abs(
                    d_t
                )
            )
        )


        # ----------------------------------------------------
        # Static memory penalty
        # ----------------------------------------------------

        static_memory = (
            lambda_1
            * q_d
        )


        # ----------------------------------------------------
        # Algorithm 1 regularization:
        #
        # lambda_2 / 2 ||u-v_t||^2
        #
        # No additional m factor.
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
            "S-MROO-MAX primal update failed: "
            f"{result.message}"
        )


    u_t = (
        result.x
    )


    # --------------------------------------------------------
    # Realized memory cost
    # --------------------------------------------------------

    d_t = memory_cost(
        u_t,
        u_prev,
        w_t,
    )


    return (
        u_t,
        v_t,
        d_t,
    )


# ============================================================
# RUN S-MROO-MAX
#
# If lambda_1 and lambda_2 are not supplied:
#
# use the theoretical parameter pair.
#
# Otherwise:
#
# use the explicitly supplied values.
# ============================================================

def run_smroo_max(
    demand,
    parameter_pair=1,
    R=None,
    lambda_1=None,
    lambda_2=None,
):

    demand = np.asarray(
        demand,
        dtype=float,
    )


    T, D = (
        demand.shape
    )


    # --------------------------------------------------------
    # Frame size
    #
    # For the full-horizon experiment we use R = T.
    # --------------------------------------------------------

    if R is None:

        R = T


    # --------------------------------------------------------
    # Compute theoretical parameters
    # --------------------------------------------------------

    (
        theoretical_lambda_1,
        theoretical_lambda_2,
        gamma_R_q,
        L_q,
    ) = get_smroo_max_parameters(

        D=D,

        R=R,

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


    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

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

        y_t = (
            demand[t]
        )


        # ----------------------------------------------------
        # Exposure vector
        # ----------------------------------------------------

        w_t = get_w_t(
            t=t,
            y_t=y_t,
        )


        # ----------------------------------------------------
        # S-MROO-MAX update
        # ----------------------------------------------------

        (
            u_t,
            v_t,
            d_t,
        ) = smroo_max_step(

            y_t=y_t,

            u_prev=u_prev,

            w_t=w_t,

            lambda_1=lambda_1,

            lambda_2=lambda_2,
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
        # Store memory vector
        # ----------------------------------------------------

        memory_history.append(
            d_t.copy()
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
        # Next round
        # ----------------------------------------------------

        u_prev = (
            u_t
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

    switching_history = np.asarray(
        switching_history
    )


    # ========================================================
    # FINAL CUMULATIVE COST
    #
    # Same evaluation objective:
    #
    # sum_t f_t
    #
    # +
    #
    # rho ||sum_t d_t||_inf
    # ========================================================

    hitting_cost_total = (
        np.sum(
            hitting_history
        )
    )


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

        "hitting_cost":
            hitting_cost_total,

        "long_term_cost":
            final_long_term_cost,

        "total_cost":
            total_cost,

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

        "R":
            R,

        "parameter_pair":
            parameter_pair,
    }