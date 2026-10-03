import sys
from pathlib import Path

import numpy as np

from scipy.optimize import (
    minimize,
    linprog,
)


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
    RHO,
    ELL,
)


# ============================================================
# AUXILIARY SET Z
#
# Same definition used by the current MROO implementation.
#
# Maximum possible scalar switching magnitude on the simplex:
#
#     Z_MAX
#       = 1/2 * (
#           largest ELL
#           + second-largest ELL
#         )
#
# Since ELL is scaled by MROO_BETA in config.py,
# Z_MAX automatically changes with beta.
# ============================================================

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
            "fun": lambda u:
                np.sum(u) - 1.0,
        }
    ]


def simplex_bounds(
    dim,
):

    return [
        (0.0, 1.0)
        for _ in range(dim)
    ]


# ============================================================
# AUXILIARY UPDATE
#
# z_t = argmin_z [
#
#     q(z) - <kappa_t, z>
#
# ]
#
# where
#
#     q(z) = rho ||z||_infinity
#
#
# Introduce scalar s such that
#
#     z_i <= s
#
# and solve
#
#     min rho*s - kappa^T z
#
# ============================================================

def compute_z_t(
    kappa_t,
    z_max=Z_MAX,
):

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )

    D = len(
        kappa_t
    )


    # --------------------------------------------------------
    # Decision variables:
    #
    # [z_1, ..., z_D, s]
    # --------------------------------------------------------

    n_variables = (
        D + 1
    )


    # --------------------------------------------------------
    # Objective:
    #
    # - kappa^T z + rho*s
    # --------------------------------------------------------

    c = np.zeros(
        n_variables,
        dtype=float,
    )

    c[:D] = (
        -kappa_t
    )

    c[D] = (
        RHO
    )


    # --------------------------------------------------------
    # Constraints:
    #
    # z_i - s <= 0
    # --------------------------------------------------------

    A_ub = np.zeros(
        (
            D,
            n_variables,
        ),
        dtype=float,
    )

    b_ub = np.zeros(
        D,
        dtype=float,
    )

    for i in range(D):

        A_ub[i, i] = 1.0

        A_ub[i, D] = -1.0


    # --------------------------------------------------------
    # Bounds:
    #
    # 0 <= z_i <= Z_MAX
    # 0 <= s   <= Z_MAX
    # --------------------------------------------------------

    bounds = (
        [
            (0.0, z_max)
            for _ in range(D)
        ]
        +
        [
            (0.0, z_max)
        ]
    )


    # --------------------------------------------------------
    # Solve LP
    # --------------------------------------------------------

    result = linprog(

        c=c,

        A_ub=A_ub,

        b_ub=b_ub,

        bounds=bounds,

        method="highs",
    )


    if not result.success:

        raise RuntimeError(
            "DMD auxiliary update failed: "
            f"{result.message}"
        )


    z_t = (
        result.x[:D]
    )


    return z_t


# ============================================================
# ONE DMD PRIMAL STEP
#
# u_t = argmin_u [
#
#     f_t(u)
#     +
#     <kappa_t, d_t(u, u_prev)>
#
# ]
#
#
# IMPORTANT:
#
# There is:
#
#   NO lambda_1
#   NO lambda_2
#   NO v_t regularization
#
# ============================================================

def dmd_step(
    y_t,
    u_prev,
    w_t,
    kappa_t,
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
    # DMD PRIMAL OBJECTIVE
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
        # Memory-cost vector
        #
        # d_t(u, u_prev)
        # ----------------------------------------------------

        d_t = memory_cost(
            u,
            u_prev,
            w_t,
        )


        # ----------------------------------------------------
        # Dual-weighted memory surrogate
        #
        # <kappa_t, d_t>
        # ----------------------------------------------------

        dual_memory = (
            np.dot(
                kappa_t,
                d_t,
            )
        )


        # ----------------------------------------------------
        # DMD OBJECTIVE
        # ----------------------------------------------------

        return (
            f
            +
            dual_memory
        )


    # --------------------------------------------------------
    # Solve primal update
    #
    # Warm start from previous allocation.
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
            "DMD primal update failed: "
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
        d_t,
    )


# ============================================================
# DUAL UPDATE
#
# g_t = z_t - d_t
#
# Euclidean mirror descent:
#
# kappa_{t+1}
#
#     = [kappa_t - eta*g_t]_+
#
#     = [kappa_t + eta*(d_t-z_t)]_+
#
# ============================================================

def dual_update(
    kappa_t,
    d_t,
    z_t,
    eta,
):

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )

    d_t = np.asarray(
        d_t,
        dtype=float,
    )

    z_t = np.asarray(
        z_t,
        dtype=float,
    )


    # --------------------------------------------------------
    # DMD subgradient
    # --------------------------------------------------------

    g_t = (
        z_t
        -
        d_t
    )


    # --------------------------------------------------------
    # Project onto nonnegative orthant
    # --------------------------------------------------------

    kappa_next = np.maximum(

        0.0,

        kappa_t
        -
        eta
        * g_t,
    )


    return (
        kappa_next,
        g_t,
    )


# ============================================================
# RUN DMD
# ============================================================

def run_dmd(
    demand,
    eta=None,
    kappa_init=None,
    z_max=Z_MAX,
):

    demand = np.asarray(
        demand,
        dtype=float,
    )


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    T, D = (
        demand.shape
    )


    # ========================================================
    # DEFAULT THEORETICAL STEP SIZE
    #
    # Current convention:
    #
    # eta = T^{-1/3}
    # ========================================================

    if eta is None:

        eta = (
            T ** (-1.0 / 3.0)
        )


    eta = float(
        eta
    )


    if eta <= 0.0:

        raise ValueError(
            "eta must be positive."
        )


    # ========================================================
    # INITIAL ALLOCATION
    #
    # u_0 = uniform simplex allocation
    # ========================================================

    u_prev = (
        np.ones(D)
        / D
    )


    # ========================================================
    # INITIAL DUAL VARIABLE
    #
    # Current convention:
    #
    # kappa_1 = (1/T) * 1
    # ========================================================

    if kappa_init is None:

        kappa_t = (
            np.ones(D)
            / T
        )

    else:

        kappa_t = np.asarray(
            kappa_init,
            dtype=float,
        )


        if kappa_t.ndim == 0:

            kappa_t = (
                np.ones(D)
                * float(kappa_t)
            )


        if kappa_t.shape != (D,):

            raise ValueError(
                f"kappa_init must have shape ({D},), "
                f"but got {kappa_t.shape}."
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

    switching_history = []

    kappa_history = []

    z_history = []

    gradient_history = []


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(T):


        # ----------------------------------------------------
        # Current demand
        # ----------------------------------------------------

        y_t = (
            demand[t]
        )


        # ----------------------------------------------------
        # Current exposure vector
        # ----------------------------------------------------

        w_t = get_w_t(
            t=t,
            y_t=y_t,
        )


        # ----------------------------------------------------
        # Save kappa used for current primal decision
        # ----------------------------------------------------

        kappa_history.append(
            kappa_t.copy()
        )


        # ====================================================
        # 1. PRIMAL UPDATE
        #
        # min f_t(u) + <kappa_t, d_t(u,u_prev)>
        # ====================================================

        (
            u_t,
            d_t,
        ) = dmd_step(

            y_t=y_t,

            u_prev=u_prev,

            w_t=w_t,

            kappa_t=kappa_t,
        )


        # ====================================================
        # 2. AUXILIARY UPDATE
        #
        # z_t = argmin q(z) - <kappa_t,z>
        # ====================================================

        z_t = compute_z_t(

            kappa_t=kappa_t,

            z_max=z_max,
        )


        # ====================================================
        # 3. DUAL UPDATE
        #
        # kappa_{t+1}
        #
        # = [kappa_t + eta(d_t-z_t)]_+
        # ====================================================

        (
            kappa_next,
            g_t,
        ) = dual_update(

            kappa_t=kappa_t,

            d_t=d_t,

            z_t=z_t,

            eta=eta,
        )


        # ====================================================
        # SAVE CURRENT STEP
        # ====================================================

        actions.append(
            u_t.copy()
        )


        hitting_history.append(

            hitting_cost(
                u_t,
                y_t,
            )

        )


        memory_history.append(
            d_t.copy()
        )


        switching_history.append(

            np.linalg.norm(
                d_t,
                ord=2,
            )

        )


        z_history.append(
            z_t.copy()
        )


        gradient_history.append(
            g_t.copy()
        )


        # ====================================================
        # NEXT ROUND
        # ====================================================

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

    z_history = np.asarray(
        z_history
    )

    gradient_history = np.asarray(
        gradient_history
    )


    # ========================================================
    # FINAL CUMULATIVE COST
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
    # RETURN
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


        "switching_history":
            switching_history,


        "kappa_history":
            kappa_history,


        "z_history":
            z_history,


        "gradient_history":
            gradient_history,


        # ----------------------------------------------------
        # Final costs
        # ----------------------------------------------------

        "hitting_cost":
            hitting_cost_total,


        "long_term_cost":
            final_long_term_cost,


        "total_cost":
            total_cost,


        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        "eta":
            eta,


        "kappa_init":
            (
                np.ones(D) / T
                if kappa_init is None
                else np.asarray(
                    kappa_init,
                    dtype=float,
                )
            ),


        "z_max":
            float(z_max),

    }