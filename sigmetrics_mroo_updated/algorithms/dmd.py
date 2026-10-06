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
# IMPORT COST FUNCTIONS
# ============================================================

from cost import (
    hitting_cost,
    memory_cost,
    long_term_cost,
)


# ============================================================
# IMPORT CONFIGURATION
# ============================================================

from config import (
    BETA,
    A_MATRICES,
)


# ============================================================
# VALIDATE MEMORY MATRICES
#
# There are exactly D memory matrices:
#
#     A_1, ..., A_D
#
# and each matrix is D x D.
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


D = int(
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
# Since u_t and u_{t-1} lie on the simplex:
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
# beta * ||A_i||_2^2
#
#
# We use:
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
    "\nDMD MEMORY CONFIG:"
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
    len(A_MATRICES),
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

        for _ in range(
            dim
        )
    ]


# ============================================================
# AUXILIARY UPDATE
#
# New long-term cost:
#
# q(z)
#
# =
#
# ||z||_infinity
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
#     ||z||_infinity
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
#     0 <= z_i <= Z_MAX
#
#
# Introduce scalar s:
#
#     z_i <= s
#
# so that:
#
#     s = ||z||_infinity
#
#
# LP:
#
# min
#
#     s - kappa_t^T z
#
#
# There is NO rho.
# ============================================================

def compute_z_t(
    kappa_t,
    z_max=Z_MAX,
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
    # -kappa_t^T z + s
    #
    # coefficient of s is 1 because:
    #
    # q(z) = ||z||_infinity
    # --------------------------------------------------------

    c = np.zeros(
        n_variables,
        dtype=float,
    )


    c[:D] = (
        -kappa_t
    )


    c[D] = 1.0


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


    for i in range(
        D
    ):

        A_ub[
            i,
            i,
        ] = 1.0

        A_ub[
            i,
            D,
        ] = -1.0


    # --------------------------------------------------------
    # Bounds:
    #
    # 0 <= z_i <= Z_MAX
    #
    # 0 <= s <= Z_MAX
    # --------------------------------------------------------

    bounds = (

        [
            (0.0, z_max)

            for _ in range(
                D
            )
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
#     <kappa_t, d_t(u, u_prev)>
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
#
#
# IMPORTANT:
#
# DMD has:
#
#     NO lambda_1
#
#     NO lambda_2
#
#     NO v_t regularization
# ============================================================

def dmd_step(
    y_t,
    u_prev,
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

    kappa_t = np.asarray(
        kappa_t,
        dtype=float,
    )


    if y_t.shape != (
        D,
    ):

        raise ValueError(
            "y_t must have shape "
            f"({D},), "
            f"but got {y_t.shape}."
        )


    if u_prev.shape != (
        D,
    ):

        raise ValueError(
            "u_prev must have shape "
            f"({D},), "
            f"but got {u_prev.shape}."
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
    # DMD primal objective
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
        # DMD objective
        # ----------------------------------------------------

        return (
            f
            +
            dual_memory
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
            "DMD primal update failed: "
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
        d_t,
    )


# ============================================================
# DUAL UPDATE
#
# g_t
#
# =
#
# z_t - d_t
#
#
# Euclidean mirror descent:
#
# kappa_{t+1}
#
# =
#
# [kappa_t - eta g_t]_+
#
#
# =
#
# [kappa_t + eta(d_t-z_t)]_+
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


    if kappa_t.shape != (
        D,
    ):

        raise ValueError(
            "kappa_t must have shape "
            f"({D},), "
            f"but got {kappa_t.shape}."
        )


    if d_t.shape != (
        D,
    ):

        raise ValueError(
            "d_t must have shape "
            f"({D},), "
            f"but got {d_t.shape}."
        )


    if z_t.shape != (
        D,
    ):

        raise ValueError(
            "z_t must have shape "
            f"({D},), "
            f"but got {z_t.shape}."
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
    # Projected Euclidean mirror-descent update
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
    # Validate demand
    # --------------------------------------------------------

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


    if demand_D != D:

        raise ValueError(
            "Demand dimension does not match "
            "the configured dimension. "
            f"Demand dimension = {demand_D}, "
            f"D = {D}."
        )


    # ========================================================
    # DEFAULT THEORETICAL STEP SIZE
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

        np.ones(
            D
        )

        / D
    )


    # ========================================================
    # INITIAL DUAL VARIABLE
    #
    # Default:
    #
    # kappa_1 = (1/T) * 1
    # ========================================================

    if kappa_init is None:

        kappa_t = (

            np.ones(
                D
            )

            / T
        )

    else:

        kappa_t = np.asarray(
            kappa_init,
            dtype=float,
        )


        # ----------------------------------------------------
        # Scalar kappa:
        #
        # k -> [k, ..., k]
        # ----------------------------------------------------

        if kappa_t.ndim == 0:

            kappa_t = (

                np.ones(
                    D
                )

                * float(
                    kappa_t
                )
            )


        if kappa_t.shape != (
            D,
        ):

            raise ValueError(
                f"kappa_init must have shape ({D},), "
                f"but got {kappa_t.shape}."
            )


    if np.any(
        ~np.isfinite(
            kappa_t
        )
    ):

        raise ValueError(
            "kappa_init must contain only finite values."
        )


    if np.any(
        kappa_t < 0.0
    ):

        raise ValueError(
            "kappa_init must be nonnegative."
        )


    # --------------------------------------------------------
    # Save the actual initial kappa used
    # --------------------------------------------------------

    initial_kappa = (
        kappa_t.copy()
    )


    # ========================================================
    # HISTORIES
    # ========================================================

    actions = []

    hitting_history = []

    memory_history = []

    kappa_history = []

    z_history = []

    gradient_history = []


    # ========================================================
    # MAIN ONLINE LOOP
    # ========================================================

    for t in range(
        T
    ):

        # ----------------------------------------------------
        # Current demand
        # ----------------------------------------------------

        y_t = (
            demand[t]
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
        # min
        #
        # f_t(u)
        #
        # +
        #
        # <kappa_t, d_t(u,u_prev)>
        # ====================================================

        (
            u_t,
            d_t,
        ) = dmd_step(

            y_t=y_t,

            u_prev=u_prev,

            kappa_t=kappa_t,
        )


        # ====================================================
        # 2. AUXILIARY UPDATE
        #
        # z_t
        #
        # =
        #
        # argmin_z
        #
        # ||z||_inf - <kappa_t,z>
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
        # =
        #
        # [kappa_t + eta(d_t-z_t)]_+
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
            u_t.copy()
        )


        kappa_t = (
            kappa_next.copy()
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


    # --------------------------------------------------------
    # Long-term cost:
    #
    # || sum_t d_t ||_inf
    #
    # =
    #
    # max_i
    #
    # sum_t
    #
    # beta / 2
    #
    # ||A_i(u_t-u_{t-1})||_2^2
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
            initial_kappa,


        "z_max":
            float(
                z_max
            ),

    }