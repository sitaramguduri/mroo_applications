import sys
import os
import numpy as np
from scipy.optimize import minimize, linprog
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from cost import hitting_cost, memory_cost


# --------------------------------------------------
# MROO parameters
# --------------------------------------------------

LAMBDA_1 = 30.0
LAMBDA_2 = 0.8
M = 100.0

RHO = 1.0

# Loose bound for Z = [0, Z_MAX]^3
Z_MAX = 600.0

KAPPA_INIT = np.array([0.05, 0.05, 0.05], dtype=float)


# --------------------------------------------------
# Simplex constraints for u
#
# u_i >= 0
# sum_i u_i = 1
# --------------------------------------------------
def simplex_constraints():
    return [{
        "type": "eq",
        "fun": lambda u: np.sum(u) - 1.0
    }]


def simplex_bounds(dim):
    return [(0.0, 1.0)] * dim


# --------------------------------------------------
# Compute v_t
#
# v_t = argmin f_t(u)
# --------------------------------------------------

def compute_v_t(y_t):

    y_t = np.asarray(y_t, dtype=float)

    result = minimize(
        fun=lambda u: hitting_cost(u, y_t),
        x0=y_t,
        method="SLSQP",
        bounds=simplex_bounds(len(y_t)),
        constraints=simplex_constraints()
    )

    if not result.success:
        raise RuntimeError(
            f"Failed to compute v_t: {result.message}"
        )

    return result.x


# --------------------------------------------------
# MROO primal update
#
# u_t = argmin [
#       f_t(u)
#       + lambda_1 <kappa_t, d_t(u,u_prev)>
#       + lambda_2*m/2 ||u-v_t||^2
# ]
# --------------------------------------------------

def primal_update(
    y_t,
    u_prev,
    w_t,
    kappa_t,
    lambda_1=LAMBDA_1,
    lambda_2=LAMBDA_2,
    m=M
):

    y_t = np.asarray(y_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)
    w_t = np.asarray(w_t, dtype=float)
    kappa_t = np.asarray(kappa_t, dtype=float)

    # Hitting-cost minimizer
    v_t = compute_v_t(y_t)

    def objective(u):

        # Hitting cost
        f = hitting_cost(u, y_t)

        # Vector-valued memory cost
        d_t = memory_cost(
            u,
            u_prev,
            w_t
        )

        # Dual-weighted memory term
        dual_memory = (
            lambda_1 *
            np.dot(kappa_t, d_t)
        )

        # Regularization toward hitting-cost minimizer
        regularization = (
            lambda_2 *
            m / 2.0 *
            np.sum((u - v_t) ** 2)
        )

        return (
            f +
            dual_memory +
            regularization
        )

    result = minimize(
        fun=objective,
        x0=u_prev,
        method="SLSQP",
        bounds=simplex_bounds(len(y_t)),
        constraints=simplex_constraints()
    )

    if not result.success:
        raise RuntimeError(
            f"MROO primal update failed: {result.message}"
        )

    return result.x, v_t


# --------------------------------------------------
# Auxiliary update
#
# z_t = argmin_z [
#       rho ||z||_infinity
#       - <kappa_t, z>
# ]
#
# subject to:
#       0 <= z_i <= Z_MAX
# --------------------------------------------------

def auxiliary_update(
    kappa_t,
    rho=RHO,
    z_max=Z_MAX
):

    kappa_t = np.asarray(kappa_t, dtype=float)

    D = len(kappa_t)

    # Introduce scalar s where
    #
    # z_i <= s
    #
    # so ||z||_infinity = s
    #
    # Optimization variables:
    # [z_1, ..., z_D, s]

    objective = np.concatenate([
        -kappa_t,
        [rho]
    ])

    # constraints:
    # z_i - s <= 0

    A_ub = np.zeros((D, D + 1))
    b_ub = np.zeros(D)

    for i in range(D):
        A_ub[i, i] = 1.0
        A_ub[i, D] = -1.0

    bounds = (
        [(0.0, z_max)] * D
        + [(0.0, z_max)]
    )

    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(
            f"Auxiliary update failed: {result.message}"
        )

    z_t = result.x[:D]

    return z_t


# --------------------------------------------------
# Dual update
#
# g_t = z_t - lambda_1 d_t
#
# For h(kappa) = 1/2 ||kappa||_2^2:
#
# kappa_{t+1}
# = projection_+(kappa_t - eta*g_t)
# --------------------------------------------------

def dual_update(
    kappa_t,
    z_t,
    d_t,
    eta,
    lambda_1=LAMBDA_1
):

    kappa_t = np.asarray(kappa_t, dtype=float)
    z_t = np.asarray(z_t, dtype=float)
    d_t = np.asarray(d_t, dtype=float)

    g_t = z_t - lambda_1 * d_t

    kappa_next = (
        kappa_t -
        eta * g_t
    )

    # Projection onto R_+^D
    kappa_next = np.maximum(
        kappa_next,
        0.0
    )

    return kappa_next, g_t


# --------------------------------------------------
# One complete MROO step
# --------------------------------------------------

def mroo_step(
    y_t,
    u_prev,
    w_t,
    kappa_t,
    eta
):

    # 1. Primal update
    u_t, v_t = primal_update(
        y_t,
        u_prev,
        w_t,
        kappa_t
    )

    # 2. Realized memory cost
    d_t = memory_cost(
        u_t,
        u_prev,
        w_t
    )

    # 3. Auxiliary update
    z_t = auxiliary_update(
        kappa_t
    )

    # 4. Dual update
    kappa_next, g_t = dual_update(
        kappa_t,
        z_t,
        d_t,
        eta
    )

    return {
        "u_t": u_t,
        "v_t": v_t,
        "d_t": d_t,
        "z_t": z_t,
        "g_t": g_t,
        "kappa_next": kappa_next
    }