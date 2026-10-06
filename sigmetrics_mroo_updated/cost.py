import numpy as np

from config import (
    P,
    C,
    BETA,
)


# ============================================================
# HITTING COST
#
# f_t(u_t)
# =
# p^T u_t
# +
# sum_i c_i (u_{t,i} - y_{t,i})^2
# ============================================================

def hitting_cost(
    u_t,
    y_t,
):
    u_t = np.asarray(
        u_t,
        dtype=float,
    )

    y_t = np.asarray(
        y_t,
        dtype=float,
    )

    provisioning_cost = np.sum(
        P * u_t
    )

    mismatch_cost = np.sum(
        C
        * (u_t - y_t) ** 2
    )

    return (
        provisioning_cost
        + mismatch_cost
    )


# ============================================================
# MEMORY COST
#
# For i = 1, ..., D:
#
# d_{t,i}
# =
# beta / 2
# *
# || A_i (u_t - u_{t-1}) ||_2^2
#
#
# If u_t is D-dimensional:
#
#     A_matrices.shape = (D, D, D)
#
# where:
#
#     A_matrices[i] = A_i
#
#
# Output:
#
#     d_t.shape = (D,)
# ============================================================

def memory_cost(
    u_t,
    u_prev,
    A_matrices,
    beta=BETA,
):
    u_t = np.asarray(
        u_t,
        dtype=float,
    )

    u_prev = np.asarray(
        u_prev,
        dtype=float,
    )

    A_matrices = np.asarray(
        A_matrices,
        dtype=float,
    )

    # --------------------------------------------------------
    # Dimension
    # --------------------------------------------------------

    D = len(
        u_t
    )

    # --------------------------------------------------------
    # Validation
    #
    # We require exactly D matrices,
    # and every A_i must be D x D.
    # --------------------------------------------------------

    if A_matrices.shape != (
        D,
        D,
        D,
    ):

        raise ValueError(
            "A_matrices must have shape "
            f"({D}, {D}, {D}), "
            f"but got {A_matrices.shape}."
        )

    if u_prev.shape != (
        D,
    ):

        raise ValueError(
            f"u_prev must have shape ({D},), "
            f"but got {u_prev.shape}."
        )

    # --------------------------------------------------------
    # Allocation change
    #
    # Delta u_t = u_t - u_{t-1}
    # --------------------------------------------------------

    delta_u = (
        u_t
        - u_prev
    )

    # --------------------------------------------------------
    # For every i:
    #
    # A_i @ Delta u_t
    #
    # Shape:
    #
    #     (D, D, D) @ (D,)
    #
    # gives
    #
    #     (D, D)
    #
    # Each row transformed[i] is:
    #
    #     A_i @ Delta u_t
    # --------------------------------------------------------

    transformed = (
        A_matrices
        @ delta_u
    )

    # --------------------------------------------------------
    # For every i:
    #
    # ||A_i Delta u_t||_2^2
    # --------------------------------------------------------

    squared_l2_norms = np.sum(
        transformed ** 2,
        axis=1,
    )

    # --------------------------------------------------------
    # d_{t,i}
    #
    # =
    #
    # beta / 2
    # *
    # ||A_i Delta u_t||_2^2
    # --------------------------------------------------------

    d_t = (
        beta
        / 2.0
        * squared_l2_norms
    )

    return d_t


# ============================================================
# LONG-TERM COST
#
# || sum_t d_t ||_infinity
#
# =
#
# max_i sum_t d_{t,i}
#
# No rho term.
# ============================================================

def long_term_cost(
    memory_cost_history,
):
    memory_cost_history = np.asarray(
        memory_cost_history,
        dtype=float,
    )

    cumulative_memory = np.sum(
        memory_cost_history,
        axis=0,
    )

    return np.max(
        cumulative_memory
    )