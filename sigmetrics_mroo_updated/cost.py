import numpy as np

from config import (
    P,
    C,
    ELL,
    RHO,
)


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


def switching_magnitude(
    u_t,
    u_prev,
):
    u_t = np.asarray(
        u_t,
        dtype=float,
    )

    u_prev = np.asarray(
        u_prev,
        dtype=float,
    )

    return (
        0.5
        * np.sum(
            ELL
            * (u_t - u_prev) ** 2
        )
    )


def memory_cost(
    u_t,
    u_prev,
    w_t,
):
    w_t = np.asarray(
        w_t,
        dtype=float,
    )

    d_bar = switching_magnitude(
        u_t,
        u_prev,
    )

    return (
        d_bar
        * w_t
    )


def long_term_cost(
    memory_cost_history,
    rho=RHO,
):
    memory_cost_history = np.asarray(
        memory_cost_history,
        dtype=float,
    )

    cumulative_memory = np.sum(
        memory_cost_history,
        axis=0,
    )

    return (
        rho
        * np.max(
            cumulative_memory
        )
    )