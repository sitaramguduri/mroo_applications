import numpy as np


# --------------------------------------------------
# Five power domains
# --------------------------------------------------

PDU_IDS = (6, 7, 8, 9, 10)
N_DOMAINS = len(PDU_IDS)

# Illustrative simulation coefficients.
# These are not supplied by Google PowerData.
# A = np.ones(N_DOMAINS, dtype=float)
MU = np.ones(N_DOMAINS, dtype=float)
SWITCHING_SCALE = 35.0
ELL = SWITCHING_SCALE * np.ones(N_DOMAINS, dtype=float)

RHO = 1.0

# Strong convexity with respect to allocation u
M_STRONG = float(np.min(MU))


# --------------------------------------------------
# Validation helpers
# --------------------------------------------------

def domain_vector(values, name):
    values = np.asarray(values, dtype=float)

    if values.shape != (N_DOMAINS,):
        raise ValueError(
            f"{name} must have shape ({N_DOMAINS},)"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must contain finite values"
        )

    return values


# --------------------------------------------------
# Hitting cost
#
# f_t(u) = 1/2 * sum_i MU_i * (u_i - y_i,t)^2
#
# y_t comes from the Google PDU trace.
# u_t is the allocation chosen by the algorithm.
# --------------------------------------------------

def hitting_cost(u, y_t):
    u = domain_vector(u, "u")
    y_t = domain_vector(y_t, "y_t")

    difference = u - y_t

    return float(
        0.5 * np.sum(MU * difference ** 2)
    )


def hitting_gradient(u, y_t):
    """Gradient of the hitting cost with respect to u."""
    u = domain_vector(u, "u")
    y_t = domain_vector(y_t, "y_t")

    return MU * (u - y_t)


# --------------------------------------------------
# Per-domain switching costs
#
# c_i,t = ELL_i/2 * (u_i,t - u_i,t-1)^2
# --------------------------------------------------

def individual_switching_costs(u, u_prev):
    u = domain_vector(u, "u")
    u_prev = domain_vector(u_prev, "u_prev")

    return 0.5 * ELL * (u - u_prev) ** 2


def switching_magnitude(u, u_prev):
    """Total adjustment burden across all domains."""
    return float(np.sum(
        individual_switching_costs(u, u_prev)
    ))


def switching_gradient(u, u_prev):
    u = domain_vector(u, "u")
    u_prev = domain_vector(u_prev, "u_prev")

    return ELL * (u - u_prev)


# --------------------------------------------------
# Exposure weights and vector-valued memory cost
#
# d_t = switching_magnitude * w_t
#
# The number of affected entities D need not equal
# the number of domains.
# --------------------------------------------------

def normalize_exposure(r_t):
    r_t = np.asarray(r_t, dtype=float)

    if r_t.ndim != 1 or r_t.size == 0:
        raise ValueError(
            "r_t must be a nonempty vector"
        )

    if not np.all(np.isfinite(r_t)) or np.any(r_t < 0):
        raise ValueError(
            "r_t must be finite and nonnegative"
        )

    norm = np.linalg.norm(r_t)

    if norm == 0:
        raise ValueError(
            "r_t must have positive norm"
        )

    return r_t / norm


def memory_cost(u, u_prev, w_t):
    w_t = np.asarray(w_t, dtype=float)

    if (
        w_t.ndim != 1
        or w_t.size == 0
        or not np.all(np.isfinite(w_t))
        or np.any(w_t < 0)
    ):
        raise ValueError(
            "w_t must be a finite nonnegative vector"
        )

    if not np.isclose(
        np.linalg.norm(w_t), 1.0,
        atol=1e-8, rtol=0.0
    ):
        raise ValueError(
            "w_t must have unit L2 norm; "
            "use normalize_exposure(r_t)"
        )

    return switching_magnitude(u, u_prev) * w_t


# --------------------------------------------------
# Long-term objective component
#
# q(mean d_t) = rho * max_k mean_t d_k,t
# --------------------------------------------------

def long_term_cost(memory_history, rho=RHO):
    history = np.asarray(memory_history, dtype=float)

    if (
        history.ndim != 2
        or history.shape[0] == 0
        or history.shape[1] == 0
    ):
        raise ValueError(
            "memory_history must have shape (T, D), "
            "with T and D positive"
        )

    if not np.all(np.isfinite(history)) or np.any(history < 0):
        raise ValueError(
            "Memory costs must be finite and nonnegative"
        )

    if not np.isfinite(rho) or rho < 0:
        raise ValueError(
            "rho must be finite and nonnegative"
        )

    return float(rho * history.mean(axis=0).max())