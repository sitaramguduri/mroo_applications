import numpy as np

# --------------------------------------------------
# Cost parameters
# --------------------------------------------------

p = np.array([0.9, 0.2, 0.1])
c = np.array([25.0, 20.0, 15.0])
l = np.array([900.0, 200.0, 100.0])


# --------------------------------------------------
# Hitting cost
# --------------------------------------------------

def hitting_cost(u_t, y_t):
    u_t = np.asarray(u_t, dtype=float)
    y_t = np.asarray(y_t, dtype=float)

    provisioning_cost = np.sum(p * u_t)

    mismatch_cost = np.sum(
        c * (u_t - y_t) ** 2
    )

    return provisioning_cost + mismatch_cost


# --------------------------------------------------
# Switching magnitude
# --------------------------------------------------

def switching_magnitude(u_t, u_prev):
    u_t = np.asarray(u_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)

    return 0.5 * np.sum(
        l * (u_t - u_prev) ** 2
    )


# --------------------------------------------------
# Vector-valued memory cost
# --------------------------------------------------

def memory_cost(u_t, u_prev, w_t):
    w_t = np.asarray(w_t, dtype=float)

    d_bar = switching_magnitude(u_t, u_prev)

    return d_bar * w_t


# --------------------------------------------------
# Long-term cost
# --------------------------------------------------

def long_term_cost(memory_cost_history, rho=1.0):
    memory_cost_history = np.asarray(
        memory_cost_history,
        dtype=float
    )

    avg_memory_cost = np.mean(
        memory_cost_history,
        axis=0
    )

    return rho * np.max(avg_memory_cost)