import numpy as np

# --------------------------------------------------
# Fixed exposure weights
# -------------------------------------------------- 

# Code completion, chatbot, non-interactive
r_fixed = np.array([0.2, 0.3, 0.5], dtype=float)

# Normalize so that ||w_t||_2 = 1
w_fixed = r_fixed / np.linalg.norm(r_fixed)


def get_w_t(t=None, y_t=None):
    """
    Returns the exposure vector w_t.

    For now, w_t is constant for all time steps.
    Later, this function can be changed to depend on
    t, y_t, priority, latency sensitivity, etc.
    """
    return w_fixed.copy()