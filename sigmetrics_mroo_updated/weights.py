import numpy as np


# ============================================================
# FIXED WEIGHTS
#
# Raw constant weight vector.
#
# This same weight is used at every time step.
# ============================================================

RAW_WEIGHT = np.array(
    [
        0.2,
        0.3,
        0.5,
    ],
    dtype=float,
)


# ============================================================
# NORMALIZE SO ||w_t||_2 = 1
# ============================================================

WEIGHT_NORM = np.linalg.norm(
    RAW_WEIGHT
)


if WEIGHT_NORM == 0.0:

    raise ValueError(
        "Weight vector cannot be all zeros."
    )


FIXED_WEIGHT = (
    RAW_WEIGHT
    / WEIGHT_NORM
)


# ============================================================
# GET w_t
#
# The weight is constant for every t.
#
# t and y_t are kept as arguments so the rest of the code
# does not need to change.
# ============================================================

def get_w_t(
    t=None,
    y_t=None,
):

    return FIXED_WEIGHT.copy()