import os
import numpy as np


# ============================================================
# BASE APPLICATION PARAMETERS
# ============================================================

P_BASE = np.array(
    [0.9, 0.2, 0.1],
    dtype=float,
)

C_BASE = np.array(
    [90.0, 70.0, 50.0],
    dtype=float,
)

ELL_BASE = np.array(
    [900.0, 200.0, 100.0],
    dtype=float,
)


# ============================================================
# DIMENSION
# ============================================================

D = len(P_BASE)


# ============================================================
# BASE THEORY PARAMETERS
# ============================================================

BASE_M = (
    2.0
    * np.min(C_BASE)
)

BASE_BETA = (
    np.max(ELL_BASE)
)


# ============================================================
# EXPERIMENT PARAMETERS
#
# Environment variables are supplied by beta/m sweep.
#
# If no environment variables are supplied:
#
# beta = 900
# m    = 30
# ============================================================

TARGET_BETA = float(
    os.environ.get(
        "MROO_BETA",
        BASE_BETA,
    )
)

TARGET_M = float(
    os.environ.get(
        "MROO_M",
        BASE_M,
    )
)


if TARGET_BETA <= 0:

    raise ValueError(
        "MROO_BETA must be positive."
    )


if TARGET_M <= 0:

    raise ValueError(
        "MROO_M must be positive."
    )


# ============================================================
# SCALE APPLICATION PARAMETERS
# ============================================================

beta_scale = (
    TARGET_BETA
    / BASE_BETA
)

m_scale = (
    TARGET_M
    / BASE_M
)


P = (
    P_BASE.copy()
)

C = (
    C_BASE
    * m_scale
)

ELL = (
    ELL_BASE
    * beta_scale
)


# ============================================================
# DERIVED THEORY VALUES
# ============================================================

M = (
    2.0
    * np.min(C)
)

BETA = (
    np.max(ELL)
)


# ============================================================
# COMMON THEORY PARAMETERS
# ============================================================

RHO = 7.0

L = 1.0

H = 1.0


# ============================================================
# MROO PARAMETERS
#
# Theoretical value:
#
# lambda_1 =
#
# 1 / (1 + L beta H^2 / m)
#
# lambda_2 = 0
# ============================================================

LAMBDA_1_THEORY = (
    1.0
    /
    (
        1.0
        +
        L
        * BETA
        * H**2
        / M
    )
)


LAMBDA_1_VALUES = [
   LAMBDA_1_THEORY
    # 0.8
]


LAMBDA_2 = 0


# ============================================================
# S-MROO-SUM PARAMETERS
#
# q(z) = rho ||z||_1
#
# L_q = rho * sqrt(D)
#
# gamma = 1
#
# Pair 1:
#
# lambda_1 =
#
# m / (
#     m * gamma
#     + beta * L_q * H^2
# )
#
# lambda_2 = 0
# ============================================================

SMROO_SUM_L_Q = (
    RHO
    * np.sqrt(D)
)

SMROO_SUM_GAMMA = 1.0


SMROO_SUM_LAMBDA_1_THEORY = (
    M
    /
    (
        M
        * SMROO_SUM_GAMMA

        +

        BETA
        * SMROO_SUM_L_Q
        * H**2
    )
)


SMROO_SUM_LAMBDA_1_VALUES = [
    SMROO_SUM_LAMBDA_1_THEORY,
]


SMROO_SUM_LAMBDA_2 = 0.0


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print(
    "\nCONFIG:"
)

print(
    "  beta =",
    BETA,
)

print(
    "  m =",
    M,
)

print(
    "  beta/m =",
    BETA / M,
)

print(
    "  C =",
    C,
)

print(
    "  ELL =",
    ELL,
)


print(
    "\nMROO:"
)

print(
    "  theoretical lambda_1 =",
    LAMBDA_1_THEORY,
)

print(
    "  lambda_1 values =",
    LAMBDA_1_VALUES,
)

print(
    "  lambda_2 =",
    LAMBDA_2,
)


print(
    "\nS-MROO-SUM:"
)

print(
    "  L_q =",
    SMROO_SUM_L_Q,
)

print(
    "  gamma =",
    SMROO_SUM_GAMMA,
)

print(
    "  theoretical lambda_1 =",
    SMROO_SUM_LAMBDA_1_THEORY,
)

print(
    "  lambda_1 values =",
    SMROO_SUM_LAMBDA_1_VALUES,
)

print(
    "  lambda_2 =",
    SMROO_SUM_LAMBDA_2,
)
# ============================================================
# S-MROO-MAX PARAMETERS
#
# q(z) = rho ||z||_inf
#
# L_q = rho
#
# For full horizon:
#
# R >> D
#
# gamma = min(R, D) = D
#
# Pair 1:
#
# lambda_1 =
#
# m / (
#     m * gamma
#     + beta * L_q * H^2
# )
#
# lambda_2 = 0
# ============================================================

SMROO_MAX_L_Q = float(
    RHO
)

SMROO_MAX_GAMMA = float(
    D
)


SMROO_MAX_LAMBDA_1_THEORY = (
    M
    /
    (
        M
        * SMROO_MAX_GAMMA

        +

        BETA
        * SMROO_MAX_L_Q
        * H**2
    )
)


SMROO_MAX_LAMBDA_1_VALUES = [
    SMROO_MAX_LAMBDA_1_THEORY
]


SMROO_MAX_LAMBDA_2 = 0.0
print(
    "\nS-MROO-MAX:"
)

print(
    "  L_q =",
    SMROO_MAX_L_Q,
)

print(
    "  gamma =",
    SMROO_MAX_GAMMA,
)

print(
    "  theoretical lambda_1 =",
    SMROO_MAX_LAMBDA_1_THEORY,
)

print(
    "  lambda_1 values =",
    SMROO_MAX_LAMBDA_1_VALUES,
)

print(
    "  lambda_2 =",
    SMROO_MAX_LAMBDA_2,
)

# ============================================================
# MEMORY COST MATRICES
#
# For each memory channel i:
#
# d_{t,i}
# =
# beta / 2
# *
# || A_i (u_t - u_{t-1}) ||_2^2
#
# A_MATRICES[i] = A_i
#
# Shape:
#     (D, D, D)
#
# For D = 3:
#     3 memory channels
#     each A_i is a 3 x 3 matrix
# ============================================================

A_1 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0, 0.0],
        [0.0, 0.0, 0],
    ],
    dtype=float,
)

A_2 = np.array(
    [
        [0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0],
    ],
    dtype=float,
)

A_3 = np.array(
    [
        [0, 0.0, 0.0],
        [0.0, 0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


A_MATRICES = np.stack(
    [
        A_1,
        A_2,
        A_3,
    ],
    axis=0,
)