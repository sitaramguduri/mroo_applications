"""Offline optimal benchmark for PDU power allocation."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from mroo import DEMAND_COLUMNS, N_DOMAINS, PDU_IDS
from power_cost import (
    ELL,
    MU,
    hitting_cost,
    individual_switching_costs,
    long_term_cost,
    memory_cost,
)


def run_offline_opt(
    trace,
    exposures,
    u_initial,
    budget,
    upper,
    rho,
):
    """Jointly optimize every allocation in one known demand window."""

    demands = trace[DEMAND_COLUMNS].to_numpy(dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    u_initial = np.asarray(u_initial, dtype=float)
    upper = np.asarray(upper, dtype=float)

    horizon = len(trace)
    if horizon == 0:
        raise ValueError("The offline window cannot be empty")
    if demands.shape != (horizon, N_DOMAINS):
        raise ValueError("Demand matrix has an invalid shape")
    if exposures.ndim != 2 or exposures.shape[0] != horizon:
        raise ValueError("Exposure horizon does not match demand horizon")
    if np.any(exposures < 0) or not np.all(np.isfinite(exposures)):
        raise ValueError("Exposures must be finite and nonnegative")
    if u_initial.shape != (N_DOMAINS,):
        raise ValueError(f"u_initial must have shape ({N_DOMAINS},)")
    if upper.shape != (N_DOMAINS,):
        raise ValueError(f"upper must have shape ({N_DOMAINS},)")
    if np.any(u_initial < 0) or np.any(u_initial > upper):
        raise ValueError("u_initial violates the allocation bounds")
    if u_initial.sum() > budget + 1e-10:
        raise ValueError("u_initial exceeds the budget")

    # U[t, i] is the allocation to PDU i at time t.
    allocation = cp.Variable((horizon, N_DOMAINS))

    # Include the fixed initial allocation in the switching differences.
    previous = cp.vstack(
        [u_initial.reshape(1, N_DOMAINS), allocation[:-1, :]]
    )
    differences = allocation - previous

    per_step_hitting = 0.5 * cp.sum(
        cp.multiply(MU, cp.square(allocation - demands)),
        axis=1,
    )
    per_step_switching = 0.5 * cp.sum(
        cp.multiply(ELL, cp.square(differences)),
        axis=1,
    )

    # d_t = switching_magnitude_t * w_t.
    memory_matrix = cp.multiply(
        cp.reshape(per_step_switching, (horizon, 1), order="C"),
        exposures,
    )
    average_entity_burden = cp.sum(memory_matrix, axis=0) / horizon

    average_hitting = cp.sum(per_step_hitting) / horizon
    offline_long_term = rho * cp.max(average_entity_burden)

    problem = cp.Problem(
        cp.Minimize(average_hitting + offline_long_term),
        [
            allocation >= 0,
            allocation <= upper,
            cp.sum(allocation, axis=1) <= budget,
        ],
    )

    # CLARABEL normally gives the most accurate solution for this convex model.
    try:
        problem.solve(
            solver=cp.CLARABEL,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            max_iter=1000,
            verbose=False,
        )
    except cp.error.SolverError:
        problem.solve(
            solver=cp.SCS,
            eps=1e-6,
            max_iters=100000,
            verbose=False,
        )

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"Offline optimization failed with status: {problem.status}"
        )

    allocations = np.asarray(allocation.value, dtype=float)
    allocations[np.abs(allocations) < 1e-10] = 0.0

    hitting_values = np.empty(horizon)
    switching_values = np.empty(horizon)
    per_domain_switching = np.empty((horizon, N_DOMAINS))
    memory_values = np.empty((horizon, exposures.shape[1]))

    u_prev = u_initial.copy()
    for t in range(horizon):
        u_t = allocations[t]
        hitting_values[t] = hitting_cost(u_t, demands[t])
        per_domain_switching[t] = individual_switching_costs(u_t, u_prev)
        switching_values[t] = per_domain_switching[t].sum()
        memory_values[t] = memory_cost(u_t, u_prev, exposures[t])
        u_prev = u_t

    evaluated_hitting = float(hitting_values.mean())
    evaluated_long_term = long_term_cost(memory_values, rho=rho)
    evaluated_total = evaluated_hitting + evaluated_long_term

    results = trace[["time"]].copy()
    for index, pdu_id in enumerate(PDU_IDS):
        results[f"u_pdu{pdu_id}"] = allocations[:, index]
        results[f"switch_pdu{pdu_id}"] = per_domain_switching[:, index]
    results["hitting_cost"] = hitting_values
    results["switching_magnitude"] = switching_values
    for entity in range(exposures.shape[1]):
        results[f"d_{entity + 1}"] = memory_values[:, entity]

    summary = {
        "average_hitting_cost": evaluated_hitting,
        "long_term_cost": evaluated_long_term,
        "total_objective": evaluated_total,
        "average_switching_magnitude": float(switching_values.mean()),
        "maximum_average_entity_burden": float(
            memory_values.mean(axis=0).max()
        ),
        "solver_status": problem.status,
        "solver_objective": float(problem.value),
    }

    if not np.isclose(
        evaluated_total,
        problem.value,
        rtol=1e-5,
        atol=1e-8,
    ):
        raise RuntimeError(
            "The recomputed OPT cost differs from the solver objective: "
            f"{evaluated_total} versus {problem.value}"
        )

    return results, summary