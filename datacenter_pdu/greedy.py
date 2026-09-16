"""Greedy hitting-cost baseline for PDU power allocation."""

from __future__ import annotations

import numpy as np

from mroo import DEMAND_COLUMNS, solve_reference
from power_cost import (
    ELL,
    N_DOMAINS,
    PDU_IDS,
    hitting_cost,
    individual_switching_costs,
    long_term_cost,
    memory_cost,
)


def run_greedy(
    trace,
    exposures,
    u_initial,
    budget,
    upper,
    rho,
):
    """Minimize only the current hitting cost at every time step."""
    demands = trace[DEMAND_COLUMNS].to_numpy(dtype=float)
    horizon = len(trace)
    entity_count = exposures.shape[1]
    u_initial = np.asarray(u_initial, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if exposures.shape[0] != horizon:
        raise ValueError("Exposure horizon does not match trace horizon")
    if u_initial.shape != (N_DOMAINS,):
        raise ValueError(f"u_initial must have shape ({N_DOMAINS},)")
    if upper.shape != (N_DOMAINS,):
        raise ValueError(f"upper must have shape ({N_DOMAINS},)")
    if np.any(u_initial < 0) or np.any(u_initial > upper):
        raise ValueError("u_initial violates domain bounds")
    if u_initial.sum() > budget + 1e-10:
        raise ValueError("u_initial exceeds budget")

    allocations = np.empty((horizon, N_DOMAINS))
    hitting_values = np.empty(horizon)
    switching_values = np.empty(horizon)
    memory_values = np.empty((horizon, entity_count))
    u_prev = u_initial.copy()

    for t in range(horizon):
        y_t = demands[t]
        u_t = solve_reference(
            y_t=y_t,
            budget=budget,
            upper=upper,
            start=u_prev,
        )
        allocations[t] = u_t
        hitting_values[t] = hitting_cost(u_t, y_t)
        switching_values[t] = individual_switching_costs(u_t, u_prev).sum()
        memory_values[t] = memory_cost(u_t, u_prev, exposures[t])
        u_prev = u_t

    average_hitting = float(hitting_values.mean())
    evaluated_long_term = float(long_term_cost(memory_values, rho=rho))

    results = trace[["time"]].copy()
    previous_allocations = np.vstack((u_initial, allocations[:-1]))
    for index, pdu_id in enumerate(PDU_IDS):
        results[f"u_pdu{pdu_id}"] = allocations[:, index]
        results[f"switch_pdu{pdu_id}"] = (
            0.5
            * ELL[index]
            * (allocations[:, index] - previous_allocations[:, index]) ** 2
        )

    results["hitting_cost"] = hitting_values
    results["switching_magnitude"] = switching_values
    for entity in range(entity_count):
        results[f"w_{entity + 1}"] = exposures[:, entity]
        results[f"d_{entity + 1}"] = memory_values[:, entity]

    summary = {
        "algorithm": "GREEDY",
        "horizon": horizon,
        "average_hitting_cost": average_hitting,
        "long_term_cost": evaluated_long_term,
        "total_objective": average_hitting + evaluated_long_term,
        "average_switching_magnitude": float(switching_values.mean()),
        "maximum_average_entity_burden": float(
            memory_values.mean(axis=0).max()
        ),
    }
    return results, summary