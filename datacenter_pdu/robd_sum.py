"""ROBD with sum switching penalty for PDU power allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mroo import (
    DEMAND_COLUMNS,
    N_DOMAINS,
    PDU_IDS,
    feasible_start,
    load_exposures,
    solve_reference,
)

from power_cost import (
    ELL,
    hitting_cost,
    hitting_gradient,
    individual_switching_costs,
    long_term_cost,
    memory_cost,
    switching_magnitude,
)


# --------------------------------------------------
# R-OBD primal decision
#
# u_t = argmin [
#     f_t(u)
#     + lambda_1 c(u, u_prev)
#     + lambda_2 c(u, v_t)
# ]
#
# c(u, a) = 1/2 sum_i ELL_i (u_i - a_i)^2
# --------------------------------------------------

def solve_robd_sum(
    y_t,
    u_prev,
    v_t,
    budget,
    upper,
    lambda_1,
    lambda_2,
):
    def objective(u):
        previous_switching = (
            0.5
            * np.sum(
                ELL * (u - u_prev) ** 2
            )
        )

        reference_switching = (
            0.5
            * np.sum(
                ELL * (u - v_t) ** 2
            )
        )

        return (
            hitting_cost(u, y_t)
            + lambda_1 * previous_switching
            + lambda_2 * reference_switching
        )

    def gradient(u):
        previous_gradient = (
            ELL * (u - u_prev)
        )

        reference_gradient = (
            ELL * (u - v_t)
        )

        return (
            hitting_gradient(u, y_t)
            + lambda_1 * previous_gradient
            + lambda_2 * reference_gradient
        )

    result = minimize(
        fun=objective,
        x0=feasible_start(
            u_prev,
            budget,
            upper,
        ),
        jac=gradient,
        method="SLSQP",
        bounds=[
            (0.0, float(limit))
            for limit in upper
        ],
        constraints=[
            {
                "type": "ineq",
                "fun": (
                    lambda u:
                    budget - np.sum(u)
                ),
                "jac": (
                    lambda u:
                    -np.ones(N_DOMAINS)
                ),
            }
        ],
        options={
            "ftol": 1e-12,
            "maxiter": 500,
        },
    )

    if not result.success:
        raise RuntimeError(
            "ROBD-SUM optimization failed: "
            f"{result.message}"
        )

    return result.x


# --------------------------------------------------
# Run ROBD-SUM
# --------------------------------------------------

def run_robd_sum(
    trace,
    exposures,
    u_initial,
    budget,
    upper,
    lambda_1,
    lambda_2,
    rho,
):
    demands = trace[
        DEMAND_COLUMNS
    ].to_numpy(dtype=float)

    horizon = len(trace)
    entity_count = exposures.shape[1]

    u_initial = np.asarray(
        u_initial,
        dtype=float,
    )

    if u_initial.shape != (N_DOMAINS,):
        raise ValueError(
            f"u_initial must have shape "
            f"({N_DOMAINS},)"
        )

    if not np.all(np.isfinite(u_initial)):
        raise ValueError(
            "u_initial must be finite"
        )

    if (
        np.any(u_initial < 0)
        or np.any(u_initial > upper)
    ):
        raise ValueError(
            "u_initial violates domain bounds"
        )

    if u_initial.sum() > budget + 1e-10:
        raise ValueError(
            "sum(u_initial) exceeds budget"
        )

    if exposures.shape[0] != horizon:
        raise ValueError(
            "Exposure horizon does not "
            "match trace horizon"
        )

    allocations = np.empty(
        (horizon, N_DOMAINS)
    )

    references = np.empty_like(
        allocations
    )

    hitting_costs = np.empty(horizon)
    switching_costs = np.empty(horizon)

    memory_costs = np.empty(
        (horizon, entity_count)
    )

    per_domain_switching = np.empty(
        (horizon, N_DOMAINS)
    )

    u_prev = u_initial.copy()

    for t in range(horizon):
        y_t = demands[t]
        w_t = exposures[t]

        # Current hitting-cost minimizer.
        v_t = solve_reference(
            y_t=y_t,
            budget=budget,
            upper=upper,
            start=u_prev,
        )

        # ROBD-SUM decision.
        u_t = solve_robd_sum(
            y_t=y_t,
            u_prev=u_prev,
            v_t=v_t,
            budget=budget,
            upper=upper,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

        allocations[t] = u_t
        references[t] = v_t

        hitting_costs[t] = hitting_cost(
            u_t,
            y_t,
        )

        per_domain_switching[t] = (
            individual_switching_costs(
                u_t,
                u_prev,
            )
        )

        switching_costs[t] = (
            switching_magnitude(
                u_t,
                u_prev,
            )
        )

        # Used only for comparison with MROO's
        # long-term objective.
        memory_costs[t] = memory_cost(
            u_t,
            u_prev,
            w_t,
        )

        u_prev = u_t

    average_hitting_cost = float(
        hitting_costs.mean()
    )

    evaluated_long_term_cost = (
        long_term_cost(
            memory_costs,
            rho=rho,
        )
    )

    total_objective = (
        average_hitting_cost
        + evaluated_long_term_cost
    )

    # --------------------------------------------------
    # Store time-step results
    # --------------------------------------------------

    results = trace[["time"]].copy()

    for index, pdu_id in enumerate(PDU_IDS):
        results[f"u_pdu{pdu_id}"] = (
            allocations[:, index]
        )

        results[f"v_pdu{pdu_id}"] = (
            references[:, index]
        )

        results[f"switch_pdu{pdu_id}"] = (
            per_domain_switching[:, index]
        )

    results["hitting_cost"] = (
        hitting_costs
    )

    results["switching_magnitude"] = (
        switching_costs
    )

    for entity in range(entity_count):
        results[f"w_{entity + 1}"] = (
            exposures[:, entity]
        )

        results[f"d_{entity + 1}"] = (
            memory_costs[:, entity]
        )

    summary = {
        "algorithm": "ROBD-SUM",
        "horizon": horizon,
        "budget": float(budget),
        "upper_bounds": upper.tolist(),
        "u_initial": u_initial.tolist(),
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "rho": float(rho),
        "average_hitting_cost": (
            average_hitting_cost
        ),
        "long_term_cost": (
            evaluated_long_term_cost
        ),
        "total_objective": (
            total_objective
        ),
        "average_sum_switching_cost": float(
            switching_costs.mean()
        ),
        "maximum_average_entity_burden": float(
            memory_costs.mean(axis=0).max()
        ),
    }

    return results, summary


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args():
    project = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trace",
        type=Path,
        default=(
            project
            / "processed_powerdata"
            / "cell_a_power_trace.csv"
        ),
    )

    parser.add_argument(
        "--exposure-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--entities",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--upper",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--u-initial",
        type=float,
        nargs="+",
        default=[0.5],
    )

    # R-OBD parameters.
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1.0,
    )

    # Used only to evaluate the common objective.
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
    )

    # Use the same final 20% used by MROO.
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            project
            / "results"
            / "robd_sum"
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()

    if not 0 < args.lambda_1 <= 1:
        raise ValueError(
            "The R-OBD paper requires "
            "0 < lambda_1 <= 1"
        )

    if args.lambda_2 < 0:
        raise ValueError(
            "lambda_2 must be nonnegative"
        )

    if not 0 < args.test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1"
        )

    trace = pd.read_csv(args.trace)

    missing_columns = [
        column
        for column in [
            "time",
            *DEMAND_COLUMNS,
        ]
        if column not in trace.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    upper = np.full(
        N_DOMAINS,
        args.upper,
        dtype=float,
    )

    if len(args.u_initial) == 1:
        u_initial = np.full(
            N_DOMAINS,
            args.u_initial[0],
            dtype=float,
        )

    elif len(args.u_initial) == N_DOMAINS:
        u_initial = np.asarray(
            args.u_initial,
            dtype=float,
        )

    else:
        raise ValueError(
            "--u-initial requires either "
            "one value or five values"
        )

    # Generate/load the full exposure sequence first
    # so the held-out portion matches MROO exactly.
    full_exposures = load_exposures(
        path=args.exposure_file,
        horizon=len(trace),
        dimension=args.entities,
        seed=args.seed,
    )

    test_size = max(
        1,
        int(len(trace) * args.test_fraction),
    )

    test_trace = (
        trace.iloc[-test_size:]
        .reset_index(drop=True)
    )

    test_exposures = (
        full_exposures[-test_size:]
    )

    results, summary = run_robd_sum(
        trace=test_trace,
        exposures=test_exposures,
        u_initial=u_initial,
        budget=args.budget,
        upper=upper,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        rho=args.rho,
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file = (
        args.output_directory
        / "robd_sum_results.csv"
    )

    summary_file = (
        args.output_directory
        / "robd_sum_summary.json"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    summary_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"\nResults saved to: {results_file}"
    )

    print(
        f"Summary saved to: {summary_file}"
    )


if __name__ == "__main__":
    main()