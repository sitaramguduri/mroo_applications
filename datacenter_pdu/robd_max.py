"""ROBD-MAX baseline for the five-PDU power-allocation experiment.

The exposure array is loaded by mroo.load_exposures so MROO, ROBD-SUM, and
ROBD-MAX use exactly the same constant exposure definition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from mroo import DEMAND_COLUMNS, load_exposures, solve_reference
from power_cost import (
    ELL,
    MU,
    N_DOMAINS,
    PDU_IDS,
    hitting_cost,
    individual_switching_costs,
    long_term_cost,
    memory_cost,
)


def solve_robd_max(
    y_t,
    u_prev,
    v_t,
    budget,
    upper,
    lambda_1,
    lambda_2,
):
    """Solve one ROBD-MAX decision as a convex optimization problem."""
    y_t = np.asarray(y_t, dtype=float)
    u_prev = np.asarray(u_prev, dtype=float)
    v_t = np.asarray(v_t, dtype=float)

    u = cp.Variable(N_DOMAINS)

    hitting = 0.5 * cp.sum(
        cp.multiply(MU, cp.square(u - y_t))
    )
    previous_max = cp.max(
        cp.multiply(0.5 * ELL, cp.square(u - u_prev))
    )
    reference_max = cp.max(
        cp.multiply(0.5 * ELL, cp.square(u - v_t))
    )

    problem = cp.Problem(
        cp.Minimize(
            hitting
            + lambda_1 * previous_max
            + lambda_2 * reference_max
        ),
        [u >= 0.0, u <= upper, cp.sum(u) <= budget],
    )

    solver_attempts = (
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-7, "max_iters": 20_000}),
    )
    errors = []

    for solver, options in solver_attempts:
        try:
            problem.solve(solver=solver, verbose=False, **options)
        except cp.error.SolverError as error:
            errors.append(f"{solver}: {error}")
            continue

        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            candidate = np.asarray(u.value, dtype=float).reshape(-1)
            if (
                candidate.shape == (N_DOMAINS,)
                and np.all(np.isfinite(candidate))
            ):
                # Remove negligible solver-tolerance violations.
                candidate = np.clip(candidate, 0.0, upper)
                if candidate.sum() > budget:
                    candidate *= budget / candidate.sum()
                return candidate

        errors.append(f"{solver}: status={problem.status}")

    raise RuntimeError(
        "ROBD-MAX optimization failed: " + "; ".join(errors)
    )


def run_robd_max(
    trace,
    exposures,
    u_initial,
    budget,
    upper,
    lambda_1,
    lambda_2,
    rho,
):
    """Run ROBD-MAX and evaluate it with the common MROO costs."""
    demands = trace[DEMAND_COLUMNS].to_numpy(dtype=float)
    horizon = len(trace)
    entity_count = exposures.shape[1]

    if exposures.shape != (horizon, entity_count):
        raise ValueError("Exposure horizon does not match the trace")

    u_initial = np.asarray(u_initial, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if u_initial.shape != (N_DOMAINS,):
        raise ValueError(f"u_initial must have shape ({N_DOMAINS},)")
    if upper.shape != (N_DOMAINS,):
        raise ValueError(f"upper must have shape ({N_DOMAINS},)")
    if not np.all(np.isfinite(u_initial)):
        raise ValueError("u_initial must be finite")
    if np.any(u_initial < 0) or np.any(u_initial > upper):
        raise ValueError("u_initial violates domain bounds")
    if u_initial.sum() > budget + 1e-10:
        raise ValueError("u_initial exceeds the total budget")

    allocations = np.empty((horizon, N_DOMAINS))
    references = np.empty_like(allocations)
    hitting_values = np.empty(horizon)
    max_switching_values = np.empty(horizon)
    total_switching_values = np.empty(horizon)
    memory_values = np.empty((horizon, entity_count))

    u_prev = u_initial.copy()

    for t in range(horizon):
        y_t = demands[t]
        w_t = exposures[t]

        v_t = solve_reference(
            y_t=y_t,
            budget=budget,
            upper=upper,
            start=u_prev,
        )
        u_t = solve_robd_max(
            y_t=y_t,
            u_prev=u_prev,
            v_t=v_t,
            budget=budget,
            upper=upper,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

        individual = individual_switching_costs(u_t, u_prev)
        allocations[t] = u_t
        references[t] = v_t
        hitting_values[t] = hitting_cost(u_t, y_t)
        max_switching_values[t] = float(individual.max())
        total_switching_values[t] = float(individual.sum())
        memory_values[t] = memory_cost(u_t, u_prev, w_t)
        u_prev = u_t

    average_hitting = float(hitting_values.mean())
    evaluated_long_term = long_term_cost(memory_values, rho=rho)

    results = trace[["time"]].copy()
    previous_allocations = np.vstack((u_initial, allocations[:-1]))

    for index, pdu_id in enumerate(PDU_IDS):
        results[f"u_pdu{pdu_id}"] = allocations[:, index]
        results[f"v_pdu{pdu_id}"] = references[:, index]
        results[f"switch_pdu{pdu_id}"] = (
            0.5
            * ELL[index]
            * (allocations[:, index] - previous_allocations[:, index]) ** 2
        )

    results["hitting_cost"] = hitting_values
    results["max_switching_cost"] = max_switching_values
    results["total_switching_cost"] = total_switching_values

    for entity in range(entity_count):
        results[f"w_{entity + 1}"] = exposures[:, entity]
        results[f"d_{entity + 1}"] = memory_values[:, entity]

    summary = {
        "algorithm": "ROBD-MAX",
        "horizon": horizon,
        "budget": float(budget),
        "upper_bounds": upper.tolist(),
        "u_initial": u_initial.tolist(),
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "rho": float(rho),
        "average_hitting_cost": average_hitting,
        "long_term_cost": float(evaluated_long_term),
        "total_objective": float(average_hitting + evaluated_long_term),
        "average_max_switching_cost": float(max_switching_values.mean()),
        "average_total_switching_cost": float(total_switching_values.mean()),
        "maximum_average_entity_burden": float(
            memory_values.mean(axis=0).max()
        ),
        "constant_exposure_vector": exposures[0].tolist(),
    }
    return results, summary


def parse_args():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=project / "processed_powerdata" / "cell_a_power_trace.csv",
    )
    parser.add_argument("--exposure-file", type=Path, default=None)
    parser.add_argument("--entities", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=float, default=3.5)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--lambda-1", type=float, default=0.3)
    parser.add_argument("--lambda-2", type=float, default=0.8)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument(
        "--u-initial",
        type=float,
        nargs="+",
        default=[0.5],
        help="One shared value or five values in PDU order 6 7 8 9 10.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "results" / "robd_max",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.entities <= 0:
        raise ValueError("entities must be positive")
    if not np.isfinite(args.upper) or args.upper <= 0:
        raise ValueError("upper must be finite and positive")
    if not np.isfinite(args.budget) or args.budget <= 0:
        raise ValueError("budget must be finite and positive")
    if not np.isfinite(args.lambda_1) or args.lambda_1 < 0:
        raise ValueError("lambda_1 must be finite and nonnegative")
    if not np.isfinite(args.lambda_2) or args.lambda_2 < 0:
        raise ValueError("lambda_2 must be finite and nonnegative")
    if not np.isfinite(args.rho) or args.rho < 0:
        raise ValueError("rho must be finite and nonnegative")
    if not 0 < args.test_fraction <= 1:
        raise ValueError("test_fraction must lie in (0, 1]")

    upper = np.full(N_DOMAINS, args.upper, dtype=float)
    if args.budget > upper.sum():
        raise ValueError("budget cannot exceed the sum of upper bounds")

    trace = pd.read_csv(args.trace)
    required = ["time", *DEMAND_COLUMNS]
    missing = [column for column in required if column not in trace]
    if missing:
        raise ValueError(f"Trace is missing columns: {missing}")
    if trace.empty:
        raise ValueError("Trace is empty")
    if not np.isfinite(trace[DEMAND_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Demand trace contains nonfinite values")

    # This is the exact loader used by MROO. With your current mroo.py it
    # returns the same constant normalized exposure vector at every time.
    full_exposures = load_exposures(
        path=args.exposure_file,
        horizon=len(trace),
        dimension=args.entities,
        seed=args.seed,
    )
    if not np.allclose(full_exposures, full_exposures[0]):
        raise RuntimeError(
            "mroo.load_exposures did not return a constant exposure sequence"
        )

    test_size = max(1, int(len(trace) * args.test_fraction))
    test_trace = trace.iloc[-test_size:].reset_index(drop=True)
    test_exposures = full_exposures[-test_size:]

    if len(args.u_initial) == 1:
        u_initial = np.full(N_DOMAINS, args.u_initial[0], dtype=float)
    elif len(args.u_initial) == N_DOMAINS:
        u_initial = np.asarray(args.u_initial, dtype=float)
    else:
        raise ValueError(
            f"--u-initial requires either one value or {N_DOMAINS} values"
        )

    results, summary = run_robd_max(
        trace=test_trace,
        exposures=test_exposures,
        u_initial=u_initial,
        budget=args.budget,
        upper=upper,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        rho=args.rho,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_file = args.output_directory / "robd_max_results.csv"
    summary_file = args.output_directory / "robd_max_summary.json"
    results.to_csv(result_file, index=False)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Constant exposure vector:", test_exposures[0])
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {result_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
