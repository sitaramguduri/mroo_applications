"""Run MROO on the processed Google cell-a PDU power trace.

Expected local files:
  processed_powerdata/cell_a_power_trace.csv
  power_cost.py

The implementation uses:
  h(kappa) = 0.5 * ||kappa||_2^2,
  q(z) = rho * ||z||_infinity,
  U = {u : 0 <= u_i <= upper_i, sum(u) <= budget}, and
  Z = [0, z_max]^D.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from power_cost import (
    ELL,
    M_STRONG,
    MU,
    N_DOMAINS,
    PDU_IDS,
    hitting_cost,
    hitting_gradient,
    individual_switching_costs,
    long_term_cost,
    memory_cost,
    normalize_exposure,
    switching_magnitude,
)


DEMAND_COLUMNS = [f"y_pdu{pdu_id}" for pdu_id in PDU_IDS]


def check_parameters(budget, upper, eta, lambda_1, lambda_2, rho):
    if upper.shape != (N_DOMAINS,):
        raise ValueError(f"upper must have shape ({N_DOMAINS},)")
    if not np.all(np.isfinite(upper)) or np.any(upper <= 0):
        raise ValueError("Every domain upper bound must be finite and positive")
    if not np.isfinite(budget) or budget <= 0 or budget > upper.sum():
        raise ValueError("budget must be positive and no larger than sum(upper)")
    for name, value in {
        "eta": eta,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "rho": rho,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if eta == 0:
        raise ValueError("eta must be positive")


def feasible_start(target, budget, upper):
    """Return a simple feasible point close to target."""
    point = np.clip(np.asarray(target, dtype=float), 0.0, upper)
    total = point.sum()
    if total > budget:
        point *= budget / total
    return point


def solve_reference(y_t, budget, upper, start):
    """v_t = argmin_{v in U} f_t(v)."""
    result = minimize(
        fun=lambda v: hitting_cost(v, y_t),
        x0=feasible_start(start, budget, upper),
        jac=lambda v: hitting_gradient(v, y_t),
        method="SLSQP",
        bounds=[(0.0, float(limit)) for limit in upper],
        constraints=[{
            "type": "ineq",
            "fun": lambda v: budget - np.sum(v),
            "jac": lambda v: -np.ones(N_DOMAINS),
        }],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Reference optimization failed: {result.message}")
    return result.x


def solve_primal(
    y_t,
    u_prev,
    v_t,
    w_t,
    kappa_t,
    budget,
    upper,
    lambda_1,
    lambda_2,
):
    """Line 4 of Algorithm 1 for the costs in power_cost.py."""
    exposure_multiplier = float(np.dot(kappa_t, w_t))

    def objective(u):
        dual_memory = exposure_multiplier * switching_magnitude(u, u_prev)
        reference = 0.5 * lambda_2 * M_STRONG * np.sum((u - v_t) ** 2)
        return hitting_cost(u, y_t) + lambda_1 * dual_memory + reference

    def gradient(u):
        memory_gradient = exposure_multiplier * ELL * (u - u_prev)
        reference_gradient = lambda_2 * M_STRONG * (u - v_t)
        return hitting_gradient(u, y_t) + lambda_1 * memory_gradient + reference_gradient

    result = minimize(
        fun=objective,
        x0=feasible_start(u_prev, budget, upper),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, float(limit)) for limit in upper],
        constraints=[{
            "type": "ineq",
            "fun": lambda u: budget - np.sum(u),
            "jac": lambda u: -np.ones(N_DOMAINS),
        }],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Primal optimization failed: {result.message}")
    return result.x


def solve_auxiliary(kappa_t, rho, z_max):
    """Solve min_{z in Z} rho*max(z)-<kappa_t,z> exactly by LP."""
    dimension = len(kappa_t)

    # Variables are [z_1, ..., z_D, s], where s >= max_k z_k.
    objective = np.concatenate((-kappa_t, [rho]))
    a_ub = np.zeros((dimension, dimension + 1))
    a_ub[:, :dimension] = np.eye(dimension)
    a_ub[:, -1] = -1.0

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=np.zeros(dimension),
        bounds=[(0.0, z_max)] * dimension + [(0.0, z_max)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Auxiliary optimization failed: {result.message}")
    return result.x[:dimension]

def load_exposures(
    path,
    horizon,
    dimension,
    seed=None,
):
    """Return the same normalized exposure vector at every time."""

    if path is not None:
        raw_vector = (
            pd.read_csv(path)
            .iloc[0]
            .to_numpy(dtype=float)
        )

        if raw_vector.shape != (dimension,):
            raise ValueError(
                "Exposure file must contain "
                f"{dimension} columns"
            )
    else:
        # Constant raw exposure profile.
        raw_vector = np.ones(
            dimension,
            dtype=float,
        )

    if not np.all(np.isfinite(raw_vector)):
        raise ValueError(
            "Exposure values must be finite"
        )

    if np.any(raw_vector < 0):
        raise ValueError(
            "Exposure values must be nonnegative"
        )

    if np.all(raw_vector == 0):
        raise ValueError(
            "At least one exposure value "
            "must be positive"
        )

    fixed_exposure = normalize_exposure(
        raw_vector
    )

    return np.tile(
        fixed_exposure,
        (horizon, 1),
    )
def run_mroo(
    trace,
    exposures,
    u_initial,
    budget,
    upper,
    eta,
    lambda_1,
    lambda_2,
    rho,
    kappa_initial,
):
    demands = trace[DEMAND_COLUMNS].to_numpy(dtype=float)
    horizon = len(trace)
    dimension = exposures.shape[1]

    if exposures.shape[0] != horizon:
        raise ValueError("Exposure horizon does not match demand horizon")
    if kappa_initial.shape != (dimension,):
        raise ValueError(f"kappa_initial must have shape ({dimension},)")
    if np.any(kappa_initial < 0) or not np.all(np.isfinite(kappa_initial)):
        raise ValueError("kappa_initial must be finite and nonnegative")

    u_initial = np.asarray(u_initial, dtype=float)
    if u_initial.shape != (N_DOMAINS,):
        raise ValueError(f"u_initial must have shape ({N_DOMAINS},)")
    if not np.all(np.isfinite(u_initial)):
        raise ValueError("u_initial must contain finite values")
    if np.any(u_initial < 0) or np.any(u_initial > upper):
        raise ValueError(
            "u_initial must satisfy 0 <= u_initial[i] <= upper[i]"
        )
    if u_initial.sum() > budget + 1e-10:
        raise ValueError("u_initial must satisfy sum(u_initial) <= budget")

    # Since 0 <= u_i <= upper_i, this bounds every possible switching magnitude.
    z_max = float(0.5 * np.sum(ELL * upper**2))

    allocations = np.empty((horizon, N_DOMAINS))
    references = np.empty_like(allocations)
    hitting = np.empty(horizon)
    switching = np.empty(horizon)
    memories = np.empty((horizon, dimension))
    auxiliary = np.empty_like(memories)
    kappas = np.empty((horizon + 1, dimension))

    kappa_t = kappa_initial.copy()
    kappas[0] = kappa_t
    u_prev = u_initial.copy()

    for t in range(horizon):
        y_t = demands[t]
        w_t = exposures[t]

        v_t = solve_reference(y_t, budget, upper, u_prev)
        u_t = solve_primal(
            y_t=y_t,
            u_prev=u_prev,
            v_t=v_t,
            w_t=w_t,
            kappa_t=kappa_t,
            budget=budget,
            upper=upper,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )
        z_t = solve_auxiliary(kappa_t, rho, z_max)
        d_t = memory_cost(u_t, u_prev, w_t)

        # Euclidean h gives projected mirror descent on R_+^D:
        # kappa_{t+1} = [kappa_t - eta * (z_t - lambda_1*d_t)]_+.
        g_t = z_t - lambda_1 * d_t
        kappa_t = np.maximum(0.0, kappa_t - eta * g_t)

        allocations[t] = u_t
        references[t] = v_t
        hitting[t] = hitting_cost(u_t, y_t)
        switching[t] = switching_magnitude(u_t, u_prev)
        memories[t] = d_t
        auxiliary[t] = z_t
        kappas[t + 1] = kappa_t
        u_prev = u_t

    average_hitting = float(hitting.mean())
    long_term = long_term_cost(memories, rho=rho)

    result = trace[["time"]].copy()
    for index, pdu_id in enumerate(PDU_IDS):
        result[f"u_pdu{pdu_id}"] = allocations[:, index]
        result[f"v_pdu{pdu_id}"] = references[:, index]
        result[f"switch_pdu{pdu_id}"] = np.vstack((
            individual_switching_costs(allocations[0], u_initial),
            [individual_switching_costs(allocations[t], allocations[t - 1])
             for t in range(1, horizon)],
        ))[:, index]
    result["hitting_cost"] = hitting
    result["switching_magnitude"] = switching
    for k in range(dimension):
        result[f"w_{k + 1}"] = exposures[:, k]
        result[f"d_{k + 1}"] = memories[:, k]
        result[f"z_{k + 1}"] = auxiliary[:, k]
        result[f"kappa_{k + 1}"] = kappas[:-1, k]

    summary = {
        "horizon": horizon,
        "budget": float(budget),
        "upper_bounds": upper.tolist(),
        "u_initial": u_initial.tolist(),
        "eta": float(eta),
        "lambda_1": float(lambda_1),
        "lambda_2": float(lambda_2),
        "rho": float(rho),
        "average_hitting_cost": average_hitting,
        "long_term_cost": long_term,
        "total_objective": average_hitting + long_term,
        "average_switching_magnitude": float(switching.mean()),
        "maximum_average_entity_burden": float(memories.mean(axis=0).max()),
        "final_kappa": kappas[-1].tolist(),
    }
    return result, summary


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
    # parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=float, default=3.5)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--lambda-1", type=float, default=8.0)
    parser.add_argument("--lambda-2", type=float, default=3.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--kappa-initial", type=float, default=0.05)
    parser.add_argument(
        "--u-initial",
        type=float,
        nargs="+",
        default=[0.5],
        help=(
            "Initial allocation u_0. Give one shared value or five values "
            "in PDU order: 6 7 8 9 10."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "results" / "mroo",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.entities <= 0:
        raise ValueError("entities must be positive")

    trace = pd.read_csv(args.trace)
    missing = [column for column in ["time", *DEMAND_COLUMNS] if column not in trace]
    if missing:
        raise ValueError(f"Trace is missing columns: {missing}")
    if trace.empty:
        raise ValueError("Trace is empty")
    if not np.isfinite(trace[DEMAND_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Demand trace contains missing or nonfinite values")

    upper = np.full(N_DOMAINS, args.upper, dtype=float)
    check_parameters(
        args.budget, upper, args.eta,
        args.lambda_1, args.lambda_2, args.rho,
    )
    exposures = load_exposures(
    args.exposure_file,
    len(trace),
    args.entities,)
    kappa_initial = np.full(args.entities, args.kappa_initial, dtype=float)

    if len(args.u_initial) == 1:
        u_initial = np.full(N_DOMAINS, args.u_initial[0], dtype=float)
    elif len(args.u_initial) == N_DOMAINS:
        u_initial = np.asarray(args.u_initial, dtype=float)
    else:
        raise ValueError(
            f"--u-initial requires either one value or {N_DOMAINS} values"
        )

    results, summary = run_mroo(
        trace=trace,
        exposures=exposures,
        u_initial=u_initial,
        budget=args.budget,
        upper=upper,
        eta=args.eta,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        rho=args.rho,
        kappa_initial=kappa_initial,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_file = args.output_directory / "mroo_results.csv"
    summary_file = args.output_directory / "mroo_summary.json"
    results.to_csv(result_file, index=False)
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {result_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
