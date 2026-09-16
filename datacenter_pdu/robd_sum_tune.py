"""Run ROBD-SUM with the fixed parameter values reported in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mroo import DEMAND_COLUMNS, N_DOMAINS, load_exposures
from robd_sum import run_robd_sum


# Fixed ROBD parameters reported in the NeurIPS draft.
LAMBDA_1 = 0.3
LAMBDA_2 = 0.8


def create_initial_allocation(values):
    if len(values) == 1:
        return np.full(N_DOMAINS, values[0], dtype=float)
    if len(values) == N_DOMAINS:
        return np.asarray(values, dtype=float)
    raise ValueError(
        f"--u-initial requires either one value or {N_DOMAINS} values"
    )


def parse_args():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        default=project / "processed_powerdata" / "cell_a_power_trace.csv",
    )
    parser.add_argument("--budget", type=float, default=3.5)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--u-initial", type=float, nargs="+", default=[0.5])
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--entities", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exposure-file", type=Path, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "results" / "robd_sum_tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trace = (
        pd.read_csv(args.trace)
        .sort_values("time")
        .reset_index(drop=True)
    )

    required_columns = ["time", *DEMAND_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in trace.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if len(trace) < 1:
        raise ValueError("The trace is empty")
    if args.entities <= 0:
        raise ValueError("--entities must be positive")
    if args.rho < 0:
        raise ValueError("--rho must be nonnegative")

    upper = np.full(N_DOMAINS, args.upper, dtype=float)
    u_initial = create_initial_allocation(args.u_initial)

    if np.any(u_initial < 0):
        raise ValueError("u_initial must be nonnegative")
    if np.any(u_initial > upper):
        raise ValueError("u_initial exceeds a domain upper bound")
    if u_initial.sum() > args.budget + 1e-10:
        raise ValueError("sum(u_initial) exceeds the budget")

    exposures = load_exposures(
        path=args.exposure_file,
        horizon=len(trace),
        dimension=args.entities,
        seed=args.seed,
    )
    if len(exposures) != len(trace):
        raise ValueError("Exposure sequence length does not match the trace")

    print("\nFixed ROBD-SUM parameters")
    print("-------------------------")
    print(f"lambda_1 = {LAMBDA_1:g}")
    print(f"lambda_2 = {LAMBDA_2:g}")
    print(f"rho = {args.rho:g}")
    print(f"trace rows = {len(trace)}")

    results, summary = run_robd_sum(
        trace=trace,
        exposures=exposures,
        u_initial=u_initial,
        budget=args.budget,
        upper=upper,
        lambda_1=LAMBDA_1,
        lambda_2=LAMBDA_2,
        rho=args.rho,
    )

    parameters = {
        "algorithm": "ROBD-SUM",
        "lambda_1": LAMBDA_1,
        "lambda_2": LAMBDA_2,
        "selection_method": "fixed_neurips_paper_parameters_no_tuning",
        "evaluated_rows": int(len(trace)),
        "fixed_parameters": {
            "budget": args.budget,
            "upper_bounds": upper.tolist(),
            "u_initial": u_initial.tolist(),
            "rho": args.rho,
            "entities": args.entities,
            "seed": args.seed,
            "exposure_file": (
                str(args.exposure_file)
                if args.exposure_file is not None
                else None
            ),
        },
        "full_trace_summary": summary,
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_file = (
        args.output_directory / "robd_sum_fixed_full_trace_results.csv"
    )
    parameter_file = args.output_directory / "best_parameters.json"

    results.to_csv(result_file, index=False)
    parameter_file.write_text(
        json.dumps(parameters, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFull-trace diagnostic result")
    print(
        "Average hitting cost = "
        f"{summary['average_hitting_cost']:.8f}"
    )
    print(f"Long-term cost = {summary['long_term_cost']:.8f}")
    print(f"Total objective = {summary['total_objective']:.8f}")
    print("\nSaved:")
    print(result_file)
    print(parameter_file)
    print("\nNow run compare.py for the 100-window box plots.")


if __name__ == "__main__":
    main()