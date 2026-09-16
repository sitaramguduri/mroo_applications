"""Run MROO with the fixed parameter values reported in the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mroo import (
    DEMAND_COLUMNS,
    N_DOMAINS,
    check_parameters,
    load_exposures,
    run_mroo,
)


# Fixed MROO parameters used for the paper's main box plot.
ETA = 1e-5
KAPPA_INITIAL_SCALAR = 0.1
LAMBDA_1 = 1.0
LAMBDA_2 = 1.0


def parse_initial(values):
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
    parser.add_argument("--exposure-file", type=Path, default=None)
    parser.add_argument("--entities", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=float, default=3.5)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--u-initial", type=float, nargs="+", default=[0.5])
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "results" / "mroo_tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trace = (
        pd.read_csv(args.trace)
        .sort_values("time")
        .reset_index(drop=True)
    )

    required = ["time", *DEMAND_COLUMNS]
    missing = [column for column in required if column not in trace.columns]
    if missing:
        raise ValueError(f"Trace is missing columns: {missing}")
    if len(trace) < 1:
        raise ValueError("The trace is empty")
    if args.entities <= 0:
        raise ValueError("--entities must be positive")

    upper = np.full(N_DOMAINS, args.upper, dtype=float)
    u_initial = parse_initial(args.u_initial)

    check_parameters(
        args.budget,
        upper,
        ETA,
        LAMBDA_1,
        LAMBDA_2,
        args.rho,
    )

    exposures = load_exposures(
        args.exposure_file,
        horizon=len(trace),
        dimension=args.entities,
        seed=args.seed,
    )

    if len(exposures) != len(trace):
        raise ValueError("Exposure sequence length does not match the trace")

    kappa_initial = np.full(
        args.entities,
        KAPPA_INITIAL_SCALAR,
        dtype=float,
    )

    print("\nFixed MROO parameters")
    print("---------------------")
    print(f"eta = {ETA:g}")
    print(f"kappa_1 = {kappa_initial.tolist()}")
    print(f"lambda_1 = {LAMBDA_1:g}")
    print(f"lambda_2 = {LAMBDA_2:g}")
    print(f"rho = {args.rho:g}")
    print(f"trace rows = {len(trace)}")

    results, summary = run_mroo(
        trace=trace,
        exposures=exposures,
        u_initial=u_initial,
        budget=args.budget,
        upper=upper,
        eta=ETA,
        lambda_1=LAMBDA_1,
        lambda_2=LAMBDA_2,
        rho=args.rho,
        kappa_initial=kappa_initial,
    )

    parameters = {
        "eta": ETA,
        "kappa_initial_scalar": KAPPA_INITIAL_SCALAR,
        "kappa_initial_vector": kappa_initial.tolist(),
        "selection_method": "fixed_paper_parameters_no_tuning",
        "evaluated_rows": int(len(trace)),
        "fixed_parameters": {
            "budget": args.budget,
            "upper_bounds": upper.tolist(),
            "u_initial": u_initial.tolist(),
            "lambda_1": LAMBDA_1,
            "lambda_2": LAMBDA_2,
            "rho": args.rho,
            "entities": args.entities,
            "exposure_seed": args.seed if args.exposure_file is None else None,
            "exposure_file": (
                str(args.exposure_file)
                if args.exposure_file is not None
                else None
            ),
        },
        "full_trace_summary": summary,
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_file = args.output_directory / "mroo_fixed_full_trace_results.csv"
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


if __name__ == "__main__":
    main()