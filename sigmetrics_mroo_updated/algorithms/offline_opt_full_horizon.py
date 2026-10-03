import sys
from pathlib import Path
import os
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ALGORITHM_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


# ============================================================
# IMPORT OFFLINE OPT
# ============================================================

from offline_opt import run_offline_opt


# ============================================================
# FILES
# ============================================================

DATA_FILE = (
    PROJECT_DIR
    / "new_results"
    / "mroo_demand_trace_norm_weekday_aligned.csv"
)

# RESULT_DIR = (
#     ALGORITHM_DIR
#     / "new_results"
# )

# RESULT_DIR.mkdir(
#     parents=True,
#     exist_ok=True
# )
DEFAULT_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
)

RESULT_DIR = Path(
    os.environ.get(
        "MROO_RESULT_DIR",
        str(DEFAULT_RESULT_DIR),
    )
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "OFFLINE OPTIMAL result directory:",
    RESULT_DIR
)
RESULT_FILE = (
    RESULT_DIR
    / "offline_opt_full_horizon_results.csv"
)


# ============================================================
# DEMAND COLUMNS
# ============================================================

DEMAND_COLUMNS = [
    "code",
    "chat",
    "non-interactive",
]


# ============================================================
# LOAD FULL 7-DAY TRACE
# ============================================================

data = pd.read_csv(
    DATA_FILE
)

demand = data[
    DEMAND_COLUMNS
].to_numpy(
    dtype=float
)

T, D = demand.shape


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "FULL-HORIZON OFFLINE OPT"
    )

    print(
        "========================================"
    )

    print(
        "Horizon length T =",
        T
    )

    print(
        "Dimension D =",
        D
    )


    # ========================================================
    # OPTIONAL SANITY CHECK
    # ========================================================

    expected_T = (
        7
        * 24
        * 60
        * 6
    )

    if T != expected_T:

        print(
            f"\nWARNING: expected {expected_T} "
            f"time steps for 7 days, found {T}."
        )


    # ========================================================
    # RUN OFFLINE OPT
    # ========================================================

    print(
        "\nSolving offline optimum over full horizon..."
    )

    result = run_offline_opt(
        demand=demand
    )
    MEMORY_FILE = (
        RESULT_DIR
        / "offline_opt_full_horizon_memory.csv"
    )

    memory_history = result[
        "memory_history"
    ]

    memory_df = pd.DataFrame(
        memory_history,
        columns=[
            "d_0",
            "d_1",
            "d_2",
        ]
    )

    memory_df.to_csv(
        MEMORY_FILE,
        index=False
    )

    print(
        "Memory history saved to:",
        MEMORY_FILE
    )
    # ACTIONS_FILE = (
    #     RESULT_DIR
    #     / "offline_opt_full_horizon_actions.csv"
    # )

    # actions_df = pd.DataFrame(
    #     result["actions"],
    #     columns=[
    #         "u_0",
    #         "u_1",
    #         "u_2",
    #     ]
    # )

    # actions_df.to_csv(
    #     ACTIONS_FILE,
    #     index=False
    # )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_row = {

        "T":
            T,

        "solver_status":
            result[
                "solver_status"
            ],

        "solver_objective":
            result[
                "solver_objective"
            ],

        "hitting_cost":
            result[
                "hitting_cost"
            ],

        "long_term_cost":
            result[
                "long_term_cost"
            ],

        "total_cost":
            result[
                "total_cost"
            ],
    }


    result_df = pd.DataFrame([
        result_row
    ])


    result_df.to_csv(
        RESULT_FILE,
        index=False
    )


    # ========================================================
    # OBJECTIVE CONSISTENCY CHECK
    # ========================================================

    objective_difference = abs(
        result_row[
            "solver_objective"
        ]
        -
        result_row[
            "total_cost"
        ]
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "OFFLINE OPT RESULT"
    )

    print(
        "========================================"
    )


    print(
        "Solver status =",
        result_row[
            "solver_status"
        ]
    )

    print(
        "Solver objective =",
        result_row[
            "solver_objective"
        ]
    )

    print(
        "Hitting cost =",
        result_row[
            "hitting_cost"
        ]
    )

    print(
        "Long-term cost =",
        result_row[
            "long_term_cost"
        ]
    )

    print(
        "Total cost =",
        result_row[
            "total_cost"
        ]
    )

    print(
        "|solver objective - recomputed total| =",
        objective_difference
    )


    print(
        "\nSaved to:",
        RESULT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()