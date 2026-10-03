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
# IMPORT GREEDY
# ============================================================

from greedy import run_greedy


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
    "GREEDY result directory:",
    RESULT_DIR
)
RESULT_FILE = (
    RESULT_DIR
    / "greedy_full_horizon_results.csv"
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
        "FULL-HORIZON GREEDY"
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
    # SANITY CHECK
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
    # RUN GREEDY ON ENTIRE HORIZON
    #
    # One continuous run.
    # No daily windows.
    # No reset every 8640 steps.
    # ========================================================

    print(
        "\nRunning Greedy over full horizon..."
    )

    result = run_greedy(
        demand=demand
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_row = {

        "T":
            T,

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
    # PRINT RESULT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "GREEDY RESULT"
    )

    print(
        "========================================"
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
        "\nSaved to:",
        RESULT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()