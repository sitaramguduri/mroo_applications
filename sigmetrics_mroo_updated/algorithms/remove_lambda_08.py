from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

SWEEP_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "beta_m_sweep_varied_lambda"
)


# ============================================================
# VALUE TO REMOVE
# ============================================================

LAMBDA_TO_REMOVE = 0.8


# ============================================================
# ALGORITHM RESULT FILES
# ============================================================

RESULT_FILES = [

    "mroo_full_horizon_results.csv",

    "smroo_sum_full_horizon_results.csv",

    "smroo_max_full_horizon_results.csv",
]


# ============================================================
# REMOVE lambda_1 = 0.8
# ============================================================

def remove_lambda_from_file(
    file_path,
):

    if not file_path.exists():

        return


    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as error:

        print(
            "Could not read:",
            file_path,
            error,
        )

        return


    if (
        len(df) == 0
        or "lambda_1" not in df.columns
    ):

        return


    lambda_values = pd.to_numeric(
        df["lambda_1"],
        errors="coerce",
    )


    remove_mask = np.isclose(
        lambda_values,
        LAMBDA_TO_REMOVE,
        rtol=1e-10,
        atol=1e-12,
    )


    removed_count = int(
        remove_mask.sum()
    )


    if removed_count == 0:

        print(
            f"No lambda={LAMBDA_TO_REMOVE} in "
            f"{file_path}"
        )

        return


    cleaned_df = (
        df.loc[
            ~remove_mask
        ]
        .reset_index(
            drop=True
        )
    )


    cleaned_df.to_csv(
        file_path,
        index=False,
    )


    print(
        f"Removed {removed_count} row(s) "
        f"with lambda_1={LAMBDA_TO_REMOVE} from:"
    )

    print(
        " ",
        file_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "REMOVING lambda_1 = 0.8"
    )

    print(
        "========================================"
    )


    setting_dirs = sorted(
        path
        for path in SWEEP_DIR.iterdir()
        if path.is_dir()
    )


    for setting_dir in setting_dirs:

        print(
            f"\nChecking {setting_dir.name}"
        )


        for filename in RESULT_FILES:

            remove_lambda_from_file(
                setting_dir
                / filename
            )


    print(
        "\n========================================"
    )

    print(
        "CLEANUP COMPLETE"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()