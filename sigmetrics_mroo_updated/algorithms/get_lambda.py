from pathlib import Path

import pandas as pd


# ============================================================
# BASE RESULT DIRECTORY
# ============================================================

ALGORITHM_DIR = Path(__file__).resolve().parent

BASE_RESULT_DIR = (
    ALGORITHM_DIR
    / "new_results"
    / "window_beta_m_sweep"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)


# ============================================================
# FIND ALL T = 17280 SETTINGS
# ============================================================

SETTING_DIRS = sorted(
    [
        path
        for path in BASE_RESULT_DIR.iterdir()
        if (
            path.is_dir()
            and "_T_17280" in path.name
            and path.name.startswith("beta_")
        )
    ]
)


if not SETTING_DIRS:
    raise RuntimeError(
        f"No T=17280 result folders found in:\n"
        f"{BASE_RESULT_DIR}"
    )


# ============================================================
# COLLECT SELECTED CONFIGURATIONS
# ============================================================

rows = []

for setting_dir in SETTING_DIRS:

    summary_file = (
        setting_dir
        / "combined_algorithm_summary.csv"
    )

    if not summary_file.exists():
        print(
            f"Skipping {setting_dir.name}: "
            f"combined_algorithm_summary.csv not found"
        )
        continue

    df = pd.read_csv(summary_file)

    # --------------------------------------------------------
    # Keep only algorithms where parameter pair matters
    # --------------------------------------------------------

    smroo_df = df[
        df["algorithm"].isin(
            [
                "S-MROO-MAX",
                "S-MROO-SUM",
            ]
        )
    ].copy()

    if len(smroo_df) == 0:
        print(
            f"Skipping {setting_dir.name}: "
            f"no S-MROO rows found"
        )
        continue

    # --------------------------------------------------------
    # Extract selected configuration
    # --------------------------------------------------------

    for _, row in smroo_df.iterrows():

        rows.append(
            {
                "folder":
                    setting_dir.name,

                "algorithm":
                    row["algorithm"],

                "beta":
                    row["beta"]
                    if "beta" in row.index
                    else None,

                "m":
                    row["m"]
                    if "m" in row.index
                    else None,

                "beta_over_m":
                    row["beta_over_m"]
                    if "beta_over_m" in row.index
                    else None,

                "parameter_pair":
                    row["parameter_pair"]
                    if "parameter_pair" in row.index
                    else None,

                "lambda_1":
                    row["lambda_1"]
                    if "lambda_1" in row.index
                    else None,

                "lambda_2":
                    row["lambda_2"]
                    if "lambda_2" in row.index
                    else None,

                "mean_hitting_cost":
                    row["mean_hitting_cost"]
                    if "mean_hitting_cost" in row.index
                    else None,

                "mean_long_term_cost":
                    row["mean_long_term_cost"]
                    if "mean_long_term_cost" in row.index
                    else None,

                "mean_total_cost":
                    row["mean_total_cost"]
                    if "mean_total_cost" in row.index
                    else None,
            }
        )


# ============================================================
# CREATE TABLE
# ============================================================

result_df = pd.DataFrame(rows)


if len(result_df) == 0:
    raise RuntimeError(
        "No selected S-MROO configurations were found."
    )


result_df = (
    result_df
    .sort_values(
        [
            "beta",
            "m",
            "algorithm",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n========================================"
)
print(
    "SELECTED S-MROO CONFIGURATIONS FOR T=17280"
)
print(
    "========================================\n"
)

print(
    result_df[
        [
            "beta",
            "m",
            "beta_over_m",
            "algorithm",
            "parameter_pair",
            "lambda_1",
            "lambda_2",
            "mean_total_cost",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# OPTIONAL: PRINT PAIR INTERPRETATION
# ============================================================

print(
    "\n========================================"
)
print(
    "PAIR INTERPRETATION"
)
print(
    "========================================"
)

for _, row in result_df.iterrows():

    pair = row["parameter_pair"]

    if pd.isna(pair):
        pair_text = "unknown"

    elif int(pair) == 1:
        pair_text = (
            "Pair 1: lambda_1 theoretical, lambda_2 = 0"
        )

    elif int(pair) == 2:
        pair_text = (
            "Pair 2: lambda_1 = 1, lambda_2 > 0"
        )

    else:
        pair_text = f"Pair {pair}"

    print(
        f"beta={row['beta']:g}, "
        f"m={row['m']:g}, "
        f"{row['algorithm']}: "
        f"{pair_text}"
    )