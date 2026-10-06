from pathlib import Path
import pandas as pd


BASE_DIR = (
    Path(__file__).resolve().parent
    / "new_results"
    / "window_beta_m_sweep"
    / "rho_7"
    / "1min_24hour"
    / "fixed_weights_w1_0p2_w2_0p3_w3_0p5"
)


folders = [
    "beta_400_m_100_T_1440",
    "beta_600_m_100_T_1440",
    "beta_1000_m_100_T_1440",
]


for folder_name in folders:

    folder = BASE_DIR / folder_name

    summary_file = (
        folder
        / "mroo_window_summary.csv"
    )

    df = pd.read_csv(
        summary_file
    )

    print(
        "\n========================================"
    )

    print(
        folder_name
    )

    print(
        "========================================"
    )

    columns = [
        "lambda_1",
        "eta",
        "kappa_init",
        "mean_hitting_cost",
        "mean_long_term_cost",
        "mean_total_cost",
    ]

    print(
        df[
            columns
        ]
        .sort_values(
            "mean_total_cost"
        )
        .to_string(
            index=False
        )
    )