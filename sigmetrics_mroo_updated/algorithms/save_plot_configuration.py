from pathlib import Path
from datetime import datetime
import ast
import shutil

import numpy as np
import pandas as pd


# ============================================================
# 1. USER SETTINGS
#
# CHANGE ONLY THIS SECTION.
#
# Run this script ONLY when you have looked at a configuration
# and decided that you want to permanently keep it for later
# paper plots.
# ============================================================


# ------------------------------------------------------------
# Experiment whose selected plotting configuration should
# be saved.
# ------------------------------------------------------------

SETTING_FOLDER = "beta_200_m_100_T_1440"


# ------------------------------------------------------------
# Give this approved configuration a useful unique name.
#
# Examples:
#
#   "beta100_m100_final"
#   "beta200_m100_best"
#   "ratio1_theory_pair"
#
# Do NOT reuse a name unless you deliberately enable overwrite.
# ------------------------------------------------------------

SAVE_NAME = "beta200_m100_good_v1"


# ------------------------------------------------------------
# Optional note for yourself.
# ------------------------------------------------------------

NOTE = (
    "Preferred configuration after inspecting box/violin plots."
)


# ------------------------------------------------------------
# Protection against accidentally replacing an approved setup.
# ------------------------------------------------------------

OVERWRITE_EXISTING_SAVE = False


# ============================================================
# 2. PATHS
# ============================================================

ALGORITHM_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


BASE_RESULT_DIR = (
    ALGORITHM_DIR
    / "matrix_memory_v1"
    / "1min_24hour"
    / "matrix_memory"
)


# ------------------------------------------------------------
# Legacy fallback
# ------------------------------------------------------------

if not BASE_RESULT_DIR.exists():

    legacy_dir = (
        ALGORITHM_DIR
        / "matrix_memory_v1"
        / "1min_24hour"
        / "matrix_weights"
    )

    if legacy_dir.exists():

        BASE_RESULT_DIR = legacy_dir


SETTING_DIR = (
    BASE_RESULT_DIR
    / SETTING_FOLDER
)


# ============================================================
# 3. FILES PRODUCED BY plot_beta_m_costs.py
#
# These contain EXACTLY the configurations that were used
# for the plot you just inspected.
# ============================================================

SELECTED_WINDOW_FILE = (
    SETTING_DIR
    / "plot_selected_algorithm_window_results.csv"
)


SELECTED_SUMMARY_FILE = (
    SETTING_DIR
    / "plot_selected_algorithm_summary.csv"
)


# ============================================================
# 4. PERMANENT APPROVED-PLOT STORAGE
# ============================================================

SAVED_ROOT = (
    BASE_RESULT_DIR
    / "saved_plot_configurations"
)


SAVED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


MASTER_SUMMARY_FILE = (
    SAVED_ROOT
    / "saved_configuration_summary.csv"
)


MASTER_WINDOW_FILE = (
    SAVED_ROOT
    / "saved_window_results.csv"
)


SNAPSHOT_ROOT = (
    SAVED_ROOT
    / "snapshots"
)


SNAPSHOT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


SNAPSHOT_DIR = (
    SNAPSHOT_ROOT
    / SAVE_NAME
)


# ============================================================
# 5. BASIC VALIDATION
# ============================================================

if not SELECTED_WINDOW_FILE.exists():

    raise FileNotFoundError(
        "\nSelected window-result file does not exist:\n"
        f"{SELECTED_WINDOW_FILE}\n\n"
        "Run plot_beta_m_costs.py first using the configuration "
        "that you want to save."
    )


if not SELECTED_SUMMARY_FILE.exists():

    raise FileNotFoundError(
        "\nSelected summary file does not exist:\n"
        f"{SELECTED_SUMMARY_FILE}\n\n"
        "Run plot_beta_m_costs.py first."
    )


if (
    SNAPSHOT_DIR.exists()
    and
    not OVERWRITE_EXISTING_SAVE
):

    raise RuntimeError(
        "\nA saved configuration with this SAVE_NAME already exists:\n"
        f"{SNAPSHOT_DIR}\n\n"
        "Choose another SAVE_NAME or set:\n\n"
        "OVERWRITE_EXISTING_SAVE = True\n\n"
        "if you intentionally want to replace it."
    )


# ============================================================
# 6. LOAD EXACT ROWS THAT GENERATED THE PLOT
# ============================================================

window_df = pd.read_csv(
    SELECTED_WINDOW_FILE,
    low_memory=False,
)


plot_summary_df = pd.read_csv(
    SELECTED_SUMMARY_FILE,
    low_memory=False,
)


# ============================================================
# 7. VALIDATE WINDOW DATA
# ============================================================

required_window_columns = {
    "algorithm",
    "window_id",
    "hitting_cost",
    "long_term_cost",
    "total_cost",
    "beta",
    "m",
}


missing = (
    required_window_columns
    -
    set(window_df.columns)
)


if missing:

    raise RuntimeError(
        "Selected result file is missing columns: "
        f"{sorted(missing)}"
    )


beta_values = (
    pd.to_numeric(
        window_df["beta"],
        errors="coerce",
    )
    .dropna()
    .unique()
)


m_values = (
    pd.to_numeric(
        window_df["m"],
        errors="coerce",
    )
    .dropna()
    .unique()
)


if len(beta_values) != 1:

    raise RuntimeError(
        "Expected exactly one beta in selected results, "
        f"found {beta_values}."
    )


if len(m_values) != 1:

    raise RuntimeError(
        "Expected exactly one m in selected results, "
        f"found {m_values}."
    )


BETA = float(beta_values[0])

M = float(m_values[0])

BETA_M_RATIO = (
    BETA / M
)


# ============================================================
# 8. HELPERS
# ============================================================

def first_valid(
    frame,
    column,
):

    if column not in frame.columns:
        return np.nan

    values = (
        frame[column]
        .dropna()
    )

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


def parse_kappa_value(
    value,
):

    if value is None:
        return []


    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):

        return (
            np.asarray(
                value,
                dtype=float,
            )
            .reshape(-1)
            .tolist()
        )


    try:

        if pd.isna(value):
            return []

    except Exception:

        pass


    text = str(value).strip()


    # --------------------------------------------------------
    # Plain scalar
    # --------------------------------------------------------

    try:

        return [
            float(text)
        ]

    except ValueError:

        pass


    # --------------------------------------------------------
    # Serialized list such as:
    #
    # [3.0, 3.0, 3.0]
    # --------------------------------------------------------

    try:

        parsed = ast.literal_eval(text)

        return (
            np.asarray(
                parsed,
                dtype=float,
            )
            .reshape(-1)
            .tolist()
        )

    except Exception:

        return []


def get_kappa_from_frame(
    frame,
):

    # --------------------------------------------------------
    # MROO may store:
    #
    # kappa_init_0
    # kappa_init_1
    # kappa_init_2
    # --------------------------------------------------------

    prefix = "kappa_init_"

    indexed_columns = []

    for column in frame.columns:

        name = str(column)

        if not name.startswith(prefix):
            continue

        suffix = name[len(prefix):]

        if suffix.isdigit():

            indexed_columns.append(
                (
                    int(suffix),
                    column,
                )
            )


    if len(indexed_columns) > 0:

        indexed_columns.sort(
            key=lambda item: item[0]
        )

        values = []

        first_row = frame.iloc[0]

        for _, column in indexed_columns:

            value = first_row[column]

            if pd.isna(value):
                continue

            values.append(
                float(value)
            )


        if len(values) > 0:

            return values


    # --------------------------------------------------------
    # DMD / serialized fallback
    # --------------------------------------------------------

    if "kappa_init" in frame.columns:

        value = first_valid(
            frame,
            "kappa_init",
        )

        return parse_kappa_value(
            value
        )


    return []


def format_kappa(
    values,
):

    if len(values) == 0:
        return ""

    return (
        "["
        +
        ", ".join(
            f"{float(value):.12g}"
            for value in values
        )
        +
        "]"
    )


# ============================================================
# 9. VERIFY ONE CONFIGURATION PER ALGORITHM
# ============================================================

algorithms = (
    window_df["algorithm"]
    .dropna()
    .unique()
    .tolist()
)


print(
    "\n========================================"
)

print(
    "CONFIGURATION TO SAVE"
)

print(
    "========================================"
)


print(
    "SAVE_NAME   =",
    SAVE_NAME,
)


print(
    "beta        =",
    BETA,
)


print(
    "m           =",
    M,
)


print(
    "beta/m      =",
    BETA_M_RATIO,
)


print(
    "algorithms  =",
    algorithms,
)


# ============================================================
# 10. BUILD PER-ALGORITHM SUMMARY
#
# We recompute means/stds from the exact window data rather
# than trusting an older summary file.
#
# This guarantees the summary corresponds to the same rows
# that will later be used in the boxplot.
# ============================================================

summary_rows = []


for algorithm in algorithms:

    algorithm_df = (
        window_df[
            window_df["algorithm"]
            ==
            algorithm
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Remove accidental duplicated windows
    # --------------------------------------------------------

    algorithm_df = (
        algorithm_df
        .drop_duplicates(
            subset=[
                "window_id",
            ],
            keep="last",
        )
        .sort_values(
            "window_id"
        )
        .reset_index(
            drop=True
        )
    )


    n_windows = (
        algorithm_df["window_id"]
        .nunique()
    )


    # --------------------------------------------------------
    # Configuration parameters
    # --------------------------------------------------------

    lambda_1 = first_valid(
        algorithm_df,
        "lambda_1",
    )


    lambda_2 = first_valid(
        algorithm_df,
        "lambda_2",
    )


    eta = first_valid(
        algorithm_df,
        "eta",
    )


    # eta is your learning / dual step size.
    learning_rate = eta


    kappa_values = (
        get_kappa_from_frame(
            algorithm_df
        )
    )


    kappa_text = (
        format_kappa(
            kappa_values
        )
    )


    config_id = first_valid(
        algorithm_df,
        "config_id",
    )


    # --------------------------------------------------------
    # Cost statistics
    # --------------------------------------------------------

    mean_hitting = (
        algorithm_df[
            "hitting_cost"
        ].mean()
    )


    std_hitting = (
        algorithm_df[
            "hitting_cost"
        ].std()
    )


    mean_long_term = (
        algorithm_df[
            "long_term_cost"
        ].mean()
    )


    std_long_term = (
        algorithm_df[
            "long_term_cost"
        ].std()
    )


    mean_total = (
        algorithm_df[
            "total_cost"
        ].mean()
    )


    std_total = (
        algorithm_df[
            "total_cost"
        ].std()
    )


    summary_rows.append(
        {
            "save_name":
                SAVE_NAME,

            "note":
                NOTE,

            "setting_folder":
                SETTING_FOLDER,

            "algorithm":
                algorithm,

            "beta":
                BETA,

            "m":
                M,

            "beta_m_ratio":
                BETA_M_RATIO,

            "n_windows":
                n_windows,

            "lambda_1":
                lambda_1,

            "lambda_2":
                lambda_2,

            "eta":
                eta,

            # Alias included deliberately because later
            # plotting/table scripts may call eta
            # "learning_rate".
            "learning_rate":
                learning_rate,

            "kappa":
                kappa_text,

            "config_id":
                config_id,

            "mean_hitting_cost":
                mean_hitting,

            "std_hitting_cost":
                std_hitting,

            "mean_long_term_cost":
                mean_long_term,

            "std_long_term_cost":
                std_long_term,

            "mean_total_cost":
                mean_total,

            "std_total_cost":
                std_total,
        }
    )


approved_summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 11. ADD SAVE METADATA TO EVERY WINDOW
#
# Keeping ALL window-level data is necessary for future
# boxplots/violin plots.
# ============================================================

saved_at = (
    datetime.now()
    .astimezone()
    .isoformat(
        timespec="seconds"
    )
)


approved_summary_df.insert(
    1,
    "saved_at",
    saved_at,
)


approved_window_df = (
    window_df.copy()
)


approved_window_df.insert(
    0,
    "save_name",
    SAVE_NAME,
)


approved_window_df.insert(
    1,
    "saved_at",
    saved_at,
)


approved_window_df.insert(
    2,
    "note",
    NOTE,
)


approved_window_df.insert(
    3,
    "setting_folder",
    SETTING_FOLDER,
)


approved_window_df.insert(
    4,
    "beta_m_ratio",
    BETA_M_RATIO,
)


# ============================================================
# 12. ADD STANDARDIZED KAPPA / LEARNING-RATE COLUMNS TO
# WINDOW DATA
#
# This makes future plotting much easier because every
# algorithm has the same metadata columns.
# ============================================================

if "learning_rate" not in approved_window_df.columns:

    if "eta" in approved_window_df.columns:

        approved_window_df[
            "learning_rate"
        ] = approved_window_df[
            "eta"
        ]

    else:

        approved_window_df[
            "learning_rate"
        ] = np.nan


approved_window_df[
    "saved_kappa"
] = ""


for algorithm in algorithms:

    mask = (
        approved_window_df[
            "algorithm"
        ]
        ==
        algorithm
    )


    algorithm_df = (
        approved_window_df[
            mask
        ]
    )


    kappa = (
        get_kappa_from_frame(
            algorithm_df
        )
    )


    approved_window_df.loc[
        mask,
        "saved_kappa",
    ] = format_kappa(
        kappa
    )


# ============================================================
# 13. DISPLAY WHAT WILL BE SAVED
# ============================================================

display_columns = [

    "algorithm",

    "beta",

    "m",

    "beta_m_ratio",

    "n_windows",

    "lambda_1",

    "lambda_2",

    "eta",

    "kappa",

    "mean_hitting_cost",

    "std_hitting_cost",

    "mean_long_term_cost",

    "std_long_term_cost",

    "mean_total_cost",

    "std_total_cost",
]


print(
    "\n========================================"
)

print(
    "APPROVED CONFIGURATION SUMMARY"
)

print(
    "========================================"
)


print(
    approved_summary_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 14. HANDLE OVERWRITE
# ============================================================

if (
    SNAPSHOT_DIR.exists()
    and
    OVERWRITE_EXISTING_SAVE
):

    shutil.rmtree(
        SNAPSHOT_DIR
    )


SNAPSHOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 15. SAVE IMMUTABLE SNAPSHOT
#
# This folder contains exactly what belonged to this approved
# configuration.
# ============================================================

SNAPSHOT_SUMMARY_FILE = (
    SNAPSHOT_DIR
    / "summary.csv"
)


SNAPSHOT_WINDOW_FILE = (
    SNAPSHOT_DIR
    / "window_results.csv"
)


approved_summary_df.to_csv(
    SNAPSHOT_SUMMARY_FILE,
    index=False,
)


approved_window_df.to_csv(
    SNAPSHOT_WINDOW_FILE,
    index=False,
)


# ============================================================
# 16. UPDATE MASTER APPROVED SUMMARY
#
# One row per algorithm per approved experiment.
# ============================================================

if MASTER_SUMMARY_FILE.exists():

    master_summary_df = (
        pd.read_csv(
            MASTER_SUMMARY_FILE,
            low_memory=False,
        )
    )

else:

    master_summary_df = (
        pd.DataFrame()
    )


# ------------------------------------------------------------
# If replacing an existing SAVE_NAME, remove old rows first.
# ------------------------------------------------------------

if (
    len(master_summary_df) > 0
    and
    "save_name"
    in master_summary_df.columns
):

    existing_match = (
        master_summary_df[
            "save_name"
        ].astype(str)
        ==
        str(SAVE_NAME)
    )


    if (
        existing_match.any()
        and
        not OVERWRITE_EXISTING_SAVE
    ):

        raise RuntimeError(
            "\nSAVE_NAME already exists in master summary:\n"
            f"{SAVE_NAME}\n"
            "Use another SAVE_NAME."
        )


    if OVERWRITE_EXISTING_SAVE:

        master_summary_df = (
            master_summary_df[
                ~existing_match
            ]
            .copy()
        )


master_summary_df = pd.concat(
    [
        master_summary_df,
        approved_summary_df,
    ],
    ignore_index=True,
    sort=False,
)


master_summary_df = (
    master_summary_df
    .sort_values(
        by=[
            "beta_m_ratio",
            "beta",
            "m",
            "save_name",
            "algorithm",
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


master_summary_df.to_csv(
    MASTER_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 17. UPDATE MASTER WINDOW RESULTS
#
# This is the IMPORTANT file for future boxplots.
#
# It keeps every window from every approved configuration.
# ============================================================

if MASTER_WINDOW_FILE.exists():

    master_window_df = (
        pd.read_csv(
            MASTER_WINDOW_FILE,
            low_memory=False,
        )
    )

else:

    master_window_df = (
        pd.DataFrame()
    )


if (
    len(master_window_df) > 0
    and
    "save_name"
    in master_window_df.columns
):

    existing_match = (
        master_window_df[
            "save_name"
        ].astype(str)
        ==
        str(SAVE_NAME)
    )


    if (
        existing_match.any()
        and
        not OVERWRITE_EXISTING_SAVE
    ):

        raise RuntimeError(
            "\nSAVE_NAME already exists in master window file:\n"
            f"{SAVE_NAME}\n"
            "Use another SAVE_NAME."
        )


    if OVERWRITE_EXISTING_SAVE:

        master_window_df = (
            master_window_df[
                ~existing_match
            ]
            .copy()
        )


master_window_df = pd.concat(
    [
        master_window_df,
        approved_window_df,
    ],
    ignore_index=True,
    sort=False,
)


sort_columns = [

    column

    for column in [
        "beta_m_ratio",
        "beta",
        "m",
        "save_name",
        "algorithm",
        "window_id",
    ]

    if column
    in master_window_df.columns
]


master_window_df = (
    master_window_df
    .sort_values(
        by=sort_columns,
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


master_window_df.to_csv(
    MASTER_WINDOW_FILE,
    index=False,
)


# ============================================================
# 18. COPY ORIGINAL PLOT-SELECTION FILES
#
# This makes the saved snapshot self-contained.
# ============================================================

ORIGINAL_SELECTED_WINDOW_COPY = (
    SNAPSHOT_DIR
    / "original_plot_selected_window_results.csv"
)


ORIGINAL_SELECTED_SUMMARY_COPY = (
    SNAPSHOT_DIR
    / "original_plot_selected_summary.csv"
)


shutil.copy2(
    SELECTED_WINDOW_FILE,
    ORIGINAL_SELECTED_WINDOW_COPY,
)


shutil.copy2(
    SELECTED_SUMMARY_FILE,
    ORIGINAL_SELECTED_SUMMARY_COPY,
)


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print(
    "\n========================================"
)

print(
    "CONFIGURATION SAVED"
)

print(
    "========================================"
)


print(
    "\nSave name:"
)

print(
    SAVE_NAME
)


print(
    "\nSnapshot directory:"
)

print(
    SNAPSHOT_DIR
)


print(
    "\nMaster configuration summary:"
)

print(
    MASTER_SUMMARY_FILE
)


print(
    "\nMaster window-level results:"
)

print(
    MASTER_WINDOW_FILE
)


print(
    "\nSnapshot summary:"
)

print(
    SNAPSHOT_SUMMARY_FILE
)


print(
    "\nSnapshot window results:"
)

print(
    SNAPSHOT_WINDOW_FILE
)


print(
    "\nRows added to master summary:",
    len(
        approved_summary_df
    ),
)


print(
    "Window rows added:",
    len(
        approved_window_df
    ),
)


print(
    "\nYou can now change the plotting configuration "
    "without losing this approved result."
)