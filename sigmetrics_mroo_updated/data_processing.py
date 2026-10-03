import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CODE_TRACE_CSV = "AzureLLMInferenceTrace_code_1week.csv"
CHAT_TRACE_CSV = "AzureLLMInferenceTrace_conv_1week.csv"


# ============================================================
# TIME RESOLUTION
#
# OLD:
# TRACE_FREQ = "10s"
#
# NEW:
# One time step = 1 minute
# ============================================================

TRACE_FREQ = "1min"


WORKLOAD_NAMES = [
    "code",
    "chat",
    "non-interactive",
]


RESULT_DIR = "new_results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# READ AND AGGREGATE TRACE
# ============================================================

def read_azure_trace_series(
    path,
    freq="1min",
):

    df = pd.read_csv(
        path
    )


    # --------------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------------

    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        utc=True,
        format="mixed",
    )


    # --------------------------------------------------------
    # Demand of each request
    #
    # Total tokens =
    # context tokens + generated tokens
    # --------------------------------------------------------

    df["token"] = (
        df["ContextTokens"]
        +
        df["GeneratedTokens"]
    )


    # --------------------------------------------------------
    # Aggregate requests into time buckets
    #
    # With freq="1min", every bucket represents one minute.
    # --------------------------------------------------------

    df["bucket"] = (
        df["TIMESTAMP"]
        .dt.floor(
            freq
        )
    )


    token_by_bucket = (

        df.groupby(
            "bucket"
        )["token"]

        .sum()

    )


    # --------------------------------------------------------
    # Add missing time slots as zero demand
    #
    # This guarantees a continuous 1-minute timeline.
    # --------------------------------------------------------

    full_index = pd.date_range(

        token_by_bucket.index.min(),

        token_by_bucket.index.max(),

        freq=freq,

        tz="UTC",

    )


    token_by_bucket = (

        token_by_bucket

        .reindex(
            full_index,
            fill_value=0.0,
        )

    )


    return (
        token_by_bucket
        .astype(int)
    )


# ============================================================
# ALIGN TRACE BY WEEKDAY
# ============================================================

def align_one_week_by_weekday(
    token_series,
    freq="1min",
):

    # --------------------------------------------------------
    # Number of time slots in one day
    #
    # For 1-minute frequency:
    #
    # 24 * 60 = 1440
    # --------------------------------------------------------

    slots_per_day = int(

        pd.Timedelta(
            "1D"
        )

        /

        pd.Timedelta(
            freq
        )

    )


    print(
        "Slots per day:",
        slots_per_day,
    )


    parts = []


    for weekday in range(7):


        part = (

            token_series[

                token_series.index.weekday
                == weekday

            ]

            .sort_index()

        )


        if len(part) != slots_per_day:

            raise ValueError(

                f"weekday={weekday} "
                f"has {len(part)} slots, "
                f"expected {slots_per_day}"

            )


        parts.append(

            part.to_numpy(
                dtype=int
            )

        )


    return np.concatenate(
        parts
    )


# ============================================================
# PROCESS CODE TRACE
# ============================================================

code_series = read_azure_trace_series(

    CODE_TRACE_CSV,

    freq=TRACE_FREQ,

)


# ============================================================
# PROCESS CHAT TRACE
# ============================================================

chat_series = read_azure_trace_series(

    CHAT_TRACE_CSV,

    freq=TRACE_FREQ,

)


# ============================================================
# ALIGN BOTH TRACES BY WEEKDAY
# ============================================================

code_trace = align_one_week_by_weekday(

    code_series,

    freq=TRACE_FREQ,

)


chat_trace = align_one_week_by_weekday(

    chat_series,

    freq=TRACE_FREQ,

)


# ============================================================
# COMMON TRACE LENGTH
# ============================================================

T = min(

    len(code_trace),

    len(chat_trace),

)


code_trace = (
    code_trace[:T]
)


chat_trace = (
    chat_trace[:T]
)


print(
    "Total aligned slots:",
    T,
)


# ============================================================
# EXPECTED 1-MINUTE WEEK LENGTH
# ============================================================

expected_week_slots = (

    7
    * 24
    * 60

)


print(
    "Expected one-week slots:",
    expected_week_slots,
)


if T != expected_week_slots:

    print(

        "WARNING: expected a full one-week "
        f"1-minute trace of {expected_week_slots} slots, "
        f"but obtained {T}."

    )


# ============================================================
# SYNTHETIC NON-INTERACTIVE WORKLOAD
# ============================================================

non_interactive_ratio = 0.25

non_interactive_noise_ratio = 0.05


rng = np.random.default_rng(
    0
)


# ------------------------------------------------------------
# Mean code + chat demand at the new 1-minute resolution
# ------------------------------------------------------------

non_interactive_base = int(

    round(

        non_interactive_ratio

        *

        np.mean(
            code_trace
            +
            chat_trace
        )

    )

)


non_interactive_noise_std = max(

    1,

    int(

        round(

            non_interactive_noise_ratio

            *

            non_interactive_base

        )

    )

)


non_interactive_noise = (

    rng.normal(

        loc=0.0,

        scale=non_interactive_noise_std,

        size=T,

    )

    .round()

    .astype(int)

)


non_interactive_trace = np.clip(

    non_interactive_base

    +

    non_interactive_noise,

    0,

    None,

).astype(int)


# ============================================================
# CREATE DEMAND MATRIX
# ============================================================

demand_arr = np.vstack(
    [
        code_trace,
        chat_trace,
        non_interactive_trace,
    ]
).T


# ============================================================
# NORMALIZE DEMAND
# ============================================================

row_sum = (

    demand_arr

    .sum(
        axis=1,
        keepdims=True,
    )

)


# ------------------------------------------------------------
# Safety check:
# avoid division by zero if all workloads were zero.
# ------------------------------------------------------------

if np.any(
    row_sum == 0
):

    raise RuntimeError(
        "Found a time slot with zero total demand "
        "across all workloads."
    )


demand_norm_arr = (

    demand_arr

    /

    row_sum

)


# ============================================================
# VERIFY NORMALIZATION
# ============================================================

normalization_check = (

    demand_norm_arr

    .sum(
        axis=1
    )

)


if not np.allclose(
    normalization_check,
    1.0,
):

    raise RuntimeError(
        "Normalized workload vectors do not sum to 1."
    )


# ============================================================
# SAVE DATAFRAMES
# ============================================================

demand_df = pd.DataFrame(

    demand_arr,

    columns=WORKLOAD_NAMES,

)


demand_norm_df = pd.DataFrame(

    demand_norm_arr,

    columns=WORKLOAD_NAMES,

)


# ------------------------------------------------------------
# Each time_slot is now one minute
# ------------------------------------------------------------

demand_df.insert(

    0,

    "time_slot",

    np.arange(
        T
    ),

)


demand_norm_df.insert(

    0,

    "time_slot",

    np.arange(
        T
    ),

)


# ============================================================
# SAVE
#
# Use separate names so the old 10-second trace is preserved.
# ============================================================

RAW_OUTPUT_FILE = os.path.join(

    RESULT_DIR,

    "mroo_demand_trace_raw_weekday_aligned_1min.csv",

)


NORM_OUTPUT_FILE = os.path.join(

    RESULT_DIR,

    "mroo_demand_trace_norm_weekday_aligned_1min.csv",

)


demand_df.to_csv(

    RAW_OUTPUT_FILE,

    index=False,

)


demand_norm_df.to_csv(

    NORM_OUTPUT_FILE,

    index=False,

)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "1-MINUTE DEMAND TRACE CREATED"
)

print(
    "========================================"
)


print(
    "\nRaw trace:"
)

print(
    RAW_OUTPUT_FILE
)


print(
    "\nNormalized trace:"
)

print(
    NORM_OUTPUT_FILE
)


print(
    "\nFirst rows:"
)

print(
    demand_norm_df.head()
)


print(
    "\nShape:",
    demand_norm_arr.shape,
)


print(
    "\nSlots per day:",
    24 * 60,
)


print(
    "Slots per week:",
    7 * 24 * 60,
)


print(
    "24-hour experiment window size:",
    24 * 60,
)