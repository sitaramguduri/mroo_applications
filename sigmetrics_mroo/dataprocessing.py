import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load Azure traces
# --------------------------------------------------
code = pd.read_csv("AzureLLMInferenceTrace_code_1week.csv")
conv = pd.read_csv("AzureLLMInferenceTrace_conv_1week.csv")

print("Code columns:", code.columns.tolist())
print("Conversation columns:", conv.columns.tolist())

# Expected columns:
# TIMESTAMP
# ContextTokens
# GeneratedTokens


# --------------------------------------------------
# 2. Convert timestamp and calculate token demand
# --------------------------------------------------
import pandas as pd

def aggregate_10_seconds(filename, output_name):
    df = pd.read_csv(filename)

    # Parse timestamps: Azure timestamps have varying fractional-second precision
    df["TIMESTAMP"] = pd.to_datetime(
        df["TIMESTAMP"],
        format="mixed",
        utc=True
    )

    # Demand of each individual request
    df["token_demand"] = (
        df["ContextTokens"] +
        df["GeneratedTokens"]
    )

    # Assign each request to a 10-second interval
    df["time_slot"] = df["TIMESTAMP"].dt.floor("10s")

    # Sum token demand in each interval
    result = (
        df.groupby("time_slot")["token_demand"]
        .sum()
        .reset_index()
        .rename(columns={"token_demand": output_name})
    )

    return result

code = aggregate_10_seconds(
    "AzureLLMInferenceTrace_code_1week.csv",
    "code_raw"
)

conv = aggregate_10_seconds(
    "AzureLLMInferenceTrace_conv_1week.csv",
    "conv_raw"
)

print(code.head())
print(conv.head())


# --------------------------------------------------
# 3. Align the two traces by weekday + time of day
# --------------------------------------------------
# The paper says the code and conversation traces cover
# different 7-day periods, so they align them by
# weekday and time of day.

def add_week_position(df):
    df = df.copy()

    weekday = df["time_slot"].dt.dayofweek       # Monday = 0
    seconds = (
        df["time_slot"].dt.hour * 3600 +
        df["time_slot"].dt.minute * 60 +
        df["time_slot"].dt.second
    )

    # unique 10-second position within a week
    df["week_slot"] = (
        weekday * (24 * 60 * 60 // 10)
        + seconds // 10
    )

    return df


code_agg = add_week_position(code)
conv_agg = add_week_position(conv)

data = pd.merge(
    code_agg[["week_slot", "code_raw"]],
    conv_agg[["week_slot", "conv_raw"]],
    on="week_slot",
    how="outer"
)

# A missing slot means zero requests
data[["code_raw", "conv_raw"]] = (
    data[["code_raw", "conv_raw"]].fillna(0)
)

data = data.sort_values("week_slot").reset_index(drop=True)


# --------------------------------------------------
# 4. Create synthetic non-interactive workload
# --------------------------------------------------

# Average combined demand of code + conversation
combined_average = (
    data["code_raw"] + data["conv_raw"]
).mean()

# Base demand = 25% of that average
base_noninteractive = 0.25 * combined_average

# Gaussian noise std = 5% of base demand
noise_std = 0.05 * base_noninteractive

np.random.seed(42)  # for reproducibility

noise = np.random.normal(
    loc=0.0,
    scale=noise_std,
    size=len(data)
)

data["noninteractive_raw"] = (
    base_noninteractive + noise
)

# Clip negative values to 0
data["noninteractive_raw"] = (
    data["noninteractive_raw"]
    .clip(lower=0)
    .round()
)

# --------------------------------------------------
# 5. Calculate normalized demand y_t
# --------------------------------------------------

data["total_raw"] = (
    data["code_raw"] +
    data["conv_raw"] +
    data["noninteractive_raw"]
)

nonzero = data["total_raw"] > 0

data["y_code"] = 0.0
data["y_conv"] = 0.0
data["y_noninteractive"] = 0.0

data.loc[nonzero, "y_code"] = (
    data.loc[nonzero, "code_raw"]
    / data.loc[nonzero, "total_raw"]
)

data.loc[nonzero, "y_conv"] = (
    data.loc[nonzero, "conv_raw"]
    / data.loc[nonzero, "total_raw"]
)

data.loc[nonzero, "y_noninteractive"] = (
    data.loc[nonzero, "noninteractive_raw"]
    / data.loc[nonzero, "total_raw"]
)
print(
    "\nAverage y sum:",
    (
        data.loc[nonzero, "y_code"] +
        data.loc[nonzero, "y_conv"] +
        data.loc[nonzero, "y_noninteractive"]
    ).mean()
)
data.to_csv("azure_y_t.csv", index=False)