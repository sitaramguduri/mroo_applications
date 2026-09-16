from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Files
# --------------------------------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent
INPUT_DIRECTORY = PROJECT_DIRECTORY / "cell_a_powerdata"

OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "processed_powerdata"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIRECTORY / "cell_a_power_trace.csv"

PDU_IDS = [6, 7, 8, 9, 10]


# --------------------------------------------------
# Load and clean each PDU trace
# --------------------------------------------------

traces = []
input_counts = {}
bad_quality_counts = {}

for pdu_id in PDU_IDS:
    name = f"pdu{pdu_id}"

    input_file = INPUT_DIRECTORY / f"cella_{name}.csv.gz"

    if not input_file.exists():
        input_file = INPUT_DIRECTORY / f"cella_{name}.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input: {input_file}")

    # Read only the columns we need.
    data = pd.read_csv(
        input_file,
        usecols=[
            "time",
            "measured_power_util",
            "bad_measurement_data"
        ],
        dtype={"time": "int64"}
    )

    original_count = len(data)

    # --------------------------------------------------
    # Remove bad-quality rows
    # --------------------------------------------------

    # IMPORTANT:
    # This assumes bad_measurement_data contains the
    # literal string "bad" for bad rows.
    data = data[
        data["bad_measurement_data"] != "bad"
    ].copy()

    bad_quality_counts[name] = original_count - len(data)

    # Quality flag is no longer needed.
    data = data.drop(columns=["bad_measurement_data"])

    # --------------------------------------------------
    # Validate measured power utilization
    # --------------------------------------------------

    data["measured_power_util"] = pd.to_numeric(
        data["measured_power_util"],
        errors="raise"
    )

    if not np.isfinite(data["measured_power_util"]).all():
        raise ValueError(
            f"{name}: missing or nonfinite utilization"
        )

    # Remove exact duplicate rows.
    data = data.drop_duplicates()

    # Do not silently combine different values
    # occurring at the same timestamp.
    if data["time"].duplicated().any():
        raise ValueError(
            f"{name}: multiple utilization values "
            "at the same timestamp"
        )

    # Rename measured power column for this PDU.
    data = data.rename(
        columns={
            "measured_power_util": f"y_{name}"
        }
    )

    # Use time as the index so the PDUs can
    # later be aligned by timestamp.
    data = data.set_index("time")

    input_counts[name] = len(data)

    traces.append(data)


# --------------------------------------------------
# Align all five PDUs by timestamp
# --------------------------------------------------

# Inner join means:
# keep a timestamp only when it exists in ALL five PDUs.
aligned = (
    pd.concat(
        traces,
        axis=1,
        join="inner"
    )
    .sort_index()
    .reset_index()
)

if aligned.empty:
    raise ValueError(
        "No timestamps are shared by all five PDUs"
    )


# --------------------------------------------------
# Save processed trace
# --------------------------------------------------

aligned.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("\nAligned timestamps:", len(aligned))

for name, count in input_counts.items():
    print(
        f"{name}: "
        f"{bad_quality_counts[name]} bad-quality rows removed, "
        f"{count} valid timestamps, "
        f"{count - len(aligned)} excluded by alignment"
    )


# --------------------------------------------------
# Check five-minute spacing
# --------------------------------------------------

# Google timestamps are in microseconds.
expected_interval = 5 * 60 * 1_000_000

time_differences = (
    aligned["time"]
    .diff()
    .dropna()
)

gap_count = int(
    (time_differences != expected_interval).sum()
)

print(
    "\nGaps between consecutive five-minute records:",
    gap_count
)

print("\nPreview:")
print(aligned.head())

print("\nSaved:", OUTPUT_FILE)