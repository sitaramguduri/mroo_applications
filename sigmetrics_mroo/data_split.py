import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VALIDATION_RATIO = 0.80

INPUT_FILE = "azure_y_t.csv"
VALIDATION_FILE = "azure_validation.csv"
TEST_FILE = "azure_test.csv"


# --------------------------------------------------
# Load full trace
# --------------------------------------------------

data = pd.read_csv(INPUT_FILE)

split_idx = int(
    VALIDATION_RATIO * len(data)
)


# --------------------------------------------------
# Chronological 80/20 split
# --------------------------------------------------

validation_data = data.iloc[
    :split_idx
].copy()

test_data = data.iloc[
    split_idx:
].copy()


# --------------------------------------------------
# Save
# --------------------------------------------------

validation_data.to_csv(
    VALIDATION_FILE,
    index=False
)

test_data.to_csv(
    TEST_FILE,
    index=False
)


print("Total slots:", len(data))
print("Validation slots:", len(validation_data))
print("Test slots:", len(test_data))

print(
    "Validation ratio:",
    len(validation_data) / len(data)
)

print(
    "Test ratio:",
    len(test_data) / len(data)
)