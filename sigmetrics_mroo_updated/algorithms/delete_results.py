from pathlib import Path


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
# DMD FILES TO DELETE
# ============================================================

DMD_FILES = {
    "dmd_window_results.csv",
    "dmd_window_summary.csv",
    "dmd_window_kappa.csv",
    "dmd_window_DONE.txt",
}


# ============================================================
# DELETE DMD FILES FROM ALL beta / m / T SETTINGS
# ============================================================

deleted_files = []


for file_path in BASE_RESULT_DIR.rglob("*"):

    if (
        file_path.is_file()
        and file_path.name in DMD_FILES
    ):

        print(
            "Deleting:",
            file_path
        )

        file_path.unlink()

        deleted_files.append(
            file_path
        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "DMD CLEANUP COMPLETE"
)

print(
    "========================================"
)

print(
    "Base directory:",
    BASE_RESULT_DIR
)

print(
    "Files deleted:",
    len(deleted_files)
)


if len(deleted_files) == 0:

    print(
        "\nNo DMD files were found."
    )