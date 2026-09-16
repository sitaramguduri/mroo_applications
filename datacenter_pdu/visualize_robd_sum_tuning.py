"""Visualize lambda_1/lambda_2 tuning results for ROBD-SUM."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent
TUNING_FILE = (
    PROJECT_DIRECTORY / "results" / "robd_sum_tuning" / "tuning_grid.csv"
)
OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY / "results" / "robd_sum_tuning" / "figures"
)
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load and validate results
# --------------------------------------------------

tuning = pd.read_csv(TUNING_FILE)

required_columns = [
    "lambda_1",
    "lambda_2",
    "average_hitting_cost",
    "long_term_cost",
    "total_objective",
]

missing = [column for column in required_columns if column not in tuning.columns]
if missing:
    raise ValueError(f"Tuning file is missing columns: {missing}")

best = tuning.loc[tuning["total_objective"].idxmin()]
best_lambda_1 = float(best["lambda_1"])
best_lambda_2 = float(best["lambda_2"])

lambda_1_values = np.sort(tuning["lambda_1"].unique())
lambda_2_values = np.sort(tuning["lambda_2"].unique())

print("Best ROBD-SUM configuration")
print("---------------------------")
print(f"lambda_1 = {best_lambda_1:g}")
print(f"lambda_2 = {best_lambda_2:g}")
print(f"Hitting cost = {best['average_hitting_cost']:.8f}")
print(f"Long-term cost = {best['long_term_cost']:.8f}")
print(f"Total objective = {best['total_objective']:.8f}")


# ==================================================
# Plot 1: Total-objective heatmap
# ==================================================

objective_matrix = (
    tuning.pivot(
        index="lambda_2",
        columns="lambda_1",
        values="total_objective",
    )
    .reindex(index=lambda_2_values, columns=lambda_1_values)
)

heatmap_figure, heatmap_axis = plt.subplots(figsize=(11, 7))

image = heatmap_axis.imshow(
    objective_matrix.to_numpy(),
    aspect="auto",
    origin="lower",
    cmap="viridis_r",
)

heatmap_axis.set_xticks(np.arange(len(lambda_1_values)))
heatmap_axis.set_xticklabels([f"{value:g}" for value in lambda_1_values])
heatmap_axis.set_yticks(np.arange(len(lambda_2_values)))
heatmap_axis.set_yticklabels([f"{value:g}" for value in lambda_2_values])
heatmap_axis.set_xlabel(r"Switching weight $\lambda_1$")
heatmap_axis.set_ylabel(r"Reference weight $\lambda_2$")
heatmap_axis.set_title("ROBD-SUM total-objective heatmap", fontweight="bold")

colorbar = heatmap_figure.colorbar(image, ax=heatmap_axis)
colorbar.set_label("Total objective")

median_objective = np.nanmedian(objective_matrix.to_numpy())
for row, lambda_2 in enumerate(lambda_2_values):
    for column, lambda_1 in enumerate(lambda_1_values):
        value = objective_matrix.loc[lambda_2, lambda_1]
        heatmap_axis.text(
            column,
            row,
            f"{value:.6f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if value > median_objective else "black",
        )

best_x = np.where(np.isclose(lambda_1_values, best_lambda_1))[0][0]
best_y = np.where(np.isclose(lambda_2_values, best_lambda_2))[0][0]

heatmap_axis.scatter(
    best_x,
    best_y,
    marker="*",
    s=350,
    color="red",
    edgecolor="white",
    linewidth=1.5,
    label="Selected configuration",
)
heatmap_axis.legend(loc="best")
heatmap_figure.tight_layout()

heatmap_file = OUTPUT_DIRECTORY / "robd_sum_tuning_heatmap.png"
heatmap_figure.savefig(heatmap_file, dpi=300, bbox_inches="tight")


# ==================================================
# Plot 2: Total objective versus lambda_2
# One curve for each lambda_1
# ==================================================

line_figure, line_axis = plt.subplots(figsize=(11, 7))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lambda_1_values)))

for color, lambda_1 in zip(colors, lambda_1_values):
    subset = (
        tuning[np.isclose(tuning["lambda_1"], lambda_1)]
        .sort_values("lambda_2")
    )

    line_axis.plot(
        subset["lambda_2"],
        subset["total_objective"],
        color=color,
        marker="o",
        linewidth=2,
        markersize=6,
        label=rf"$\lambda_1={lambda_1:g}$",
    )

line_axis.scatter(
    best_lambda_2,
    best["total_objective"],
    marker="*",
    s=280,
    color="red",
    edgecolor="black",
    linewidth=1,
    zorder=10,
    label="Selected configuration",
)

line_axis.set_xlabel(r"Reference weight $\lambda_2$")
line_axis.set_ylabel("Total objective")
line_axis.set_title(
    r"ROBD-SUM total objective versus $\lambda_2$ for each $\lambda_1$",
    fontweight="bold",
)
line_axis.grid(alpha=0.25)
line_axis.legend(title=r"Switching weight $\lambda_1$", fontsize=8)
line_figure.tight_layout()

line_file = OUTPUT_DIRECTORY / "robd_sum_total_objective_curves.png"
line_figure.savefig(line_file, dpi=300, bbox_inches="tight")


# ==================================================
# Plot 3: Hitting cost versus long-term cost
# ==================================================

ordered = (
    tuning.sort_values(["lambda_2", "lambda_1"])
    .reset_index(drop=True)
)

ordered["configuration"] = [
    rf"$\lambda_1={lambda_1:g}$" + "\n" + rf"$\lambda_2={lambda_2:g}$"
    for lambda_1, lambda_2 in zip(ordered["lambda_1"], ordered["lambda_2"])
]

bar_figure, bar_axis = plt.subplots(figsize=(16, 7))
x_positions = np.arange(len(ordered))
bar_width = 0.38

bar_axis.bar(
    x_positions - bar_width / 2,
    ordered["average_hitting_cost"],
    width=bar_width,
    color="#4C78A8",
    label="Average hitting cost",
)

bar_axis.bar(
    x_positions + bar_width / 2,
    ordered["long_term_cost"],
    width=bar_width,
    color="#F58518",
    label="Long-term cost",
)

bar_axis.set_xticks(x_positions)
bar_axis.set_xticklabels(
    ordered["configuration"],
    rotation=55,
    ha="right",
    fontsize=7,
)
bar_axis.set_xlabel("ROBD-SUM parameter configuration")
bar_axis.set_ylabel("Cost")
bar_axis.set_title(
    "ROBD-SUM hitting cost versus long-term cost",
    fontweight="bold",
)
bar_axis.grid(axis="y", alpha=0.25)
bar_axis.legend()

selected_positions = ordered.index[
    np.isclose(ordered["lambda_1"], best_lambda_1)
    & np.isclose(ordered["lambda_2"], best_lambda_2)
]

if len(selected_positions) != 1:
    raise ValueError("Could not uniquely locate the selected configuration")

selected_position = int(selected_positions[0])
bar_axis.axvspan(
    selected_position - 0.5,
    selected_position + 0.5,
    color="red",
    alpha=0.10,
)
bar_axis.text(
    selected_position,
    max(
        ordered.loc[selected_position, "average_hitting_cost"],
        ordered.loc[selected_position, "long_term_cost"],
    ),
    "Selected",
    ha="center",
    va="bottom",
    color="red",
    fontweight="bold",
)

bar_figure.tight_layout()
bar_file = OUTPUT_DIRECTORY / "robd_sum_hitting_vs_long_term.png"
bar_figure.savefig(bar_file, dpi=300, bbox_inches="tight")


plt.show()

print("\nSaved figures:")
print(heatmap_file)
print(line_file)
print(bar_file)
