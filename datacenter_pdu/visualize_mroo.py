from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# Load tuning results
# --------------------------------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent

TUNING_FILE = (
    PROJECT_DIRECTORY
    / "results"
    / "mroo_tuning"
    / "tuning_grid.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "results"
    / "mroo_tuning"
    / "figures"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

tuning = pd.read_csv(TUNING_FILE)

required_columns = [
    "eta",
    "kappa_initial",
    "average_hitting_cost",
    "long_term_cost",
    "total_objective",
]

missing_columns = [
    column
    for column in required_columns
    if column not in tuning.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# --------------------------------------------------
# Find best configuration
# --------------------------------------------------

best_index = tuning["total_objective"].idxmin()
best = tuning.loc[best_index]

best_eta = float(best["eta"])
best_kappa = float(best["kappa_initial"])

print("Best configuration")
print("------------------")
print(f"eta = {best_eta:g}")
print(f"kappa_1 = {best_kappa:g}")
print(
    f"Hitting cost = "
    f"{best['average_hitting_cost']:.8f}"
)
print(
    f"Long-term cost = "
    f"{best['long_term_cost']:.8f}"
)
print(
    f"Total objective = "
    f"{best['total_objective']:.8f}"
)


# --------------------------------------------------
# Prepare values
# --------------------------------------------------

eta_values = np.sort(
    tuning["eta"].unique()
)

kappa_values = np.sort(
    tuning["kappa_initial"].unique()
)

objective_matrix = (
    tuning.pivot(
        index="kappa_initial",
        columns="eta",
        values="total_objective",
    )
    .reindex(
        index=kappa_values,
        columns=eta_values,
    )
)

ordered = (
    tuning.sort_values(
        ["kappa_initial", "eta"]
    )
    .reset_index(drop=True)
)

ordered["configuration"] = [
    (
        rf"$\eta={eta:.0e}$"
        + "\n"
        + rf"$\kappa_1={kappa:g}$"
    )
    for eta, kappa in zip(
        ordered["eta"],
        ordered["kappa_initial"],
    )
]


# ==================================================
# Create one figure containing three plots
# ==================================================

heatmap_figure, heatmap_axis = plt.subplots(
    figsize=(10, 7)
)

line_figure, line_axis = plt.subplots(
    figsize=(11, 7)
)

bar_figure, bar_axis = plt.subplots(
    figsize=(15, 7)
)

# ==================================================
# Plot 1: Total-objective heatmap
# ==================================================

image = heatmap_axis.imshow(
    objective_matrix.to_numpy(),
    aspect="auto",
    origin="lower",
    cmap="viridis_r",
)

heatmap_axis.set_xticks(
    np.arange(len(eta_values))
)

heatmap_axis.set_xticklabels(
    [f"{eta:.0e}" for eta in eta_values]
)

heatmap_axis.set_yticks(
    np.arange(len(kappa_values))
)

heatmap_axis.set_yticklabels(
    [f"{kappa:g}" for kappa in kappa_values]
)

heatmap_axis.set_xlabel(
    r"Learning rate $\eta$"
)

heatmap_axis.set_ylabel(
    r"Initial dual value $\kappa_1$"
)

heatmap_axis.set_title(
    "A. Total-objective heatmap",
    fontweight="bold",
)

colorbar = heatmap_figure.colorbar(
    image,
    ax=heatmap_axis,
    fraction=0.046,
    pad=0.04,
)

colorbar.set_label("Total objective")

median_value = np.nanmedian(
    objective_matrix.to_numpy()
)

for row, kappa in enumerate(kappa_values):
    for column, eta in enumerate(eta_values):
        value = objective_matrix.loc[
            kappa,
            eta
        ]

        heatmap_axis.text(
            column,
            row,
            f"{value:.6f}",
            ha="center",
            va="center",
            fontsize=8,
            color=(
                "white"
                if value > median_value
                else "black"
            ),
        )

best_eta_position = np.where(
    np.isclose(eta_values, best_eta)
)[0][0]

best_kappa_position = np.where(
    np.isclose(kappa_values, best_kappa)
)[0][0]

heatmap_axis.scatter(
    best_eta_position,
    best_kappa_position,
    marker="*",
    s=350,
    color="red",
    edgecolor="white",
    linewidth=1.5,
    label="Best",
)

heatmap_axis.legend(loc="best")


# ==================================================
# Plot 2: Total objective versus kappa_1
# ==================================================

colors = plt.cm.viridis(
    np.linspace(0.1, 0.9, len(eta_values))
)

for color, eta in zip(colors, eta_values):
    eta_results = (
        tuning[
            np.isclose(tuning["eta"], eta)
        ]
        .sort_values("kappa_initial")
    )

    line_axis.plot(
        eta_results["kappa_initial"],
        eta_results["total_objective"],
        color=color,
        marker="o",
        linewidth=2,
        markersize=6,
        label=rf"$\eta={eta:.0e}$",
    )

line_axis.scatter(
    best_kappa,
    best["total_objective"],
    marker="*",
    s=280,
    color="red",
    edgecolor="black",
    linewidth=1,
    zorder=10,
    label="Selected configuration",
)

line_axis.set_xlabel(
    r"Initial dual value $\kappa_1$"
)

line_axis.set_ylabel("Total objective")

line_axis.set_title(
    r"B. Total objective versus $\kappa_1$ for each $\eta$",
    fontweight="bold",
)

line_axis.grid(alpha=0.25)
line_axis.legend(
    title="Learning rate",
    fontsize=9,
)


# ==================================================
# Plot 3: Hitting versus long-term cost
# ==================================================

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
    rotation=50,
    ha="right",
    fontsize=8,
)

bar_axis.set_xlabel(
    "Parameter configuration"
)

bar_axis.set_ylabel("Cost")

bar_axis.set_title(
    "C. Hitting cost versus long-term cost",
    fontweight="bold",
)

bar_axis.grid(
    axis="y",
    alpha=0.25,
)

bar_axis.legend()


# Highlight the best configuration.
best_position = ordered.index[
    np.isclose(ordered["eta"], best_eta)
    & np.isclose(
        ordered["kappa_initial"],
        best_kappa,
    )
][0]

bar_axis.axvspan(
    best_position - 0.5,
    best_position + 0.5,
    color="red",
    alpha=0.10,
)

bar_axis.text(
    best_position,
    max(
        ordered.loc[
            best_position,
            "average_hitting_cost",
        ],
        ordered.loc[
            best_position,
            "long_term_cost",
        ],
    ),
    "Selected",
    ha="center",
    va="bottom",
    color="red",
    fontweight="bold",
)


# --------------------------------------------------
# Save complete figure
# --------------------------------------------------

# --------------------------------------------------
# Save heatmap
# --------------------------------------------------

heatmap_figure.tight_layout()

heatmap_file = (
    OUTPUT_DIRECTORY
    / "mroo_tuning_heatmap.png"
)

heatmap_figure.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight",
)


# --------------------------------------------------
# Save line plot
# --------------------------------------------------

line_figure.tight_layout()

line_file = (
    OUTPUT_DIRECTORY
    / "mroo_total_objective_curves.png"
)

line_figure.savefig(
    line_file,
    dpi=300,
    bbox_inches="tight",
)

# --------------------------------------------------
# Save bar graph
# --------------------------------------------------

bar_figure.tight_layout()

bar_file = (
    OUTPUT_DIRECTORY
    / "mroo_hitting_vs_long_term.png"
)

bar_figure.savefig(
    bar_file,
    dpi=300,
    bbox_inches="tight",
)


# Display the three separate figures
plt.show()

print("\nSaved:")
print(heatmap_file)
print(line_file)
print(bar_file)
