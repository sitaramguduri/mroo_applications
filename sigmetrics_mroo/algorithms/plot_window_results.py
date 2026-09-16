import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    "window_algorithm_results.csv"
)

algorithm_order = [
    "GRD",
    "ROBD-SUM",
    "ROBD-MAX",
    "MROO"
]


metrics = [
    ("hitting_cost", "Hitting Cost"),
    ("long_term_cost", "Long-term Cost"),
    ("total_cost", "Total Cost")
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


for ax, (metric, ylabel) in zip(
    axes,
    metrics
):

    plot_data = []

    for algorithm in algorithm_order:

        values = data[
            data["algorithm"] == algorithm
        ][metric].values

        plot_data.append(values)

    ax.boxplot(
        plot_data,
        tick_labels=algorithm_order,
        showfliers=True
    )

    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )


plt.tight_layout()

plt.savefig(
    "algorithm_cost_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()