import pandas as pd
import matplotlib.pyplot as plt

validation_results_df = pd.read_csv(
    "mroo_validation_results.csv"
)

eta_values = validation_results_df["eta"].unique()

for eta in eta_values:
    subset = validation_results_df[
        validation_results_df["eta"] == eta
    ].sort_values("kappa_init")

    plt.plot(
        subset["kappa_init"],
        subset["total_cost"],
        marker="o",
        label=f"eta={eta}"
    )

plt.xlabel("Initial kappa value")
plt.ylabel("Total objective")
plt.title("MROO Validation Tuning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()