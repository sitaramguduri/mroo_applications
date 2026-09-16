import sys
import os
import numpy as np
import pandas as pd

from mroo import mroo_step
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
from weights import get_w_t
from cost import hitting_cost, long_term_cost


# --------------------------------------------------
# Load SAME validation/test split used by all algorithms
# --------------------------------------------------

validation_data = pd.read_csv(
    "../azure_validation.csv"
)

test_data = pd.read_csv(
    "../azure_test.csv"
)

validation_demand = validation_data[
    ["y_code", "y_conv", "y_noninteractive"]
].to_numpy()

test_demand = test_data[
    ["y_code", "y_conv", "y_noninteractive"]
].to_numpy()
# --------------------------------------------------
# Run one MROO configuration
# --------------------------------------------------

def run_mroo_on_trace(
    demand,
    eta,
    kappa_init
):
    u_prev = np.array([
        1/3,
        1/3,
        1/3
    ], dtype=float)

    kappa_t = np.asarray(
        kappa_init,
        dtype=float
    ).copy()

    hitting_history = []
    memory_history = []

    for t in range(len(demand)):

        y_t = demand[t]

        w_t = get_w_t(t, y_t)

        result = mroo_step(
            y_t=y_t,
            u_prev=u_prev,
            w_t=w_t,
            kappa_t=kappa_t,
            eta=eta
        )

        u_t = result["u_t"]
        d_t = result["d_t"]

        # actual hitting cost
        f_t = hitting_cost(
            u_t,
            y_t
        )

        hitting_history.append(f_t)
        memory_history.append(d_t)

        # update state
        u_prev = u_t
        kappa_t = result["kappa_next"]

    avg_hitting_cost = np.mean(
        hitting_history
    )

    q_cost = long_term_cost(
        memory_history
    )

    total_cost = (
        avg_hitting_cost +
        q_cost
    )

    return {
        "avg_hitting_cost": avg_hitting_cost,
        "long_term_cost": q_cost,
        "total_cost": total_cost,
        "final_kappa": kappa_t
    }


# --------------------------------------------------
# Hyperparameter grid
# --------------------------------------------------

eta_values = [
    1e-6
]

kappa_values = [
    np.array([0.05, 0.05, 0.05])
]


# --------------------------------------------------
# Tune ONLY on validation data
# --------------------------------------------------

validation_results = []

for eta in eta_values:

    for kappa_init in kappa_values:

        print(
            f"Validation run: "
            f"eta={eta}, "
            f"kappa_init={kappa_init}"
        )

        result = run_mroo_on_trace(
            demand=validation_demand,
            eta=eta,
            kappa_init=kappa_init
        )

        validation_results.append({
            "eta": eta,
            "kappa_init": kappa_init[0],
            "avg_hitting_cost":
                result["avg_hitting_cost"],
            "long_term_cost":
                result["long_term_cost"],
            "total_cost":
                result["total_cost"]
        })


validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df.to_csv(
    "mroo_validation_results.csv",
    index=False
)

print("\nValidation results:")
print(validation_results_df)


# --------------------------------------------------
# Select best MROO configuration
# --------------------------------------------------

best_idx = (
    validation_results_df["total_cost"]
    .idxmin()
)

best_row = validation_results_df.loc[
    best_idx
]

best_eta = best_row["eta"]
best_kappa_value = best_row["kappa_init"]

best_kappa = np.array([
    best_kappa_value,
    best_kappa_value,
    best_kappa_value
])

print("\nBest validation configuration:")
print(best_row)


# --------------------------------------------------
# FINAL evaluation on test data
# --------------------------------------------------

test_result = run_mroo_on_trace(
    demand=test_demand,
    eta=best_eta,
    kappa_init=best_kappa
)

print("\nFinal MROO test result:")
print("eta =", best_eta)
print("kappa_1 =", best_kappa)

print(
    "Average hitting cost =",
    test_result["avg_hitting_cost"]
)

print(
    "Long-term cost =",
    test_result["long_term_cost"]
)

print(
    "Total cost =",
    test_result["total_cost"]
)

# --------------------------------------------------
# Save final MROO result
# --------------------------------------------------

mroo_result = pd.DataFrame([{
    "algorithm": "MROO",
    "eta": best_eta,
    "kappa_init": best_kappa[0],
    "avg_hitting_cost": test_result["avg_hitting_cost"],
    "long_term_cost": test_result["long_term_cost"],
    "total_cost": test_result["total_cost"]
}])

mroo_result.to_csv(
    "mroo_best_result.csv",
    index=False
)

print("\nSaved final MROO result:")
print(mroo_result)