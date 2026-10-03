import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# INPUT
# ============================================================

CSV_FILE = Path(
    # "../new_results/mroo_window_summary.csv"   # CHANGE THIS IF NEEDED
    "../algorithms/new_results/window_beta_m_sweep/rho_7/1min_24hour/fixed_weights_w1_0p2_w2_0p3_w3_0p5/beta_75_m_150_T_1440"
)

OUTPUT_DIR = Path(
    "mroo_parameter_plots"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(CSV_FILE)

# Only compare runs that finished all 100 windows.
df = df[
    df["n_windows"] == 100
].copy()

df = df.sort_values(
    ["eta", "kappa_init"]
)


print(
    df[
        [
            "eta",
            "kappa_init",
            "mean_hitting_cost",
            "mean_long_term_cost",
            "mean_total_cost",
        ]
    ]
)


# ============================================================
# HELPER: HEATMAP
# ============================================================

def make_heatmap(
    value_column,
    title,
    output_file,
):

    pivot = df.pivot_table(
        index="eta",
        columns="kappa_init",
        values=value_column,
        aggfunc="mean",
    )

    # Sort numerically
    pivot = pivot.sort_index(
        ascending=True
    )

    pivot = pivot.reindex(
        sorted(pivot.columns),
        axis=1,
    )

    values = pivot.to_numpy()

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
    )

    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(
            len(pivot.columns)
        )
    )

    ax.set_xticklabels(
        [
            f"{x:g}"
            for x in pivot.columns
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(pivot.index)
        )
    )

    ax.set_yticklabels(
        [
            f"{x:g}"
            for x in pivot.index
        ]
    )

    ax.set_xlabel(
        r"$\kappa_{\mathrm{init}}$"
    )

    ax.set_ylabel(
        r"$\eta$"
    )

    ax.set_title(
        title
    )


    # --------------------------------------------------------
    # WRITE COST INSIDE EACH CELL
    # --------------------------------------------------------

    for i in range(
        len(pivot.index)
    ):

        for j in range(
            len(pivot.columns)
        ):

            value = values[i, j]

            if np.isfinite(value):

                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )


    # --------------------------------------------------------
    # COLOR BAR
    # --------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Mean cost"
    )


    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# TOTAL COST HEATMAP
# ============================================================

make_heatmap(
    value_column="mean_total_cost",
    title=(
        r"MROO Mean Total Cost vs "
        r"$\eta$ and $\kappa_{\mathrm{init}}$"
    ),
    output_file="total_cost_heatmap.png",
)


# ============================================================
# LONG-TERM COST HEATMAP
# ============================================================

make_heatmap(
    value_column="mean_long_term_cost",
    title=(
        r"MROO Mean Long-Term Cost vs "
        r"$\eta$ and $\kappa_{\mathrm{init}}$"
    ),
    output_file="long_term_cost_heatmap.png",
)


# ============================================================
# TOTAL COST VS KAPPA
#
# One line for each eta
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for eta, group in df.groupby(
    "eta"
):

    group = group.sort_values(
        "kappa_init"
    )

    plt.plot(
        group["kappa_init"],
        group["mean_total_cost"],
        marker="o",
        label=fr"$\eta={eta:g}$",
    )


plt.xlabel(
    r"$\kappa_{\mathrm{init}}$"
)

plt.ylabel(
    "Mean Total Cost"
)

plt.title(
    r"Effect of $\kappa_{\mathrm{init}}$ on Total Cost"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "total_cost_vs_kappa.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# LONG-TERM COST VS KAPPA
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for eta, group in df.groupby(
    "eta"
):

    group = group.sort_values(
        "kappa_init"
    )

    plt.plot(
        group["kappa_init"],
        group["mean_long_term_cost"],
        marker="o",
        label=fr"$\eta={eta:g}$",
    )


plt.xlabel(
    r"$\kappa_{\mathrm{init}}$"
)

plt.ylabel(
    "Mean Long-Term Cost"
)

plt.title(
    r"Effect of $\kappa_{\mathrm{init}}$ on Long-Term Cost"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "long_term_cost_vs_kappa.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# PRINT BEST CONFIGURATION
# ============================================================

best = df.loc[
    df["mean_total_cost"].idxmin()
]

print("\n========================================")
print("BEST CONFIGURATION")
print("========================================")

print(
    "eta =",
    best["eta"]
)

print(
    "kappa_init =",
    best["kappa_init"]
)

print(
    "Mean hitting cost =",
    best["mean_hitting_cost"]
)

print(
    "Mean long-term cost =",
    best["mean_long_term_cost"]
)

print(
    "Mean total cost =",
    best["mean_total_cost"]
)