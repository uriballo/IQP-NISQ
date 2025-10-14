import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
from matplotlib.ticker import MaxNLocator

# ---------------------------
# Global style config (same as in your sim_vs_eval script)
# ---------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 28,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.fontsize": 24,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.loc": "best",
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


# ---------------------------
# File paths
# ---------------------------
INPUT_CSV = Path("results/best_nisq.csv")
OUT_DIR = Path("plots/scaling")
(OUT_DIR / "png").mkdir(parents=True, exist_ok=True)

# ---------------------------
# Colors/linestyles
# ---------------------------
colors = {"Sparse": "#1f77b4", "Medium": "#ff7f0e", "Dense": "#2ca02c"}
linestyles = {"Sparse": "-", "Medium": "--", "Dense": "-."}
markers = {"Sparse": "o", "Medium": "s", "Dense": "D"}

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(INPUT_CSV)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# ---------------------------
# Metrics
# ---------------------------
diff_metrics = [
    "density_diff",
    "var_degree_dist_diff",
    "tri_diff",
    "bipartite_diff",
]
percent_metrics = [
    "memorized_adj_pct",
    "memorized_iso_pct",
    "diversity_adj_pct",
    "diversity_iso_pct",
    "generated_bipartite",
]

# ---------------------------
# Custom titles and labels
# ---------------------------
custom_labels = {
    "density_diff": {
        "title": r"{Density}",
        "ylabel": r"{Difference}",
    },
    "var_degree_dist_diff": {
        "title": r"{Degree Distribution Variance}",
        "ylabel": r"{Difference}",
    },
    "tri_diff": {
        "title": r"{Triangle Count}",
        "ylabel": r"{Difference}",
    },
    "bipartite_diff": {
        "title": r"{Bipartite Generation Accuracy}",
        "ylabel": r"{Difference}",
    },
    "memorized_adj_pct": {
        "title": r"{Memorization by Adjacency}",
        "ylabel": r"{Memorization (\%) – Adjacency}",
    },
    "memorized_iso_pct": {
        "title": r"{Memorization}",
        "ylabel": r"{Recall (\%)}",
    },
    "diversity_adj_pct": {
        "title": r"{Diversity by Adjacency}",
        "ylabel": r"{Diversity (\%) – Adjacency}",
    },
    "diversity_iso_pct": {
        "title": r"{Diversity}",
        "ylabel": r"{Uniqueness (\%)}",
    },
    "generated_bipartite": {
        "title": r"{Bipartite Generation}",
        "ylabel": r"{Accuracy (\%)}",
    },
}

# ---------------------------
# Plot helper
# ---------------------------
def make_plot(sub, gtype, metric, abs_error=False):
    fig, ax = plt.subplots(figsize=(7, 7))  # consistent with sim_vs_eval

    # Use manual labels
    label_cfg = custom_labels.get(metric, {})
    title = label_cfg.get("title", metric)
    ylabel = label_cfg.get("ylabel", metric)
    #%if abs_error:  %    ylabel += " (Absolute)"

    # plot lines
    for density in ["Sparse", "Medium", "Dense"]:
        dsub = sub[sub["density_category"] == density]
        if dsub.empty:
            continue
        ax.plot(
            dsub["num_nodes"],
            dsub[metric].abs() if abs_error else dsub[metric],
            marker=markers[density],
            linestyle=linestyles[density],
            color=colors[density],
            markersize=7,
            linewidth=2,
            label=density,
        )

    ax.set_xlabel(r"Nodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Baselines and axis scaling
    if metric in diff_metrics:
        if abs_error:
            ax.set_ylim(0, 100 if metric == "bipartite_diff" else None)
        else:
            ylim = np.nanmax(np.abs(sub[metric]))
            ax.set_ylim(-ylim *1.2, ylim * 1.2)
            ax.axhline(0, color="k", linestyle=":", linewidth=1)
    elif metric == "generated_bipartite":
        ax.set_ylim(0, 100)
        ax.axhline(100, color="k", linestyle=":", linewidth=1)
    elif "diversity" in metric or "memorized" in metric:
        ax.set_ylim(0, 100)
        if "memorized" in metric:
            ax.axhline(0, color="k", linestyle=":", linewidth=1)
        else:
            ax.axhline(100, color="k", linestyle=":", linewidth=1)
    
    ax.legend(frameon=True)
    fig.tight_layout()

    # Save
    fname = f"{gtype}_{metric}{'_abs' if abs_error else ''}"
    fig.savefig(OUT_DIR / "png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved {fname}")


# ---------------------------
# Generate plots
# ---------------------------
for gtype in ["ER", "Bipartite"]:
    sub = df[df["graph_type"] == gtype]
    if sub.empty:
        continue

    # diffs (signed and absolute)
    for m in diff_metrics:
        make_plot(sub, gtype, m, abs_error=False)
        make_plot(sub, gtype, m, abs_error=True)

    # percent metrics
    for m in percent_metrics:
        make_plot(sub, gtype, m, abs_error=False)