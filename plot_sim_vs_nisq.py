import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---------------------------
# Global style configuration
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
# Parse arguments
# ---------------------------
parser = argparse.ArgumentParser(
    description="Plot Simulation vs Evaluation for features"
)
parser.add_argument(
    "--results-csv",
    type=Path,
    default=Path("results/analysis_summary_8N.csv"),
    help="Path to a single analysis_summary_{nodes}N.csv"
)
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("plots"),
    help="Directory to save plots"
)
args = parser.parse_args()

# ---------------------------
# Load results
# ---------------------------
df = pd.read_csv(args.results_csv)

# ---------------------------
# Infer run_type from run_id path
# ---------------------------
# Simulation paths contain "simulation_results/", evaluation contain "evaluation_results/"
df["run_type"] = np.where(
    df["run_id"].str.contains("simulation_results/"), "Simulation", "Evaluation"
)

# Normalize run_id so we can match sim⇔eval pairs
df["run_id_normalized"] = (
    df["run_id"]
    .str.replace(r"^.*(?:simulation_results|evaluation_results)/", "", regex=True)
)

# ---------------------------
# Output directories
# ---------------------------
out_dir = args.out_dir
(out_dir / "png").mkdir(parents=True, exist_ok=True)

# ---------------------------
# Helper to pivot features
# ---------------------------
def pivot_feature(df, feature):
    sim_df = (
        df[df["run_type"] == "Simulation"]
        [["run_id_normalized", "density_category", "graph_type", feature]]
        .rename(columns={feature: "sim_value"})
    )
    eval_df = (
        df[df["run_type"] == "Evaluation"]
        [["run_id_normalized", "graph_type", feature]]
        .rename(columns={feature: "eval_value"})
    )
    merged = pd.merge(sim_df, eval_df,
                      on=["run_id_normalized", "graph_type"])
    merged["sim_value"] = pd.to_numeric(merged["sim_value"], errors="coerce")
    merged["eval_value"] = pd.to_numeric(merged["eval_value"], errors="coerce")
    return merged.dropna(subset=["sim_value", "eval_value"])

# ---------------------------
# Features to plot
# ---------------------------
features = [
    "generated_density",
    "generated_var_degree_dist",
    "generated_tri",
    "generated_bipartite",
    "mmd_density",
    "mmd_var_degree_dist",
    "mmd_tri",
    "mmd_bipartite",
    "mmd_samples",
    "memorized_adj_pct",
    "memorized_iso_pct",
    "diversity_adj_pct",
    "diversity_iso_pct",
]

# ---------------------------
# Feature titles
# ---------------------------
title_map = {
    "generated_density": "Avg. Density",
    "generated_var_degree_dist": "Avg. Variance Deg. Dist.",
    "generated_tri": "Avg. Triangle Count",
    "generated_bipartite": "Bipartite Accuracy",
    "mmd_density": "Density",
    "mmd_var_degree_dist": "Variance Deg. Dist.",
    "mmd_tri": "Triangle Count",
    "mmd_bipartite": "Bipartiteness",
    "mmd_samples": "Samples",
    "memorized_adj_pct": "Memorization \% by Adjacency Check",
    "memorized_iso_pct": "Memorization",
    "diversity_adj_pct": "Diversity \% by Adjacency Check",
    "diversity_iso_pct": "Diversity",
}

# ---------------------------
# Visual encodings
# ---------------------------
colors = {"Sparse": "#1f77b4", "Medium": "#ff7f0e", "Dense": "#2ca02c"}
shapes = {"ER": "o", "Bipartite": "s"}

# ---------------------------
# Generate plots
# ---------------------------
for feature in features:
    merged = pivot_feature(df, feature)
    if merged.empty:
        print(f"⚠️ No matching sim/eval pairs for {feature}")
        continue

    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter points by density_category & graph_type
    for density in colors:
        for gtype in shapes:
            sub = merged[
                (merged["density_category"] == density)
                & (merged["graph_type"] == gtype)
            ]
            if sub.empty:
                continue
            ax.scatter(
                sub["sim_value"],
                sub["eval_value"],
                marker=shapes[gtype],
                color=colors[density],
                s=100,
                edgecolor="k",
                linewidth=0.6,
                alpha=0.9,
                zorder=3,
            )
    if feature in ["generated_density", "generated_var_degree_dist", 
    "generated_tri","generated_bipartite",
            "memorized_adj_pct",
    "memorized_iso_pct",
    "diversity_adj_pct",
    "diversity_iso_pct"]:
        low, high = min(merged["sim_value"].min(), merged["eval_value"].min()), max(merged["sim_value"].max(), merged["eval_value"].max())
        ax.plot([low, high], [low, high], "--", color="brown",
            linewidth=2, label="Ideal", zorder=2)
    else:
        low, high = 0, max(merged["sim_value"].max(), merged["eval_value"].max())
        ax.plot([low, high], [low, high], "--", color="brown",
                linewidth=2, label="Ideal", zorder=2)

    # Labels & title
    ax.set_xlabel(r"Simulation")
    ax.set_ylabel(r"NISQ")
    title = title_map.get(feature, feature.replace("_", " ").title())
    ax.set_title(title, pad=12)

    # Legend handles
    density_handles = [
        Line2D([0], [0],
               marker="o", color="w",
               markerfacecolor=colors[d],
               markersize=9, label=d)
        for d in colors
    ]
    gtype_handles = [
        Line2D([0], [0],
               marker=shapes[g], color="k",
               markerfacecolor="none",
               markeredgecolor="k",
               markersize=9, linestyle="None", label=g)
        for g in shapes
    ]
    ideal_handle = Line2D([0], [0],
                          linestyle="--", color="brown",
                          linewidth=2, label="Ideal")

    ax.legend(
        handles=density_handles + gtype_handles + [ideal_handle],
        frameon=True,
    )

    fig.tight_layout()
    # Save
    fname = feature
    fig.savefig(out_dir / "png" / f"sim_vs_eval_{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plots for {feature}")