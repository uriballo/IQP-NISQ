import argparse
import logging
import re
from pathlib import Path
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---------------------------
# Global style config
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
    description="Plot scaling of best‐run metrics across node counts"
)
parser.add_argument(
    "--results-dir", type=Path, default=Path("results/"),
    help="Directory containing analysis_summary_{N}N.csv files"
)
parser.add_argument(
    "--out-dir", type=Path, default=Path("plots/scaling"),
    help="Directory to save scaling plots"
)
args = parser.parse_args()

# ---------------------------
# Prepare output dirs
# ---------------------------
(args.out_dir / "png").mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load all per-node CSVs
# ---------------------------
all_files = list(args.results_dir.glob("analysis_summary_*N.csv"))
pattern = re.compile(r"analysis_summary_(\d+)N\.csv$")
dfs = []
for f in all_files:
    m = pattern.search(f.name)
    if not m:
        continue
    tmp = pd.read_csv(f)
    tmp["num_nodes"] = int(m.group(1))
    dfs.append(tmp)
if not dfs:
    logging.error("No analysis_summary_{N}N.csv files found")
    exit(1)
df = pd.concat(dfs, ignore_index=True)

# ---------------------------
# Infer run_type & normalize ID
# ---------------------------
df["run_type"] = np.where(
    df["run_id"].str.contains("simulation_results/"), "Simulation", "Evaluation"
)
df["run_id_norm"] = (
    df["run_id"]
    .str.replace(r"^.*(?:simulation_results|evaluation_results)/", "", regex=True)
)

# ---------------------------
# Drop Simulation for 8-node
# ---------------------------
mask_8n_sim = df["run_type"] == "Simulation"
df = df[~mask_8n_sim]

# ---------------------------
# Compute density_diff for ER selection
# ---------------------------
df["density_diff"] = (df["generated_density"] - df["density_value"]).abs()

# ---------------------------
# Select best runs per group
# ---------------------------
def pick_best(group):
    if group["graph_type"].iloc[0] == "Bipartite":
        # maximize generated_bipartite
        return group.loc[group["generated_bipartite"].idxmax()]
    else:
        # ER: minimize density_diff
        return group.loc[group["density_diff"].idxmin()]

best_df = df.groupby(
    ["graph_type", "density_category", "num_nodes"], as_index=False
).apply(pick_best).reset_index(drop=True)

# ---------------------------
# Convert bipartite fractions to percentages
# ---------------------------
if "generated_bipartite" in best_df.columns:
    best_df["generated_bipartite"] *= 100
    best_df["bipartite_value"] *= 100

# ---------------------------
# Metrics to plot
# ---------------------------
mmd_metrics = [
    "mmd_density", "mmd_var_degree_dist", "mmd_tri",
    "mmd_bipartite", "mmd_samples"
]
percent_metrics = [
    "generated_bipartite",
    "memorized_adj_pct", "memorized_iso_pct",
    "diversity_adj_pct", "diversity_iso_pct"
]

# ---------------------------
# Titles & labels
# ---------------------------
title_map = {
    "mmd_density": ("Density Distribution", r"MMD"),
    "mmd_var_degree_dist": ("Variance Degree Distribution", r"MMD"),
    "mmd_tri": ("Triangle Count Distribution", r"MMD"),
    "mmd_bipartite": ("Bipartiteness Distribution", r"MMD"),
    "mmd_samples": ("Generated Samples", r"MMD"),
    "generated_bipartite": ("Generated Bipartite \%", r"Bipartite (\%)"),
    "memorized_adj_pct": ("Memorization by Adjacency Check", r"Memorization (\%)"),
    "memorized_iso_pct": ("Memorization by Isomorphism Check", r"Memorization (\%)"),
    "diversity_adj_pct": ("Diversity by Adjacency Check", r"Diversity (\%)"),
    "diversity_iso_pct": ("Diversity by Isomorphism Check", r"Diversity (\%)"),
}

# ---------------------------
# Visual encodings
# ---------------------------
colors = {"Sparse": "#1f77b4", "Medium": "#ff7f0e", "Dense": "#2ca02c"}
linestyles = {"Sparse": "-", "Medium": "--", "Dense": "-."}
markers = {"Sparse": "o", "Medium": "s", "Dense": "D"}

# ---------------------------
# Plot helper
# ---------------------------
def make_plot(subdf: pd.DataFrame, gtype: str, metric: str):
    fig, ax = plt.subplots(figsize=(7,7))
    title, ylabel = title_map.get(metric, (metric, metric))
    ax.set_title(title, pad=12)
    ax.set_xlabel(r"Nodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel(ylabel)

    for density in colors:
        dsub = subdf[subdf["density_category"] == density]
        if dsub.empty:
            continue
        vals = dsub[metric]
        ax.plot(
            dsub["num_nodes"], vals,
            marker=markers[density],
            linestyle=linestyles[density],
            color=colors[density],
            markersize=7, linewidth=2,
            label=density
        )

    if metric in mmd_metrics:
        ymax = subdf[metric].max()
        ymin = subdf[metric].min()
        ax.set_ylim(0, ymax*1.2)
    else:
        ax.set_ylim(0, 100)
    
    ax.legend(frameon=True)
    fig.tight_layout()

    name = f"{gtype}_{metric}"
    fig.savefig(args.out_dir/"png"/f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved {gtype} {metric}")

# ---------------------------
# Generate scaling plots
# ---------------------------
for gtype in ["ER", "Bipartite"]:
    sdf = best_df[best_df["graph_type"] == gtype]
    if sdf.empty:
        continue
    for metric in mmd_metrics + percent_metrics:
        if metric in sdf.columns:
            make_plot(sdf, gtype, metric)