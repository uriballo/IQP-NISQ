import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import re
import itertools

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
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",

})

# ---------------------------
# Function to extract run_id from path
# ---------------------------
def extract_run_id(path: str) -> str:
    """
    Extract run_id from path of format .../run_<run_id>/...
    """
    match = re.search(r'/run_([^/]+)/', path)
    if match:
        return match.group(1)
    return None

# ---------------------------
# Function to merge NISQ and Sim CSVs
# ---------------------------
def load_and_merge(nisq_csv, sim_csv):
    nisq_df = pd.read_csv(nisq_csv)
    sim_df = pd.read_csv(sim_csv)

    # Extract run_id
    nisq_df['run_id'] = nisq_df['path'].apply(extract_run_id)
    sim_df['run_id'] = sim_df['path'].apply(extract_run_id)

    # Merge on dataset_name and run_id
    merged_df = pd.merge(
        nisq_df,
        sim_df,
        on=['dataset_name', 'run_id'],
        suffixes=('_nisq', '_sim')
    )

    # Drop any unmatched rows
    merged_df = merged_df.dropna(subset=['path_nisq', 'path_sim'])

    return merged_df

def clean_dataset_name(name: str) -> str:
    """
    Converts '8N_Name_Density' -> 'Name-Density'
    """
    # Remove the initial '8N_'
    name = re.sub(r'^\d+N_', '', name)
    # Replace underscores with dashes
    name = name.replace('_', '-')
    name = name.replace('Bipartite', 'BP')
    return name

def plot_sim_vs_nisq(df, metric_sim_col, metric_nisq_col, xlabel=None, ylabel=None, title=None, save_path=None):
    # Define colorblind-friendly palette and markers
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    markers = ['X', 's', 'D', '^', 'v', 'P', '*', 'X']  
    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)

    # Assign a color/marker to each dataset
    dataset_styles = {}
    for dataset in df['dataset_name'].unique():
        dataset_styles[dataset] = {
            'color': next(color_cycle),
            'marker': next(marker_cycle)
        }

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot each dataset
    for dataset, group in df.groupby('dataset_name'):
        style = dataset_styles[dataset]
        ax.scatter(
            group[metric_sim_col],
            group[metric_nisq_col],
            label=clean_dataset_name(dataset),
            color=style['color'],
            marker=style['marker'],
            s=100,
            edgecolor='k',
            alpha=0.8
        )

    # y=x reference line
    all_values = np.concatenate([df[metric_sim_col].values, df[metric_nisq_col].values])
    min_val, max_val = np.min(all_values), np.max(all_values)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label=r"Ideal")

    ax.set_xlabel(xlabel if xlabel else f"Simulated {metric_sim_col}")
    ax.set_ylabel(ylabel if ylabel else f"NISQ {metric_nisq_col}")
    ax.set_title(title if title else f"{metric_nisq_col} Comparison: NISQ vs Simulated", pad=20)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
            ncol=4, frameon=True, framealpha=0.9, fontsize=16)

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    nisq_csv = "results/analysis/nisq/analysis_8N.csv"
    sim_csv = "results/analysis/simulations/analysis_8N.csv"

    # Load and merge dynamically
    merged_df = load_and_merge(nisq_csv, sim_csv)

    # Plot comparisons
    plot_sim_vs_nisq(
        merged_df,
        metric_sim_col="gen_density_sim",
        metric_nisq_col="gen_density_nisq",
        xlabel=r"Simulated $\mathbb{E}[\rho]$",
        ylabel=r"NISQ $\mathbb{E}[\rho]$",
        title=r"Generated Density: NISQ vs Simulation",
        save_path="plots/comparisons/gen_density_comparison.pdf"
    )

    plot_sim_vs_nisq(
        merged_df,
        metric_sim_col="mmd_sim",
        metric_nisq_col="mmd_nisq",
        xlabel=r"Simulated MMD$(q_\theta, p)$",
        ylabel=r"NISQ MMD$(q_\theta, p)$",
        title=r"MMD: NISQ vs Simulation",
        save_path="plots/comparisons/mmd_comparison.pdf"
    )

    plot_sim_vs_nisq(
        merged_df,
        metric_sim_col="gen_bipartivity_sim",
        metric_nisq_col="gen_bipartivity_nisq",
        xlabel=r"Simulated $\mathbb{E}[\beta]$",
        ylabel=r"NISQ $\mathbb{E}[\beta]$",
        title=r"Bipartivity: NISQ vs Simulation",
        save_path="plots/comparisons/gen_bipartivity_comparison.pdf"
    )

    plot_sim_vs_nisq(
        merged_df,
        metric_sim_col="gen_bipartite_percent_sim",
        metric_nisq_col="gen_bipartite_percent_nisq",
        xlabel=r"Simulated Accuracy (\%)",
        ylabel=r"NISQ Accuracy (\%)",
        title=r"Bipartite Accuracy: NISQ vs Simulation",
        save_path="plots/comparisons/gen_bipartite_comparison.pdf"
    )

