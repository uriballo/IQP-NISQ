import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import binom
from src.utils.utils import vec_to_graph

# ---------------------------
# Global style config
# ---------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 26,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.fontsize": 20,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.loc": "best",
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.markersize": 6,
})

empirical_color = "#56B4E9"  # sky blue
binomial_color = "#D55E00"   # reddish orange
markers = ['s', 'v', 'D']
line_styles = ['-', '--', '-.']
# ---------------------------
# Functions
# ---------------------------

def plot_degree_distribution(file_path, p, n, dataset_name, graph_type="ER", simulations = False):
    """Plot single-degree distribution from .npy samples, show TVD, save plot."""
    out_dir = "plots/degree_distributions"
    os.makedirs(out_dir, exist_ok=True)

    # Load graphs
    binary_graphs = np.load(file_path, allow_pickle=True)
    graphs = [vec_to_graph(g, n) for g in binary_graphs]

    # Aggregate degrees
    all_degrees = [d for g in graphs for _, d in g.degree()]

    # Empirical distribution
    k_values = np.arange(n)
    degree_counts = np.bincount(all_degrees, minlength=n)
    empirical_probs = degree_counts / np.sum(degree_counts)

    # Binomial distribution
    binom_probs = binom.pmf(k_values, n=n-1, p=p)

    # Total Variation Distance
    tvd = 0.5 * np.sum(np.abs(empirical_probs - binom_probs))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.bar(k_values, empirical_probs, color=empirical_color, alpha=0.6, edgecolor="k", linewidth=0.5)
    ax.plot(k_values, binom_probs, color=binomial_color, marker="D", linestyle="-", linewidth=2)
    ax.set_xlabel(r"Degree $k$")
    ax.set_ylabel(r"Probability $p(k)$")
    ax.set_title(rf"Generated Degree Distribution", pad=25)
    ax.text(0.5, 1.02, rf"TVD = {tvd:.4f}", transform=ax.transAxes,
            ha="center", fontsize=18)

    # Legend
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=empirical_color,
               markeredgecolor='k', markersize=10, label="Empirical"),
        Line2D([0], [0], marker='D', color=binomial_color, linestyle='-', linewidth=2,
               markersize=8, label="Target")
    ]
    ax.legend(handles=handles, frameon=True, shadow=True)

    fig.tight_layout(rect=[0,0,1,0.93])
    if simulations:
        save_path = os.path.join(out_dir, f"{dataset_name}_simulated_deg_dist.pdf")
    else:
        save_path = os.path.join(out_dir, f"{dataset_name}_deg_dist.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot for {dataset_name} (TVD={tvd:.4f}) at {save_path}")


def plot_multiple_degree_distributions_samples(file_paths, dataset_labels, n_nodes_list, p_list, densities, simulations = False):
    """Combined histogram for multiple .npy sample datasets with colored target markers,
    horizontal legend, and average TVD as subtitle.
    """
    mpl.rcParams.update({
        "font.size": 34,
        "axes.labelsize": 34,
        "axes.titlesize": 38,
        "legend.fontsize": 28,
        "xtick.labelsize": 34,
        "ytick.labelsize": 34,
    })

    fig, ax = plt.subplots(figsize=(14, 9))
    colors = ["#e69F00", "#0072b2", "#cc79a7"]
    bar_width = 0.2
    n_datasets = len(file_paths)
    offsets = np.linspace(-bar_width*(n_datasets-1)/2, bar_width*(n_datasets-1)/2, n_datasets)

    all_handles = []
    tvd_list = []
    target_handle = Line2D([0], [0], marker='D', color='0.5', linestyle='-', linewidth=2, markersize=10, label="Target")
    all_handles.append(target_handle)

    for i, (file_path, density, n_nodes, p) in enumerate(zip(file_paths, densities, n_nodes_list, p_list)):
        # Load and convert graphs
        binary_graphs = np.load(file_path, allow_pickle=True)
        graphs = [vec_to_graph(g, n_nodes) for g in binary_graphs]

        # Aggregate degrees
        all_degrees = [d for g in graphs for _, d in g.degree()]
        degree_counts = np.bincount(all_degrees, minlength=n_nodes)
        empirical_probs = degree_counts / degree_counts.sum()

        # Binomial distribution
        k_values = np.arange(n_nodes)
        binomial_probs = binom.pmf(k_values, n=n_nodes-1, p=p)

        # Compute TVD
        tvd = 0.5 * np.sum(np.abs(empirical_probs - binomial_probs))
        tvd_list.append(tvd)

        # Plot bars and binomial lines
        ax.bar(k_values + offsets[i], empirical_probs, width=bar_width,
               color=colors[i % len(colors)], alpha=0.6, edgecolor="k")
        ax.plot(k_values + offsets[i], binomial_probs, color=colors[i % len(colors)],
                linestyle='-', linewidth=2)

        # Fill area under binomial line
        ax.fill_between(k_values + offsets[i], 0, binomial_probs, color=colors[i % len(colors)], alpha=0.2)

        # Target markers (same color as dataset)
        mid_index = n_nodes // 2  # place one marker in middle for clarity
        # Instead of a single mid_index marker, plot markers on all degrees:
        marker = markers[i % len(markers)]
        ax.plot(k_values + offsets[i], binomial_probs,
                marker=marker, color=colors[i % len(colors)],
                linestyle='None', markersize=8)

        handle = Line2D([0], [0], color=(colors[i % len(colors)], 0.6), lw=12, linestyle='-', label=density)
        all_handles.append(handle)

        print(f"{density} -> Nodes: {n_nodes}, p = {p:.4f}, TVD = {tvd:.4f}")


    # Horizontal legend below plot
    ax.legend(handles=all_handles[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.25),
              ncol=len(all_handles), frameon=True, framealpha=0.9)
    qubits = n_nodes_list[0] * (n_nodes_list[0] - 1) // 2
    ax.set_xlabel(r"Degree $k$")
    ax.set_ylabel(r"Probability $p(k)$")
    ax.set_title(fr"Degree Distributions: ${qubits}-$qubit Erd\H{{o}}s-Rényi Models", pad=50)


    # Average TVD subtitle
    avg_tvd = np.mean(tvd_list)
    ax.text(0.5, 1.05, f"Average TVD = {avg_tvd:.4f}", transform=ax.transAxes,
            ha="center", fontsize=28)

    # Save figure
    out_dir = "plots/degree_distributions"
    os.makedirs(out_dir, exist_ok=True)
    if simulations:
        save_path = os.path.join(out_dir, f"combined_deg_dist_simulated_{n_nodes_list[0]}N.pdf")
    else:
        save_path = os.path.join(out_dir, f"combined_deg_dist_{n_nodes_list[0]}N.pdf")
    fig.tight_layout(rect=[0,0,1,0.88])  # leave space for legend
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot for {n_nodes_list[0]} nodes at {save_path}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Load CSV and filter ER datasets
    df = pd.read_csv("results/analysis/nisq/best_runs.csv", sep=",")
    er_df = df[df['dataset_name'].str.contains("ER")]

    # Plot individual degree distributions
    for _, row in er_df.iterrows():
        dataset_name = row['dataset_name']
        path = row['path']
        n = int(dataset_name.split('N')[0])
        p = row['ref_density']
        plot_degree_distribution(path, p, n, dataset_name)

    # Plot combined degree distributions per node count
    node_counts = er_df['dataset_name'].apply(lambda x: int(x.split('N')[0])).unique()
    for n_nodes in node_counts:
        subset = er_df[er_df['dataset_name'].str.startswith(str(n_nodes))]
        file_paths = subset['path'].tolist()
        densities = subset['density'].tolist()  # Only density for legend
        n_list = [int(l.split('N')[0]) for l in subset['dataset_name']]
        p_list = subset['ref_density'].tolist()
        plot_multiple_degree_distributions_samples(file_paths, subset['density'].tolist(), n_list, p_list, densities)
