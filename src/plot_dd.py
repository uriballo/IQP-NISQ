import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import binom
from src.utils.utils import vec_to_graph

# ---------------------------
# Global Matplotlib style
# ---------------------------
def setup_pra_plot():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "axes.grid.which": "major",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    })


# ---------------------------
# Helper function
# ---------------------------
def compute_degree_distribution(file_path, n_nodes, p):
    """Load binary graphs from .npy, compute empirical degree distribution, binomial reference, and TVD."""
    binary_graphs = np.load(file_path, allow_pickle=True)
    graphs = [vec_to_graph(g, n_nodes) for g in binary_graphs]

    all_degrees = [d for g in graphs for _, d in g.degree()]
    degree_counts = np.bincount(all_degrees, minlength=n_nodes)
    empirical_probs = degree_counts / degree_counts.sum()

    k_values = np.arange(n_nodes)
    binomial_probs = binom.pmf(k_values, n=n_nodes - 1, p=p)
    tvd = 0.5 * np.sum(np.abs(empirical_probs - binomial_probs))

    return k_values, empirical_probs, binomial_probs, tvd

# ---------------------------
# Main plotting function
# ---------------------------
def plot_combined_ER_distributions(csv_path="results/analysis/nisq/best_runs.csv", out_dir="plots/two_column"):
    os.makedirs(out_dir, exist_ok=True)

    # Load CSV and filter ER datasets
    df = pd.read_csv(csv_path)
    er_df = df[df['dataset_name'].str.contains("ER")]

    # Extract unique node counts (e.g., 8N, 10N, 14N, 18N)
    node_counts = sorted(er_df['dataset_name'].apply(lambda x: int(x.split('N')[0])).unique())
    n_panels = len(node_counts)
    print(f"Found {n_panels} ER dataset sizes: {node_counts}")

    # --- Figure setup ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharey=True)
    axes = axes.flatten()
    
    colors = ['#377eb8', '#ff7f00', '#4daf4a']
    markers = ['s', 'v', 'D']

    avg_tvds = []

    # --- Loop over datasets (one subplot per node count) ---
    for i, (n_nodes, ax) in enumerate(zip(node_counts, axes)):
        subset = er_df[er_df['dataset_name'].str.startswith(str(n_nodes))]
        densities = subset['density'].tolist()
        file_paths = subset['path'].tolist()
        p_list = subset['ref_density'].tolist()

        tvd_list = []
        data_triplets = []

        for j, (path, p) in enumerate(zip(file_paths, p_list)):
            k, empirical, binomial, tvd = compute_degree_distribution(path, n_nodes, p)
            tvd_list.append(tvd)
            data_triplets.append((k, empirical, binomial))
            
            # Bar and line plot
            color = colors[j % len(colors)]
            offset = (j - 1) * 0.2  # center bars
            ax.bar(k + offset, empirical, width=0.2, color=color, alpha=0.6, edgecolor="k")
            ax.plot(k, binomial, color=color, lw=2)
            ax.fill_between(k, 0, binomial, color=color, alpha=0.2)
            ax.plot(k, binomial, marker=markers[j % len(markers)], linestyle="None", color=color, markersize=6)

        avg_tvd = np.mean(tvd_list)
        avg_tvds.append(avg_tvd)
        ax.minorticks_on()
        # Labels and titles
        qubits = n_nodes * (n_nodes - 1) // 2
        ax.set_title(fr"({chr(97+i)}) {qubits}-qubit Erd\H{{o}}s-Rényi models")
        ax.text(0.5, 0.93, f"Average TVD = {avg_tvd:.4f}", transform=ax.transAxes, ha="center")
        #ax.set_xlabel(r"Degree $k$")
        #if i % 2 == 0:
        #    ax.set_ylabel(r"Probability $p(k)$")
        ax.set_xlim(-0.5, n_nodes - 0.5)

    # --- Shared legend ---
    handles = [
        Line2D([0], [0], color=colors[2], lw=4, label="Sparse"),
        Line2D([0], [0], color=colors[1], lw=4, label="Medium"),
        Line2D([0], [0], color=colors[0], lw=4, label="Dense"),
        Line2D([0], [0], color="black", marker="x", markersize=6, linestyle="-", lw=1, label="Target"),

    ]
    fig.legend(handles=handles,  bbox_to_anchor=(0.5, 0.01), loc="lower center", ncol=4, frameon=False, fontsize=16)

    # --- Shared axis labels ---
    fig.text(0.5, 0.1, r"Degree $k$", ha="center", fontsize=17)
    fig.text(-0.005, 0.5, r"Probability $p(k)$", va="center", rotation="vertical", fontsize=17)

    # --- Layout and save ---
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    save_path = os.path.join(out_dir, "degree_distributions.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved combined figure to {save_path}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    setup_pra_plot()
    plot_combined_ER_distributions()
