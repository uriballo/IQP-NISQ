import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------
# Academic-style plotting config
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
# Plotting function
# ---------------------------
def plot_metric_scaling(df, metric_col, ylabel=None, title=None, save_name=None,
                        baseline_col=None, baseline_shade=0.05, x_ticks=[8, 10, 14, 18]  # Force these x-axis ticks
):
    """
    Plot the scaling of a metric across datasets.

    Args:
        df (pd.DataFrame): DataFrame containing 'dataset_name', 'density', metric_col, optional baseline_col.
        metric_col (str): Column name of the metric to plot (e.g., 'gen_density').
        ylabel (str, optional): Label for y-axis.
        title (str, optional): Plot title.
        save_name (str, optional): Path to save the figure.
        baseline_col (str, optional): Column for baseline reference (dashed line).
        baseline_shade (float, optional): Fractional shading around baseline.
        x_ticks (list, optional): Explicit x-axis ticks to display.
    """
    densities = df['density'].unique()
    colors = ["#e69F00", "#0072b2", "#cc79a7"][::-1]
    markers = ['s', 'v', 'D']
    line_styles = ['-', '--', '-.']

    fig, ax = plt.subplots(figsize=(7, 7))
    y_max = 0.0
    for i, density in enumerate(densities[::-1]):
        subset = df.loc[df['density'] == density].copy()
        subset['N'] = subset['dataset_name'].str.extract(r'(\d+)').astype(int)
        subset = subset.sort_values('N')

        x = subset['N'].values
        y = subset[metric_col].values
        y_max = max(np.max(y), y_max)

        # Main line
        ax.plot(x, y, marker=markers[i % len(colors)], color=colors[i % len(colors)],
                linestyle=line_styles[i % len(colors)], linewidth=2, markersize=8, label=f"{density}")


    ax.set_ylim(0.0, 1.1 * y_max)
    ax.set_xlabel(r"Nodes $M$")
    ax.set_ylabel(ylabel if ylabel else metric_col)
    ax.set_title(title if title else f"{metric_col} Scaling Across Datasets", pad=20)

    ax.set_xticks(x_ticks)

    # Horizontal legend below plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=len(densities), frameon=True, framealpha=0.9)

    fig.tight_layout(rect=[0,0,1,0.9])

    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        fig.savefig(save_name, bbox_inches="tight")
        print(f"Saved plot at {save_name}")
    plt.close(fig)

def plot_metric_scaling_merged(df_list, labels, metric_col, ylabel=None, title=None, save_name=None,
                               x_ticks=[8, 10, 14, 18]):
    """
    Plot the scaling of a metric for multiple datasets on the same axes.

    Each (dataset, density) combination has a distinct color, marker, and line style.
    Legend uses simple labels like 'ER-Dense' or 'BP-Sparse', displayed in two rows below the axes.
    Figure width is dynamically adjusted to match legend width.
    """
    base_colors = ["#e69F00", "#0072b2", "#cc79a7", "#56B4E9", "#D55E00", "#009E73", "#F0E442"]
    base_markers = ['s', 'v', 'D', '^', 'o', 'P', '*']
    base_lines = ['-', '--', '-.', ':', '-', '--', '-.']

    # Collect all unique (dataset, density) combinations
    combinations = []
    for j, df in enumerate(df_list):
        dataset_type = labels[j]
        densities = sorted(df['density'].unique())
        for density in densities:
            combinations.append((dataset_type, density))

    # Map each combination to a unique style
    style_map = {}
    for idx, (dataset, density) in enumerate(combinations):
        style_map[(dataset, density)] = {
            'color': base_colors[idx % len(base_colors)],
            'marker': base_markers[idx % len(base_markers)],
            'linestyle': base_lines[idx % len(base_lines)]
        }

    # Dynamic figure width based on legend
    total_combinations = len(combinations)
    ncol = int(np.ceil(total_combinations / 2))
    fig_width = 7
    fig_height = 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_max = 0.0
    all_labels = []

    # Plot each combination
    for j, df in enumerate(df_list):
        dataset_type = labels[j]
        densities = sorted(df['density'].unique())
        for density in densities:
            subset = df[df['density'] == density].copy()
            subset['N'] = subset['dataset_name'].str.extract(r'(\d+)').astype(int)
            subset = subset.sort_values('N')

            x = subset['N'].values
            y = subset[metric_col].values
            y_max = max(np.max(y), y_max)

            label = f"{dataset_type}-{density}"
            all_labels.append(label)

            style = style_map[(dataset_type, density)]
            ax.plot(x, y, marker=style['marker'], color=style['color'],
                    linestyle=style['linestyle'], linewidth=2, markersize=8, label=label)

    ax.set_ylim(0.0, 1.1 * y_max)
    ax.set_xlabel(r"Nodes $M$")
    ax.set_ylabel(ylabel if ylabel else metric_col)
    ax.set_title(title if title else f"{metric_col} Scaling", pad=20)
    ax.set_xticks(x_ticks)

    # Place legend below axes in 2 rows
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=ncol, frameon=True, framealpha=0.9)


    if save_name:
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        fig.savefig(save_name, bbox_inches="tight")
        print(f"Saved merged plot at {save_name}")
    plt.close(fig)



# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Load dataset CSV
    df = pd.read_csv("results/analysis/nisq/best_runs.csv", sep=",")
    baselines_df = pd.read_csv("results/analysis/baseline_bp.csv", sep=",")

    # Merge baseline BP
    df = pd.merge(df, baselines_df[['dataset_name','baseline_bp_percent']], on="dataset_name", how="left")

    # Filter only Bipartite datasets
    bipartite_df = df[df['dataset_name'].str.contains("Bipartite")]
    erdosrenyi_df = df[df['dataset_name'].str.contains("ER")]

    out_dir = "plots/metric_scaling"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Density scaling
    plot_metric_scaling(
        bipartite_df,
        metric_col="density_error",
        ylabel=r"$\Delta\mathbb E[\rho]$",
        title=r"Scaling of the Expected Density Difference",
        save_name=os.path.join(out_dir, "density_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        erdosrenyi_df,
        metric_col="density_error",
        ylabel=r"$\Delta\mathbb E[\rho]$",
        title=r"Scaling of the Expected Density Difference",
        save_name=os.path.join(out_dir, "density_scaling_er.pdf")
    )

    plot_metric_scaling(
        bipartite_df,
        metric_col="mmd",
        ylabel=r"MMD$(q_\theta, p)$",
        title=r"Scaling of the MMD",
        save_name=os.path.join(out_dir, "mmd_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        erdosrenyi_df,
        metric_col="mmd",
        ylabel=r"MMD$(q_\theta, p)$",
        title=r"Scaling of the MMD",
        save_name=os.path.join(out_dir, "mmd_scaling_er.pdf")
    )

    plot_metric_scaling(
        bipartite_df,
        metric_col="gen_bipartivity",
        ylabel=r"$\mathbb E[\beta]$",
        title=r"Scaling of the Spectral Bipartivy",
        save_name=os.path.join(out_dir, "bipartivity_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        bipartite_df,
        metric_col="gen_bipartite_percent",
        ylabel=r"Accuracy (\%)",
        title=r"Scaling of the Bipartite Generation Accuracy",
        save_name=os.path.join(out_dir, "bipartite_generation_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        bipartite_df,
        metric_col="memorized",
        ylabel=r"Memorization (\%)",
        title=r"Scaling of the Memorisation (BP)",
        save_name=os.path.join(out_dir, "memorization_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        erdosrenyi_df,
        metric_col="memorized",
        ylabel=r"Memorization (\%)",
        title=r"Scaling of the Memorisation (ER)",
        save_name=os.path.join(out_dir, "memorization_scaling_er.pdf")
    )

    plot_metric_scaling(
        bipartite_df,
        metric_col="diversity",
        ylabel=r"Diversity (\%)",
        title=r"Scaling of the Sample Diversity (BP)",
        save_name=os.path.join(out_dir, "diversity_scaling_bipartite.pdf")
    )

    plot_metric_scaling(
        erdosrenyi_df,
        metric_col="diversity",
        ylabel=r"Diversity (\%)",
        title=r"Scaling of the Sample Diversity (ER)",
        save_name=os.path.join(out_dir, "diversity_scaling_er.pdf")
    )

    # Merge ER and BP for each metric
    metrics = [
        ("mmd", r"MMD$(q_\theta, p)$", "the MMD"),
        ("memorized", r"Memorization (\%)", "Memorisation"),
        ("diversity", r"Diversity (\%)", "Sample Diversity")
    ]

    for metric_col, ylabel, title in metrics:
        plot_metric_scaling_merged(
            df_list=[bipartite_df, erdosrenyi_df],
            labels=["BP", "ER"],
            metric_col=metric_col,
            ylabel=ylabel,
            title=f"Scaling of {title}",
            save_name=os.path.join(out_dir, f"{metric_col}_scaling_merged.pdf")
        )


