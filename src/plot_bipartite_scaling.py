import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ==============================================================
# 1. PRA-style configuration (Computer Modern via LaTeX)
# ==============================================================

def setup_pra_plot():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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


# ==============================================================
# 2. Helper: plot a single metric on a given Axes
# ==============================================================

def plot_metric_on_axis(ax, csv_path, metric_column, ylabel, title, baseline_csv=None,
                        xlim=(0, 0.55), ylim=(0, 1.05), show_xlabel=True):
    """
    Plot metric for several node counts onto `ax`.
    If show_xlabel is False, this function will NOT set the x-axis label
    (useful when using fig.supxlabel for a shared x-label).
    """

    df = pd.read_csv(csv_path)
    if metric_column not in df.columns:
        raise ValueError(f"CSV does not contain '{metric_column}' column")

    df = df.rename(columns={
        "mean_density_in_bin": "density",
        metric_column: "metric",
        "nodes": "nodes"
    })

    valid_nodes = [8, 10, 14, 18]
    df = df[df["nodes"].isin(valid_nodes)]
    df = df.groupby(["nodes", "density"], as_index=False)["metric"].mean()
    df["qubits"] = df["nodes"] * (df["nodes"] - 1) // 2

    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']
    line_styles = ['solid', 'dashed', 'dashdot', (5, (10, 3))]  # custom styles

    line_handles = []  # store handles for legend

    for i, n in enumerate(valid_nodes):
        subset = df[df["nodes"] == n].sort_values("density")
        line, = ax.plot(
            subset["density"],
            subset["metric"],
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.25,
            label=fr"{n*(n-1)//2}\,qubits"
        )
        line_handles.append(line)

    # Optional baseline overlay
    if baseline_csv is not None and Path(baseline_csv).exists():
        df_base = pd.read_csv(baseline_csv)
        if metric_column in df_base.columns:
            df_base = df_base.rename(columns={
                "mean_density_in_bin": "density",
                metric_column: "metric",
                "nodes": "nodes"
            })
            df_base = df_base[df_base["nodes"].isin(valid_nodes)]
            df_base = df_base.groupby(["nodes", "density"], as_index=False)["metric"].mean()
            df_base["qubits"] = df_base["nodes"] * (df_base["nodes"] - 1) // 2

            for i, n in enumerate(valid_nodes):
                subset_base = df_base[df_base["nodes"] == n].sort_values("density")
                ax.plot(
                    subset_base["density"],
                    subset_base["metric"],
                    color=colors[i % len(colors)],
                    linestyle='dotted',
                    linewidth=1.2,
                    alpha=0.8
                )

    # only set x label if requested (we will use fig.supxlabel for a shared one)
    if show_xlabel:
        ax.set_xlabel(r"Empirical Density $\rho$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', length=3, width=0.6)

    return ax, line_handles


# ==============================================================
# 3. Combined figure: two plots, shared legend and shared x label
# ==============================================================

def plot_combined_two_column(
    csv1, metric1, ylabel1, title1,
    csv2, metric2, ylabel2, title2,
    baseline_csv=None,
    save_path="plots/pra_two_column/combined_two_panel.pdf"
):
    setup_pra_plot()
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # create two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=False)

    # Plot without per-axis x-labels (we will place a single supxlabel)
    _, handles_left = plot_metric_on_axis(
        axes[0], csv1, metric1, ylabel1, title1,
        baseline_csv=baseline_csv, xlim=(0, 0.5), ylim=(0, 105),
        show_xlabel=False
    )
    _, handles_right = plot_metric_on_axis(
        axes[1], csv2, metric2, ylabel2, title2,
        baseline_csv=baseline_csv, xlim=(0, 0.55), ylim=(0.45, 1.05),
        show_xlabel=False
    )

    # Use left handles for legend (solid lines for qubits)
    handles = handles_left.copy()
    labels = [h.get_label() for h in handles]

    # Add baseline cue
    baseline_handle = Line2D([0], [0], color='black', linestyle='dotted', linewidth=1.2, alpha=0.8)
    handles.append(baseline_handle)
    labels.append("baseline")

    # Shared legend below both panels. Put legend near bottom (adjust y if needed).
    # We'll place legend lower than the shared xlabel so the xlabel sits between panels and legend.
    legend_y = 0.1  # tune this if overlapping on your system
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, legend_y),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.4
    )

    # Single centered x-label for both panels (place above the legend)
    fig.supxlabel(r"Empirical Density $\rho$", y=0.1)  # adjust y if overlapping

    # adjust layout: leave extra bottom room for xlabel + legend
    fig.tight_layout(pad=0.6, rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved combined two-panel figure: {save_path}")


# ==============================================================
# 4. Run both plots combined
# ==============================================================

if __name__ == "__main__":
    plot_combined_two_column(
        csv1="results/analysis/per_bin_bipartite_accuracy.csv",
        metric1="bipartite_percent",
        ylabel1=r"Bipartite Accuracy (\%)",
        title1=r"Bipartite Generation Scaling Across Densities",

        csv2="results/analysis/per_bin_bipartivity.csv",
        metric2="mean_bipartivity_in_bin",
        ylabel2=r"Mean Bipartivity",
        title2=r"Bipartivity Scaling Across Densities",

        baseline_csv="results/analysis/finite_baseline_per_bin.csv",
        save_path="plots/two_column/combined_scaling_figure.pdf"
    )
