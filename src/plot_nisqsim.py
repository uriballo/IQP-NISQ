#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import re
import math
from matplotlib.lines import Line2D

# ==============================================================
# 1. PRA-style configuration
# ==============================================================

def setup_pra_plot():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size":11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.5,
        "axes.grid.which": "major",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    })

# ==============================================================
# 2. Utility
# ==============================================================

def extract_run_id(path: str) -> str:
    m = re.search(r'/run_([^/]+)/', path)
    return m.group(1) if m else None

def load_and_merge(nisq_csv, sim_csv):
    nisq_df = pd.read_csv(nisq_csv)
    sim_df = pd.read_csv(sim_csv)
    nisq_df['run_id'] = nisq_df['path'].apply(extract_run_id)
    sim_df['run_id'] = sim_df['path'].apply(extract_run_id)
    merged = pd.merge(nisq_df, sim_df,
                      on=['dataset_name','run_id'],
                      suffixes=('_nisq','_sim'))
    return merged.dropna(subset=['path_nisq','path_sim'])

# ==============================================================
# 3. Combined plot
# ==============================================================

def plot_combined_sim_vs_nisq(df, comparisons, save_path):
    setup_pra_plot()

    df["dataset_type"] = np.where(df["dataset_name"].str.contains("Bipartite"), "BP", "ER")
    # pick one density column (NISQ preferred)
    if "density_nisq" in df.columns:
        df["density"] = df["density_nisq"].astype(str).str.lower()
    elif "density_sim" in df.columns:
        df["density"] = df["density_sim"].astype(str).str.lower()
    else:
        raise KeyError("Neither 'density_nisq' nor 'density_sim' found in dataframe.")


    dataset_colors = {"BP": '#377eb8', "ER": '#ff7f00'}
    dataset_line = {"BP": "solid", "ER": "dashed"}
    density_markers = {"sparse": "o", "medium": "s", "dense": "D"}

    n = len(comparisons)
    n_cols = 2 if n > 1 else 1
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5*n_cols, 3.0*n_rows),
                             sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, (sim_col, nisq_col, xlabel, ylabel, title) in enumerate(comparisons):
        ax = axes[i]

        for dataset_type in df["dataset_type"].unique():
            color = dataset_colors[dataset_type]
            subset_dt = df[df["dataset_type"] == dataset_type]
            for density in sorted(subset_dt["density"].unique()):
                marker = density_markers[density]
                subset = subset_dt[subset_dt["density"] == density]
                ax.scatter(subset[sim_col], subset[nisq_col],
                           color=color, marker=marker,
                           s=22, alpha=0.85, zorder=3, label=None)

        # y=x reference line — no label
        sim_vals = df[sim_col].to_numpy()
        nisq_vals = df[nisq_col].to_numpy()
        valid = np.isfinite(sim_vals) & np.isfinite(nisq_vals)
        if np.any(valid):
            vmin = min(sim_vals[valid].min(), nisq_vals[valid].min())
            vmax = max(sim_vals[valid].max(), nisq_vals[valid].max())
            margin = 0.05 * (vmax - vmin)
            lims = (vmin - margin, vmax + margin)
        else:
            lims = (0, 1)

        ax.plot(lims, lims, 'r--', lw=1.0, label=None)
        ax.set_xlim(lims)
        ax.minorticks_on()
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)

        #ax.text(0.02, 0.96, f"({chr(97+i)})",
        #        transform=ax.transAxes,
        #        fontsize=9, va="top", ha="left", weight="bold")

    # remove unused axes
    for j in range(len(comparisons), len(axes)):
        fig.delaxes(axes[j])

    # ==========================================================
    # 4. Clean two-tier legend (manual)
    # ==========================================================

    # Graph type (color)
    bp = Line2D([0], [0], color=dataset_colors["BP"], lw=4, label="Bipartite")
    er = Line2D([0], [0], color=dataset_colors["ER"], lw=4, label=r"Erd\H{o}s-Rényi")

    # Density (marker)
    sparse = Line2D(
        [0], [0],
        color="black",
        marker=density_markers["sparse"],
        linestyle="None",
        markersize=5,
        markerfacecolor="none",  # make marker hollow
        markeredgecolor="black", # outline color
        label="Sparse"
    )

    medium = Line2D(
        [0], [0],
        color="black",
        marker=density_markers["medium"],
        linestyle="None",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="black",
        label="Medium"
    )

    dense = Line2D(
        [0], [0],
        color="black",
        marker=density_markers["dense"],
        linestyle="None",
        markersize=5,
        markerfacecolor="none",
        markeredgecolor="black",
        label="Dense"
    )

    # sparse = Line2D([], [], color='black', marker=density_markers["sparse"],
    #                 linestyle='None', markersize=4, label='Sparse')
    # medium = Line2D([], [], color='black', marker=density_markers["medium"],
    #                 linestyle='None', markersize=4, label='Medium')
    # dense = Line2D([], [], color='black', marker=density_markers["dense"],
    #                linestyle='None', markersize=4, label='Dense')

    last_ax = axes[-1]
    ax.minorticks_on()
    fig.legend(
        handles=[bp, er],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=2, frameon=False,
        columnspacing=0.8, handlelength=1.4, handletextpad=0.4, labelspacing=0.2
    )
    fig.legend(
        handles=[sparse, medium, dense],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=3, frameon=False,
        columnspacing=0.8, handlelength=1.2, handletextpad=0.35, labelspacing=0.12
    )

    # leg1 = last_ax.legend(handles=[bp, er],
    #                       loc='upper center',
    #                       bbox_to_anchor=(-0.5, -0.25),
    #                       ncol=2, frameon=False, columnspacing=0.8,
    #                       handlelength=1.8, handletextpad=0.4,
    #                       labelspacing=0.2, title_fontsize=8)
    # leg2 = last_ax.legend(handles=[sparse, medium, dense],
    #                       loc='upper center',
    #                       bbox_to_anchor=(-0.5, -0.35),
    #                       ncol=3, frameon=False, columnspacing=0.8,
    #                       handlelength=1.2, handletextpad=0.35,
    #                       labelspacing=0.12, title_fontsize=8)

    #last_ax.add_artist(leg1)

    fig.tight_layout(pad=0.5)
    #plt.subplots_adjust(bottom=0.32, hspace=0.4, wspace=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"✅ Saved clean two-tier legend figure → {save_path}")


# ==============================================================
# 5. Example usage
# ==============================================================

if __name__ == "__main__":
    nisq_csv = "results/analysis/nisq/analysis_8N.csv"
    sim_csv  = "results/analysis/simulations/analysis_8N_filtered.csv"
    merged_df = load_and_merge(nisq_csv, sim_csv)

    comparisons = [
        ("gen_density_sim", "gen_density_nisq",
         r"Simulated $\mathbb{E}[\rho]$", r"NISQ $\mathbb{E}[\rho]$", "(a) Generated density"),
        ("gen_bipartite_percent_sim", "gen_bipartite_percent_nisq",
         r"Simulated Accuracy (\%)", r"NISQ Accuracy (\%)", "(b) Bipartite accuracy"),
        ("gen_bipartivity_sim", "gen_bipartivity_nisq",
         r"Simulated $\mathbb{E}[\beta]$", r"NISQ $\mathbb{E}[\beta]$", "(c) Spectral bipartivity"),
        ("mmd_sim", "mmd_nisq",
         r"Simulated MMD$(q_\theta,p)$", r"NISQ MMD$(q_\theta,p)$", "(d) MMD"),
    ]

    out_path = "plots/two_column/combined_sim_vs_nisq.pdf"
    plot_combined_sim_vs_nisq(merged_df, comparisons, out_path)
