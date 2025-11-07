import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import os
import numpy as np

def setup_pra_plot():
    """Standardized matplotlib style for publication-ready plots."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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

def main():
    setup_pra_plot()

    csv_path = "results/analysis/nisq/best_runs.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at: {csv_path}")

    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception as e:
        print(f"Warning: autodetect failed ({e}), retrying with comma separator")
        df = pd.read_csv(csv_path, sep=",")

    df.columns = [c.strip() for c in df.columns]

    if "dataset_name" not in df.columns:
        raise KeyError(f"'dataset_name' not found; columns = {df.columns.tolist()}")

    df["N"] = df["dataset_name"].str.extract(r"(\d+)N").astype(int)
    df["graph_type"] = df["dataset_name"].apply(lambda x: "BP" if "Bipartite" in x else "ER")
    df["density_type"] = df["dataset_name"].str.extract(r"_(Dense|Medium|Sparse)")[0].str.lower()

    N_to_qubits = {8: 28, 10: 45, 14: 91, 18: 153}
    df = df[df["N"].isin(N_to_qubits.keys())]
    df["Qubits"] = df["N"].map(N_to_qubits)
    dataset_colors = {"BP": '#377eb8', "ER": '#ff7f00'}
    density_markers = {"dense": "D", "medium": "s", "sparse": "o"}

    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    for graph_type, color in dataset_colors.items():
        sub = df[df["graph_type"] == graph_type]
        for d_type, marker in density_markers.items():
            data = sub[sub["density_type"] == d_type].sort_values("N")
            if not data.empty:
                linestyle = ":" if graph_type == "ER" else "-"  
                ax.plot(
                    data["Qubits"],           
                    np.abs(data["mmd"]),
                    marker=marker,
                    markersize=4,
                    color=color,
                    linestyle=linestyle,
                    markerfacecolor=color,      
                    markeredgecolor=color,
                    label=f"{graph_type}-{d_type}"
                )

    ax.set_xlabel(r"Qubits")
    ax.set_ylabel(r"MMD($q_\theta, p$)")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xticks(list(N_to_qubits.values()))
    ax.set_xlim(min(N_to_qubits.values()) - 5, max(N_to_qubits.values()) + 10)

    bp = Line2D([0], [0], color=dataset_colors["BP"], lw=4, label="Bipartite")
    er = Line2D([0], [0], linestyle=':', color=dataset_colors["ER"], lw=4, label=r"Erd\H{o}s-Rényi")

    sparse = Line2D([], [], color="black", marker=density_markers["sparse"],
                    linestyle="None", markersize=5, markerfacecolor="none",
                    markeredgecolor="black", label="Sparse")
    medium = Line2D([], [], color="black", marker=density_markers["medium"],
                    linestyle="None", markersize=5, markerfacecolor="none",
                    markeredgecolor="black", label="Medium")
    dense = Line2D([], [], color="black", marker=density_markers["dense"],
                   linestyle="None", markersize=5, markerfacecolor="none",
                   markeredgecolor="black", label="Dense")

    fig.legend(
        handles=[bp, er],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.06),
        ncol=2, frameon=False,
        columnspacing=0.8, handlelength=2.0, handletextpad=0.4, labelspacing=0.2
    )
    fig.legend(
        handles=[sparse, medium, dense],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.00),
        ncol=3, frameon=False,
        columnspacing=0.8, handlelength=1.4, handletextpad=0.35, labelspacing=0.12
    )

    fig.tight_layout()

    save_dir = "plots/single_column"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mmd_scaling.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    print(f"Saved high-resolution figure to {save_path}")

if __name__ == "__main__":
    main()
