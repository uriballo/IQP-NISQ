#!/usr/bin/env python3
"""
Unified Graph Generation Analysis Script
----------------------------------------
Processes all generated model results in `results/`,
matches them to datasets in `data/raw_data/`,
computes metrics (CSV_COLUMNS),
creates one CSV per node count,
and generates scaling + degree distribution plots.

Assumes dataset naming like:
    data/raw_data/{n}N_{ER|Bipartite}_{Sparse|Medium|Dense}.pkl
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import networkx as nx
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

from src.utils.metrics import estrada_bipartivity
from src.utils.utils import setup_logging, vec_to_graph, graph_to_vec
from src.datasets.er import ErdosRenyiGraphDataset
from src.datasets.bipartites import BipartiteGraphDataset
from iqpopt.gen_qml.sample_methods import mmd_loss_samples
from iqpopt.gen_qml.utils import median_heuristic
import jax.numpy as jnp

# ---------------- Constants ---------------- #

DATASET_SUMMARY_PATH = Path("data/datasets_summary.yml")
TRAINING_DATA_DIR = Path("data/raw_data")
DEFAULT_RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
(PLOTS_DIR / "png").mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "run_id", "run_type", "num_nodes", "graph_type", "density_category",
    "generated_density", "density_value",
    "generated_bipartite", "bipartite_value", "bipartivity",
    "mmd_samples", "memorized_iso_pct", "diversity_iso_pct",
]

# ---------------- Utils ---------------- #

def extract_graph_type_and_density(name: str):
    gt = "Bipartite" if "Bipartite" in name or "BP" in name else "ER"
    for c in ["Sparse", "Medium", "Dense"]:
        if c in name:
            return gt, c
    return gt, None


def safe_sigma(ref: np.ndarray) -> float:
    sigma = median_heuristic(ref)
    return float(sigma if np.isfinite(sigma) and sigma > 0 else 1.0)


def sample_mmd(gen: np.ndarray, ref: np.ndarray) -> float:
    if ref is None or len(ref) == 0:
        return 0.0
    sigma = safe_sigma(ref)
    return float(mmd_loss_samples(jnp.array(gen), jnp.array(ref), sigma))


def load_training_data(dataset_name: str, nodes: int) -> Optional[np.ndarray]:
    path = TRAINING_DATA_DIR / f"{dataset_name}.pkl"
    if not path.exists():
        return None
    gt, _ = extract_graph_type_and_density(dataset_name)
    ds = ErdosRenyiGraphDataset.from_file(str(path)) if gt == "ER" else BipartiteGraphDataset.from_file(str(path))
    return np.array([graph_to_vec(g, nodes) for g in ds.graphs if g.number_of_nodes() == nodes])


def load_dataset_summaries(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return {d["name"]: d for d in raw}


def calc_memorized_iso_pct(gen: np.ndarray, train: np.ndarray, nodes: int) -> float:
    if train is None or len(gen) == 0:
        return 0.0
    count = 0
    for vec in gen:
        g = vec_to_graph(vec, nodes)
        if any(nx.is_isomorphic(g, vec_to_graph(t, nodes)) for t in train):
            count += 1
    return count / len(gen) * 100


def calc_diversity_iso_pct(gen: np.ndarray, nodes: int) -> float:
    if len(gen) <= 1:
        return 100.0
    unique = []
    for vec in gen:
        g = vec_to_graph(vec, nodes)
        if not any(nx.is_isomorphic(g, h) for h in unique):
            unique.append(g)
    return len(unique) / len(gen) * 100


def find_nodes_from_path(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)N_", str(path.resolve()))
    return int(m.group(1)) if m else None


# ---------------- Metric Processing ---------------- #

def process_model_dir(model_dir: Path, base_dir: Path, ds_summary: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    parts = model_dir.parts
    run_id = str(model_dir.relative_to(base_dir))
    match = re.search(r"(\d+N_(Bipartite|ER)_(Sparse|Medium|Dense))", str(model_dir))
    if not match:
        return None
    key = match.group(1)
    if key not in ds_summary:
        return None

    nodes = find_nodes_from_path(model_dir)
    ds_info = ds_summary[key]
    files = list(model_dir.glob("*samples*.npy"))
    if not files:
        return None
    gen = np.vstack([np.load(f) for f in files])
    train = load_training_data(key, nodes)
    num_edges = nodes * (nodes - 1) / 2

    dens_gen, bip_gen, bip_cont = [], [], []
    for v in gen:
        g = vec_to_graph(v, nodes)
        dens_gen.append(len(g.edges()) / num_edges)
        bip_gen.append(1 if nx.is_bipartite(g) else 0)
        bip_cont.append(estrada_bipartivity(g))

    dens_ref = []
    if train is not None:
        for v in train:
            g = vec_to_graph(v, nodes)
            dens_ref.append(len(g.edges()) / num_edges)

    mmd_s = sample_mmd(gen, train)
    mem_iso = calc_memorized_iso_pct(gen, train, nodes)
    div_iso = calc_diversity_iso_pct(gen, nodes)

    return {
        "run_id": run_id,
        "run_type": "Evaluation",
        "num_nodes": nodes,
        "graph_type": ds_info["graph_type"],
        "density_category": ds_info["density_category"],
        "generated_density": np.mean(dens_gen),
        "density_value": np.mean(dens_ref) if dens_ref else ds_info["density_value"],
        "generated_bipartite": np.mean(bip_gen),
        "bipartite_value": ds_info.get("bipartite_value", 0.0),
        "bipartivity": np.mean(bip_cont),
        "mmd_samples": mmd_s,
        "memorized_iso_pct": mem_iso,
        "diversity_iso_pct": div_iso,
    }


# ---------------- Plotting ---------------- #

def make_scaling_plots(df: pd.DataFrame):
    mpl.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.size": 24,
        "axes.labelsize": 24, "axes.titlesize": 28,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.fontsize": 24, "legend.frameon": True, "legend.framealpha": 0.9,
        "axes.grid": True, "grid.color": "0.85", "grid.linestyle": "-", "grid.linewidth": 0.8,
    })
    colors = {"Sparse": "#1f77b4", "Medium": "#ff7f0e", "Dense": "#2ca02c"}
    ls = {"Sparse": "-", "Medium": "--", "Dense": "-."}
    markers = {"Sparse": "o", "Medium": "s", "Dense": "D"}

    metrics = ["generated_density", "generated_bipartite", "bipartivity", "memorized_iso_pct", "diversity_iso_pct"]
    for gtype in ["ER", "Bipartite"]:
        sub = df[df["graph_type"] == gtype]
        if sub.empty:
            continue
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(7, 7))
            for cat in ["Sparse", "Medium", "Dense"]:
                dsub = sub[sub["density_category"] == cat]
                if dsub.empty:
                    continue
                ax.plot(dsub["num_nodes"], dsub[metric], marker=markers[cat],
                        linestyle=ls[cat], color=colors[cat], label=cat)
            ax.set_xlabel("Nodes")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel(metric)
            ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "png" / f"{gtype}_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def plot_degree_distribution(path: Path, p: float, n: int, label: str):
    data = np.load(path, allow_pickle=True)
    graphs = [vec_to_graph(g, n) for g in data]
    all_deg = [d for g in graphs for _, d in g.degree()]
    k = np.arange(0, n)
    emp = np.bincount(all_deg, minlength=n) / len(all_deg)
    theo = binom.pmf(k, n - 1, p)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(k, emp, color="#56B4E9", alpha=0.6, edgecolor="k", linewidth=0.5)
    ax.plot(k, theo, color="#D55E00", marker="o", linestyle="-", linewidth=2)
    ax.set_xlabel(r"Degree $k$")
    ax.set_ylabel(r"Probability $p(k)$")
    ax.set_title(rf"Degree Distribution: {label}")
    ax.legend([
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#56B4E9",
               markeredgecolor="k", markersize=10, label="Empirical"),
        Line2D([0], [0], marker="o", color="#D55E00", label="Binomial")
    ])
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "png" / f"degdist_{label}.png", dpi=300)
    plt.close(fig)


# ---------------- Main ---------------- #

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    ds_summary = load_dataset_summaries(DATASET_SUMMARY_PATH)
    all_files = list(args.results_dir.rglob("*samples*.npy"))
    model_dirs = sorted({f.parent for f in all_files})
    rows = [r for d in model_dirs if (r := process_model_dir(d, args.results_dir, ds_summary))]

    if not rows:
        logging.warning("No valid results found.")
        return

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    for n, grp in df.groupby("num_nodes"):
        out_csv = args.results_dir / f"analysis_summary_{n}N.csv"
        grp.to_csv(out_csv, index=False, float_format="%.4f")
        logging.info(f"Saved {len(grp)} rows → {out_csv}")

    make_scaling_plots(df)


if __name__ == "__main__":
    main()
