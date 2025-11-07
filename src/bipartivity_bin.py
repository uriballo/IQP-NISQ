#!/usr/bin/env python3
"""
Compute per-density-bin mean bipartivity for the selected "best" model runs.

Selection: exactly one run per (nodes, density_category).
Default selection order:
  1) highest gen_bipartite_percent
  2) highest gen_bipartivity
  3) lowest mmd

Output CSV columns:
  dataset_name, nodes, density_category, run_path, num_samples,
  bin_idx, bin_left, bin_right, bin_count,
  mean_density_in_bin, mean_bipartivity_in_bin

Usage:
  python scripts/compute_per_bin_bipartivity.py \
    --best_csv results/analysis/nisq/best_runs.csv \
    --summary data/datasets_summary.yml \
    --out results/analysis/per_bin_bipartivity.csv \
    --bins 10
"""
import argparse
import yaml
import numpy as np
import pandas as pd
import gc
import networkx as nx
import re
from pathlib import Path
from src.datasets.bipartites import BipartiteGraphDataset
from src.utils.utils import vec_to_graph
from src.utils.metrics import estrada_bipartivity  


# ------------------------------------------------------------
# Dataset summary loader
# ------------------------------------------------------------
def load_dataset_summary(summary_path: Path) -> dict:
    with open(summary_path, "r") as f:
        summary_list = yaml.safe_load(f)
    return {d["name"]: d for d in summary_list}


# ------------------------------------------------------------
# Reference pooled densities (for consistent global binning)
# ------------------------------------------------------------
def compute_reference_pooled_densities(dataset_summary: dict, verbose=True):
    pooled = []
    for name, info in dataset_summary.items():
        if info.get("graph_type") != "Bipartite":
            continue
        nodes = info["nodes"]
        dens_cat = info["density_category"]
        pkl_path = Path(f"data/raw_data/{nodes}N_Bipartite_{dens_cat}.pkl")
        if not pkl_path.exists():
            if verbose:
                print(f"[WARN] reference pkl not found {pkl_path} — skipping")
            continue
        try:
            ds = BipartiteGraphDataset.from_file(str(pkl_path))
            graphs = ds.graphs
            pooled.extend([nx.density(g) for g in graphs])
            del ds, graphs
            gc.collect()
        except Exception as e:
            print(f"[ERROR] loading reference {pkl_path}: {e}")
            continue
    if not pooled:
        raise RuntimeError("No reference densities found for Bipartite datasets.")
    return np.asarray(pooled, dtype=float)


# ------------------------------------------------------------
# Compute per-sample densities and bipartivity
# ------------------------------------------------------------
def compute_per_bin_for_run(samples_npy: Path, nodes: int, bin_edges: np.ndarray):
    """
    Load generated samples (.npy of adjacency vectors), convert to graphs,
    compute per-sample density and bipartivity (0–1),
    and return arrays: densities, bipartivities, bin_idx
    """
    samples_vec = np.load(str(samples_npy))
    if samples_vec.ndim == 1:
        samples_vec = samples_vec[None, :]
    n_samples = samples_vec.shape[0]

    # Convert to graphs
    samples_graphs = [vec_to_graph(samples_vec[i], nodes) for i in range(n_samples)]

    densities = np.array([nx.density(g) for g in samples_graphs], dtype=float)

    # Compute bipartivity values for each graph
    bipartivities = np.zeros(n_samples, dtype=float)
    for i, g in enumerate(samples_graphs):
        try:
            bipartivities[i] = float(estrada_bipartivity(g))
        except Exception as e:
            print(f"[WARN] Bipartivity computation failed for sample {i}: {e}")
            bipartivities[i] = np.nan

    # Assign bins
    bin_idx = np.digitize(densities, bin_edges, right=False) - 1
    nbins = len(bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    del samples_graphs
    gc.collect()
    return densities, bipartivities, bin_idx


# ------------------------------------------------------------
# Infer dataset metadata if missing
# ------------------------------------------------------------
def infer_nodes_and_density_from_name(dataset_name: str):
    m = re.match(r"(\d+)N_.*_(Dense|Medium|Sparse)$", dataset_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


# ------------------------------------------------------------
# Select best runs (one per nodes × density_category)
# ------------------------------------------------------------
def select_best_runs(df_best: pd.DataFrame, dataset_summary: dict):
    def lookup_info(row):
        name = row.get("dataset_name")
        info = dataset_summary.get(name)
        if info:
            return info.get("nodes"), info.get("density_category")
        else:
            return infer_nodes_and_density_from_name(name)

    nodes_list, dens_cat_list = [], []
    for _, r in df_best.iterrows():
        n, d = lookup_info(r)
        nodes_list.append(n)
        dens_cat_list.append(d)
    df_best = df_best.copy()
    df_best["nodes"] = nodes_list
    df_best["density_category"] = dens_cat_list

    df_best = df_best[~df_best["nodes"].isna() & ~df_best["density_category"].isna()].copy()
    if df_best.empty:
        raise SystemExit("[ERROR] No valid rows with nodes & density_category found in best_runs.csv")

    df_best = df_best[df_best["dataset_name"].str.contains("Bipartite", na=False)].copy()
    if df_best.empty:
        raise SystemExit("[ERROR] No Bipartite rows found in best_runs.csv")

    sort_cols, ascending = [], []
    if "gen_bipartite_percent" in df_best.columns:
        sort_cols.append("gen_bipartite_percent"); ascending.append(False)
    if "gen_bipartivity" in df_best.columns:
        sort_cols.append("gen_bipartivity"); ascending.append(False)
    if "mmd" in df_best.columns:
        sort_cols.append("mmd"); ascending.append(True)
    if not sort_cols:
        sort_cols = ["num_samples"]; ascending = [False]

    df_best = df_best.sort_values(by=["nodes", "density_category"] + sort_cols, ascending=[True, True] + ascending)
    selected = df_best.drop_duplicates(subset=["nodes", "density_category"], keep="first").copy()
    return selected.reset_index(drop=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_csv", type=Path, default=Path("results/analysis/nisq/best_runs.csv"))
    p.add_argument("--summary", type=Path, default=Path("data/datasets_summary.yml"))
    p.add_argument("--out", type=Path, default=Path("results/analysis/per_bin_bipartivity.csv"))
    p.add_argument("--bins", type=int, default=100)
    args = p.parse_args()

    if not args.best_csv.exists():
        raise SystemExit(f"[ERROR] best_csv not found: {args.best_csv}")
    if not args.summary.exists():
        raise SystemExit(f"[ERROR] summary not found: {args.summary}")

    dataset_summary = load_dataset_summary(args.summary)
    df_best = pd.read_csv(args.best_csv)
    selected_df = select_best_runs(df_best, dataset_summary)
    print(f"[INFO] Selected {len(selected_df)} best runs (one per nodes × density_category).")

    # Global bin edges based on reference datasets
    print("[INFO] Loading reference Bipartite datasets for global density bins...")
    pooled_ref = compute_reference_pooled_densities(dataset_summary)
    bin_edges = np.histogram_bin_edges(pooled_ref, bins=args.bins)
    print(f"[INFO] Using {len(bin_edges)-1} bins.")

    records = []
    for _, row in selected_df.iterrows():
        dataset_name = row["dataset_name"]
        run_path = Path(row["path"])
        if not run_path.exists():
            print(f"[WARN] Missing samples for {dataset_name}: {run_path} — skipping")
            continue

        ref_info = dataset_summary.get(dataset_name)
        if ref_info:
            nodes = int(ref_info["nodes"])
            density_category = ref_info.get("density_category", "")
        else:
            nodes, density_category = infer_nodes_and_density_from_name(dataset_name)
            if nodes is None:
                print(f"[WARN] Cannot infer nodes for {dataset_name}, skipping.")
                continue

        try:
            densities, bipartivities, bin_idx = compute_per_bin_for_run(run_path, nodes, bin_edges)
        except Exception as e:
            print(f"[ERROR] Failed processing {dataset_name}: {e}")
            continue

        nbins = len(bin_edges) - 1
        n_samples = len(densities)

        for b in range(nbins):
            mask = (bin_idx == b)
            cnt = int(mask.sum())
            if cnt == 0:
                rec = {
                    "dataset_name": dataset_name,
                    "nodes": nodes,
                    "density_category": density_category,
                    "run_path": str(run_path),
                    "num_samples": n_samples,
                    "bin_idx": b,
                    "bin_left": float(bin_edges[b]),
                    "bin_right": float(bin_edges[b + 1]),
                    "bin_count": cnt,
                    "mean_density_in_bin": np.nan,
                    "mean_bipartivity_in_bin": np.nan,
                }
            else:
                mean_d = float(np.nanmean(densities[mask]))
                mean_bip = float(np.nanmean(bipartivities[mask]))
                rec = {
                    "dataset_name": dataset_name,
                    "nodes": nodes,
                    "density_category": density_category,
                    "run_path": str(run_path),
                    "num_samples": n_samples,
                    "bin_idx": b,
                    "bin_left": float(bin_edges[b]),
                    "bin_right": float(bin_edges[b + 1]),
                    "bin_count": cnt,
                    "mean_density_in_bin": mean_d,
                    "mean_bipartivity_in_bin": mean_bip,
                }
            records.append(rec)

        del densities, bipartivities, bin_idx
        gc.collect()
        print(f"[INFO] Processed {dataset_name} ({n_samples} samples) → {nbins} bins")

    out_df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, float_format="%.6f", na_rep="NA")
    print(f"[INFO] Wrote per-bin bipartivity CSV → {args.out}")
    print(f"[INFO] Rows: {len(out_df)}")

if __name__ == "__main__":
    main()
