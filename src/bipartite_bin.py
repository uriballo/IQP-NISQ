#!/usr/bin/env python3
"""
Compute per-density-bin bipartite accuracy for the selected "best" model runs.

Selection: exactly one run per (nodes, density_category).
Default selection order:
  1) highest gen_bipartite_percent
  2) highest gen_bipartivity
  3) lowest mmd

Output CSV columns:
  dataset_name, nodes, density_category, run_path, num_samples,
  bin_idx, bin_left, bin_right, bin_count, bipartite_count, bipartite_percent, mean_density_in_bin

Usage:
  python scripts/compute_per_bin_bp_accuracy.py \
    --best_csv results/analysis/nisq/best_runs.csv \
    --summary data/datasets_summary.yml \
    --out results/analysis/per_bin_bipartite_accuracy.csv \
    --bins 10
"""
import argparse
import yaml
import numpy as np
import pandas as pd
import gc
import networkx as nx
import re
from src.datasets.bipartites import BipartiteGraphDataset
from src.datasets.er import ErdosRenyiGraphDataset
from src.utils.utils import vec_to_graph
from pathlib import Path

def load_dataset_summary(summary_path: Path) -> dict:
    with open(summary_path, "r") as f:
        summary_list = yaml.safe_load(f)
    # map by name for quick lookup
    return {d["name"]: d for d in summary_list}

def compute_reference_pooled_densities(dataset_summary: dict, verbose=True):
    """
    Load all Bipartite reference datasets and return pooled array of densities.
    """
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
            for g in graphs:
                pooled.append(nx.density(g))
            # free
            del ds, graphs
            gc.collect()
        except Exception as e:
            print(f"[ERROR] loading reference {pkl_path}: {e}")
            continue
    if len(pooled) == 0:
        raise RuntimeError("No reference densities found for Bipartite datasets.")
    return np.asarray(pooled, dtype=float)

def compute_per_bin_for_run(samples_npy: Path, nodes: int, bin_edges: np.ndarray):
    """
    Load generated samples (.npy of vectors), convert to graphs, compute per-sample density and bipartiteness,
    and return arrays: bin_idx (0..nbins-1), densities, is_bipartite (bool)
    """
    samples_vec = np.load(str(samples_npy))
    # Ensure samples_vec shape is (n_samples, vector_len). If it's a single sample vector, make 2D.
    if samples_vec.ndim == 1:
        samples_vec = samples_vec[None, :]
    n_samples = samples_vec.shape[0]

    # convert to graphs
    samples_graphs = [vec_to_graph(samples_vec[i], nodes) for i in range(n_samples)]

    # per-sample densities and bipartite flag
    densities = np.array([nx.density(g) for g in samples_graphs], dtype=float)
    is_bip = np.array([1 if nx.is_bipartite(g) else 0 for g in samples_graphs], dtype=int)

    # assign bins: bin i corresponds to [edges[i], edges[i+1]) for i=0..nbins-2, last bin includes right edge
    bin_idx = np.digitize(densities, bin_edges, right=False) - 1
    # fix edge-cases: digitize returns nbins for values == bin_edges[-1], clip to nbins-1
    nbins = len(bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    # free graphs
    del samples_graphs
    gc.collect()

    return densities, is_bip, bin_idx

def infer_nodes_and_density_from_name(dataset_name: str):
    """
    Try to infer nodes and density_category from dataset_name if summary lookup fails.
    Expected pattern: '<nodes>N_<Type>_<Density>' e.g. '8N_Bipartite_Sparse'
    """
    m = re.match(r"(\d+)N_.*_(Dense|Medium|Sparse)$", dataset_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)

def select_best_runs(df_best: pd.DataFrame, dataset_summary: dict):
    """
    Return a DataFrame with exactly one row per (nodes, density_category) chosen by the selection policy.
    """
    # Add nodes and density_category columns from dataset_summary (if available)
    def lookup_info(row):
        name = row.get('dataset_name')
        info = dataset_summary.get(name)
        if info:
            return info.get('nodes'), info.get('density_category')
        else:
            return infer_nodes_and_density_from_name(name)

    nodes_list = []
    dens_cat_list = []
    for _, r in df_best.iterrows():
        n, d = lookup_info(r)
        nodes_list.append(n)
        dens_cat_list.append(d)
    df_best = df_best.copy()
    df_best['nodes'] = nodes_list
    df_best['density_category'] = dens_cat_list

    # filter only rows where we have nodes and density_category (we need both to group)
    df_best = df_best[~df_best['nodes'].isna() & ~df_best['density_category'].isna()].copy()
    if df_best.empty:
        raise SystemExit("[ERROR] No valid rows with nodes & density_category found in best_runs.csv")

    # Only Bipartite datasets
    df_best = df_best[df_best['dataset_name'].str.contains("Bipartite", na=False)].copy()
    if df_best.empty:
        raise SystemExit("[ERROR] No Bipartite rows found in best_runs.csv")

    # Determine sort priority columns depending on availability
    sort_cols = []
    ascending = []

    if 'gen_bipartite_percent' in df_best.columns:
        sort_cols.append('gen_bipartite_percent')
        ascending.append(False)  # higher better
    if 'gen_bipartivity' in df_best.columns:
        sort_cols.append('gen_bipartivity')
        ascending.append(False)  # higher better
    if 'mmd' in df_best.columns:
        sort_cols.append('mmd')
        ascending.append(True)   # lower better

    if not sort_cols:
        # fallback: use num_samples if nothing else
        sort_cols = ['num_samples']
        ascending = [False]

    # Sort by grouping keys then our selection criteria (so best is first)
    df_best = df_best.sort_values(by=['nodes', 'density_category'] + sort_cols, ascending=[True, True] + ascending)
    # Now drop duplicates keeping first occurrence of each (nodes,density_category)
    selected = df_best.drop_duplicates(subset=['nodes', 'density_category'], keep='first').copy()

    # Reindex and return
    selected = selected.reset_index(drop=True)
    return selected

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_csv", type=Path, default=Path("results/analysis/nisq/best_runs.csv"))
    p.add_argument("--summary", type=Path, default=Path("data/datasets_summary.yml"))
    p.add_argument("--out", type=Path, default=Path("results/analysis/per_bin_bipartite_accuracy.csv"))
    p.add_argument("--bins", type=int, default=100, help="Number of density bins (global).")
    args = p.parse_args()

    if not args.best_csv.exists():
        raise SystemExit(f"[ERROR] best_csv not found: {args.best_csv}")
    if not args.summary.exists():
        raise SystemExit(f"[ERROR] summary not found: {args.summary}")

    dataset_summary = load_dataset_summary(args.summary)
    df_best = pd.read_csv(args.best_csv)

    # Select exactly one best run per (nodes, density_category)
    selected_df = select_best_runs(df_best, dataset_summary)
    print(f"[INFO] Selected {len(selected_df)} best runs (one per nodes x density_category).")

    # Build pooled reference densities and global bin edges (for Bipartite refs)
    print("[INFO] Loading reference Bipartite datasets to compute pooled densities for bin edges...")
    pooled_ref = compute_reference_pooled_densities(dataset_summary)
    bin_edges = np.histogram_bin_edges(pooled_ref, bins=args.bins)
    print(f"[INFO] Using {len(bin_edges)-1} bins with edges: {bin_edges}")

    records = []
    for idx, row in selected_df.iterrows():
        dataset_name = row['dataset_name']
        run_path = Path(row['path'])
        if not run_path.exists():
            print(f"[WARN] samples .npy not found for {dataset_name}: {run_path} — skipping")
            continue

        # lookup nodes and density_category from summary (should exist)
        ref_info = dataset_summary.get(dataset_name)
        if ref_info is None:
            # try to infer
            nodes, density_category = infer_nodes_and_density_from_name(dataset_name)
            if nodes is None:
                print(f"[WARN] no summary and cannot infer nodes for {dataset_name} — skipping")
                continue
        else:
            nodes = int(ref_info['nodes'])
            density_category = ref_info.get('density_category', '')

        # compute per-sample metrics and bins
        try:
            densities, is_bip, bin_idx = compute_per_bin_for_run(run_path, nodes, bin_edges)
        except Exception as e:
            print(f"[ERROR] failed to process samples for {dataset_name} at {run_path}: {e}")
            continue

        nbins = len(bin_edges) - 1
        n_samples = densities.shape[0]

        for b in range(nbins):
            mask = (bin_idx == b)
            cnt = int(mask.sum())
            if cnt == 0:
                # still record empty bin for completeness (count=0)
                rec = {
                    "dataset_name": dataset_name,
                    "nodes": nodes,
                    "density_category": density_category,
                    "run_path": str(run_path),
                    "num_samples": n_samples,
                    "bin_idx": int(b),
                    "bin_left": float(bin_edges[b]),
                    "bin_right": float(bin_edges[b + 1]),
                    "bin_count": cnt,
                    "bipartite_count": 0,
                    "bipartite_percent": float("nan"),
                    "mean_density_in_bin": float("nan"),
                }
            else:
                bip_cnt = int(is_bip[mask].sum())
                bip_pct = float(bip_cnt) / float(cnt) * 100.0
                mean_d = float(densities[mask].mean())
                rec = {
                    "dataset_name": dataset_name,
                    "nodes": nodes,
                    "density_category": density_category,
                    "run_path": str(run_path),
                    "num_samples": n_samples,
                    "bin_idx": int(b),
                    "bin_left": float(bin_edges[b]),
                    "bin_right": float(bin_edges[b + 1]),
                    "bin_count": cnt,
                    "bipartite_count": bip_cnt,
                    "bipartite_percent": bip_pct,
                    "mean_density_in_bin": mean_d,
                }
            records.append(rec)

        # small memory cleanup
        del densities, is_bip, bin_idx
        gc.collect()
        print(f"[INFO] processed {dataset_name} ({n_samples} samples) → recorded {nbins} bins")

    # Make DataFrame and save
    out_df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, float_format="%.6f", na_rep="NA")
    print(f"[INFO] Wrote per-bin bipartite accuracy CSV → {args.out}")
    print(f"[INFO] Rows: {len(out_df)} (selected_runs * bins)")

if __name__ == "__main__":
    main()
