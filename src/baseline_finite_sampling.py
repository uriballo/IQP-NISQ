#!/usr/bin/env python3
import numpy as np
import networkx as nx
import gc
import yaml
from pathlib import Path
from src.datasets.bipartites import BipartiteGraphDataset
from src.utils.utils import vec_to_graph
from src.utils.metrics import estrada_bipartivity

NUM_SAMPLES = 1000000  # number of random graphs per dataset
BINS = 100          # number of density bins

def load_dataset_summary(summary_path: Path):
    with open(summary_path, "r") as f:
        summary_list = yaml.safe_load(f)
    return {d["name"]: d for d in summary_list}

def compute_reference_pooled_densities(dataset_summary: dict):
    pooled = []
    for name, info in dataset_summary.items():
        if info.get("graph_type") != "Bipartite":
            continue
        nodes = info["nodes"]
        dens_cat = info["density_category"]
        pkl_path = Path(f"data/raw_data/{nodes}N_Bipartite_{dens_cat}.pkl")
        if not pkl_path.exists():
            continue
        try:
            ds = BipartiteGraphDataset.from_file(str(pkl_path))
            graphs = ds.graphs
            pooled.extend([nx.density(g) for g in graphs])
            del ds, graphs
            gc.collect()
        except Exception as e:
            print(f"[ERROR] loading {pkl_path}: {e}")
            continue
    if not pooled:
        raise RuntimeError("No reference densities found.")
    return np.array(pooled, dtype=float)

def compute_per_bin_baseline(dataset_path: str, nodes: int, density: float, bin_edges: np.ndarray, num_samples=NUM_SAMPLES):
    L = nodes * (nodes - 1) // 2
    bin_counts = np.zeros(len(bin_edges)-1, dtype=int)
    bip_counts = np.zeros(len(bin_edges)-1, dtype=int)
    bin_density_sum = np.zeros(len(bin_edges)-1, dtype=float)
    bipartivity_sum = np.zeros(len(bin_edges)-1, dtype=float)  # NEW: sum of bipartivity per bin

    for _ in range(num_samples):
        vec = (np.random.rand(L) < density).astype(np.uint8)
        g = vec_to_graph(vec, nodes)
        dens = nx.density(g)
        bidx = np.digitize(dens, bin_edges, right=False) - 1
        bidx = np.clip(bidx, 0, len(bin_edges)-2)
        bin_counts[bidx] += 1
        bip_counts[bidx] += int(nx.is_bipartite(g))
        bin_density_sum[bidx] += dens

        # Compute bipartivity and accumulate
        try:
            bipartivity_sum[bidx] += float(estrada_bipartivity(g))
        except Exception as e:
            print(f"[WARN] bipartivity computation failed: {e}")

    records = []
    for b in range(len(bin_edges)-1):
        cnt = bin_counts[b]
        if cnt == 0:
            mean_d = np.nan
            bip_pct = np.nan
            mean_bip = np.nan
        else:
            mean_d = bin_density_sum[b] / cnt
            bip_pct = bip_counts[b] / cnt * 100
            mean_bip = bipartivity_sum[b] / cnt  # NEW: mean bipartivity per bin
        records.append({
            "bin_idx": b,
            "bin_left": float(bin_edges[b]),
            "bin_right": float(bin_edges[b+1]),
            "bin_count": int(cnt),
            "bipartite_count": int(bip_counts[b]),
            "bipartite_percent": float(bip_pct),
            "mean_density_in_bin": float(mean_d),
            "mean_bipartivity_in_bin": float(mean_bip),  # NEW COLUMN
        })
    return records

def run_finite_baseline_bins(summary_path: Path, bins=BINS, num_samples=NUM_SAMPLES):
    dataset_summary = load_dataset_summary(summary_path)
    pooled_ref = compute_reference_pooled_densities(dataset_summary)
    bin_edges = np.histogram_bin_edges(pooled_ref, bins=bins)

    all_records = []
    for name, ds in dataset_summary.items():
        if ds["graph_type"] != "Bipartite":
            continue
        nodes = ds["nodes"]
        density = ds["average_density"]
        dataset_file = Path(f"data/raw_data/{nodes}N_Bipartite_{ds['density_category']}.pkl")
        if not dataset_file.exists():
            print(f"[WARN] Dataset not found: {dataset_file}")
            continue
        recs = compute_per_bin_baseline(str(dataset_file), nodes, density, bin_edges, num_samples=num_samples)
        for r in recs:
            all_records.append({**r, "dataset_name": name, "nodes": nodes, "density_category": ds["density_category"]})
        print(f"[INFO] Processed baseline for {name}")

    import pandas as pd
    df = pd.DataFrame(all_records)
    out_csv = Path("results/analysis/finite_baseline_per_bin.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f", na_rep="NA")
    print(f"[INFO] Saved per-bin finite baseline → {out_csv}")

if __name__ == "__main__":
    summary_path = Path("data/datasets_summary.yml")
    run_finite_baseline_bins(summary_path)
