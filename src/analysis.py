import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import networkx as nx
import yaml

# Local imports
from src.utils.metrics import average_edge_prob
from src.utils.utils import setup_logging, vec_to_graph, graph_to_vec
from src.datasets.bipartites import BipartiteGraphDataset
from src.datasets.er import ErdosRenyiGraphDataset

# JAX-based MMD utilities
import jax.numpy as jnp
from iqpopt.gen_qml.sample_methods import mmd_loss_samples
from iqpopt.gen_qml.utils import median_heuristic

# ----------------- Constants ----------------- #

DEFAULT_RESULTS_DIR = Path("results")
DATASET_SUMMARY_PATH = Path("data/datasets_summary.yml")
TRAINING_DATA_DIR = Path("data/raw_data")

CSV_COLUMNS = [
    "run_id", "run_type", "num_nodes", "graph_type", "density_category",
    # density
    "generated_density", "density_value", "mmd_density",
    # var-degree-dist
    "generated_var_degree_dist", "var_degree_dist_value", "mmd_var_degree_dist",
    # triangles
    "generated_tri", "tri_value", "mmd_tri",
    # bipartite
    "generated_bipartite", "bipartite_value", "mmd_bipartite",
    # samples-level
    "mmd_samples",
    # memorization/diversity
    "memorized_adj_pct", "memorized_iso_pct",
    "diversity_adj_pct", "diversity_iso_pct",
]

# ----------------- Data Containers ----------------- #

class DatasetInfo:
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "")
        self.nodes = data.get("nodes", 0)
        self.graph_type = data.get("graph_type", "")
        self.density_category = data.get("density_category", "")
        self.density_value = data.get("density_value", 0.0)
        self.var_degree_dist_value = data.get("var_degree_dist_value", 0.0)
        self.tri_value = data.get("tri_value", 0.0)
        self.bipartite_value = data.get("bipartite_value", 0.0)

class MemorizationMetrics:
    def __init__(self, adj_pct=0.0, iso_pct=0.0):
        self.adj_pct = adj_pct
        self.iso_pct = iso_pct

class DiversityMetrics:
    def __init__(self, adj_pct=0.0, iso_pct=0.0):
        self.adj_pct = adj_pct
        self.iso_pct = iso_pct

# ----------------- Helpers ----------------- #

def extract_graph_type_and_density(name: str) -> Tuple[Optional[str], Optional[str]]:
    gt = None
    if "Bipartite" in name or "BP" in name: gt = "BP"
    elif "ER" in name: gt = "ER"
    dc = None
    for c in ["Sparse","Medium","Dense"]:
        if c in name: dc = c.lower()
    return gt, dc

def safe_sigma(ref: np.ndarray) -> float:
    """Compute σ = median_heuristic(ref); fallback to 1.0 if non-finite or ≤0."""
    sigma = median_heuristic(ref)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return float(sigma)

def scalar_mmd(gen: np.ndarray, ref: np.ndarray) -> float:
    """1D MMD between gen and ref, σ from ref only."""
    if len(ref)==0 or len(gen)==0:
        return 0.0
    g = gen.reshape(-1,1)
    r = ref.reshape(-1,1)
    sigma = safe_sigma(r)
    return float(mmd_loss_samples(jnp.array(r), jnp.array(g), sigma))

def sample_mmd(gen: np.ndarray, ref: np.ndarray) -> float:
    """MMD over adjacency vectors, σ from ref only."""
    if ref is None or len(ref)==0 or len(gen)==0:
        return 0.0
    # normalize by dimension
    d = gen.shape[1]
    g = gen.astype(float)/d
    r = ref.astype(float)/d
    sigma = safe_sigma(r)
    return float(mmd_loss_samples(jnp.array(r), jnp.array(g), sigma))

def calculate_memorization_metrics(generated: np.ndarray, training: Optional[np.ndarray], nodes: int) -> MemorizationMetrics:
    if training is None or len(generated)==0:
        return MemorizationMetrics()
    num = len(generated)
    adj_count = 0
    iso_count = 0
    for vec in generated:
        if any(np.array_equal(vec, t) for t in training):
            adj_count += 1
        try:
            g = vec_to_graph(vec, nodes)
            if any(nx.is_isomorphic(g, vec_to_graph(t, nodes)) for t in training):
                iso_count += 1
        except:
            pass
    return MemorizationMetrics(adj_count/num*100, iso_count/num*100)

def calculate_diversity_metrics(generated: np.ndarray, nodes: int) -> DiversityMetrics:
    n = len(generated)
    if n<=1:
        return DiversityMetrics(100.0, 100.0)
    unique_adj = len({tuple(v) for v in generated})
    adj_pct = unique_adj/n*100
    unique_iso = []
    for vec in generated:
        g = vec_to_graph(vec, nodes)
        if not any(nx.is_isomorphic(g, h) for h in unique_iso):
            unique_iso.append(g)
    iso_pct = len(unique_iso)/n*100
    return DiversityMetrics(adj_pct, iso_pct)

# ----------------- Loaders ----------------- #

def load_training_data(dataset_name: str, nodes: int) -> Optional[np.ndarray]:
    p = TRAINING_DATA_DIR / f"{dataset_name}.pkl"
    if not p.exists():
        return None
    gt, _ = extract_graph_type_and_density(dataset_name)
    if gt=="BP":
        ds = BipartiteGraphDataset.from_file(str(p))
    elif gt=="ER":
        ds = ErdosRenyiGraphDataset.from_file(str(p))
    else:
        return None
    if not getattr(ds, "graphs", None):
        return None
    vecs = []
    for g in ds.graphs:
        if g.number_of_nodes()==nodes:
            vecs.append(graph_to_vec(g, nodes))
    return np.array(vecs) if vecs else None

def load_dataset_summaries(path: Path) -> List[DatasetInfo]:
    with open(path,"r") as f:
        raw = yaml.safe_load(f)
    return [DatasetInfo(d) for d in raw]

# ----------------- Processing ----------------- #

def get_reliable_identifiers(model_dir: Path, base_dir: Path) -> Tuple[Optional[str],Optional[str]]:
    try:
        run_id = str(model_dir.relative_to(base_dir))
        parts = model_dir.parts
        if len(parts)>2:
            key = f"{parts[-3]}_{parts[-1]}"
            if re.match(r"\d+N_(?:Bipartite|ER)_(?:Dense|Medium|Sparse)", key):
                return run_id, key
        return run_id, None
    except:
        return None, None

def find_nodes_from_path(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)N_", str(path.resolve()))
    return int(m.group(1)) if m else None

def process_model_directory(
    model_dir: Path,
    base_results_dir: Path,
    ds_summaries: List[DatasetInfo]
) -> Optional[Dict[str,Any]]:
    run_id, ds_key = get_reliable_identifiers(model_dir, base_results_dir)
    if not run_id or not ds_key:
        return None
    nodes = find_nodes_from_path(model_dir)
    ds_info = next((d for d in ds_summaries if d.name==ds_key), None)
    if ds_info is None:
        return None

    # load generated samples
    files = list(model_dir.glob("*samples*.npy"))
    if not files:
        return None
    gen = np.vstack([np.load(f) for f in files])

    # per-graph feature lists
    dens_gen=[]; var_gen=[]; tri_gen=[]; bip_gen=[]
    num_edges = nodes*(nodes-1)/2
    for v in gen:
        g = vec_to_graph(v, nodes)
        dens_gen.append(len(g.edges())/num_edges)
        var_gen.append(np.var([d for _,d in g.degree()]))
        tri_gen.append(sum(nx.triangles(g).values())//3)
        bip_gen.append(1 if nx.is_bipartite(g) else 0)

    # reference distributions
    train = load_training_data(ds_key, nodes)
    dens_ref=[]; var_ref=[]; tri_ref=[]; bip_ref=[]
    if train is not None:
        for v in train:
            g = vec_to_graph(v, nodes)
            dens_ref.append(len(g.edges())/num_edges)
            var_ref.append(np.var([d for _,d in g.degree()]))
            tri_ref.append(sum(nx.triangles(g).values())//3)
            bip_ref.append(1 if nx.is_bipartite(g) else 0)

    # compute MMDs
    mmd_d = scalar_mmd(np.array(dens_gen), np.array(dens_ref)) if dens_ref else 0.0
    mmd_v = scalar_mmd(np.array(var_gen), np.array(var_ref)) if var_ref else 0.0
    mmd_t = scalar_mmd(np.array(tri_gen), np.array(tri_ref)) if tri_ref else 0.0
    mmd_b = scalar_mmd(np.array(bip_gen), np.array(bip_ref)) if bip_ref else 0.0
    mmd_s = sample_mmd(gen, train)

    # memorization/diversity
    mem = calculate_memorization_metrics(gen, train, nodes)
    div = calculate_diversity_metrics(gen, nodes)

    return {
        "run_id": run_id,
        "run_type": "Evaluation",
        "num_nodes": nodes,
        "graph_type": ds_info.graph_type,
        "density_category": ds_info.density_category,
        "generated_density": np.mean(dens_gen),
        "density_value": np.mean(dens_ref) if dens_ref else ds_info.density_value,
        "mmd_density": mmd_d,
        "generated_var_degree_dist": np.mean(var_gen),
        "var_degree_dist_value": np.mean(var_ref) if var_ref else ds_info.var_degree_dist_value,
        "mmd_var_degree_dist": mmd_v,
        "generated_tri": np.mean(tri_gen),
        "tri_value": np.mean(tri_ref) if tri_ref else ds_info.tri_value,
        "mmd_tri": mmd_t,
        "generated_bipartite": np.mean(bip_gen),
        "bipartite_value": np.mean(bip_ref) if bip_ref else ds_info.bipartite_value,
        "mmd_bipartite": mmd_b,
        "mmd_samples": mmd_s,
        "memorized_adj_pct": mem.adj_pct,
        "memorized_iso_pct": mem.iso_pct,
        "diversity_adj_pct": div.adj_pct,
        "diversity_iso_pct": div.iso_pct,
    }

def validate_results(df: pd.DataFrame) -> bool:
    miss = set(CSV_COLUMNS) - set(df.columns)
    if miss:
        logging.error(f"Missing columns: {miss}")
        return False
    return True

def main():
    setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--results-base-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()

    ds_summ = load_dataset_summaries(DATASET_SUMMARY_PATH)
    all_files = list(args.results_base_dir.rglob("*samples*.npy"))
    model_dirs = sorted({f.parent for f in all_files})
    rows=[]
    for d in model_dirs:
        r = process_model_directory(d, args.results_base_dir, ds_summ)
        if r: rows.append(r)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if not validate_results(df): return

    # split by node count
    for n, grp in df.groupby("num_nodes"):
        out = args.results_base_dir / f"analysis_summary_{n}N.csv"
        grp.to_csv(out, index=False, float_format="%.4f")
        logging.info(f"Saved {len(grp)} rows to {out}")

if __name__=="__main__":
    main()