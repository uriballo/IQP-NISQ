import yaml
from pathlib import Path
import numpy as np
from jax import numpy as jnp
import pandas as pd
from collections import defaultdict
from src.utils.metrics import compute_average_density, compute_bipartite_percentage, compute_average_bipartivity, compute_memorization, compute_diversity
from src.utils.utils import vec_to_graph, graph_to_vec
from src.datasets.er import ErdosRenyiGraphDataset
from src.datasets.bipartites import BipartiteGraphDataset
from iqpopt.gen_qml.sample_methods import mmd_loss_samples
from iqpopt.gen_qml.utils import median_heuristic

def load_dataset_summary(summary_path: Path) -> dict:
    with open(summary_path, "r") as f:
        summary_list = yaml.safe_load(f)
    return {d["name"]: d for d in summary_list}

def process_all_runs(
    summary_path: Path,
    results_base_dir: Path,
    out_dir: Path = Path("results/analysis")
):
    dataset_summary = load_dataset_summary(summary_path)
    grouped_by_nodes = defaultdict(list)

    for base_folder in results_base_dir.iterdir():
        if not base_folder.is_dir():
            continue

        for run_folder in base_folder.iterdir():
            if not run_folder.is_dir():
                continue

            run_name = run_folder.name

            for density_folder_name in ["Dense", "Medium", "Sparse"]:
                density_folder = run_folder / density_folder_name
                if not density_folder.exists():
                    continue

                npy_files = list(density_folder.glob("*.npy"))
                if not npy_files:
                    continue
                npy_file = npy_files[0]

                dataset_name = f"{base_folder.name}_{density_folder_name}"
                if dataset_name not in dataset_summary:
                    print(f"[WARNING] No summary for {dataset_name}")
                    continue
                ref_info = dataset_summary[dataset_name]

                node_count = ref_info["nodes"]
                dataset_type = ref_info["graph_type"]
                density_category = ref_info["density_category"]

                dataset_path = f"data/raw_data/{node_count}N_{dataset_type}_{density_category}.pkl"

                if dataset_type == "Bipartite":
                    dataset = BipartiteGraphDataset(1, 0.1).from_file(dataset_path, verbose=False)
                    data_graphs = dataset.graphs
                    data_vecs = dataset.vectors
                else:
                    dataset = ErdosRenyiGraphDataset(1, 0.1).from_file(dataset_path, verbose=False)
                    data_graphs = dataset.graphs
                    data_vecs = dataset.vectors
                
                try:
                    samples_vec = np.load(npy_file)
                    samples_graph = [vec_to_graph(sample, node_count) for sample in samples_vec]
                except Exception as e:
                    print(f"[ERROR] Failed to load {npy_file}: {e}")
                    continue

                sigma = median_heuristic(data_vecs) 
                mmd = mmd_loss_samples(jnp.array(data_vecs, dtype=jnp.float32), jnp.array(samples_vec,  dtype=jnp.float32), sigma)

                grouped_by_nodes[ref_info["nodes"]].append({
                    "dataset_name": dataset_name,
                    "density": density_folder_name,
                    "path": str(npy_file),
                    "num_samples": samples_vec.shape[0],
                    "ref_density": ref_info["average_density"],
                    "ref_bipartite_percent": ref_info["bipartite_percentage"],
                    "ref_bipartivity": ref_info["average_bipartiteness"],
                    "gen_density": compute_average_density(samples_graph),
                    "gen_bipartite_percent": compute_bipartite_percentage(samples_graph),
                    "gen_bipartivity": compute_average_bipartivity(samples_graph),
                    "memorized": compute_memorization(samples_graph, data_graphs),
                    "diversity": compute_diversity(samples_graph),
                    "mmd": mmd,
                })

    # Export CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    fieldnames = [
        "dataset_name", "density", "ref_density", "ref_bipartite_percent", "ref_bipartivity",
        "gen_density", "gen_bipartite_percent", "gen_bipartivity",
        "path", "num_samples", "mmd", "memorized", "diversity"
    ]

    for nodes, records in grouped_by_nodes.items():
        if not records:
            continue
        csv_path = out_dir / f"analysis_{nodes}N.csv"
        df = pd.DataFrame(records)[fieldnames]
        df.to_csv(csv_path, index=False, float_format="%.4f", na_rep="NA")
        all_dfs.append(df)
        print(f"[INFO] Wrote {len(df)} entries → {csv_path}")

    return pd.DataFrame(), grouped_by_nodes, dataset_summary

# ---------------- Main ---------------- #

if __name__ == "__main__":
    summary_path = Path("data/datasets_summary.yml")
    results_base_dir = Path("results/evaluation_results")
    out_dir = Path("results/analysis/nisq")

    all_df, grouped_by_nodes, dataset_summary = process_all_runs(
        summary_path, results_base_dir, out_dir
    )