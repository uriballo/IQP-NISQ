import numpy as np
import networkx as nx
from pathlib import Path
from src.utils.utils import vec_to_graph
from src.datasets.bipartites import BipartiteGraphDataset

NUM_SAMPLES = 1000000 
TARGET_NODES = {8, 10, 14, 18}  # only these node counts

def compute_baseline_bp(dataset_path: str, density: float, num_samples: int = NUM_SAMPLES) -> float:
    """
    Generates random graphs using the dataset's density and computes
    the bipartite percentage.
    """
    dataset = BipartiteGraphDataset(1, 0.1).from_file(dataset_path, verbose=False)
    N = dataset.nodes
    L = N * (N - 1) // 2  # strict upper triangle length

    bipartite_count = 0
    for _ in range(num_samples):
        random_vec = (np.random.rand(L) < density).astype(np.uint8)
        graph = vec_to_graph(random_vec, N)
        if nx.is_bipartite(graph):
            bipartite_count += 1

    return bipartite_count / num_samples * 100  # percentage

def run_baseline_bp_for_all_datasets(dataset_summary_path: Path):
    import yaml
    with open(dataset_summary_path, "r") as f:
        datasets = yaml.safe_load(f)

    results = {}
    for ds in datasets:
        # Only bipartite datasets with target node counts
        if ds["graph_type"] != "Bipartite" or ds["nodes"] not in TARGET_NODES:
            continue

        name = ds["name"]
        node_count = ds["nodes"]
        density_category = ds["density_category"]
        density = ds["average_density"]

        dataset_file = f"data/raw_data/{node_count}N_Bipartite_{density_category}.pkl"
        dataset_file_path = Path(dataset_file)
        if not dataset_file_path.exists():
            print(f"[WARNING] Dataset file not found: {dataset_file}")
            continue

        bp_percent = compute_baseline_bp(str(dataset_file_path), density)
        results[name] = {
            "dataset_file": dataset_file,
            "density": density,
            "baseline_bp_percent": bp_percent,
        }
        print(f"{name}: Baseline BP% = {bp_percent:.2f}%")

    return results

if __name__ == "__main__":
    dataset_summary_path = Path("data/datasets_summary.yml")
    baseline_results = run_baseline_bp_for_all_datasets(dataset_summary_path)

    import pandas as pd
    df = pd.DataFrame.from_dict(baseline_results, orient="index")
    df.to_csv("results/analysis/baseline_bp.csv", index_label="dataset_name", float_format="%.4f")
    print("[INFO] Baseline bipartite percentages saved to results/analysis/baseline_bp.csv")