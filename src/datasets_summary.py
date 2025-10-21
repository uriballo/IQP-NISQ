import argparse
import logging
from pathlib import Path
import re
import numpy as np
import yaml
import networkx as nx
from typing import List, Dict, Any
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.datasets.bipartites import BipartiteGraphDataset
from src.datasets.er import ErdosRenyiGraphDataset
from src.utils.metrics import estrada_bipartivity  


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_dataset_paths() -> List[Path]:
    """Finds all dataset .pkl files in the data/raw_data/ directory."""
    data_dir = Path("data/raw_data")
    if not data_dir.is_dir():
        print(f"Error: Directory not found at '{data_dir.resolve()}'")
        return []
    return sorted(list(data_dir.glob("*.pkl")))


def parse_filename(path: Path) -> Dict[str, Any]:
    """Parses dataset parameters from a filename like '10N_Bipartite_Dense.pkl'."""
    match = re.match(r"(\d+)N_(Bipartite|ER)_(.*)", path.stem)
    if not match:
        return None
    nodes, graph_type, density_category = match.groups()
    return {"nodes": int(nodes), "graph_type": graph_type, "density_category": density_category}


def load_dataset_from_path(path: Path) -> Any:
    """Loads a dataset object from the correct class."""
    params = parse_filename(path)
    if not params:
        print(f"Warning: Could not parse filename: {path.name}")
        return None

    dataset_type = params["graph_type"]
    try:
        if dataset_type == "Bipartite":
            return BipartiteGraphDataset.from_file(str(path))
        elif dataset_type == "ER":
            return ErdosRenyiGraphDataset.from_file(str(path))
        else:
            print(f"Warning: Unknown dataset type '{dataset_type}' for {path.name}")
            return None
    except Exception as e:
        print(f"ERROR: Failed to load dataset {path.name}. Error: {e}")
        return None


# -----------------------------------------------------------------------------
# Graph analysis core
# -----------------------------------------------------------------------------
def process_graphs(graphs: List[nx.Graph]) -> Dict[str, float]:
    """Compute density, bipartiteness, and other basic stats."""
    num_graphs = len(graphs)
    if num_graphs == 0:
        return {
            "average_density": 0.0,
            "bipartite_percentage": 0.0,
            "average_bipartiteness": 0.0,
        }

    density_sum = 0.0
    bipartite_count = 0
    bipartiteness_sum = 0.0

    for i, graph in enumerate(graphs):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{num_graphs} graphs processed")

        # Basic density
        density_sum += nx.density(graph)

        # True bipartite check
        if nx.is_bipartite(graph):
            bipartite_count += 1

        bipartiteness_sum += estrada_bipartivity(graph)


    # Averages
    density_value = density_sum / num_graphs
    bipartite_value = (bipartite_count / num_graphs) * 100
    bipartiteness_value = bipartiteness_sum / num_graphs

    return {
        "average_density": density_value,
        "bipartite_percentage": bipartite_value,
        "average_bipartiteness": bipartiteness_value,
    }


# -----------------------------------------------------------------------------
# Dataset-level analysis
# -----------------------------------------------------------------------------
def analyze_dataset_selective(dataset_path: Path) -> dict:
    """Analyze one dataset, computing summary statistics."""
    print(f"Analyzing: {dataset_path.name}")
    params = parse_filename(dataset_path)
    if not params:
        return None

    dataset_obj = load_dataset_from_path(dataset_path)
    if not dataset_obj:
        return None

    try:
        graphs = dataset_obj.graphs
        if not graphs or not isinstance(graphs[0], nx.Graph):
            raise AttributeError
    except AttributeError:
        print(f"Error: {dataset_path.name} has no valid '.graphs' attribute.")
        return None

    num_elements = len(graphs)

    # Compute stats
    stats = process_graphs(graphs)
    del dataset_obj, graphs
    gc.collect()

    result = {
        "name": dataset_path.stem,
        "nodes": params["nodes"],
        "graph_type": params["graph_type"],
        "density_category": params["density_category"],
        "num_samples": num_elements, 
    }
    result.update({k: float(v) for k, v in stats.items()})
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """Analyze all datasets in data/raw_data and write summary YAML."""
    parser = argparse.ArgumentParser(description="Dataset analyzer with bipartiteness metrics")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    args = parser.parse_args()

    dataset_paths = get_dataset_paths()
    if not dataset_paths:
        print("No datasets found in 'data/raw_data/'.")
        return

    print(f"Found {len(dataset_paths)} datasets. Starting analysis...")

    all_stats = []
    if args.parallel:
        with ProcessPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
            futures = {executor.submit(analyze_dataset_selective, path): path for path in dataset_paths}
            for future in as_completed(futures):
                stats = future.result()
                if stats:
                    all_stats.append(stats)
    else:
        for path in dataset_paths:
            stats = analyze_dataset_selective(path)
            if stats:
                all_stats.append(stats)
            gc.collect()

    # Sort for readability
    all_stats.sort(key=lambda s: (s["nodes"], s["graph_type"], s["name"]))

    # Save to YAML
    summary_file = Path("data/datasets_summary.yml")
    with open(summary_file, "w") as f:
        yaml.dump(all_stats, f, indent=2, sort_keys=False)

    print(f"\n✅ Analysis complete. Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
