import pathlib
import re
import networkx as nx
import numpy as np
import yaml
import multiprocessing
from typing import List, Dict, Any
from datasets.bipartites import BipartiteGraphDataset
from datasets.er import ErdosRenyiGraphDataset

def get_dataset_paths() -> List[pathlib.Path]:
    """Finds all dataset .pkl files in the data/raw_data/ directory."""
    data_dir = pathlib.Path("data/raw_data")
    if not data_dir.is_dir():
        print(f"Error: Directory not found at '{data_dir.resolve()}'")
        return []
    return sorted(list(data_dir.glob("*.pkl")))

def parse_filename(path: pathlib.Path) -> Dict[str, Any]:
    """Parses dataset parameters from a filename like '10N_Bipartite_Dense.pkl'."""
    match = re.match(r"(\d+)N_(Bipartite|ER)_(.*)", path.name)
    if not match:
        return None
    
    nodes, type, _ = match.groups()
    return {"nodes": int(nodes), "type": type}

def load_dataset_from_path(path: pathlib.Path) -> Any:
    """Dynamically loads a dataset object from a file path."""
    params = parse_filename(path)
    if not params:
        print(f"Warning: Could not parse filename: {path.name}")
        return None

    nodes = params["nodes"]
    dataset_type = params["type"]
    
    if dataset_type == "Bipartite":
        loader = BipartiteGraphDataset(nodes=nodes, edge_prob=0.1)
    elif dataset_type == "ER":
        loader = ErdosRenyiGraphDataset(nodes=nodes, edge_prob=0.1)
    else:
        return None
        
    return loader.from_file(path)

def analyze_dataset(dataset_path: pathlib.Path) -> dict:
    """Loads a dataset and computes summary statistics."""
    print(f"Analyzing: {dataset_path.name}")
    params = parse_filename(dataset_path)
    dataset_obj = load_dataset_from_path(dataset_path)

    if not dataset_obj or not params:
        return None

    try:
        graphs = dataset_obj.graphs
    except AttributeError:
        print(f"Error: The loaded object for {dataset_path.name} lacks a '.graphs' attribute.")
        return None

    if not graphs or not isinstance(graphs[0], nx.Graph):
        print(f"Warning: Failed to get graphs from {dataset_path.name}.")
        return None

    num_elements = len(graphs)
    
    unique_hashes = {nx.weisfeiler_lehman_graph_hash(g) for g in graphs}
    num_unique = len(unique_hashes)
    
    bipartite_count = sum(1 for g in graphs if nx.is_bipartite(g))
    avg_density = np.mean([nx.density(g) for g in graphs]) if num_elements > 0 else 0
    
    return {
        "nodes": params["nodes"],
        "type": params["type"],
        "name": dataset_path.stem,
        "elements": int(num_elements),
        "unique_isomorphic": int(num_unique),
        "bipartite_percentage": float((bipartite_count / num_elements) * 100) if num_elements > 0 else 0.0,
        "average_edge_prob": float(avg_density),
    }

def main():
    """Main function to discover, analyze, print, and save a report on all datasets."""
    dataset_paths = get_dataset_paths()
    
    if not dataset_paths:
        print("No datasets found in 'data/raw_data/' to analyze.")
        return

    with multiprocessing.Pool() as pool:
        all_stats = pool.map(analyze_dataset, dataset_paths)
    
    all_stats = [stats for stats in all_stats if stats]
    
    all_stats.sort(key=lambda s: (s['nodes'], s['type']))
            
    # --- Print Report to Console ---
    print("\n--- 📊 Dataset Summary ---")
    print(f"{'Dataset Name':<28} | {'Total':>8} | {'Unique':>8} | {'% Bipartite':>12} | {'Avg Density':>12}")
    print("-" * 75)
    for stats in all_stats:
        print(
            f"{stats['name']:<28} | "
            f"{stats['elements']:>8} | "
            f"{stats['unique_isomorphic']:>8} | "
            f"{stats['bipartite_percentage']:>11.2f}% | "
            f"{stats['average_edge_prob']:>11.4f}"
        )
    print("-" * 75)

    results_dir = pathlib.Path("data")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "datasets_summary.yml"
    
    with open(output_path, "w") as f:
        yaml_stats = [
            {k: v for k, v in s.items() if k not in ['nodes', 'type']} 
            for s in all_stats
        ]
        yaml.dump(yaml_stats, f, indent=2, sort_keys=False)
        
    print(f"\n✅ Summary saved to: {output_path}")

if __name__ == "__main__":
    main()