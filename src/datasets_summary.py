import argparse
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd
import yaml
import networkx as nx
from typing import List, Dict, Any, Generator
import gc
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

from src.datasets.bipartites import BipartiteGraphDataset
from src.datasets.er import ErdosRenyiGraphDataset

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
    return {
        "nodes": int(nodes), 
        "graph_type": graph_type, 
        "density_category": density_category
    }

def load_dataset_from_path(path: Path) -> Any:
    """Dynamically loads a dataset object from a file path using the correct classes."""
    params = parse_filename(path)
    if not params:
        print(f"Warning: Could not parse filename: {path.name}")
        return None

    dataset_type = params["graph_type"]
    
    try:
        print(f"  -> Loading {path.name} using {dataset_type} class...")
        if dataset_type == "Bipartite":
            return BipartiteGraphDataset.from_file(str(path))
        elif dataset_type == "ER":
            return ErdosRenyiGraphDataset.from_file(str(path))
        else:
            print(f"Warning: Unknown dataset type '{dataset_type}' for {path.name}")
            return None
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {path.resolve()} using its class. Error: {e}")
        return None

def count_triangles_fast(graph: nx.Graph) -> float:
    """Fast triangle counting using NetworkX's optimized method."""
    return sum(nx.triangles(graph).values()) // 3

def count_4_cycles_fast(graph: nx.Graph) -> float:
    """Fast 4-cycle (square) counting using matrix multiplication."""
    try:
        adj = nx.adjacency_matrix(graph, dtype=np.int32)
        adj_squared = adj @ adj
        
        # Count 4-cycles: Tr(A^4) - 2*edges - sum(deg_i^2)  
        trace_a4 = (adj_squared @ adj_squared).diagonal().sum()
        two_edges = 2 * graph.number_of_edges()
        deg_squared_sum = sum(d*d for _, d in graph.degree())
        
        cycles_4 = (trace_a4 - two_edges - deg_squared_sum) // 8
        return max(0, cycles_4)
    except Exception as e:
        print(f"    Warning: 4-cycle counting failed: {e}")
        return 0.0

def count_cycles_8n_intensive(graph: nx.Graph) -> Dict[str, float]:
    """
    INTENSIVE 8-cycle analysis specifically for 8-node graphs.
    No limits, no timeouts - compute ALL cycles up to 8-cycles no matter the cost.
    """
    cycle_counts = {}
    max_length = 8
    
    # Initialize counts up to 8-cycles
    for length in range(3, max_length + 1):
        cycle_counts[f"{length}_cycle"] = 0.0
    
    if graph.number_of_nodes() < 3:
        cycle_counts.update({
            "odd_cycles_total": 0.0,
            "even_cycles_total": 0.0,
            "total_cycles": 0.0,
            "odd_even_ratio": 0.0,
            "odd_total_ratio": 0.0
        })
        return cycle_counts
    
    print(f"    🔥 INTENSIVE 8N ANALYSIS: Computing ALL cycles up to 8-cycles (no limits)")
    
    try:
        # Fast triangle counting
        cycle_counts["3_cycle"] = count_triangles_fast(graph)
        print(f"       3-cycles (triangles): {cycle_counts['3_cycle']}")
        
        # Fast 4-cycle counting
        cycle_counts["4_cycle"] = count_4_cycles_fast(graph)
        print(f"       4-cycles (squares): {cycle_counts['4_cycle']}")
        
        # INTENSIVE cycle counting for 5-8 cycles - NO LIMITS
        directed_g = graph.to_directed()
        cycle_count = 0
        
        # NO LIMITS for 8N analysis
        max_cycles_to_check = float('inf')  # No cycle limit
        timeout = float('inf')  # No timeout
        
        start_time = time.time()
        cycles_by_length = {length: 0 for length in range(5, 9)}
        
        print(f"       Starting exhaustive cycle search for 5-8 cycles...")
        
        try:
            for cycle in nx.simple_cycles(directed_g):
                cycle_length = len(cycle)
                if 5 <= cycle_length <= 8:
                    cycles_by_length[cycle_length] += 1
                    
                cycle_count += 1
                
                # Progress reporting for long computations
                if cycle_count % 100000 == 0:
                    elapsed = time.time() - start_time
                    print(f"       Progress: {cycle_count:,} cycles processed in {elapsed:.1f}s")
                    for length in range(5, 9):
                        if cycles_by_length[length] > 0:
                            print(f"         {length}-cycles found so far: {cycles_by_length[length] // 2}")
            
            # Convert counts and divide by 2 for undirected graph
            for length in range(5, 9):
                cycle_counts[f"{length}_cycle"] = cycles_by_length[length] / 2.0
                print(f"       {length}-cycles (final): {cycle_counts[f'{length}_cycle']}")
            
            elapsed = time.time() - start_time
            print(f"    ✅ INTENSIVE ANALYSIS COMPLETE: {cycle_count:,} total cycles processed in {elapsed:.1f}s")
                    
        except Exception as e:
            print(f"    Warning: Intensive cycle counting interrupted: {e}")
        
        del directed_g
            
    except Exception as e:
        print(f"    Warning: Could not complete intensive cycle counting: {e}")
    
    # Calculate aggregate metrics
    odd_total = sum(cycle_counts.get(f"{length}_cycle", 0.0) 
                   for length in range(3, max_length + 1, 2))
    
    even_total = sum(cycle_counts.get(f"{length}_cycle", 0.0) 
                    for length in range(4, max_length + 1, 2))
    
    total_cycles = odd_total + even_total
    
    # Calculate ratios
    odd_even_ratio = odd_total / even_total if even_total > 0 else (999.0 if odd_total > 0 else 0.0)
    odd_total_ratio = odd_total / total_cycles if total_cycles > 0 else 0.0
    
    cycle_counts.update({
        "odd_cycles_total": float(odd_total),
        "even_cycles_total": float(even_total),
        "total_cycles": float(total_cycles),
        "odd_even_ratio": float(odd_even_ratio) if not np.isinf(odd_even_ratio) else 999.0,
        "odd_total_ratio": float(odd_total_ratio)
    })
    
    return cycle_counts

def get_no_cycle_data() -> Dict[str, float]:
    """
    Return empty cycle data for datasets where we skip cycle analysis.
    """
    cycle_data = {}
    
    # Set all cycle counts to 0
    for length in range(3, 9):  # 3 to 8 for display consistency
        cycle_data[f"{length}_cycle"] = 0.0
    
    cycle_data.update({
        "odd_cycles_total": 0.0,
        "even_cycles_total": 0.0,
        "total_cycles": 0.0,
        "odd_even_ratio": 0.0,
        "odd_total_ratio": 0.0
    })
    
    return cycle_data

def process_graphs_selective(graphs: List[nx.Graph], is_8n: bool = False) -> Dict[str, float]:
    """
    Process graphs with selective strategy: 
    - 8N: intensive 8-cycle analysis
    - Others: NO cycle analysis, only basic metrics
    """
    num_graphs = len(graphs)
    
    # Initialize accumulators for basic metrics
    density_sum = 0.0
    triangle_sum = 0.0
    bipartite_count = 0
    degree_sum = 0.0
    degree_sq_sum = 0.0
    degree_count = 0
    
    # Initialize cycle count accumulators (only used for 8N)
    if is_8n:
        cycle_accumulators = {}
        for length in range(3, 9):  # 3 to 8
            cycle_accumulators[f"{length}_cycle"] = 0.0
        
        cycle_accumulators.update({
            "odd_cycles_total": 0.0,
            "even_cycles_total": 0.0,
            "total_cycles": 0.0,
            "odd_even_ratio": 0.0,
            "odd_total_ratio": 0.0
        })
    else:
        # For non-8N, use empty cycle data
        cycle_accumulators = get_no_cycle_data()
    
    # Process each graph
    for i, graph in enumerate(graphs):
        if (i + 1) % 10 == 0 or is_8n:
            print(f"    Progress: {i + 1}/{num_graphs} graphs processed")
        
        # Basic metrics (always computed)
        density_sum += nx.density(graph)
        
        if nx.is_bipartite(graph):
            bipartite_count += 1
        
        # Degree statistics (always computed)
        degrees = [d for n, d in graph.degree()]
        for degree in degrees:
            degree_sum += degree
            degree_sq_sum += degree * degree
            degree_count += 1
        
        if is_8n:
            # ONLY for 8N: Intensive cycle analysis
            cycle_counts = count_cycles_8n_intensive(graph)
            for key, value in cycle_counts.items():
                if key in cycle_accumulators:
                    cycle_accumulators[key] += value
            
            # Use triangle count from cycle counting
            triangle_sum += cycle_counts.get("3_cycle", 0.0)
            del cycle_counts
        else:
            # For non-8N: Only compute triangles for the triangle metric, no cycle data stored
            triangle_sum += count_triangles_fast(graph)
            print(f"    Skipping cycle analysis for non-8N dataset")
        
        del degrees
    
    # Calculate averages for basic metrics
    density_value = density_sum / num_graphs
    triangle_value = triangle_sum / num_graphs
    bipartite_value = (bipartite_count / num_graphs) * 100
    
    mean_degree = degree_sum / degree_count if degree_count > 0 else 0.0
    mean_degree_sq = degree_sq_sum / degree_count if degree_count > 0 else 0.0
    var_degree_dist_value = mean_degree_sq - (mean_degree * mean_degree)
    
    if is_8n:
        # Average cycle counts for 8N datasets
        for key in cycle_accumulators:
            if key.endswith('_cycle') or key.endswith('_total'):
                cycle_accumulators[key] /= num_graphs
        
        # Recalculate ratios from averaged totals
        odd_total = cycle_accumulators["odd_cycles_total"]
        even_total = cycle_accumulators["even_cycles_total"]
        total_cycles = odd_total + even_total
        
        cycle_accumulators["odd_even_ratio"] = (
            odd_total / even_total if even_total > 0 
            else (999.0 if odd_total > 0 else 0.0)
        )
        cycle_accumulators["odd_total_ratio"] = (
            odd_total / total_cycles if total_cycles > 0 else 0.0
        )
    
    return {
        "density_value": density_value,
        "var_degree_dist_value": var_degree_dist_value,
        "tri_value": triangle_value,
        "bipartite_value": bipartite_value,
        **cycle_accumulators
    }

def analyze_dataset_selective(dataset_path: Path) -> dict:
    """Dataset analysis with selective strategy: intensive 8N, basic for others."""
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
        print(f"Error: The loaded object for {dataset_path.name} lacks a valid '.graphs' attribute.")
        return None

    num_elements = len(graphs)
    dataset_name = dataset_path.stem
    max_nodes = params["nodes"]
    
    # Determine if this is an 8N dataset
    is_8n = (max_nodes == 8)
    
    if is_8n:
        analysis_type = "🔥 INTENSIVE 8N"
        print(f"  -> {analysis_type} analysis of {num_elements} graphs (8-cycles EXHAUSTIVE - no limits)")
    else:
        analysis_type = "BASIC ONLY"
        print(f"  -> {analysis_type} analysis of {num_elements} graphs with {max_nodes} nodes (NO cycle analysis)")
    
    # Process with selective strategy
    stats = process_graphs_selective(graphs, is_8n)
    
    del dataset_obj, graphs
    gc.collect()

    result = {
        "name": dataset_name,
        "nodes": params["nodes"], 
        "graph_type": params["graph_type"],
        "density_category": params["density_category"],
    }
    
    result.update({k: float(v) for k, v in stats.items()})
    return result

def main():
    """Main function with selective analysis: intensive 8N, basic for all others."""
    parser = argparse.ArgumentParser(description="Selective dataset analysis: intensive 8N only")
    parser.add_argument("--parallel", action="store_true", 
                       help="Use parallel processing")
    args = parser.parse_args()
    
    dataset_paths = get_dataset_paths()
    if not dataset_paths:
        print("No datasets found in 'data/raw_data/' to analyze.")
        return

    summary_file = Path("data/datasets_summary.yml")
    
    print("🎯 SELECTIVE ANALYSIS MODE:")
    print("  • 8N datasets: INTENSIVE 8-cycle analysis (exhaustive, no limits)")
    print("  • All other datasets: Basic metrics only (NO cycle analysis)")
    
    # Check if we have 8N datasets
    has_8n_datasets = any(parse_filename(path) and parse_filename(path)['nodes'] == 8 
                         for path in dataset_paths)
    if has_8n_datasets:
        print("🔥 8N datasets detected - intensive analysis will take significant time!")
    else:
        print("ℹ️  No 8N datasets found - only basic analysis will be performed")
    
    if args.parallel:
        # For 8N analysis, use fewer workers
        max_workers = 2 if has_8n_datasets else 4
        print(f"Using parallel processing with {max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=min(max_workers, multiprocessing.cpu_count())) as executor:
            futures = {executor.submit(analyze_dataset_selective, path): path for path in dataset_paths}
            all_stats = []
            
            for future in as_completed(futures):
                stats = future.result()
                if stats:
                    all_stats.append(stats)
    else:
        # Sequential processing
        all_stats = []
        for dataset_path in dataset_paths:
            stats = analyze_dataset_selective(dataset_path)
            if stats:
                all_stats.append(stats)
            gc.collect()
    
    all_stats.sort(key=lambda s: (s['nodes'], s['graph_type'], s['name']))
    
    # Display results - show cycle columns but they'll be 0 for non-8N
    print("\n" + "="*140)
    print("--- 📊 Selective Analysis: 8N Intensive + Others Basic Only ---")
    print("="*140)
    
    # Create header with cycle columns - separate odd and even
    base_header = f"{'Dataset Name':<28} | {'Type':<4} | {'Density':>9} | {'%Bip':>6} | {'Tri':>6}"
    
    # Cycles 3,5,7 (odd) and 4,6,8 (even)
    odd_header_str = "  3 |   5 |   7"
    even_header_str = "  4 |   6 |   8"
    ratio_header_str = "O/E | O/T"
    
    full_header = f"{base_header} | {odd_header_str} | {even_header_str} | {ratio_header_str}"
    print(full_header)
    print("-" * len(full_header))
    
    last_nodes = None
    for stats in all_stats:
        if last_nodes is not None and stats['nodes'] != last_nodes:
            print("-" * len(full_header))
        
        # Base stats
        base_row = (
            f"{stats['name']:<28} | "
            f"{stats['graph_type']:<4} | "
            f"{stats['density_value']:>9.4f} | "
            f"{stats['bipartite_value']:>5.1f}% | "
            f"{stats['tri_value']:>6.1f}"
        )
        
        # Cycle stats - will be 0 for non-8N datasets
        cycle_3 = stats.get("3_cycle", 0.0)
        cycle_5 = stats.get("5_cycle", 0.0)
        cycle_7 = stats.get("7_cycle", 0.0)
        odd_row = f"{cycle_3:>4.1f} | {cycle_5:>4.1f} | {cycle_7:>4.1f}"
        
        cycle_4 = stats.get("4_cycle", 0.0)
        cycle_6 = stats.get("6_cycle", 0.0)
        cycle_8 = stats.get("8_cycle", 0.0)
        even_row = f"{cycle_4:>4.1f} | {cycle_6:>4.1f} | {cycle_8:>4.1f}"
        
        # Ratio stats
        odd_even_ratio = stats.get('odd_even_ratio', 0.0)
        odd_total_ratio = stats.get('odd_total_ratio', 0.0)
        
        odd_even_str = f"{odd_even_ratio:>3.1f}" if odd_even_ratio < 99 else "∞" if odd_even_ratio > 0 else "0.0"
        ratio_row = f"{odd_even_str} | {odd_total_ratio:>3.2f}"
        
        # Mark 8N datasets with special indicator
        is_8n_dataset = stats['nodes'] == 8
        prefix = "🔥 " if is_8n_dataset else "   "
        suffix = " (BASIC ONLY)" if not is_8n_dataset else ""
        
        full_row = f"{prefix}{base_row} | {odd_row} | {even_row} | {ratio_row}{suffix}"
        print(full_row)
        last_nodes = stats['nodes']
    
    print("="*140)

    # Print interpretation help
    print(f"\n💡 Selective Analysis Results:")
    print("🔥 8N datasets: Complete intensive 8-cycle analysis performed")
    print("   Other datasets: BASIC metrics only (cycle columns show 0)")
    print("- For 8N datasets:")
    print("  • ODD cycles (3,5,7): Bipartite graphs should have ZERO")
    print("  • EVEN cycles (4,6,8): Both bipartite and ER graphs can have these")
    print("  • O/E ratio: Odd/Even cycle ratio (should be 0.0 for bipartite)")
    print("  • O/T ratio: Odd/Total cycle ratio (should be 0.0 for bipartite)")
    print("- For non-8N datasets: Only density, bipartiteness %, triangles, and degree variance computed")

    results_dir = Path("data")
    results_dir.mkdir(exist_ok=True)
    
    with open(summary_file, "w") as f:
        yaml.dump(all_stats, f, indent=2, sort_keys=False)
        
    print(f"\n✅ Selective analysis complete! Results saved to: {summary_file}")

if __name__ == "__main__":
    main()