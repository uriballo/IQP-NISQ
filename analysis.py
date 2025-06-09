import os
from tqdm import tqdm
from datasets.bipartites import BipartiteGraphDataset
from datasets.er import ErdosRenyiGraphDataset
import re
from datasets.utils import vec_to_graph
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

# =============================================================================
# Data Loading (Corrected)
# =============================================================================

# Define the base path for the datasets
datasets_path = "datasets/raw_data/"
samples_path = "results/samples/max_weights"

# --- Configuration for Programmatic Loading ---
load_config = {
    "nodes": [6, 7, 8, 10, 14, 18],
    "models": {
        "BP": {"class": BipartiteGraphDataset, "fname": "Bipartite"},
        "ER": {"class": ErdosRenyiGraphDataset, "fname": "ER"},
    },
    "densities": {"S": "Sparse", "M": "Medium", "D": "Dense"},
}

# This dictionary will hold all the loaded datasets
datasets = {}
print("--- Starting Dataset Loading from .pkl files ---")
combinations = [
    (n, m_key, d_key)
    for n in load_config["nodes"]
    for m_key in load_config["models"]
    for d_key in load_config["densities"]
]
for nodes, model_prefix, density_prefix in tqdm(
    combinations, desc="Loading datasets"
):
    model_info = load_config["models"][model_prefix]
    density_name = load_config["densities"][density_prefix]
    key = f"{model_prefix}{nodes}N{density_prefix}"
    filename = f"{nodes}N_{model_info['fname']}_{density_name}.pkl"
    full_path = os.path.join(datasets_path, filename)
    try:
        DatasetClass = model_info["class"]
        datasets[key] = (
            DatasetClass(nodes=nodes, edge_prob=0.1)
            .from_file(full_path)
            .graphs
        )
    except FileNotFoundError:
        print(f"\nWarning: File not found for '{key}'. Path: '{full_path}'")
    except Exception as e:
        print(f"\nError loading '{key}': {e}")

print(f"\n--- Dataset Loading Complete ---")
print(f"Total datasets loaded: {len(datasets)}")

# --- Sample Loading ---
graph_samples = {}
node_pattern = re.compile(r"(?:BP|ER)(\d+)N")
print(f"\n--- Starting Sample Loading from .npy files in '{samples_path}' ---")
try:
    sample_filenames = [
        f for f in os.listdir(samples_path) if f.endswith(".npy")
    ]
    for filename in tqdm(sample_filenames, desc="Loading samples"):
        match = node_pattern.search(filename)
        if not match:
            continue
        num_nodes = int(match.group(1))
        key = os.path.splitext(filename)[0]
        full_path = os.path.join(samples_path, filename)
        try:
            vectors = np.load(full_path)
            graph_samples[key] = [
                vec_to_graph(vec, num_nodes) for vec in vectors
            ]
        except Exception as e:
            print(f"\nError loading or processing '{filename}': {e}")
except FileNotFoundError:
    print(f"\nError: Directory not found at '{samples_path}'.")

print(f"\n--- Sample Loading Complete ---")
print(f"Total sample sets loaded: {len(graph_samples)}")


# =============================================================================
# Analysis and Testing Functions
# =============================================================================
def test_bipartite_proportion(graphs: list) -> float:
    if not graphs: return 0.0
    return sum(1 for g in graphs if nx.is_bipartite(g)) / len(graphs)

def test_non_isomorphic_proportion(graphs: list) -> float:
    if not graphs: return 0.0
    unique_hashes = {nx.weisfeiler_lehman_graph_hash(g) for g in graphs}
    return len(unique_hashes) / len(graphs)

def test_average_edge_probability(graphs: list, graph_type: str) -> float:
    if not graphs: return 0.0
    total_edges = sum(g.number_of_edges() for g in graphs)
    num_graphs, nodes = len(graphs), graphs[0].number_of_nodes()
    if graph_type == "ER": max_edges = nodes * (nodes - 1) / 2
    elif graph_type == "BP": max_edges = (nodes // 2) * (nodes - (nodes // 2))
    else: return 0.0
    if max_edges == 0: return 0.0
    return total_edges / (max_edges * num_graphs)

def test_average_components(graphs: list) -> float:
    if not graphs: return 0.0
    return sum(nx.number_connected_components(g) for g in graphs) / len(graphs)

def calculate_js_divergence(graphs1: list, graphs2: list) -> float:
    if not graphs1 or not graphs2: return np.nan
    degrees1 = [d for g in graphs1 for _, d in g.degree()]
    degrees2 = [d for g in graphs2 for _, d in g.degree()]
    max_degree = max(max(degrees1) if degrees1 else 0, max(degrees2) if degrees2 else 0)
    bins = np.arange(0, max_degree + 2)
    p, _ = np.histogram(degrees1, bins=bins, density=True)
    q, _ = np.histogram(degrees2, bins=bins, density=True)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m + 1e-10) + entropy(q, m + 1e-10))

# --- NEW ACADEMIC METRIC FUNCTIONS ---
def test_average_clustering(graphs: list) -> float:
    """Calculates the average clustering coefficient across a list of graphs."""
    if not graphs: return 0.0
    return np.mean([nx.average_clustering(g) for g in graphs])

def test_average_shortest_path(graphs: list) -> float:
    """Calculates the average shortest path length on the LCC of each graph."""
    if not graphs: return 0.0
    path_lengths = []
    for g in graphs:
        if nx.is_connected(g):
            path_lengths.append(nx.average_shortest_path_length(g))
        else:
            # Calculate on the largest connected component
            lcc_nodes = max(nx.connected_components(g), key=len)
            lcc = g.subgraph(lcc_nodes)
            path_lengths.append(nx.average_shortest_path_length(lcc))
    return np.mean(path_lengths)

def test_average_diameter(graphs: list) -> float:
    """Calculates the average diameter on the LCC of each graph."""
    if not graphs: return 0.0
    diameters = []
    for g in graphs:
        if nx.is_connected(g):
            diameters.append(nx.diameter(g))
        else:
            # Calculate on the largest connected component
            lcc_nodes = max(nx.connected_components(g), key=len)
            lcc = g.subgraph(lcc_nodes)
            diameters.append(nx.diameter(lcc))
    return np.mean(diameters)

def test_average_spectral_gap(graphs: list) -> float:
    """Calculates the average spectral gap (algebraic connectivity)."""
    if not graphs: return 0.0
    spectral_gaps = []
    for g in graphs:
        if g.number_of_nodes() > 1:
            # The second smallest eigenvalue of the Laplacian
            gap = nx.laplacian_spectrum(g)[1]
            spectral_gaps.append(gap)
    return np.mean(spectral_gaps) if spectral_gaps else 0.0


# =============================================================================
# Main Analysis Loop to Build and Print Table
# =============================================================================
print("\n--- Running Comprehensive Analysis and Building Results Table ---")

results_list = []

for sample_key in tqdm(sorted(graph_samples.keys()), desc="Analyzing samples"):
    source_type = "Simulated" if "_simulated" in sample_key else "Hardware"
    truth_key = sample_key.replace("_simulated", "")
    if truth_key not in datasets: continue

    sample_graphs = graph_samples[sample_key]
    truth_graphs = datasets[truth_key]
    graph_type = "BP" if "BP" in truth_key else "ER"

    # Run all tests and store in a dictionary
    results = {
        "Dataset": truth_key,
        "Source": source_type,
        "Bipartite % (Truth)": test_bipartite_proportion(truth_graphs) * 100,
        "Bipartite % (Sample)": test_bipartite_proportion(sample_graphs) * 100,
        "Non-Iso % (Truth)": test_non_isomorphic_proportion(truth_graphs) * 100,
        "Non-Iso % (Sample)": test_non_isomorphic_proportion(sample_graphs) * 100,
        "Avg Edge Prob (Truth)": test_average_edge_probability(truth_graphs, graph_type),
        "Avg Edge Prob (Sample)": test_average_edge_probability(sample_graphs, graph_type),
        "Avg Components (Truth)": test_average_components(truth_graphs),
        "Avg Components (Sample)": test_average_components(sample_graphs),
        "Avg Clustering (Truth)": test_average_clustering(truth_graphs),
        "Avg Clustering (Sample)": test_average_clustering(sample_graphs),
        "Avg Shortest Path (Truth)": test_average_shortest_path(truth_graphs),
        "Avg Shortest Path (Sample)": test_average_shortest_path(sample_graphs),
        "Avg Diameter (Truth)": test_average_diameter(truth_graphs),
        "Avg Diameter (Sample)": test_average_diameter(sample_graphs),
        "Avg Spectral Gap (Truth)": test_average_spectral_gap(truth_graphs),
        "Avg Spectral Gap (Sample)": test_average_spectral_gap(sample_graphs),
        "JS Divergence (Degree)": calculate_js_divergence(truth_graphs, sample_graphs),
    }
    results_list.append(results)

df_results = pd.DataFrame(results_list)
df_results.set_index(["Dataset", "Source"], inplace=True)
df_results.sort_index(inplace=True)

# --- Format the DataFrame for display ---
df_display = df_results.copy()
percent_cols = ["Bipartite % (Truth)", "Bipartite % (Sample)", "Non-Iso % (Truth)", "Non-Iso % (Sample)"]
float4_cols = ["Avg Edge Prob (Truth)", "Avg Edge Prob (Sample)", "JS Divergence (Degree)", "Avg Clustering (Truth)", "Avg Clustering (Sample)", "Avg Spectral Gap (Truth)", "Avg Spectral Gap (Sample)"]
float2_cols = ["Avg Components (Truth)", "Avg Components (Sample)", "Avg Shortest Path (Truth)", "Avg Shortest Path (Sample)", "Avg Diameter (Truth)", "Avg Diameter (Sample)"]

for col in percent_cols: df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%")
for col in float4_cols: df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
for col in float2_cols: df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")

# --- Display Final Table ---
print("\n\n" + "=" * 30 + " COMPREHENSIVE ANALYSIS TABLE " + "=" * 30)
print(df_display.to_markdown())