import argparse
import logging
from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.utils import setup_logging

def extract_node_count(dataset_name: str) -> int:
    """Extracts the integer node count from a name like '8N_Bipartite_Sparse'."""
    try:
        return int(dataset_name.split('N')[0])
    except (ValueError, IndexError):
        return 0

def calculate_random_bipartite_baseline(
    node_counts: list, all_summaries: list, num_samples: int = 512
) -> list:
    """
    For each node count, generates random graphs using the average edge probability
    from the corresponding Bipartite datasets to find the baseline.
    """
    logging.info(f"Calculating random bipartite baseline for nodes: {node_counts}...")
    baseline_percentages = []

    for n in node_counts:
        # Find all Bipartite datasets for the current node count 'n'
        relevant_probs = [
            summary['average_edge_prob'] for summary in all_summaries
            if summary['name'].startswith(f"{n}N_Bipartite")
        ]

        if relevant_probs:
            target_edge_prob = np.mean(relevant_probs)
        else:
            target_edge_prob = 0.5 
        
        logging.info(f"Using average edge probability of {target_edge_prob:.4f} for {n}-node baseline.")

        bipartite_count = 0
        for _ in range(num_samples):
            # Create a G(n,p) random graph with the target edge probability
            G = nx.fast_gnp_random_graph(n, target_edge_prob)
            if nx.is_bipartite(G):
                bipartite_count += 1
        baseline_percentages.append((bipartite_count / num_samples) * 100)
        
    return baseline_percentagess


def plot_bipartite_scaling(df: pd.DataFrame, output_dir: Path):
    """
    Generates a line plot for the generated bipartite percentage on real hardware
    for bipartite datasets.
    """
    logging.info("Generating bipartite percentage scaling plot...")
    # 1. Filter data: Only 'Evaluation' runs on 'Bipartite' datasets
    plot_df = df[
        (df['run_type'] == 'Evaluation') &
        (df['dataset_name'].str.contains('Bipartite'))
    ].copy()

    if plot_df.empty:
        logging.warning("No data found for bipartite scaling plot. Skipping.")
        return

    # 2. Setup plot
    plt.style.use('seaborn-v0_8_whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Draw lines for each model type (Sparse, Medium, Dense)
    sns.lineplot(
        data=plot_df, x='nodes', y='generated_bipartite_pct', hue='model',
        style='model', markers=True, dashes=False, lw=2.5, ax=ax
    )

    # 4. Calculate and draw the random baseline
    node_counts = sorted(plot_df['nodes'].unique())
    baseline_values = calculate_random_bipartite_baseline(node_counts)
    ax.plot(node_counts, baseline_values, label='Random Baseline', color='grey', linestyle='--', zorder=1)

    # 5. Customize axes and title
    ax.set_title('Bipartite Graph Percentage (Hardware Evaluation)', fontsize=16, pad=20)
    ax.set_ylabel('Generated Bipartite Graphs (%)', fontsize=12)
    ax.set_xlabel('Graph Size', fontsize=12)
    ax.legend(title='Model Type')

    # Create custom x-axis labels
    tick_labels = {
        n: f"{n} Nodes\n({n*(n-1)//2} Qubits)" for n in node_counts
    }
    ax.set_xticks(node_counts)
    ax.set_xticklabels([tick_labels[n] for n in node_counts])

    # 6. Save figure
    save_path = output_dir / "bipartite_percentage_scaling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Plot saved to {save_path}")

def plot_edge_error_scaling(df: pd.DataFrame, output_dir: Path):
    """
    Generates a line plot for the absolute edge probability error on real hardware
    for all dataset types.
    """
    logging.info("Generating edge probability error scaling plot...")
    # 1. Filter data: Only 'Evaluation' runs
    plot_df = df[df['run_type'] == 'Evaluation'].copy()

    if plot_df.empty:
        logging.warning("No data found for edge error scaling plot. Skipping.")
        return

    # 2. Setup plot
    plt.style.use('seaborn-v0_8_whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Draw lines for each model type
    sns.lineplot(
        data=plot_df, x='nodes', y='abs_edge_prob_diff', hue='model',
        style='model', markers=True, dashes=False, lw=2.5, ax=ax
    )

    # 4. Customize axes and title
    ax.set_title('Absolute Edge Probability Error (Hardware Evaluation)', fontsize=16, pad=20)
    ax.set_ylabel('Absolute Error in Edge Probability', fontsize=12)
    ax.set_xlabel('Graph Size', fontsize=12)
    ax.legend(title='Model Type')

    # Create custom x-axis labels
    node_counts = sorted(plot_df['nodes'].unique())
    tick_labels = {
        n: f"{n} Nodes\n({n*(n-1)//2} Qubits)" for n in node_counts
    }
    ax.set_xticks(node_counts)
    ax.set_xticklabels([tick_labels[n] for n in node_counts])

    # 5. Save figure
    save_path = output_dir / "edge_probability_error_scaling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"✅ Plot saved to {save_path}")


def main(args: argparse.Namespace):
    """Main function to load data and generate plots."""
    setup_logging()
    input_csv = args.input_csv

    if not input_csv.is_file():
        logging.error(f"Input CSV file not found: {input_csv}")
        return

    logging.info(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    df['nodes'] = df['dataset_name'].apply(extract_node_count)
    df['abs_edge_prob_diff'] = df['edge_prob_diff'].abs()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_bipartite_scaling(df, output_dir)
    plot_edge_error_scaling(df, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scaling plots from analysis summary CSV."
    )
    parser.add_argument(
        "results/master_analysis_summary.csv",
        type=Path,
        help="Path to the master_analysis_summary.csv file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to save the generated plots. Default: './results/plots'"
    )
    args = parser.parse_args()
    main(args)