import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.metrics import average_edge_prob, bipartite_proportion
from src.utils.utils import setup_logging


def get_dataset_summary(model_dir: Path, all_summaries: list) -> dict | None:
    """Determines the correct dataset summary by parsing the directory structure."""
    try:
        grandparent_name = model_dir.parent.parent.name
        density_name = model_dir.name
        dataset_key = f"{grandparent_name}_{density_name}"

        for summary in all_summaries:
            if summary['name'] == dataset_key:
                return summary
        logging.warning(f"No matching summary found for key: {dataset_key}")
        return None
    except Exception as e:
        logging.error(f"Could not determine dataset key for {model_dir}: {e}")
        return None


def find_nodes_from_config(model_dir: Path) -> int | None:
    """Reads the hyperparams.yml to find the number of nodes."""
    try:
        config_path_parts = list(model_dir.resolve().parts)

        if 'evaluation_results' in config_path_parts:
            source_folder = 'evaluation_results'
        elif 'simulation_results' in config_path_parts:
            source_folder = 'simulation_results'
        else:
            raise ValueError("Path does not contain 'evaluation_results' or 'simulation_results'")

        idx = config_path_parts.index(source_folder)
        config_path_parts[idx] = 'trained_params'

        config_dir = Path(*config_path_parts[:-1])
        config_file_path = config_dir / model_dir.name / 'hyperparams.yml'

        with open(config_file_path, 'r') as f:
            metadata = yaml.safe_load(f)
        return metadata['nodes']
    except (ValueError, FileNotFoundError):
        logging.warning(f"Could not find corresponding hyperparams.yml for model '{model_dir.name}'. Check directory structure.")
        return None


def main(args: argparse.Namespace):
    """Main function to find, analyze, and summarize all experiment results."""
    setup_logging()
    base_results_dir = args.results_base_dir
    if not base_results_dir.is_dir():
        logging.error(f"Base results directory not found: {base_results_dir}")
        return

    try:
        summary_path = Path('data/datasets_summary.yml')
        with open(summary_path, 'r') as f:
            all_dataset_summaries = yaml.safe_load(f)
        logging.info(f"Successfully loaded {len(all_dataset_summaries)} dataset summaries.")
    except FileNotFoundError:
        logging.error(f"Dataset summary file not found at: {summary_path}")
        return

    # --- Find all model directories across all runs ---
    logging.info(f"Scanning for all results in '{base_results_dir}'...")
    all_sample_files = list(base_results_dir.rglob("*samples*.npy"))
    if not all_sample_files:
        logging.error("No sample files found anywhere in the results directory.")
        return
    # Get the unique parent directories of all found sample files
    model_dirs = sorted(list(set(p.parent for p in all_sample_files)))
    logging.info(f"Found {len(model_dirs)} total models to analyze.")

    # --- Process each model and collect results for the master CSV ---
    all_results = []
    for model_dir in model_dirs:
        logging.info(f"--- Processing: {model_dir.relative_to(base_results_dir)} ---")

        nodes = find_nodes_from_config(model_dir)
        sample_files = list(model_dir.glob("*samples*.npy")) # Re-list for the current dir
        if nodes is None or not sample_files:
            continue

        generated_samples = np.vstack([np.load(f) for f in sample_files])
        dataset_info = get_dataset_summary(model_dir, all_dataset_summaries)
        if not dataset_info:
            logging.warning(f"Skipping model {model_dir.name} due to missing dataset info.")
            continue

        # --- Calculate metrics ---
        dataset_bipartite_pct = dataset_info['bipartite_percentage']
        dataset_edge_prob = dataset_info['average_edge_prob']
        gen_bipartite_pct = bipartite_proportion(generated_samples, nodes) * 100
        gen_edge_prob = average_edge_prob(generated_samples, nodes)

        bipartite_diff = gen_bipartite_pct - dataset_bipartite_pct
        edge_prob_diff = gen_edge_prob - dataset_edge_prob

        run_type = 'Simulation' if 'simulation_results' in str(model_dir) else 'Evaluation'

        all_results.append({
            'run_type': run_type,
            'dataset_name': dataset_info['name'],
            'model': model_dir.name,
            'num_samples': len(generated_samples),
            'bipartite_diff': bipartite_diff,
            'edge_prob_diff': edge_prob_diff,
            'generated_bipartite_pct': gen_bipartite_pct,
            'dataset_bipartite_pct': dataset_bipartite_pct,
            'generated_edge_prob': gen_edge_prob,
            'dataset_edge_prob': dataset_edge_prob,
            'full_path': str(model_dir)
        })

    if not all_results:
        logging.warning("No results were generated. CSV file will not be created.")
        return

    df = pd.DataFrame(all_results)
    output_path = base_results_dir / 'master_analysis_summary.csv'
    df.to_csv(output_path, index=False, float_format='%.4f')
    logging.info(f"✅ Analysis complete. Master summary saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze all quantum experiment results and generate a master summary CSV."
    )
    parser.add_argument(
        "--results-base-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing 'evaluation_results' and 'simulation_results'. Default: './results'"
    )
    args = parser.parse_args()
    main(args)