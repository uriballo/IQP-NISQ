import argparse
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path

import iqpopt as iqp
import numpy as np

from src.utils.utils import aachen_connectivity, efficient_connectivity_gates, setup_logging

def create_circuit(num_qubits: int, num_layers: int) -> iqp.IqpSimulator:
    """Creates an IQP circuit simulator with a fixed connectivity."""
    grid_conn = aachen_connectivity()
    gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
    return iqp.IqpSimulator(num_qubits, gates)


def simulate_model(model_dir: Path, shots: int):
    """Loads a model, simulates it using the .sample() method, and saves the results."""
    logging.info(f"--- Processing model in: {model_dir} ---")

    # Load metadata to check the number of nodes
    config_file = model_dir / 'hyperparams.yml'
    if not config_file.is_file():
        logging.warning(f"No hyperparams file found in {model_dir}. Skipping.")
        return

    with open(config_file, 'r') as f:
        metadata = yaml.safe_load(f)

    nodes = metadata.get('nodes')
    if nodes != 8:
        logging.info(f"Skipping model with {nodes} nodes (only simulating for 8).")
        return

    logging.info(f"Found 8-node model to simulate: {model_dir.name}")

    # Load model parameters
    params_file = model_dir / "params.npy"
    if not params_file.exists():
        logging.warning(f"No params.npy file found in {model_dir}. Skipping.")
        return
    params = np.load(params_file)

    num_layers = metadata['hyperparameters']['num_layers']
    num_qubits = nodes * (nodes - 1) // 2

    logging.info(f"Model details: {nodes} Nodes, {num_layers} Layers")
    circuit = create_circuit(num_qubits, num_layers)

    try:
        path_parts = list(model_dir.resolve().parts)
        idx = path_parts.index('trained_params')
        path_parts[idx] = 'simulation_results'
        output_save_dir = Path(*path_parts)
    except ValueError:
        logging.warning("Could not find 'trained_params' in the path. Saving results in a new 'simulation_results' subdir.")
        output_save_dir = model_dir.parent / "simulation_results" / model_dir.name

    output_save_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Simulating circuit with {shots} shots...")
    samples = circuit.sample(params=params, shots=shots)
    logging.info(f"Received {len(samples)} samples from simulation.")

    date_str = datetime.now().strftime("%Y%m%d")
    save_path = output_save_dir / f"simulated_samples_{date_str}.npy"
    np.save(save_path, samples)
    logging.info(f"✅ Simulated samples saved to: {save_path}")


def main(args: argparse.Namespace):
    """Finds all 8-node models in a directory and runs simulations for them."""
    setup_logging()

    base_params_dir = args.base_dir
    if not base_params_dir.is_dir():
        logging.error(f"Base directory not found at: {base_params_dir}")
        return

    # Recursively find all 'hyperparams.yml' files to identify all models
    logging.info(f"Scanning for models in: {base_params_dir}")
    model_configs = list(base_params_dir.rglob('hyperparams.yml'))

    if not model_configs:
        logging.error("No 'hyperparams.yml' files found in the specified directory.")
        return

    model_dirs = sorted([p.parent for p in model_configs])
    logging.info(f"Found {len(model_dirs)} potential models. Will filter for 8-node versions.")

    for model_dir in model_dirs:
        simulate_model(model_dir, args.shots)

    logging.info("\n===== 🎉 All simulations complete! =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate trained IQP models."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/trained_params"),
        help="Base directory containing all trained model parameters. Default: './trained_params'"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=512,
        help="Number of shots to use for sampling. Default: 8192"
    )

    args = parser.parse_args()
    main(args)