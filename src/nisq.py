# evaluate_on_nisq.py
import argparse
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
import iqpopt as iqp
import numpy as np
import pennylane as qml
from dotenv import load_dotenv
from qiskit_ibm_runtime import QiskitRuntimeService

from src.utils.utils import (
    create_circuit,
    load_model_from_dir,
    setup_logging,
    setup_output_directory,
)

def setup_ibmq_backend(min_qubits: int, shots: int, save: bool = False) -> qml.device:
    """Sets up the connection to IBMQ and returns a PennyLane device."""
    logging.info("Setting up connection to IBM Quantum...")
    load_dotenv('.env')

    ibm_token = os.getenv('IBM_TOKEN')
    instance = os.getenv("INSTANCE")

    if not ibm_token or not instance:
        raise ValueError("IBM_TOKEN and INSTANCE must be set in your .env file.")

    if save:
        QiskitRuntimeService.save_account(channel="ibm_quantum", token=ibm_token, overwrite=True)

    service = QiskitRuntimeService(channel="ibm_cloud", token=ibm_token, instance=instance)

    logging.info(f"Finding least busy backend with at least {min_qubits} qubits...")
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    logging.info(f"Selected backend: {backend.name}")

    device = qml.device(
        'qiskit.remote',
        wires=backend.num_qubits,
        backend=backend,
        shots=shots,
        twirling={"enable_gates": True, "enable_measure": True},
        dynamical_decoupling={"enable": True}
    )

    logging.info(f"PennyLane device for remote backend '{backend.name}' created successfully.")
    return device


def evaluate_model(model_dir: Path, base_dir: Path, qiskit_device: qml.device, shots: int):
    """Loads a single model, evaluates it on hardware, and saves the samples."""
    logging.info(f"--- Evaluating model: {model_dir.relative_to(base_dir)} ---")

    params, metadata = load_model_from_dir(model_dir)
    if params is None:
        return

    nodes = metadata['nodes']
    num_layers = metadata['hyperparameters']['num_layers']
    num_qubits = nodes * (nodes - 1) // 2

    logging.info(f"Model details: {nodes} Nodes, {num_layers} Layers")
    circuit = create_circuit(num_qubits, num_layers)
    output_save_dir = setup_output_directory(model_dir, "evaluation")

    logging.info(f"Executing circuit on {qiskit_device.backend.name} with {shots} shots...")
    samples = circuit.sample_qiskit(params=params, device=qiskit_device, shots=shots)
    logging.info(f"Received {len(samples)} samples.")

    date_str = datetime.now().strftime("%Y%m%d")
    save_path = output_save_dir / f"samples_{date_str}.npy"
    np.save(save_path, samples)
    logging.info(f"✅ Samples saved to: {save_path}")


def main(args: argparse.Namespace):
    """Finds and evaluates selected models on real hardware."""
    setup_logging()
    base_dir = args.base_dir.resolve()
    if not base_dir.is_dir():
        logging.error(f"Base directory not found at: {base_dir}")
        return

    # 1. Discover all models by finding their hyperparams.yml files
    all_models = [p.parent for p in base_dir.rglob('hyperparams.yml')]
    logging.info(f"Found {len(all_models)} total structured models in '{base_dir}'.")

    # 2. Filter models based on run type and include/exclude criteria
    models_to_run = []
    for model_path in all_models:
        with open(model_path / 'hyperparams.yml', 'r') as f:
            meta = yaml.safe_load(f)

        # Filter by run type (legacy vs new)
        is_legacy = 'source_file' in meta.get('hyperparameters', {})
        if args.run_type == 'legacy' and not is_legacy:
            continue
        if args.run_type == 'new' and is_legacy:
            continue

        # Filter by node count
        node_count_str = f"{meta.get('nodes', 0)}N"
        if args.include and node_count_str not in args.include:
            continue
        if args.exclude and node_count_str in args.exclude:
            continue
        
        models_to_run.append(model_path)

    if not models_to_run:
        logging.warning("No models match the selection criteria. Nothing to evaluate.")
        return

    logging.info(f"--- Preparing to evaluate {len(models_to_run)} model(s) ---")

    # 3. Set up the hardware backend once for all evaluations
    num_qubits_required = 153 # Sufficient for up to 18-node graphs
    qiskit_device = setup_ibmq_backend(min_qubits=num_qubits_required, shots=args.shots)

    # 4. Loop through and evaluate only the selected models
    for model_dir in sorted(models_to_run):
        evaluate_model(model_dir, base_dir, qiskit_device, args.shots)

    logging.info("\n===== 🎉 All selected evaluations complete! =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate selected trained models on IBM Quantum hardware."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/"),
        help="Base directory containing all structured model runs."
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=512,
        help="Number of shots to use for sampling."
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=['all', 'legacy', 'new'],
        default='all',
        help="Type of runs to evaluate."
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--include",
        nargs='+',
        help="List of node counts to run (e.g., --include 8N 10N)."
    )
    group.add_argument(
        "--exclude",
        nargs='+',
        help="List of node counts to exclude (e.g., --exclude 14N 18N)."
    )

    args = parser.parse_args()
    main(args)