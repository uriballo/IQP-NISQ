import logging
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import iqpopt as iqp
from iqpopt.utils import initialize_from_data
from iqpopt.gen_qml.utils import median_heuristic
import optuna
import jax
from jax import numpy as jnp
import numpy as np
from sklearn.model_selection import RepeatedKFold

from src.utils.utils import aachen_connectivity, efficient_connectivity_gates, setup_logging
from src.datasets.bipartites import BipartiteGraphDataset
from src.datasets.er import ErdosRenyiGraphDataset

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file and sets up paths."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['dataset_dir'] = Path(config['dataset_dir'])
    config['results_dir'] = Path(config['results_dir'])
    config['hpo_log_dir'] = Path(config.get('hpo_log_dir', config['results_dir'] / 'hpo_logs'))
    config['final_models_dir'] = Path(config.get('final_models_dir', config['results_dir'] / 'final_models'))
    config['config_path'] = config_path
    return config

def load_dataset(cfg: Dict[str, Any], connectivity: str, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Loads and shuffles a full dataset for a specific connectivity."""
    DATASET_CLASSES = {"Bipartite": BipartiteGraphDataset, "ER": ErdosRenyiGraphDataset}
    graph_type = cfg.get("graph_type")
    if graph_type not in DATASET_CLASSES:
        raise ValueError(f"Unknown graph_type '{graph_type}'.")

    dataset_class = DATASET_CLASSES[graph_type]
    ds_path = cfg["dataset_dir"] / f"{cfg['nodes']}N_{graph_type}_{connectivity}.pkl"
    logging.info(f"Loading dataset from {ds_path}")
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {ds_path}")

    dataset = dataset_class(nodes=cfg['nodes'], edge_prob=0.01).from_file(str(ds_path))
    vectors = jnp.array(dataset.vectors.copy())
    return jax.random.permutation(key, vectors)

def create_circuit(num_qubits: int, num_layers: int) -> iqp.IqpSimulator:
    """Creates an IQP circuit simulator with a fixed connectivity."""
    grid_conn = aachen_connectivity()
    gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
    return iqp.IqpSimulator(num_qubits, gates, device='lightning.qubit')

def run_training(params: jnp.ndarray, circuit: iqp.IqpSimulator, train_ds: jnp.ndarray,
                 hyperparams: Dict[str, Any], n_iters: int, cfg: Dict[str, Any], key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, float]:
    """A generic training loop that returns final parameters and loss."""
    sigma = median_heuristic(train_ds) * hyperparams["sigma_multiplier"]
    loss_kwargs = {
        "params": params, "iqp_circuit": circuit, "ground_truth": train_ds,
        "sigma": [sigma], "n_ops": cfg["n_ops"], "n_samples": cfg["n_samples"], "key": key,
    }
    trainer = iqp.Trainer(cfg["optimizer"], iqp.gen_qml.mmd_loss_iqp, stepsize=hyperparams["learning_rate"])
    trainer.train(n_iters=n_iters, loss_kwargs=loss_kwargs, turbo=1)
    return trainer.final_params, trainer.losses[-1]


def objective_cv(trial: optuna.Trial, num_qubits: int, dataset: jnp.ndarray,
                 cfg: Dict[str, Any], key: jax.random.PRNGKey) -> float:
    """Objective function that dynamically tunes or fixes hyperparameters based on the config."""
    
    ALL_HYPERPARAMS = [
        "learning_rate", 
        "sigma_multiplier", 
        "num_layers", 
        "initialization_multiplier"
    ]

    hp = {}
    space = cfg.get("hpo_search_space", {})
    fixed = cfg.get("fixed_params", {})

    for param_name in ALL_HYPERPARAMS:
        if param_name in space:
            if param_name == "learning_rate":
                hp[param_name] = trial.suggest_float(param_name, *space[param_name], log=True)
            elif param_name == "num_layers":
                hp[param_name] = trial.suggest_int(param_name, *space[param_name])
            else: 
                hp[param_name] = trial.suggest_float(param_name, *space[param_name])
        elif param_name in fixed:
            hp[param_name] = fixed[param_name]
        else:
            raise ValueError(
                f"Configuration Error: Hyperparameter '{param_name}' must be defined in "
                f"either 'hpo_search_space' or 'fixed_params' in your config file."
            )

    logging.info(f"Trial {trial.number}: Starting with HP: {hp}")

    rkf = RepeatedKFold(n_splits=cfg["hpo_cv_folds"], n_repeats=2, random_state=cfg["random_seed"])
    num_total_folds = rkf.get_n_splits(dataset)
    fold_keys = jax.random.split(key, num_total_folds)

    try:
        fold_losses = []
        for i, (train_index, val_index) in enumerate(rkf.split(dataset)):
            train_ds, val_ds = dataset[train_index], dataset[val_index]
            
            circuit = create_circuit(num_qubits, hp["num_layers"])
            params_init = initialize_from_data(circuit.gates, train_ds) * hp["initialization_multiplier"]

            final_params, _ = run_training(
                params_init, circuit, train_ds, hp, cfg["hpo_iterations"], cfg, fold_keys[i]
            )

            val_loss = iqp.gen_qml.mmd_loss_iqp(
                params=final_params, 
                iqp_circuit=circuit, 
                ground_truth=val_ds,
                sigma=[median_heuristic(train_ds) * hp["sigma_multiplier"]],
                n_ops=cfg["n_ops"], 
                n_samples=cfg["n_samples"], 
                key=fold_keys[i],
            )
            
            if jnp.isnan(val_loss) or jnp.isinf(val_loss):
                logging.warning(f"Trial {trial.number}, Fold {i+1}: Unstable validation loss.")
                return float('inf')
            fold_losses.append(val_loss)

            trial.report(float(jnp.mean(jnp.array(fold_losses))), step=i)
            if trial.should_prune():
                logging.info(f"Trial {trial.number} pruned at fold {i+1}.")
                raise optuna.exceptions.TrialPruned()

        average_loss = jnp.mean(jnp.array(fold_losses))
        logging.info(f"Trial {trial.number} -> Avg. CV Loss: {average_loss:.8f}")
        return float(average_loss)

    except optuna.exceptions.TrialPruned:
        return float('inf') 
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        return float('inf')

def run_hpo_for_connectivity(connectivity: str, cfg: Dict[str, Any], key: jax.random.PRNGKey) -> Dict[str, Any]:
    """Runs an HPO session for a single, specific connectivity."""
    logging.info(f"--- 🚀 Starting HPO for {connectivity} ---")
    
    try:
        dataset = load_dataset(cfg, connectivity, key)
        num_qubits = dataset.shape[1]
    except FileNotFoundError as e:
        logging.error(f"Cannot run HPO for '{connectivity}', dataset not found. Aborting. Details: {e}")
        return None

    cfg['hpo_log_dir'].mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{cfg['hpo_log_dir']}/all_hpo_studies.db"
    study_name = f"{cfg['nodes']}N_{cfg['graph_type']}_{connectivity}_run_{cfg['run_id']}"
    
    pruner = optuna.pruners.HyperbandPruner()
    
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg["random_seed"]),
        pruner=pruner, 
        load_if_exists=True
    )
    study.set_user_attr("config_file", cfg["config_path"])
    study.set_user_attr("run_id", cfg["run_id"])
    study.set_user_attr("connectivity", connectivity)

    def _objective(trial: optuna.Trial) -> float:
        trial_key = jax.random.fold_in(key, trial.number)
        return objective_cv(trial, num_qubits, dataset, cfg, trial_key)
    
    logging.info(f"Starting/Resuming HPO for study '{study_name}'.")
    logging.info(f"All studies are stored in DB: {storage_name}")
    logging.info(f"Running {cfg['hpo_trials']} trials with {cfg['hpo_cv_folds']} folds and pruning enabled...")
    study.optimize(_objective, n_trials=cfg['hpo_trials'])
    
    logging.info(f"--- ✅ HPO for {connectivity} Finished ---")
    logging.info(f"Best trial: {study.best_trial.number} with value {study.best_value:.8f}")
    logging.info(f"Best hyperparameters found: {study.best_params}")
    return study.best_params

def train_and_save_final_model(connectivity: str, best_hp: Dict[str, Any], cfg: Dict[str, Any], key: jax.random.PRNGKey):
    """Loads data, runs final training on the full dataset, and saves the model."""
    logging.info(f"\n--- 🚂 Starting Final Training for Connectivity: {connectivity} ---")
    data_key, train_key = jax.random.split(key)

    try:
        train_ds = load_dataset(cfg, connectivity, data_key)
        num_qubits = train_ds.shape[1]
        logging.info(f"Training final model on full dataset of size {len(train_ds)}.")
    except FileNotFoundError as e:
        logging.warning(f"Could not find dataset for '{connectivity}'. Skipping. Details: {e}")
        return

    # MODIFICATION: Combine the best HPs from the study with the fixed HPs from the config.
    full_hyperparams = best_hp.copy()
    full_hyperparams.update(cfg.get("fixed_params", {}))

    final_circuit = create_circuit(num_qubits, full_hyperparams["num_layers"])
    initial_params = initialize_from_data(final_circuit.gates, train_ds) * full_hyperparams["initialization_multiplier"]
    
    logging.info(f"Training for {cfg['final_train_iterations']} iterations...")
    final_params, final_loss = run_training(
        initial_params, final_circuit, train_ds, full_hyperparams,
        cfg["final_train_iterations"], cfg, train_key
    )
    logging.info(f"Final training complete with loss: {final_loss:.8f}")

    output_dir = cfg["final_models_dir"] / f"{cfg['nodes']}N_{cfg['graph_type']}" / f"run_{cfg['run_id']}" / connectivity
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "params.npy", final_params)
    
    metadata = {
        "run_id": cfg["run_id"],
        "nodes": cfg["nodes"],
        "graph_type": cfg["graph_type"],
        "connectivity": connectivity,
        "final_training_loss": float(final_loss),
        "hyperparameters": full_hyperparams
    }
    with open(output_dir / "hyperparams.yml", "w") as f:
        yaml.dump(metadata, f, indent=2)
        
    logging.info(f"✅ Final model for '{connectivity}' saved to: {output_dir}")

# =================================================================
# Main Execution Block
# =================================================================

def main(config_path: str):
    """Main execution function to run HPO and final training for each connectivity."""
    setup_logging()
    cfg = load_config(config_path)
    
    cfg['run_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Generated unique Run ID: {cfg['run_id']}")

    master_key = jax.random.PRNGKey(cfg["random_seed"])
    
    connectivities_to_run = ["Sparse", "Medium", "Dense"]
    
    for i, conn in enumerate(connectivities_to_run):
        conn_key = jax.random.fold_in(master_key, i)
        hpo_key, final_train_key = jax.random.split(conn_key)
        
        logging.info(f"\n===== 🚀 Starting Full Workflow for Connectivity: {conn} =====")
        
        best_hyperparams = run_hpo_for_connectivity(conn, cfg, hpo_key)
        
        if best_hyperparams is None:
            logging.warning(f"Skipping final training for {conn} due to HPO failure.")
            continue
            
        train_and_save_final_model(conn, best_hyperparams, cfg, final_train_key)

    logging.info(f"\n===== 🎉 All Runs for {cfg['nodes']}-Node {cfg['graph_type']} (Run ID: {cfg['run_id']}) Complete =====")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IQP graph training with robust HPO using k-fold CV.")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the configuration YAML file. (default: config.yaml)"
    )
    args = parser.parse_args()
    main(config_path=args.config)