import os
import tempfile
import functools

import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
import jax
import optuna
import mlflow
 
from iqpopt import iqp_optimizer
import iqpopt as iqp
from iqpopt import utils
from iqpopt import training
import iqpopt.gen_qml as genq
from iqpopt.gen_qml.utils import median_heuristic
from iqpopt.utils import initialize_from_data, local_gates

from datasets.bipartites import BipartiteGraphDataset 

#CONFIGURATION
n_trials = 10 #Trials de optuna
project_name = 'Prueba'


#-------Objective Function for Optuna------
def objective(trial,n_qubits, flat_ds, base_sigma):
    #Hyperparameters
    n_iters = trial.suggest_int('n_iters', 100, 2000)
    sigma_proportion = trial.suggest_float("sigma proportion", 0.1, 5.0)
    learning_rate = trial.suggest_float("learning rate",1e-5,1e-2)

    # Calculate actual sigma for this trial
    sigma = float(base_sigma * sigma_proportion)

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: n_iters={n_iters:.2e}, sigma_prop={sigma_proportion:.2f} (sigma={sigma:.4f}), learning rate={learning_rate}")

    #Start mlflow run
    with mlflow.start_run(nested=True):
        #Log params
        mlflow.log_params(trial.params)
        mlflow.log_param("Sigma",sigma)
        mlflow.log_param("iters",n_iters)
        mlflow.log_param("Learning rate",learning_rate)
        try:
            #Setup Model based on Trial Parameters
            gates = local_gates(n_qubits, max_weight=3)
            #Re-initialize parameters for each trial, especially if max_weight changes structure
            params_init = initialize_from_data(gates, flat_ds)
            circuit = iqp.IqpSimulator(n_qubits, gates)
            loss = genq.mmd_loss_iqp # MMD loss
 
            loss_kwargs = {
                "params": params_init,
                "iqp_circuit": circuit,
                "ground_truth": flat_ds, # samples from ground truth distribution
                "sigma": sigma,
                "n_ops": 1000,
                "n_samples": 1000,
                "key": jax.random.PRNGKey(trial.number), # Use trial number for key variation
            }
 
            #Train the Model
            trainer = iqp.Trainer("Adam", loss, stepsize=learning_rate)
            trainer.train(n_iters=n_iters, loss_kwargs=loss_kwargs, turbo=10)

            #Evaluate and Log Results
            final_loss = trainer.losses[-1]
            if jnp.isnan(final_loss) or jnp.isinf(final_loss):
                print(f"Trial {trial.number} resulted in NaN/Inf loss. Pruning.")
                # Report a high loss to Optuna and stop the trial
                raise optuna.exceptions.TrialPruned()
    
            print(f"  Trial {trial.number} Final Loss: {final_loss:.6f}")
            mlflow.log_metric("final_loss", float(final_loss))

            # Log loss curve as an artifact
            fig, ax = plt.subplots()
            ax.plot(trainer.losses)
            ax.set_title(f"Trial {trial.number} Loss Curve")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("MMD Loss")
            # Save plot temporarily and log it
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name)
                mlflow.log_artifact(tmpfile.name, "plots")
            plt.close(fig) # Close the plot to free memory
            os.remove(tmpfile.name) # Clean up the temporary file
            #Return the metric for Optuna to minimize
            return float(final_loss)
        
        except Exception as e:
            print(f"Trial {trial.number} failed with exception: {e}")
            mlflow.log_metric("final_loss", float('inf')) # Log high loss on failure
            # You might want to return a large value or re-raise depending on Optuna's handling
            # Returning float('inf') tells Optuna this trial was unsuccessful
            return float('inf') # Or raise optuna.exceptions.TrialPruned()
        

#Main Execution
bipartite_ds = BipartiteGraphDataset(
    num_samples = 500,
    num_vertices = 7,
    ratio_bipartite = 0.9,
    edge_prob = 0.35,
    ensure_connected = True,
    seed = 42,
)
data,labels = bipartite_ds.get_all_data()
n_qubits = 21
bsigma = median_heuristic(data)

mlflow.set_experiment(project_name)
print(f"Logging MLflow results to experiment: '{project_name}'")

objective_with_data = functools.partial(
        objective,
        n_qubits=n_qubits,
        flat_ds= data, # Note: train_ds isn't directly used in objective currently
        base_sigma= bsigma
    )
print(f"Starting Optuna optimization with {n_trials} trials...")
study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), # Example pruner
        study_name="Prueba" # Optional: Name for storage/resuming
    )
# Start a parent MLflow run to encompass the whole study (optional)
with mlflow.start_run(run_name="Optuna_Study"):
    mlflow.log_param("n_trials", n_trials)
    mlflow.log_param("base_sigma", bsigma)
    study.optimize(objective_with_data, n_trials=n_trials)
    # Log best trial results to the parent run
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_trial_value", study.best_trial.value)
    mlflow.set_tag("optimization_status", "completed")

# 4. Show Results
print("\n--- Optimization Finished ---")
print(f"Number of finished trials: {len(study.trials)}")
 
best_trial = study.best_trial
print(f"Best trial value (min loss): {best_trial.value:.6f}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")
 
print("\n--- MLflow UI ---")
print("To view the detailed results for each trial, run 'mlflow ui' in your terminal")
print(f"and navigate to the experiment '{project_name}'.")