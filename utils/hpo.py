import optuna
import jax
import iqpopt.gen_qml as genq
from jax import numpy as jnp
import iqpopt as iqp
from iqpopt.utils import initialize_from_data
import numpy as np
from utils.nisq import efficient_connectivity_gates

def objective(trial: optuna.Trial, 
              grid_connectivity, 
              num_qubits,
              base_sigma, 
              train_ds,
              n_iters_hpo = 500,
              n_ops = 1000,
              n_samples = 1000) -> float:
              
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 0.1, log=True)
    sigma_multiplier = trial.suggest_float("sigma_multiplier", 0.1, 2.0)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    initialization_multiplier = trial.suggest_float("initialization_multiplier", 1e-4, np.pi)

    trial_key = jax.random.PRNGKey(42 + trial.number)
    
    gates = efficient_connectivity_gates(grid_connectivity, num_qubits, num_layers) 
    iqp_circ = iqp.IqpSimulator(num_qubits, gates, device='lightning.qubit')
    sigma = base_sigma * sigma_multiplier
    params_init = initialize_from_data(gates, train_ds) * initialization_multiplier

    loss_kwarg = {
        "params": params_init,
        "iqp_circuit": iqp_circ, 
        "ground_truth": train_ds,
        "sigma": [sigma],
        "n_ops": n_ops,   
        "n_samples": n_samples, 
        "key": trial_key,  
    }

    trainer = iqp.Trainer("Adam", genq.mmd_loss_iqp, stepsize=learning_rate)

    print(f"Trial {trial.number}:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Sigma Multiplier: {sigma_multiplier}")
    print(f"  Initialization Multiplier: {initialization_multiplier}")
    print(f"  Number of Layers: {num_layers}")

    try:
        trainer.train(n_iters=n_iters_hpo, loss_kwargs=loss_kwarg, turbo=1)
        final_loss = trainer.losses[-1]

        if jnp.isnan(final_loss) or jnp.isinf(final_loss) or final_loss > 1e10: 
            print(f"Trial {trial.number} resulted in unstable loss: {final_loss}")
            return float('inf') 

        print(f"Trial {trial.number} final loss: {final_loss:.8f}")
        return float(final_loss)

    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        return float('inf') 

def run_hpo(
    grid_connectivity,
    num_qubits,
    base_sigma,
    train_ds,
    n_trials: int = 100,
    n_iters_hpo: int = 500,
    n_ops: int = 2000,
    n_samples: int = 2000,
    direction: str = "minimize",
) -> optuna.study.Study:
    """
    Runs an Optuna HPO experiment.
    """

    def _objective(trial: optuna.Trial) -> float:
        # simply forward to existing objective signature
        return objective(
            trial,
            grid_connectivity,
            num_qubits,
            base_sigma,
            train_ds,
            n_iters_hpo=n_iters_hpo,
            n_ops=n_ops,
            n_samples=n_samples,
        )

    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=n_trials)

    return study