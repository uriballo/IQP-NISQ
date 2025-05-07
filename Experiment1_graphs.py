import matplotlib.pyplot as plt
import numpy as np
import iqpopt as iqp
from iqpopt.utils import initialize_from_data, local_gates
import iqpopt.gen_qml as genq
from iqpopt.gen_qml.utils import median_heuristic
import jax
import jax.numpy as jnp
import json
import pandas as pd
from datasets.random_graphs import ErdosRenyiDataset


probs = np.linspace(0.05, 0.97,40)
results = []
final_params = []
#Circuito IQP
nqubits = 15
gates = local_gates(n_qubits=nqubits,max_weight=2)
circ = iqp.IqpSimulator(nqubits,gates,device='lightning.qubit')
loss = genq.mmd_loss_iqp


for p in probs:
    dataset = ErdosRenyiDataset(num_nodes=6,
                                 num_samples=300,
                                 er_prob=p,
                                 seed=42)
    X = []
    train = dataset.get_train_data()
    X = train[0]
    
    #Training
    sigma = median_heuristic(X)*4
    params_init = initialize_from_data(gates, jnp.array(X)) 
    loss_kwarg = {
        "params": params_init,
        "iqp_circuit": circ,
        "ground_truth": jnp.array(X), 
        "sigma": [sigma],
        "n_ops": 1000,
        "n_samples": 1000,
        "key": jax.random.PRNGKey(42),}

    trainer = iqp.Trainer("Adam", loss, stepsize=0.003)
    trainer.train(n_iters= 600,loss_kwargs=loss_kwarg)
    trained_params = trainer.final_params
    check = circ.sample(trained_params, shots = 5000)
    sample_prob = []
    for v in check:
        sample_prob.append(sum(v)/15)
    #Mean and Standard Error
    mean = np.mean(sample_prob)
    err = np.std(sample_prob,ddof=0) / np.sqrt(len(sample_prob))
    #Record of results
    results.append({"Prob": p,"Mean Prob":mean,"Standard Err": err})
    final_params.append({"Prob":p,"Final Params":trained_params})

# Al final, lo conviertes en DataFrame
df = pd.DataFrame(results)
for item in final_params:
    item["Final Params"] = json.dumps(item["Final Params"].tolist())

param = pd.DataFrame(final_params)
param = pd.DataFrame(final_params)

# Guardas en CSV
df.to_csv("Experiment1A.csv", index=False)
param.to_csv("Experiment1_paramsA.csv",index=False)


