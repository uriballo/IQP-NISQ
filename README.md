# Latent IQP

Idea: use a bernoulli autoencoder to obtain a low-dim binary latent representation of classical data and use a iqp circuit to learn that latent distribution.

Keywords: 
- Binary-Latent Autoencoder.
- IQP Circuits.
- Generative modelling.
- Implicit generative model.
- MMD loss
 

`python3.12`

- datasets/
- plots/
- weights/

- src/
    - utils/
        - `plotter.py`
            - reconstruction, rec + latent
            - circuit model (qml draw)
        - `trainer.py``
            - train iqp
            - train autoencoder
    - autoencoders/
        - `bvae.py`
        - `bae.py`
    - iqps/
        - `iqp.py`
    - `autoencoder-manager.py`
        - train autoencoder
        - load weights
        - get latent from autoencoder
        - decode
    - `main.py
