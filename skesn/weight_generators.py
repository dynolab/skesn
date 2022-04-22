import numpy as np


def standart_weights_generator(random_state, n_reservoir: int, sparsity: float, spectral_radius: float, n_endo: int, n_exo: int):
    W_in = random_state.rand(n_reservoir, n_endo) * 2 - 1

    # initialize recurrent weights:
    # begin with a random matrix centered around zero:
    W = random_state.rand(n_reservoir, n_reservoir) - 0.5
    # delete the fraction of connections given by (self.sparsity):
    W[random_state.rand(*W.shape) < sparsity] = 0
    # compute the spectral radius of these weights:
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    # rescale them to reach the requested spectral radius:
    W = W * (spectral_radius / radius)

    W_c = None
    if n_exo > 0:
        W_c = random_state.rand(n_reservoir, n_exo) * 2 - 1
    return W_in, W, W_c
