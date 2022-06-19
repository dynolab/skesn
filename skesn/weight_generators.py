import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm
import matplotlib.pyplot as plt


def standart_weights_generator(random_state, n_reservoir: int, sparsity: float, spectral_radius: float, endo_states: np.ndarray, exo_states: any):
    n_endo = endo_states.shape[2]
    n_exo = 0 if exo_states is None else exo_states.shape[-1]

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

    return W_in, W

def optimal_weights_generator(verbose = 2, range_generator = np.linspace, 
    steps = 100, hidden_std = 0.5, find_optimal_input = True, thinning_step = 10):
    def _generator(random_state, n_reservoir: int, sparsity: float, spectral_radius: any, 
        endo_states: np.ndarray, exo_states: any = None, controller_inst = None):
        # Preparing 
        n_endo = endo_states.shape[2]
        n_exo = 0 if exo_states is None else exo_states.shape[-1]

        endo_states = endo_states.reshape((-1, n_endo))
        endo_states = endo_states[::thinning_step]
        if(exo_states is not None):
            exo_states = exo_states.reshape((-1, n_exo))
            exo_states = exo_states[::thinning_step]

        if(type(spectral_radius) in [int, float]):
            if(range_generator is np.linspace):
                spectral_radius = range_generator(spectral_radius/1e8, spectral_radius*2, steps)
            elif(range_generator is np.logspace):
                r_pow = np.log10(spectral_radius)
                spectral_radius = range_generator(r_pow-8, r_pow+0.3, steps)
        elif(type(spectral_radius) is tuple):
            spectral_radius = range_generator(*spectral_radius, steps)
        spectral_radius = spectral_radius.reshape((steps, 1, 1))

        # -----------------------------
        # Generate optimal W_in
        # -----------------------------
        if(verbose > 0): print("\n------------Reservoir searching------------")
        if(find_optimal_input):
            if(verbose > 0): print("Input matrix generation...")

            # Initialization
            W_in = random_state.uniform(-1, 1, (steps, n_reservoir, n_endo)) * spectral_radius
            W_in[random_state.rand(*W_in.shape) < 0.3] = 0

            # Get activations 
            #   Compute input pre-activations
            input_preact = np.einsum('ijk,lk->ilj', W_in, endo_states)
            if(exo_states is not None and controller_inst is not None):
                input_preact = controller_inst.preact(input_preact, exo_states)
            #   Generate hidden pre-activations: half of them are zeros, 
            #   the other half are random numbers with a uniform distribution.
            hidden_preact = random_state.normal(0, hidden_std, input_preact.shape)
            hidden_preact[random_state.rand(*hidden_preact.shape) < sparsity] = 0
            preact = input_preact + hidden_preact
            act = np.tanh(preact).reshape((steps, -1))

            # Get the values of the quality metric
            qualities = np.zeros((steps,))
            target = random_state.uniform(-1, 1, act.shape[1:])
            if(verbose > 0): pbar = tqdm(total=steps,position=0)

            for i in range(steps):
                qualities[i] = ks_2samp(act[i], target).pvalue
                if(verbose > 0): pbar.update()

            if(verbose > 0): pbar.close()

            # Optimal matrices
            idx = np.argmax(qualities)
            W_in = W_in[idx]

            if(verbose > 0): print("Optimal scale: %lf" % (spectral_radius[idx,0,0]))

            # Draw plots
            if(verbose > 1):
                plt.figure(figsize=(12,3))
                if(find_optimal_input): 
                    plt.subplot(1, 4, 1)
                    plt.semilogy(spectral_radius[:,0,0], qualities)
                    plt.semilogy(spectral_radius[:,0,0][idx], qualities[idx], "o")
                    plt.xlabel("scales")
                    plt.ylabel("metric (input)")
                    plt.subplot(1, 4, 2)
                    plt.hist(act[idx])
                    plt.xlabel("activations")
                    plt.ylabel("counts")
        else:
            W_in = random_state.uniform(-1, 1, (n_reservoir, n_endo))
            W_in[random_state.rand(*W_in.shape) < 0.3] = 0


        # -----------------------------
        # Generate optimal W
        # -----------------------------

        if(verbose > 0): print("Hidden matrix generation...")

        # Initialization
        W = random_state.uniform(-1, 1, (steps, n_reservoir, n_reservoir)) * spectral_radius
        W[random_state.rand(*W.shape) < 0.3] = 0
        
        # Get activations
        #   Compute input pre-activations 
        input_preact = endo_states @ W_in.T
        act = np.zeros((steps, endo_states.shape[0], n_reservoir))

        #   Compute hidden pre-activations
        if(verbose > 0): pbar = tqdm(total=steps + endo_states.shape[0], position=0)
        for i in range(1, endo_states.shape[0]):
            act[:, i] = np.tanh(input_preact[i] + \
                np.einsum('ijk,ik->ij', W, act[:, i-1]))
            if(verbose > 0): pbar.update()
        act = act.reshape((steps, -1))

        # Get the values of the quality metric
        qualities = np.zeros((steps,))
        target = random_state.uniform(-1, 1, act.shape[1:])   
        for i in range(steps):
            qualities[i] = ks_2samp(act[i], target).pvalue
            if(verbose > 0): pbar.update()

        if(verbose > 0): pbar.close()

        # Optimal matrices
        idx = np.argmax(qualities)
        W = W[idx]

        if(verbose > 0): print("Optimal scale: %lf" % (spectral_radius[idx,0,0]))
        if(verbose > 0): print("-------------------------------------------\n")

        # Draw plots
        if(verbose > 1): 
            if(find_optimal_input): _n = 2
            else:
                plt.figure(figsize=(6,3))
                _n = 0
            
            plt.subplot(1, _n+2, _n+1)
            plt.semilogy(spectral_radius[:,0,0], qualities)
            plt.semilogy(spectral_radius[:,0,0][idx], qualities[idx], "o")
            plt.xlabel("scales")
            plt.ylabel("metric (hidden)")
            plt.subplot(1, _n+2, _n+2)
            plt.hist(act[idx])
            plt.xlabel("activations")
            plt.ylabel("counts")

        return W_in, W
    return _generator