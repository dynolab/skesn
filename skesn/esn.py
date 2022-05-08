import numpy as np
from tqdm import tqdm

from skesn.base import BaseForecaster
from skesn.misc import correct_dimensions, identity
from skesn.weight_generators import standart_weights_generator

from enum import Enum
update_modes = Enum("update_modes", "synchronization transfer_learning refit")


ACTIVATIONS = {
    'identity': {
        'direct': identity,
        'inverse': identity,
    },
    'tanh': {
        'direct': np.tanh,
        'inverse': np.arctanh,  
    },
    'relu': {
        'direct': lambda x: np.maximum(0.,x),
        'inverse': lambda x: np.maximum(0.,x),
    },
    'leaky_relu': {
        'direct': lambda x: np.maximum(0.,x) + np.minimum(0.,x)*0.001,
        'inverse': lambda x: np.maximum(0.,x) + np.minimum(0.,x)*0.001,
    },
    'flat_relu': {
        'direct': lambda x: np.minimum(np.maximum(0.,x), 1.),
        'inverse': lambda x: np.minimum(np.maximum(0.,x), 1.),
    },
    'gauss': {
        'direct': lambda x: np.exp(-x**2/2),
        'inverse': lambda x: -(np.log(x) * 2)**0.5,
    },
    'rel_gauss': {
        'direct': lambda x: np.maximum(np.minimum(x/2+0.5, 0.5),-0.5)+np.maximum(np.minimum(-x/2+0.5, 0.5),-0.5),
        'inverse': lambda x: -2*x+2
    },
    'sin': {
        'direct': np.sin,
        'inverse': np.arcsin
    },
}

class EsnForecaster(BaseForecaster):
    """Echo State Network time-forecaster.

    Parameters
    ----------
    spectral_radius : float
        Spectral radius of the recurrent weight matrix W
    sparsity : float
        Proportion of elements of the weight matrix W set to zero
    regularization : {'noise', 'l2', None}
        Type of regularization
    lambda_r : float
        Regularization parameter value which will be multiplied
        with the regularization term
    noise : float
        Noise stregth; added to each reservoir state element for regularization
    in_activation : {'identity', 'tanh'}
        Input activation function (applied to the linear
        combination of input, reservoir and control states)
    out_activation : {'identity', 'tanh'}
        Output activation function (applied to the readout)
    use_additive_noise_when_forecasting : bool
        Whether additive noise (same as used for regularization) is added
        before activation in the forecasting mode
    inverse_out_activation : function
        Inverse of the output activation function
    random_state : positive integer seed, np.rand.RandomState object,
                   or None to use numpy's builting RandomState.
        Used as a seed to randomize matrices W_in, W and W_c
    use_bias : bool
        Whether a bias term is added before activation
    """
    def __init__(self,
                 n_reservoir=200,
                 spectral_radius=0.95,
                 sparsity=0,
                 regularization='noise',
                 lambda_r=0.001,
                 in_activation='tanh',
                 out_activation='identity',
                 use_additive_noise_when_forecasting=True,
                 random_state=None,
                 use_bias=True):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.regularization = regularization
        self.lambda_r = lambda_r
        self.use_additive_noise_when_forecasting = \
            use_additive_noise_when_forecasting
        if self.regularization == 'l2':
            self.use_additive_noise_when_forecasting = False
        self.in_activation = in_activation
        self.out_activation = out_activation
        self.random_state = random_state
        self.use_bias = use_bias

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state is not None:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand
        super().__init__()

    def get_fitted_params(self):
        """Get fitted parameters. Overloaded method from BaseForecaster

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return {
            'W_in': self.W_in_,
            'W': self.W_,
            'W_c': self.W_c_,
            'W_out': self.W_out_,
        }

    def _fit(self, y, X=None,
             initialization_strategy=standart_weights_generator,
             inspect=False):
        """Fit forecaster to training data. Overloaded method from
        BaseForecaster.
        Generates random recurrent matrix weights and fits the readout weights
        to the available time series (endogeneous time series).
        After that the function of calculating the optimal matrix W_out is called.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : array-like, shape (batch_size x n_timesteps x n_inputs)
            Time series to which to fit the forecaster
            or a sequence of them (batches).
        X : array-like or function, shape (batch_size x n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to fit to or a sequence of them (batches).
            Can also be understood as a control signal
        initialization_strategy: function
            A function generating random matrices W_in, W and W_c
        inspect : bool
            Whether to show a visualisation of the collected reservoir states

        Returns
        -------
        self : returns an instance of self.
        """
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='3D')
        self.W_in_, self.W_, self.W_c_ = \
            initialization_strategy(self.random_state_,
                                    self.n_reservoir,
                                    self.sparsity,
                                    self.spectral_radius,
                                    endo_states,
                                    exo_states=exo_states)
        
        return self._update_via_refit(endo_states, exo_states, inspect)

    def _predict(self, n_timesteps, X=None, inspect=False):
        """Forecast time series at further time steps.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        n_timesteps : int
            Forecasting horizon
        X : array-like or function, shape (n_timesteps x n_controls), optional
            Exogeneous time series to fit to.
            Can also be understood as a control signal

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh

        Returns:
            Array of output activations
        """
        if X is None or X[0] is None:  # TODO: some sklearn subroutines like gridsearchcv
            # cannot pass X=None and pass array of None. That's why we check X[0] here
            exo_states = None
        elif isinstance(X, np.ndarray):
            exo_states = correct_dimensions(X)
        if n_timesteps is None:  # TODO: this is actually a stupid bypass because
            # some sklearn subroutines like gridsearchcv cannot pass any kwargs to predict()
            if X is not None:
                n_timesteps = len(X)
            else:
                raise ValueError('No way to deduce the number of time steps: both n_timesteps and X are set to None')
        n_endo = self.last_endo_state_.shape[0]
        n_reservoir = self.last_reservoir_state_.shape[0]

        endo_states = np.vstack([self.last_endo_state_,
                                 np.zeros((n_timesteps, n_endo))])
        reservoir_states = np.vstack([self.last_reservoir_state_,
                                      np.zeros((n_timesteps, n_reservoir))])
        if inspect:
            print("predict...")
            pbar = tqdm(total=n_timesteps, position=0, leave=True)
        for n in range(n_timesteps):
            reservoir_state = reservoir_states[n, :]
            endo_state = endo_states[n, :]
            exo_state = None
            if exo_states is None:
                exo_state = None
            elif isinstance(exo_states, np.ndarray):
                exo_state = exo_states[n, :]
            else:  # exo_states is assumed to be a callable
                exo_state = exo_states(n, endo_state)
            reservoir_states[n + 1, :] = \
                self._iterate_reservoir_state(reservoir_state,
                                              endo_state,
                                              exo_state)
            endo_states[n + 1, :] = \
                ACTIVATIONS[self.out_activation]['direct'](np.dot(self.W_out_,
                                                                  reservoir_states[n + 1, :]))
            if self.use_bias:
                endo_states[n + 1, 0] = 1  # bias
            if inspect:
                pbar.update(1)
        if inspect:
            pbar.close()
        if self.use_bias:
            return endo_states[1:, 1:]
        else:
            return endo_states[1:]

    def _update(self, y, X=None, mode='synchronization', inspect=False):
        """Update the model to incremental training data.
        Depending on the mode, it can be done as synchronization
        or transfer learning.

        Writes to self:
            If mode == 'synchronization'
                updates last_reservoir_state_ and last_endo_state_
            If mode == 'transfer_learning'
                updates W_out_, last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to update the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to update to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        if mode == update_modes.synchronization:
            return self._update_via_synchronization(y, X)
        elif mode == update_modes.transfer_learning:
            return self._update_via_transfer_learning(y, X, mu=1e-8, inspect=inspect)
        elif mode == update_modes.refit:
            endo_states, exo_states = \
                self._treat_dimensions_and_bias(y, X, representation='3D')
            return self._update_via_refit(endo_states, exo_states, inspect)

    def _update_via_refit(self, endo_states, exo_states=None, inspect=False):
        """Refit forecaster to training data. 
        The model can be fitted based on a single time series (batch_size == 1)
        or a sequence of disconnected time series (batch_size > 1).
        Optionally, a multivariate control signal (exogeneous time series)
        can be passed which we will also be included into fitting process.
        Note that in this case, the control signal must also be passed during
        prediction.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        endo_states : array-like, shape (batch_size x n_timesteps x n_inputs)
            Time series to which to fit the forecaster
            or a sequence of them (batches).
        exo_states : array-like or function, shape (batch_size x n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to fit to or a sequence of them (batches).
            Can also be understood as a control signal
        inspect : bool
            Whether to show a visualisation of the collected reservoir states

        Returns
        -------
        self : returns an instance of self.
        """
        n_batches = endo_states.shape[0]
        n_timesteps = endo_states.shape[1]
        reservoir_states = np.zeros((n_batches, n_timesteps, self.n_reservoir))

        if inspect:
            print("fitting...")
            pbar = tqdm(total=n_batches*n_timesteps,
                        position=0,
                        leave=True)

        for b in range(n_batches):
            for n in range(1, n_timesteps):
                if exo_states is None:
                    reservoir_states[b, n, :] = \
                        self._iterate_reservoir_state(reservoir_states[b, n - 1],
                                                      endo_states[b, n - 1, :])
                else:
                    reservoir_states[b, n, :] = \
                        self._iterate_reservoir_state(reservoir_states[b, n - 1],
                                                      endo_states[b, n - 1, :],
                                                      exo_states[b, n, :])
                if inspect:
                    pbar.update(1)
        if inspect:
            pbar.close()

        reservoir_states = np.reshape(reservoir_states,
                                      (-1, reservoir_states.shape[-1]))
        endo_states = np.reshape(endo_states,
                                 (-1, endo_states.shape[-1]))
        if inspect:
            print("solving...")
        if self.regularization == 'l2':
            idenmat = self.lambda_r * np.identity(self.n_reservoir)
            U = np.dot(reservoir_states.T, reservoir_states) + idenmat
            self.W_out_ = np.linalg.solve(U, reservoir_states.T @ endo_states).T
        elif self.regularization == 'noise' or self.regularization is None:
            # same formulas as above but with lambda = 0
            U = np.dot(reservoir_states.T, reservoir_states)
            self.W_out_ = np.linalg.solve(U, reservoir_states.T @ endo_states).T
        else:
            raise ValueError(f'Unknown regularization: {self.regularization}')
        # remember the last state for later:
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_state_ = endo_states[-1, :]
        if exo_states is None:
            self.last_exo_state_ = 0
        else:
            raise NotImplementedError('forgot to implement')
        return self

    def _update_via_synchronization(self, y, X=None):
        """Update the model to incremental training data
        by synchnorizing the reservoir state with the
        given time series.

        Writes to self:
            updates last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to synchronize the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to synchronize to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='2D')
        n_timesteps = y.shape[0]
        reservoir_states = np.zeros((n_timesteps, self.n_reservoir))
        for n in range(1, n_timesteps):
            if X is None:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1, :],
                                                  endo_states[n - 1, :])
            else:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1, :],
                                                  endo_states[n - 1, :],
                                                  exo_states[n - 1, :])
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_state_ = endo_states[-1, :]
        return self

    def _update_via_transfer_learning(self, y, X=None, mu=1e-8, inspect=False):
        """Update the model to incremental training data using transfer
        learning.

        Writes to self:
            updates W_out_, last_reservoir_state_ and last_endo_state_

        Parameters
        ----------
        y : array-like, shape (n_timesteps x n_inputs)
            Time series to which to synchronize the forecaster.
        X : array-like or function, shape (n_timesteps x n_controls), optional (default=None)
            Exogeneous time series to synchronize to.
            Can also be understood as a control signal

        Returns
        -------
        self
        """
        endo_states, exo_states = \
            self._treat_dimensions_and_bias(y, X, representation='2D')
        n_timesteps = endo_states.shape[0]

        # step the reservoir through the given input,output pairs:
        reservoir_states = np.zeros((n_timesteps, self.n_reservoir))

        if inspect:
            print("transfer...")
            pbar = tqdm(total=n_timesteps - 1,
                        position=0,
                        leave=True)

        for n in range(1, n_timesteps):
            if(exo_states is None):
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1],
                                                  endo_states[n - 1, :])
            else:
                reservoir_states[n, :] = \
                    self._iterate_reservoir_state(reservoir_states[n - 1],
                                                  endo_states[n - 1, :],
                                                  exo_states[n, :])
            if inspect:
                pbar.update(1)
        if inspect:
            pbar.close()

        identmat = mu * np.identity(self.n_reservoir)
        if inspect:
            print("solving...")
        R = reservoir_states.T @ reservoir_states
        dW = np.linalg.solve(R + identmat, reservoir_states.T @ endo_states - (self.W_out_ @ R).T).T
        self.W_out_ += dW

        # remember the last state for later:
        self.last_reservoir_state_ = reservoir_states[-1, :]
        self.last_endo_input_ = endo_states[-1, :]
        return self

    def _iterate_reservoir_state(self, reservoir_state, endo_state,
                                 exo_state=None,
                                 forecasting_mode=False):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and
        output patterns.
        """
        n_reservoir = reservoir_state.shape[0]
        preactivation = np.dot(self.W_, reservoir_state) + np.dot(self.W_in_,
                                                                  endo_state)
        if exo_state is not None:
            preactivation += np.dot(self.W_c_, exo_state)
        s = ACTIVATIONS[self.in_activation]['direct'](preactivation)
        if (forecasting_mode and self.use_additive_noise_when_forecasting) or \
           (not forecasting_mode and self.regularization == 'noise'):
            s += self.lambda_r * (self.random_state_.rand(n_reservoir) - 0.5)
        return s

    def _treat_dimensions_and_bias(self, y, X=None, representation='2D'):
        """Transform array shapes following the specified representation
        and add the bias term if necessary.

        Parameters
        ----------
        y : array-like
            Time series of endogeneous states.
        X : array-like or function, optional (default=None)
            Time series of exogeneous states.
            Can also be understood as a control signal

        Returns
        -------
        endo_states, exo_states
        """
        y = correct_dimensions(y, representation=representation)
        if y is None:
            raise ValueError(f'Inconsistent combination of y shape '
                             f'and {representation} representation')
        if X is None or X[0] is None:
            X = None
        elif isinstance(X, np.ndarray):
            X = correct_dimensions(X, representation=representation)
        exo_states = X
        endo_states = y
        if(self.use_bias):
            ones_shape = None
            if representation == '2D':
                ones_shape = (endo_states.shape[0], 1)
            elif representation == '3D':
                ones_shape = (endo_states.shape[0], endo_states.shape[1], 1)
            else:
                raise ValueError(f'Unsupported representation: '
                                 f'{representation}')
            endo_states = np.concatenate((np.ones(ones_shape), endo_states),
                                         axis=-1)
        return endo_states, exo_states
