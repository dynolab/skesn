import numpy as np

class Controller:
    def __init__(self):
        """
        Class constructor. It is used to set all the 
        necessary parameters of the controller.
        """
        raise NotImplementedError()

    def initialize(self, esn, sig):
        """The function is called when the reservoir 
        matrices in ESNForecaster are initialized.

        Parameters
        ----------
        esn : object of class ESNForecaster 
              Controlled ESN.
        sig : array-like or function, shape (batch_size x n_timesteps x n_controls)
              Control signal.
        """
        raise NotImplementedError()

    def preact(self, pre, sig):
        """A function that injects unique behavior 
        during the pre-activation phase of the 
        reservoir (when training and predicting). 

        Parameters
        ----------
        pre : array-like, shape (n_reservoir)
              Calculated preactivations of the 
              reservoir activation function.
        sig : array-like or function, shape (n_controls)
              Control signal.

        Returns
        -------
        The override function should return a new pre-activation, shape (n_reservoir)
        """
        raise NotImplementedError()

    def prereg(self, hiddens, targets, sig):
        """A function that injects a unique behavior 
        before the linear regression phase. 

        Parameters
        ----------
        hiddens : array-like, shape (batch_size x n_timesteps x n_reservoir)
                  Collected hidden states.
        targets : array-like or function, shape (batch_size x n_timesteps x n_inputs)
                  Training dataset.
        sig     : array-like or function, shape (batch_size x n_timesteps x n_controls)
                  Control signal.

        Returns
        -------
        The override function should return a new set 
        of hidden states, shape (batch_size x n_timesteps x n_reservoir)
        """
        raise NotImplementedError()

class InjectedController(Controller):
    """Controller that injects the control signal into the reservoir.
    """

    def __init__(self): pass

    def initialize(self, esn, sig):
        """The function generates a weight matrix 
        W_c and saves the ESN model
        """
        if(sig is None): return None
        self.esn = esn
        self.W_c = esn.random_state_.randn(esn.n_reservoir, sig.shape[-1]) / np.sqrt(sig.shape[-1]*3)

    def preact(self, pre, sig):
        """The function adds a linear transformation of 
        the control signal to the existing pre-activation
        """
        return pre + np.dot(self.W_c, sig)

    def prereg(self, hiddens, targets, sig): 
        return hiddens

class HomotopyController(Controller):
    """Controller that controls ESN with Homotopy.

    Parameters
    ----------
    use_transfer_learning : bool
        Should transfer learning be used to calculate 
        matrices of continuous transformation
    anchor_signal : float or list
        The relative position of the anchor signal between 
        the minimum and maximum (should be between 0 and 1). 
        The matrix calculated with this signal is used 
        as a basis for transfer learning. 
    mu : float
        Parameter regulating transfer learning. 
    """

    def _rbf_phi(self, x):
        return  -np.sqrt(1+(self.eps * x**2))# np.exp(-self.eps*x**2)# x**2 * np.log(x + self.eps)#

    def _rbf_memorize(self, x, y):
        # self._rbf_x = x
        # H = np.array([np.linalg.norm(x[i] - x, axis = 1) for i in range(len(x))])
        # # print(H.max())
        # H = self._rbf_phi(H)
        # self._rbf_weights = np.tensordot(np.linalg.inv(H.T @ H) @ H.T, y, axes=[1,0])
        y = np.array(y)

        h = x[1:,0]-x[:-1,0]
        w = y[1:]-y[:-1]
        # self.c = 3*(w.T/h**2).T
        # self.d = -2*(w.T/h**3).T
        # self.k = (w.T/h**2).T

        self.k = (w.T/h).T

        self.x = x
        self.y = y


    def _rbf_call(self, x):
        # h = np.array([self._rbf_phi(np.linalg.norm(x - self._rbf_x, axis=1))])
        # return np.tensordot(h, self._rbf_weights, axes=[1,0])
        
        # idx = np.argmin(x > self.x)-1
        # return self.y[idx] + self.c[idx] * (x-self.x[idx])**2 + self.d[idx] * (x-self.x[idx])**3

        idx = np.argmin(x > self.x)-1
        return self.y[idx] + self.k[idx] * (x - self.x[idx])


    def __init__(self, use_transfer_learning = True, anchor_signal = 0.5, mu=1e-2, eps = 1e-6): 
        """Simple parameter initialization.
        """
        self.use_tl_ = use_transfer_learning
        self.anc_sig_ = anchor_signal
        self.mu = mu
        self.eps = eps

    def initialize(self, esn, sig): 
        """The function saves the ESN model
        """
        self.esn = esn

    def preact(self, pre, sig): 
        """The function changes the W_out matrix of the ESN 
        according to the control signal using a continuous 
        transformation between the existing weight matrices 
        during the prediction phase at the preactivation step.
        """
        if(hasattr(self, "weights") and hasattr(self.esn, "W_out_")):
            s = sig[0]

            # # Extrapolation
            # if(s < self.signals[0]):
            #     t = (self.signals[0] - s) / (self.signals[1] - self.signals[0])
            #     self.esn.W_out_ = self.weights[0] + (self.weights[0] - self.weights[1]) * t
            # elif(s > self.signals[-1]):
            #     t = (s - self.signals[-1]) / (self.signals[-1] - self.signals[-2])
            #     self.esn.W_out_ = self.weights[-1]+ (self.weights[-1]- self.weights[-2])* t
            # # Interpolation
            # else:
            #     for idx in range(len(self.signals)-1):
            #         if(s > self.signals[idx] and \
            #             s < self.signals[idx+1]):
            #             break
            #     t = (s - self.signals[idx]) / (self.signals[idx+1] - self.signals[idx])
            #     self.esn.W_out_ = self.weights[idx] + (self.weights[idx+1] - self.weights[idx]) * t
            self.esn.W_out_ = self._rbf_call(s)
        return pre

    def prereg(self, hiddens, targets, sig): 
        """The function calculates and stores all 
        the matrices of the continuous transformation 
        for the different control signals.
        """
        if(sig is None): return hiddens
        signals = np.unique(sig[:, 0], axis=0)
        _min, _max = signals.min(axis=0), signals.max(axis=0)
        P = _min + self.anc_sig_ * (_max - _min)
        idx = np.linalg.norm(signals - P, axis=-1).argmin()

        # print("Anchor signal: %s" % str(signals[idx]))

        # First calculate the anchor matrix for transfer
        signals[[0, idx]] = signals[[idx, 0]]

        self.weights = []
        main_ids = []
        for i in range(len(signals)):
            ids = []
            for j in range(len(hiddens)):
                if(sig[j, 0, 0] == signals[i]): ids.append(j)
            z = hiddens[ids].reshape((-1, hiddens.shape[-1]))
            y = targets[ids].reshape((-1, targets.shape[-1]))
            if(i == 0 or not self.use_tl_):
                if 'l2' in self.esn.regularization:
                    idenmat = self.esn.lambda_r * np.identity(z.shape[-1])
                    U = np.dot(z.T, z) + idenmat
                    W = np.linalg.solve(U, z.T @ y).T
                elif 'noise' in self.esn.regularization or self.esn.regularization is None:
                    U = np.dot(z.T, z)
                    W = np.linalg.solve(U, z.T @ y).T
                else:
                    raise ValueError(f'Unknown regularization: {self.esn.regularization}')
            else:
                identmat = self.mu * np.identity(z.shape[-1])
                R = z.T @ z
                dW = np.linalg.solve(R + identmat, z.T @ y - (self.weights[0] @ R).T).T
                W = self.weights[0] + dW
            self.weights.append(W)

        signals[[0, idx]] = signals[[idx, 0]]
        self.weights[0], self.weights[idx] = self.weights[idx], self.weights[0]
        # self.signals = signals
        self._rbf_memorize(signals, self.weights)
        
        return hiddens