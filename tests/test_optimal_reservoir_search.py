import unittest

import numpy as np

from skesn.esn import EsnForecaster
from skesn.weight_generators import optimal_weights_generator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import isclose

def _lorenz(x_0, dt, t_final):
    sigma_ = 10.
    beta_ = 8./3.
    rho_ = 28.

    def rhs(x):
        f_ = np.zeros(3)
        f_[0] = sigma_ * (x[1] - x[0])
        f_[1] = rho_ * x[0] - x[0] * x[2] - x[1]
        f_[2] = x[0] * x[1] - beta_ * x[2]
        return f_

    times = np.arange(0, t_final, dt)
    ts = np.zeros((len(times), 3))
    ts[0, :] = x_0
    cur_x = x_0
    dt_integr = 10**(-3)
    n_timesteps = int(np.ceil(dt / dt_integr))
    dt_integr = dt / n_timesteps
    for i in range(1, n_timesteps*len(times)):
        cur_x = cur_x + dt_integr * rhs(cur_x)
        saved_time_i = i*dt_integr / dt
        if isclose(saved_time_i, np.round(saved_time_i)):
            saved_time_i = int(np.round(i*dt_integr / dt))
            ts[saved_time_i, :] = cur_x
    return ts, times

class OptResFunctionalCheck(unittest.TestCase):

    def test_logging(self):
        X = np.sin(np.linspace(0, 3 * np.pi, 100))
        model = EsnForecaster(
            n_reservoir=100,
            spectral_radius=1.,
            sparsity=0.9,
            regularization='l2',
            lambda_r=1e-4,
            in_activation='tanh',
            random_state=0,
        )
        
        for verbose in range(3):
            for find_i in [True, False]:
                model.fit(X, inspect = False, initialization_strategy = optimal_weights_generator(
                    verbose = verbose,
                    range_generator=np.linspace,
                    steps = 100,
                    hidden_std = 0.5,
                    find_optimal_input = find_i,
                    thinning_step = 1,
                ))
                plt.show()
        
        self.assertTrue(True)
        
    def test_parameters(self):
        configs = [
            {
                'range_generator': np.logspace,
                'steps': 100,
                'thinning_step': 1,
            },
            {
                'range_generator': np.linspace,
                'steps': 100,
                'thinning_step': 1,
            },
            {
                'range_generator': np.linspace,
                'steps': 200,
                'thinning_step': 1,
            },
            {
                'range_generator': np.linspace,
                'steps': 100,
                'thinning_step': 10,
            },
        ]

        X = np.sin(np.linspace(0, 3 * np.pi, 100))
        model = EsnForecaster(
            n_reservoir=100,
            spectral_radius=1.,
            sparsity=0.9,
            regularization='l2',
            lambda_r=1e-4,
            in_activation='tanh',
            random_state=0,
        )

        for kwargs in configs:
            model.fit(X, inspect = False, initialization_strategy = optimal_weights_generator(
                verbose = 2,
                hidden_std = 0.5,
                find_optimal_input = True,
                **kwargs,
            ))
            plt.show()
        
        self.assertTrue(True)

    def test_improvement(self):
        np.random.seed(0)

        data, time = _lorenz(np.random.rand(3,), 2e-3, 40)
        data = data[10000::10]
        time = time[:-10000:10]

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        model = EsnForecaster(
            n_reservoir=100,
            spectral_radius=0.99,
            sparsity=0.9,
            regularization='l2',
            lambda_r=1e-4,
            in_activation='tanh',
            random_state=0,
        )

        model.fit(data[:1000])
        p = model.predict(250)
        err_bad = mean_squared_error(p, data[750:])
        print("!!! MSE before optimization: %lf" % (err_bad, ))

        model.spectral_radius = (0,0.2)
        model.random_state_ = np.random.RandomState(0)
        model.fit(data[:750], inspect = False, initialization_strategy = optimal_weights_generator(
            verbose = 2,
            range_generator=np.linspace,
            steps = 100,
            hidden_std = 0.5,
            find_optimal_input = False,
            thinning_step = 10,
        ))
        plt.show()
        p = model.predict(250)
        err_good = mean_squared_error(p, data[750:])
        print("!!! MSE after optimization: %lf" % (err_good, ))

        self.assertGreater(err_bad, err_good)