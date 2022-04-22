import unittest
from math import isclose

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from skesn.esn import EsnForecaster
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin


class ValidationFunctionalCheck(unittest.TestCase):
    def test_rolling_forecasting_origin(self):
        times = np.arange(0, 200, dtype=np.float64)
        time_series = exp_sin(times)
        model = EsnForecaster(
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0,
            regularization='noise',
            lambda_r=0.001,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=True,
            random_state=None,
            use_bias=True
        )
        v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                      n_test_timesteps=10,
                                                      n_splits=10)
        v.evaluate(model, y=time_series, X=None)
        self.assertEqual(True, True)

    def test_grid_search(self):
        times = np.arange(0, 200, dtype=np.float64)
        time_series = exp_sin(times)
        model = EsnForecaster(
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0,
            regularization='noise',
            lambda_r=0.0,
            #regularization='l2',
            #lambda_r=0.001,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=False,
            random_state=None,
            use_bias=True
        )
        v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                      n_test_timesteps=10,
                                                      n_splits=15,
                                                      metric=mean_squared_error)
        res, best_model = v.grid_search(model,
                                        param_grid=dict(
                                            n_reservoir=[200, 300],
                                            spectral_radius=[0.5, 0.95],
                                        ),
                                        y=time_series,
                                        X=None)
        self.assertEqual(True, True)


def exp_sin(t):
    return np.exp(-0.01*t) * np.sin(0.1*t)
