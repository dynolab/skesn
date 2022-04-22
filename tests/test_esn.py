import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from skesn.esn import EsnForecaster
from skesn.misc import SklearnWrapperForForecaster, build_target_scaler


class EsnFunctionalCheck(unittest.TestCase):
    y_matrices = [
        {
            'original': np.array([1., 2., 3., 4.]),
            '2D': {
                'with_bias': 
                    np.array([
                        [1, 1.],
                        [1, 2.],
                        [1, 3.],
                        [1, 4.]
                    ]),
                'without_bias':
                    np.array([
                        [1.],
                        [2.],
                        [3.],
                        [4.]
                    ]),
            },
            '3D': {
                'with_bias': 
                    np.array([[
                        [1, 1.],
                        [1, 2.],
                        [1, 3.],
                        [1, 4.]
                    ]]),
                'without_bias':
                    np.array([[
                        [1.],
                        [2.],
                        [3.],
                        [4.]
                    ]]),
            }
            
        },
        {
            'original': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '2D': {
                'with_bias':
                    np.array([
                        [1., 1., 2., 3., 4.],
                        [1., 5., 6., 7., 8.],
                    ]),
                'without_bias':
                    np.array([
                        [1., 2., 3., 4.],
                        [5., 6., 7., 8.],
                    ]),
            },            
            '3D': {
                'with_bias':
                    np.array([[
                        [1., 1., 2., 3., 4.],
                        [1., 5., 6., 7., 8.],
                    ]]),
                'without_bias':
                    np.array([[
                        [1., 2., 3., 4.],
                        [5., 6., 7., 8.],
                    ]]),
            },
        },
        {
            'original': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
            '2D': {
                'with_bias': None,
                'without_bias': None,
            },
            '3D': {
                'with_bias':
                    np.array([
                        [
                            [1., 1., 2., 3., 4.],
                            [1., 5., 6., 7., 8.],
                        ],
                        [
                            [1., 9., 10., 11., 12.],
                            [1., 13., 14., 15., 16.],
                        ],
                    ]),
                'without_bias':
                    np.array([
                        [
                            [1., 2., 3., 4.],
                            [5., 6., 7., 8.],
                        ],
                        [
                            [9., 10., 11., 12.],
                            [13., 14., 15., 16.],
                        ],
                    ]),
            }
        },
    ]
    X_matrices = [
        {
            'original': None,
            '2D': None,
            '3D': None,
        },
        {
            'original': np.array([1., 2., 3., 4.]),
            '2D': np.array([
                [1.],
                [2.],
                [3.],
                [4.]
            ]),
            '3D': np.array([[
                [1.],
                [2.],
                [3.],
                [4.]
            ]]),
        },
        {
            'original': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '2D': np.array([
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]),
            '3D': np.array([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
            ]]),
        },
        {
            'original': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
            '2D': None,
            '3D': np.array([
                [
                    [1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                ],
                [
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.],
                ],
            ]),
        },
    ]

    def test_treat_dimensions_and_bias(self):
        def _to_words(use_bias_):
            return 'with_bias' if use_bias_ else 'without_bias'
        for use_bias in (True, False):
            model = EsnForecaster(use_bias=use_bias)
            for y, X in zip(self.y_matrices, self.X_matrices):
                for repr in ('2D', '3D'):
                    if y[repr][_to_words(use_bias)] is not None:
                        y_corrected, X_corrected = model._treat_dimensions_and_bias(y['original'], X['original'],
                                                                                                       representation=repr)
                        for arr_corrected, arr_true in ((y_corrected, y[repr][_to_words(use_bias)]), 
                                                        (X_corrected, X[repr])):
                            if arr_true is None:
                                self.assertEqual(arr_corrected, arr_true)
                            else:
                                np.testing.assert_allclose(arr_corrected, arr_true)
                    else:
                        self.assertRaises(ValueError, model._treat_dimensions_and_bias, 
                                          y['original'], X['original'], representation=repr)

    def test_fit_predict_on_constant_timeseries(self):
        time_series = np.ones((100,), dtype=np.float64)
        model = EsnForecaster(
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0,
            regularization='l2',
            lambda_r=0.001,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=True,
            random_state=None,
            use_bias=True
        )

        model.fit(time_series)
        self.assertEqual(model, model.fit(time_series))
        n_timesteps = 10
        np.testing.assert_allclose(np.ones((n_timesteps, 1)),
                                   model.predict(n_timesteps),
                                   rtol=10**(-4),
                                   atol=10**(-4))

    def test_fit_predict_on_exponentially_decaying_timeseries(self):
        times = np.arange(0, 200, dtype=np.float64)
        time_series = exp_sin(times)
        model = EsnForecaster(
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0,
            #regularization='noise',
            regularization='l2',
            lambda_r=0.001,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=True,
            random_state=None,
            use_bias=True
        )

        model.fit(time_series)
        self.assertEqual(model, model.fit(time_series))
        n_timesteps = 10
        further_times = np.arange(200, 200 + n_timesteps, dtype=np.float64)
        np.testing.assert_allclose(exp_sin(further_times),
                                   model.predict(n_timesteps)[:, 0],
                                   rtol=10**(-2),
                                   atol=10**(-2))

    def test_pipeline_fit_predict_on_exponentially_decaying_timeseries(self):
        times = np.arange(0, 200, dtype=np.float64)
        time_series = exp_sin(times)
        esn_ = EsnForecaster(
            n_reservoir=200,
            spectral_radius=0.95,
            sparsity=0,
            #regularization='noise',
            regularization='l2',
            lambda_r=0.001,
            in_activation='tanh',
            out_activation='identity',
            use_additive_noise_when_forecasting=True,
            random_state=None,
            use_bias=True
        )

        pipe = Pipeline(steps=[
            ('forecaster', SklearnWrapperForForecaster(esn_)),
        ])
        model = build_target_scaler(pipe)
        self.assertEqual(model, model.fit(X=None, y=time_series))
        n_timesteps = 10
        further_times = np.arange(200, 200 + n_timesteps, dtype=np.float64)
        np.testing.assert_allclose(exp_sin(further_times),
                                   model.predict(X=None, n_timesteps=n_timesteps),
                                   rtol=10**(-2),
                                   atol=10**(-2))


def exp_sin(t):
    return np.exp(-0.01*t) * np.sin(0.1*t)
