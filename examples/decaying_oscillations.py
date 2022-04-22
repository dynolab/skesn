import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from skesn.esn import EsnForecaster
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin


def _exp_sin(t):
    return np.exp(-0.01*t) * np.sin(0.1*t)


if __name__ == '__main__':
    times = np.arange(0, 200, dtype=np.float64)
    time_series = _exp_sin(times)
    model = EsnForecaster(
        n_reservoir=200,
        #spectral_radius=0.95,
        spectral_radius=0.2,
        sparsity=0.8,
        #regularization='noise',
        #lambda_r=0.001,
        regularization='l2',
        lambda_r=0.001,
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
    fig, (ax, ax_metric) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    initial_test_times = []
    metric_values = []
    times = np.arange(len(time_series))
    ax.plot(times, time_series, linewidth=2)
    for test_index, y_pred, y_true in v.prediction_generator(model,
                                                             y=time_series,
                                                             X=None):
        ax.plot(times[test_index], y_pred, color='tab:orange', linewidth=2)
        ax.plot([times[test_index][0]], [y_pred[0]], 'o', color='tab:orange')
        initial_test_times.append(times[test_index][0])
        metric_values.append(v.metric(y_true, y_pred))
    ax.set_ylabel(r'$X_t$', fontsize=12)
    ax_metric.semilogy(initial_test_times, metric_values, 'o--')
    ax_metric.set_xlabel(r'$t$', fontsize=12)
    ax_metric.set_ylabel(r'MSE', fontsize=12)
    ax.grid()
    ax_metric.grid()
    plt.tight_layout()
    plt.show()
