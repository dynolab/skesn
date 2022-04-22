from math import isclose
import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from skesn.esn import EsnForecaster
from skesn.cross_validation import ValidationBasedOnRollingForecastingOrigin


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


if __name__ == '__main__':
    coord_names = [r'$X_t$', r'$Y_t$', r'$Z_t$']
    esn_dt = 0.01
    model = EsnForecaster(
        n_reservoir=300,
        spectral_radius=0.2,
        sparsity=0.8,
        #regularization='noise',
        #lambda_r=0.1,
        regularization='l2',
        lambda_r=0.0001,
        in_activation='tanh',
        out_activation='identity',
        use_additive_noise_when_forecasting=False,
        random_state=None,
        use_bias=True)
    
    # Plot short-term forecasting skill assessment based on rolling forecasting origin
    test_time_length = 5
    n_splits = 19
    n_ics = 100
    random_ics = np.random.rand(n_ics, 3)  # (0; 1)
    random_ics -= 0.5  # (-0.5; 0.5)
    random_ics[:, 2] *= 10.  # z: (-5; 5)
    random_ics[:, 2] += 25.  # z: (20; 30)
    metric_values = np.zeros((random_ics.shape[0], n_splits))
    ss = StandardScaler()
    ts_for_plotting = None
    times_for_plotting = None
    forecasting_origins = np.zeros((n_splits,))
    forecasting_origins_times = np.zeros((n_splits,))
    for i in range(random_ics.shape[0]):
        print(f'Processing IC #{i+1} out of {random_ics.shape[0]}')
        ts, times = _lorenz(random_ics[i, :], dt=esn_dt, t_final=100)
        if ts_for_plotting is None:
            ts_for_plotting = ts
            times_for_plotting = times
        ts = ss.fit_transform(ts)
        v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                      n_test_timesteps=int(test_time_length / esn_dt),
                                                      n_splits=n_splits,
                                                      metric=mean_squared_error)
        j = 0
        for test_index, y_pred, y_true in v.prediction_generator(model,
                                                                 y=ts,
                                                                 X=None):
            y_pred = ss.inverse_transform(y_pred)
            y_true = ss.inverse_transform(y_true)
            metric_values[i, j] = v.metric(y_true[:, 0], y_pred[:, 0])
            if i == 0:
                forecasting_origins[j] = y_pred[0, 0]
                forecasting_origins_times[j] = times[test_index][0]
            j += 1
    fig, (ax, ax_metric) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax.plot(times_for_plotting, ts_for_plotting[:, 0], linewidth=2)
    ax.plot(forecasting_origins_times, forecasting_origins, 'o')
    ax_metric.boxplot(metric_values, positions=forecasting_origins_times,
                      widths=1.5)
    ax_metric.set_yscale('log')
    #xticks = ax_metric.get_xticks()
    #ax_metric.set_xticks([0.] + list(xticks) + [100.])
    ax_metric.set_xlabel(r'$t$', fontsize=12)
    ax_metric.set_ylabel('MSE', fontsize=12)
    ax_metric.grid()
    ax.grid()
    plt.tight_layout()
    plt.show()
    print(metric_values)
