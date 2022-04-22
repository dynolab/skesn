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
    ts, times = _lorenz([0.1, 0.2, 25.], dt=esn_dt, t_final=100)
    ss = StandardScaler()
    ts = ss.fit_transform(ts)
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
    
    # Plot long-term prediction
    n_prediction = int(ts.shape[0] / 3)
    dt = times[1] - times[0]
    future_times = np.arange(times[-1] + dt, times[-1] + dt * (n_prediction + 1), dt)
    model.fit(ts)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    for i in range(ts.shape[1]):
        axes[i].plot(times, ss.inverse_transform(ts)[:, i],
                     linewidth=2,
                     label='True')
        axes[i].plot(future_times, ss.inverse_transform(model.predict(n_prediction))[:, i],
                     linewidth=2,
                     label='Prediction')
        ylim = axes[i].get_ylim()
        axes[i].fill_between(times, y1=ylim[0], y2=ylim[1], color='tab:blue', alpha=0.2)
        axes[i].set_ylabel(coord_names[i], fontsize=12)
        axes[i].grid()
        axes[i].legend()
    axes[0].set_title('ESN long-term prediction of the Lorenz system', fontsize=16)
    axes[-1].set_xlabel(r'$t$', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Plot short-term forecasting skill assessment based on rolling forecasting origin
    test_time_length = 5
    v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                  n_test_timesteps=int(test_time_length / esn_dt),
                                                  n_splits=18,
                                                  metric=mean_squared_error)
    fig, (ax, ax_metric) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    initial_test_times = []
    metric_values = []
    ax.plot(times, ss.inverse_transform(ts)[:, 0], linewidth=2)
    for test_index, y_pred, y_true in v.prediction_generator(model,
                                                             y=ts,
                                                             X=None):
        y_pred = ss.inverse_transform(y_pred)
        y_true = ss.inverse_transform(y_true)
        ax.plot(times[test_index], y_pred[:, 0], color='tab:orange', linewidth=2)
        ax.plot([times[test_index][0]], [y_pred[0, 0]], 'o', color='tab:red')
        initial_test_times.append(times[test_index][0])
        metric_values.append(v.metric(y_true[:, 0], y_pred[:, 0]))
    ax.set_ylabel(r'$X_t$', fontsize=12)
    ax_metric.semilogy(initial_test_times, metric_values, 'o--')
    ax_metric.set_xlabel(r'$t$', fontsize=12)
    ax_metric.set_ylabel(r'MSE', fontsize=12)
    ax.grid()
    ax_metric.grid()
    plt.tight_layout()
    plt.show()

    # Plot hyperparameter grid search results based on rolling forecasting origin
    test_time_length = 2
    n_splits = 40
    v = ValidationBasedOnRollingForecastingOrigin(n_training_timesteps=None,
                                                  n_test_timesteps=int(test_time_length / esn_dt),
                                                  n_splits=n_splits,
                                                  metric=mean_squared_error)
    summary, best_model = v.grid_search(model,
                                        param_grid=dict(
                                            spectral_radius=[0.5, 0.95],
                                            sparsity=[0, 0.8],
                                            lambda_r=[0.01, 0.001, 0.0001]),
                                        y=ts,
                                        X=None)
    summary_df = pd.DataFrame(summary).sort_values('rank_test_score')
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
    table_rows = []
    param_names = list(summary_df.iloc[0]['params'].keys())
    ranks = []
    test_scores = []
    for i in range(len(summary_df)):
        ranks.append(int(summary_df.iloc[i]['rank_test_score']))
        table_rows.append(list(summary_df.iloc[i]['params'].values()))
        test_scores.append(np.abs(np.array([float(summary_df.iloc[i][f'split{j}_test_score']) for j in range(n_splits)])))
    ax.boxplot(test_scores)
    ax.set_yscale('log')
    ax.set_xticks([])
    ax.set_ylabel('MSE', fontsize=12)
    ax.grid()
    table_rows = [*zip(*table_rows)]
    the_table = ax.table(cellText=table_rows,
                         rowLabels=param_names,
                         colLabels=ranks,
                         loc='bottom')
    plt.tight_layout()
    plt.show()
