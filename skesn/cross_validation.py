import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, \
    mean_squared_error, make_scorer
import pandas as pd

from skesn.base import BaseForecaster
from skesn.misc import SklearnWrapperForForecaster


class ValidationBasedOnRollingForecastingOrigin:
    """We fix the test size (this defines multi-step forecasting) and gradually increase training size.
    Optionally, we can fix training size too.
    Optionally, we can fix the training time series and validate only against rolling forecasting origin.

    See https://otexts.com/fpp3/tscv.html
    """
    def __init__(self, metric=mean_absolute_percentage_error,
                 n_training_timesteps=None,
                 n_test_timesteps=10,
                 n_splits=10):
        self.metric = metric
        self.n_training_timesteps = n_training_timesteps
        self.n_test_timesteps = n_test_timesteps
        self.n_splits = n_splits

    def evaluate(self, forecaster, y, X):
        return [self.metric(y_true, y_pred)
                for _, y_pred, y_true in self.prediction_generator(forecaster, y, X)]

    def prediction_generator(self, forecaster, y, X):
        ts_cv = TimeSeriesSplit(n_splits=self.n_splits,
                                test_size=self.n_test_timesteps)
        for train_index, test_index in ts_cv.split(y):
            if X is None:
                forecaster.fit(y=y[train_index],
                               X=None)
                y_pred = forecaster.predict(self.n_test_timesteps,
                                            X=None)
            else:
                forecaster.fit(y=y[train_index], X=X[train_index])
                y_pred = forecaster.predict(self.n_test_timesteps,
                                            X=X[test_index])
            yield test_index, y_pred, y[test_index]

    def grid_search(self, forecaster, param_grid, y, X):
        """
        For error metrics (they are assumed here), score values will be negative even for
        non-negative metrics. You need to compute absolute values of the scores to get
        the expected values.
        """
        if X is None: # TODO: some sklearn subroutines like gridsearchcv
            # cannot pass X=None so we need to pass array of None
            X = [None for _ in range(len(y))]
        if issubclass(type(forecaster), BaseForecaster):
            forecaster = SklearnWrapperForForecaster(forecaster)
            param_grid = {f'custom_estimator__{k}': v for k, v in param_grid.items()}
        scorer = make_scorer(self.metric, greater_is_better=False)
        grid = GridSearchCV(forecaster, param_grid,
                            scoring=scorer,
                            cv=TimeSeriesSplit(n_splits=self.n_splits,
                                               test_size=self.n_test_timesteps))
        grid.fit(X=X, y=y)
        return grid.cv_results_, grid.best_estimator_
