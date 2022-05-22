from contextlib import contextmanager
from urllib.parse import non_hierarchical

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator as _BaseEstimator


### COPY FROM sktime.base._base.py ###
class BaseEstimator(_BaseEstimator):
    """Base class for defining estimators in sktime.

    Extends sktime's BaseObject to include basic functionality for fittable estimators.
    """

    def __init__(self):
        self._is_fitted = False
        super(BaseEstimator, self).__init__()

    @property
    def is_fitted(self):
        """Whether `fit` has been called."""
        return self._is_fitted

    def check_is_fitted(self):
        """Check if the estimator has been fitted.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This instance of {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )


### HEAVILY MODIFIED COPY FROM sktime.forecasting.base._base.py ###
class BaseForecaster(BaseEstimator):
    """Base forecaster template class.

    The base forecaster specifies the methods and method
    signatures that all forecasters have to implement.

    Specific implementations of these methods is deferred to concrete
    forecasters.
    """

    def __init__(self):
        self._is_fitted = False

        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self._cutoff = None  # reference point for relative fh

        super(BaseForecaster, self).__init__()

    def fit(self, y, X=None, **kwargs):
        """Fit forecaster to training data.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets self._is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh if fh is passed.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        self : Reference to self.
        """
        # If fit is called, fitted state is re-set
        self._is_fitted = False

        # Pass to inner fit
        self._fit(y=y, X=X, **kwargs)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(
        self,
        n_timesteps,
        X=None,
        **kwargs
    ):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores fh to self.fh if fh is passed and has not been passed previously.

        Parameters
        ----------
        n_timesteps : int
            Number of time steps to forecast for
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        """
        # handle inputs

        self.check_is_fitted()

        # this is how it is supposed to be after the refactor is complete and effective
        y_pred = self._predict(n_timesteps=n_timesteps, X=X, **kwargs)
        return y_pred

    def fit_predict(
        self, y, X=None, n_timesteps=None, fit_kwargs={}, predict_kwargs={}
    ):
        """Fit and forecast time series at future horizon.

        State change:
            Changes state to "fitted".

        Writes to self:
            Sets is_fitted flag to True.
            Writes self._y and self._X with `y` and `X`, respectively.
            Sets self.cutoff and self._cutoff to last index seen in `y`.
            Sets fitted model attributes ending in "_".
            Stores fh to self.fh.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to and to predict from
            if self.get_tag("X-y-must-have-same-index"),
            X.index must contain y.index and fh.index

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        """
        # if fit is called, fitted state is re-set
        self._is_fitted = False

        # apply fit and then predict
        self._fit(y=y, X=X, **fit_kwargs)
        self._is_fitted = True
        # call the public predict to avoid duplicating output conversions
        #  input conversions are skipped since we are using X_inner
        return self.predict(
            n_timesteps=n_timesteps, X=X, **predict_kwargs
        )

    def update(self, y, X=None, mode=None):
        """Update cutoff value and, optionally, fitted parameters.

        If no estimator-specific update method has been implemented,
        default fall-back is as follows:
            update_params=True: fitting to all observed data so far
            update_params=False: updates cutoff and remembers data only

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self. cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, or 2D np.ndarray optional (default=None)
            Exogeneous time series to fit to
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        self.check_is_fitted()

        # checks and conversions complete, pass to inner fit
        self._update(y=y, X=X, mode=mode)

        return self

#    def update_predict(
#        self,
#        y,
#        cv=None,
#        X=None,
#        update_params=True,
#    ):
#        """Make predictions and update model iteratively over the test set.
#
#        State required:
#            Requires state to be "fitted".
#
#        Accesses in self:
#            Fitted model attributes ending in "_".
#            Pointers to seen data, self._y and self.X
#            self.cutoff, self._is_fitted
#            If update_params=True, model attributes ending in "_".
#
#        Writes to self:
#            Update self._y and self._X with `y` and `X`, by appending rows.
#            Updates self.cutoff and self._cutoff to last index seen in `y`.
#            If update_params=True,
#                updates fitted model attributes ending in "_".
#
#        Parameters
#        ----------
#        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
#            Time series to which to fit the forecaster.
#            if self.get_tag("scitype:y")=="univariate":
#                must have a single column/variable
#            if self.get_tag("scitype:y")=="multivariate":
#                must have 2 or more columns
#            if self.get_tag("scitype:y")=="both": no restrictions apply
#        cv : temporal cross-validation generator, optional (default=None)
#        X : pd.DataFrame, or 2D np.ndarray optional (default=None)
#            Exogeneous time series to fit to and predict from
#            if self.get_tag("X-y-must-have-same-index"),
#            X.index must contain y.index and fh.index
#        update_params : bool, optional (default=True)
#
#        Returns
#        -------
#        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
#            Point forecasts at fh, with same index as fh
#            y_pred has same type as y
#
#        TODO
#        ----
#        It is now the same as update_predict_single. Different
#        versions should be implemented in future
#        """
#        self.check_is_fitted()
#
#        # input checks and minor coercions on X, y
#        X_inner, y_inner = self._check_X_y(X=X, y=y)
#
#        cv = check_cv(cv)
#
#        return self._predict_moving_cutoff(
#            y=y_inner,
#            cv=cv,
#            X=X_inner,
#            update_params=update_params,
#        )

    def update_predict_single(
        self,
        y=None,
        n_timesteps=None,
        X=None,
        mode=None,
    ):
        """Update model with new data and make forecasts.

        This method is useful for updating and making forecasts in a single step.

        If no estimator-specific update method has been implemented,
        default fall-back is first update, then predict.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".
            Pointers to seen data, self._y and self.X
            self.cutoff, self._is_fitted
            If update_params=True, model attributes ending in "_".

        Writes to self:
            Update self._y and self._X with `y` and `X`, by appending rows.
            Updates self. cutoff and self._cutoff to last index seen in `y`.
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Target time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to fit to and to predict from
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain y.index and fh.index

        Returns
        -------
        y_pred : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Point forecasts at fh, with same index as fh
            y_pred has same type as y
        """
        self.check_is_fitted()

        return self._update_predict_single(
            y=y,
            n_timesteps=n_timesteps,
            X=X,
            mode=mode,
        )

    def predict_residuals(self, n_timesteps, y=None, X=None, **kwargs):
        """Return residuals of time series forecasts.

        Residuals will be computed for forecasts at y.index.

        If fh must be passed in fit, must agree with y.index.
        If y is an np.ndarray, and no fh has been passed in fit,
        the residuals will be computed at a fh of range(len(y.shape[0]))

        State required:
            Requires state to be "fitted".
            If fh has been set, must correspond to index of y (pandas or integer)

        Accesses in self:
            Fitted model attributes ending in "_".
            self.cutoff, self._is_fitted

        Writes to self:
            Stores y.index to self.fh if has not been passed previously.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, np.ndarray (1D or 2D), or None
            Time series with ground truth observations, to compute residuals to.
            Must have same type, dimension, and indices as expected return of predict.
            if None, the y seen so far (self._y) are used, in particular:
                if preceded by a single fit call, then in-sample residuals are produced
                if fit requires fh, it must have pointed to index of y in fit
        X : pd.DataFrame, or 2D np.ndarray, optional (default=None)
            Exogeneous time series to predict from
            if self.get_tag("X-y-must-have-same-index"), X.index must contain fh.index

        Returns
        -------
        y_res : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Forecast residuals at fh, with same index as fh
            y_pred has same type as y passed in fit (most recently)
        """
        # if no y is passed, the so far observed y is used
        if y is None:
            y = self._y

        y_pred = self.predict(n_timesteps=n_timesteps, X=X, **kwargs)
        y_res = y - y_pred

        return y_res

    def score(self, y, X=None, n_timesteps=None):
        """Scores forecast against ground truth, using MAPE.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            Time series to score
            if self.get_tag("scitype:y")=="univariate":
                must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, array-like or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, or 2D np.array, optional (default=None)
            Exogeneous time series to score
            if self.get_tag("X-y-must-have-same-index"), X.index must contain y.index

        Returns
        -------
        score : float
            MAPE loss of self.predict(fh, X) with respect to y_test.
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        # symmetric=True is default for mean_absolute_percentage_error
        from sklearn.metrics import (
            mean_absolute_percentage_error,
        )

        if n_timesteps is None:
            n_timesteps = y.shape[0]
        return mean_absolute_percentage_error(y, self.predict(n_timesteps, X))

    def get_fitted_params(self):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        raise NotImplementedError("abstract method")

    @property
    def cutoff(self):
        """Cut-off = "present time" state of forecaster.

        Returns
        -------
        cutoff : int
        """
        return self._cutoff

    def _set_cutoff(self, cutoff):
        """Set and update cutoff.

        Parameters
        ----------
        cutoff: pandas compatible index element

        Notes
        -----
        Set self._cutoff is to `cutoff`.
        """
        self._cutoff = cutoff


    def _fit(self, y, X):
        """Fit forecaster to training data.

            core logic

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : returns an instance of self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, n_timesteps, X=None):
        """Forecast time series at future horizon.

            core logic

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
            - Will be removed in v 0.10.0
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals - deprecate in v 0.10.1

        """
        raise NotImplementedError("abstract method")

    def _update(self, y, X=None, mode=None):
        """Update time series to incremental training data.

        Writes to self:
            If update_params=True,
                updates fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : series of a type in self.get_tag("y_inner_mtype")
            Point forecasts at fh, with same index as fh
        y_pred_int : pd.DataFrame - only if return_pred_int=True
            Prediction intervals
        """
        # default to re-fitting if update is not implemented
        print(
            f"NotImplementedWarning: {self.__class__.__name__} "
            f"does not have a custom `update` method implemented. "
            f"{self.__class__.__name__} will be refit each time "
            f"`update` is called."
        )
        # refit with updated data, not only passed data
        self.fit(self._y, self._X)
        # todo: should probably be self._fit, not self.fit
        # but looping to self.fit for now to avoid interface break

        return self

    def _update_predict_single(
        self,
        y,
        n_timesteps,
        X=None,
        mode=None,
    ):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.
        """
        self.update(y, X, mode=mode)
        return self.predict(n_timesteps, X)


class NotFittedError(Exception):
    pass
