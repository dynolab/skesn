from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import TransformedTargetRegressor


#class TargetScaler(TransformerMixin, BaseEstimator):
#    def __init__(self, shift=0., scale=1.) -> None:
#        self.shift_ = shift
#        self.scale_ = scale
#        super().__init__()
#
#    def fit(self, X, y=None):
#        return self
#
#    def transform(self, X, y=None):
#        X

class SklearnWrapperForForecaster(RegressorMixin, BaseEstimator):
    def __init__(self, custom_estimator):
        self.custom_estimator = custom_estimator

    def fit(self, X, y, **kwargs):
        self.custom_estimator.fit(y, X, **kwargs)
        return self

    def predict(self, X, n_timesteps=None, **kwargs):
        return self.custom_estimator.predict(n_timesteps, X, **kwargs)


def scale_target(y, shift=0., scale=1.):
    return (y - shift) / scale


def inverse_scale_target(y, shift=0., scale=1.):
    return y * scale + shift


def build_target_scaler(pipe_or_estimator, shift=0., scale=1.):
    return TransformedTargetRegressor(pipe_or_estimator,
                                      func=partial(scale_target,
                                                   shift=shift,
                                                   scale=scale),
                                      inverse_func=partial(inverse_scale_target,
                                                           shift=shift,
                                                           scale=scale))


def correct_dimensions(arr: np.ndarray, representation='2D'):
    """Transforms arr to appropriate representation.
    2D representation implies shape (n_timesteps, n_states).
        If 1D array, we assume a shape (n_timesteps=len(arr), n_states=1)
        If 2D array, do nothing
        If >2D array, throw exception
    3D representation implies shape (n_batches, n_timesteps, n_states).
        If 1D array, we assume a shape (n_batches=1, n_timesteps=len(arr), n_states=1)
        If 2D array, we assume a shape (n_batches=1, n_timesteps=arr.shape[0], n_states=arr.shape[1])
        If 3D array, do nothing
        If >3D array, throw exception

    Parameters
    ----------
    arr : 1D, 2D or 3D array or None
        Numpy array to be corrected
    representation : {'2D', '3D'}
        Type of representation the array should be recasted to

    Returns
    -------
    None if arr is None, else corrected numpy array
    """
    if arr is None:
        return None
    corrected_arr = None
    if isinstance(arr, np.ndarray):
        if representation == '2D':
            if arr.ndim == 1:
                corrected_arr = np.reshape(arr, (len(arr), 1))
            elif arr.ndim == 2:
                corrected_arr = arr
            else:
                ValueError(f'2D representation is not compatible with array dimension {arr.ndim}')
        elif representation == '3D':
            if arr.ndim == 1:
                corrected_arr = np.reshape(arr, (1, len(arr), 1))
            elif arr.ndim == 2:
                corrected_arr = np.reshape(arr, (1, arr.shape[0], arr.shape[1]))
            elif arr.ndim == 3:
                corrected_arr = arr
            else:
                ValueError(f'3D representation is not compatible with array dimension {arr.ndim}')
    else:
        raise ValueError('Unsupported array type')
    return corrected_arr


def identity(x):
    return x
