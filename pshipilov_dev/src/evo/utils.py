from deap import tools

import numpy as np

from .. import dump

import sklearn.metrics as metrics

from skesn.esn import EsnForecaster

from ..config import Config
from ..lorenz import get_lorenz_data, data_to_train, train_to_data

def normalize_name(name: str):
    return name.lower().strip()

def _get_data_set():
    if Config.Evaluate.Model == 'lorenz':
        return get_lorenz_data(
            Config.Models.Lorenz.Ro,
            Config.Models.Lorenz.N,
            Config.Models.Lorenz.Dt,
            Config.Models.Lorenz.RandSeed,
        )
    raise 'unknown evaluate model'

def _split_data_set(data: np.ndarray):
    train_data: np.ndarray = None
    if Config.Evaluate.Opts.SparsityTrain > 0:
        train_data = data[..., :Config.Models.Lorenz.N//2:Config.Evaluate.Opts.SparsityTrain]
    else:
        train_data = data[..., :Config.Models.Lorenz.N//2]
    valid_data = data[..., Config.Models.Lorenz.N//2:]
    return train_data, valid_data

# Args:
# evaluate_kvargs must contains:
# disable_dump: bool = default(False) - disabled dump train and valid data sets
def wrap_esn_evaluate_f(esn_creator_by_ind_f, **evaluate_kvargs):
    train_data, valid_data = _split_data_set(_get_data_set())
    if not evaluate_kvargs.get('disable_dump', False):
        dump.do_np_arr(train_data=train_data, valid_data=valid_data)

    evaluate_f = map_evaluate_f(
        map_metric_f(),
        data_to_train(train_data).T,
        valid_data,
        **evaluate_kvargs,
    )

    return lambda ind: evaluate_f(esn_creator_by_ind_f(ind))

# Mapping functions

def map_metric_f():
    norm_name = normalize_name(Config.Evaluate.Metric)

    if norm_name == 'mse':
        return metrics.mean_squared_error
    raise 'unknown evaluate metric'

def map_select_f(select: str):
    if not isinstance(select, str):
        raise Exception(f'select should be a string')
    if not select.startswith('sel'):
        raise Exception(f'unknown select "{select}", it should start with "sel"')
    if hasattr(tools, select):
        return getattr(tools, select)
    raise Exception(f'unknown select "{select}"')

def map_mate_f(crossing: str):
    if not isinstance(crossing, str):
        raise Exception(f'crossing should be a string')
    if not crossing.startswith('cx'):
        raise Exception(f'unknown crossing "{crossing}", it should start with "cx"')
    if hasattr(tools, crossing):
        return getattr(tools, crossing)
    raise Exception(f'unknown crossing "{crossing}"')

def map_mutate_f(mutate: str):
    if not isinstance(mutate, str):
        raise Exception(f'mutate should be a string')
    if not mutate.startswith('mut'):
        raise Exception(f'unknown mutate "{mutate}", it should start with "mut"')
    if hasattr(tools, mutate):
        return getattr(tools, mutate)
    raise Exception(f'unknown mutate "{mutate}"')

def map_evaluate_f(metric_f, fit_data: np.ndarray, valid_data: np.ndarray, **evaluate_kvargs):
    if Config.Evaluate.Steps < 0:
        raise 'bad evaluate.steps config field value, must be greater then 0'

    if Config.Evaluate.Steps > 1:
        n = valid_data.shape[1] // Config.Evaluate.Steps
        idxs = [int(idx) for idx in np.linspace(0, valid_data.shape[1], n, True)]
        def _valid_multi_f(model: EsnForecaster):
            model.fit(fit_data)
            predict_data = np.ndarray(len(idxs) - 1)
            for i in range(1, len(idxs)):
                predict_data[i - 1] = train_to_data(model.predict(idxs[i] - idxs[i - 1], True, False).T)
            return metric_f(valid_data, predict_data),
        return _valid_multi_f

    def _valid_one_f(model: EsnForecaster):
        model.fit(fit_data)
        predict_data = np.ndarray(len(valid_data[0]))
        for i in range(len(valid_data[0])):
            predict_data[i] = train_to_data(model.predict(1, True, Config.Esn.Inspect).T)
        return metric_f(valid_data, predict_data),
    return _valid_one_f
