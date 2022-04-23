from deap import tools

import numpy as np

from .. import dump

from skesn.esn import EsnForecaster

from ..config import Config
from ..lorenz import get_lorenz_data, data_to_train, train_to_data

__dumped: bool = False

def map_config_scoring_f(scoring: str, valid_data, train_data, valid_multi_n=None):
    global __dumped

    data = get_lorenz_data(
        Config.Models.Lorenz.Ro,
        Config.Models.Lorenz.N,
        Config.Models.Lorenz.Dt,
        Config.Models.Lorenz.RandSeed,
    )
    train_data = data[..., :Config.Models.Lorenz.N//2:10]
    valid_data = data[..., Config.Models.Lorenz.N//2:]
    fit_data = data_to_train(train_data).T

    if not __dumped:
        dump.do_np_arr(train_data=train_data, valid_data=valid_data)
        __dumped = True

    if scoring == 'train':
        def _train_f(model: EsnForecaster):
            return model.fit(fit_data).mean()**0.5,
        return _train_f
    elif scoring == 'valid_one':
        def _valid_one_f(model: EsnForecaster):
            model.fit(fit_data)
            max_i = len(valid_data[0])
            err = np.ndarray(max_i)
            for i in range(max_i):
                predict = train_to_data(model.predict(1, True, Config.Esn.Inspect).T)
                err[i] = (([[v] for v in valid_data[:,i]] - predict)**2).mean()**0.5
            return err.mean(),
        return _valid_one_f
    elif scoring == 'valid_multi':
        h = valid_multi_n
        if h is None:
            raise Exception('argument "valid_multi_n" had to provided when "valid_multi" is bound')
        n = valid_data.shape[1] // h
        idxs = [int(idx) for idx in np.linspace(0, valid_data.shape[1], n, True)]
        def _valid_multi_f(model: EsnForecaster):
            model.fit(fit_data)
            err = np.ndarray(len(idxs) - 1)
            for i in range(1, len(idxs)):
                predict = train_to_data(model.predict(idxs[i] - idxs[i - 1], True, False).T)
                err[i - 1] = ((([[v] for v in valid_data[:,i]] - predict)**2).mean()**0.5)
            return err.mean(),
        return _valid_multi_f

    raise(Exception(f'"scoring" has unknown value ({scoring})'))

def wrap_scoring_f(model_creator_by_ind: list, scoring: str, **scoring_opts):
    return lambda ind: map_config_scoring_f(scoring, **scoring_opts)(model_creator_by_ind(ind))

def map_select_f(select: str, *args, **kvargs):
    if not isinstance(select, str):
        raise Exception(f'select should be a string')
    if not select.startswith('sel'):
        raise Exception(f'unknown select "{select}", it should start with "sel"')
    if hasattr(tools, select):
        return getattr(tools, select)
    raise Exception(f'unknown select "{select}"')

def map_crossing_f(crossing: str, *args, **kvargs):
    if not isinstance(crossing, str):
        raise Exception(f'crossing should be a string')
    if not crossing.startswith('cx'):
        raise Exception(f'unknown crossing "{crossing}", it should start with "cx"')
    if hasattr(tools, crossing):
        return getattr(tools, crossing)
    raise Exception(f'unknown crossing "{crossing}"')


def map_mutate_f(mutate: str, *args, **kvargs):
    if not isinstance(mutate, str):
        raise Exception(f'mutate should be a string')
    if not mutate.startswith('mut'):
        raise Exception(f'unknown mutate "{mutate}", it should start with "mut"')
    if hasattr(tools, mutate):
        return getattr(tools, mutate)
    raise Exception(f'unknown mutate "{mutate}"')
