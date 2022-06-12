from typing import Any, Dict
import numpy as np

from .lorenz import train_to_data

from skesn.esn import EsnForecaster

def valid_multi_f(valid_multi_n, model: EsnForecaster, valid_data):
    h = valid_multi_n
    n = valid_data.shape[1] // h
    idxs = [int(idx) for idx in np.linspace(0, valid_data.shape[1], n, True)]
    err = np.ndarray(len(idxs) - 1)
    predict = []
    for i in range(1, len(idxs)):
        predict = train_to_data(model.predict(idxs[i] - idxs[i - 1], True, False).T)
        err[i - 1] = ((([[v] for v in valid_data[:,i]] - predict)**2).mean()**0.5)
    return err.mean()

def get_necessary_arg(args, name1, name2=None):
    if hasattr(args, name1):
        return getattr(args, name1)
    elif name2 is not None:
        if hasattr(args, name2):
            return getattr(args, name2)
        raise 'unknown arg names: {0}, {1}'.format(name1, name2)
    raise 'unknown arg name: {0}'.format(name1)

def try_get_param_by_kvargs(
    kvargs: Dict[str, Any],
    name: str,
    default: Any=None,
) -> Any:
    return kvargs[name] if name in kvargs else default
