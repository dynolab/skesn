from typing import Any, Dict, List, Union
import numpy as np

from src.config import KVArgConfigSection, EvoLimitGenConfigField

from .lorenz import train_to_data

from scipy.stats import loguniform
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

def get_optional_arg(
    args,
    name1: str,
    name2: str=None,
    default: Any=None,
) -> Any:
    if hasattr(args, name1):
        ret = getattr(args, name1)
        if ret is not None:
            return ret
    elif name2 is not None:
        if hasattr(args, name2):
            ret = getattr(args, name2)
            if ret is not None:
                return ret
    return default

def kv_config_arr_to_kvargs(
    args: List[KVArgConfigSection],
) -> Dict[str, Any]:
    ret = {}
    for kv in args:
        ret[kv.Key] = kv.Val
    return ret

_KVARGS_ARGS = 'args'

def get_args_via_kvargs(kvargs):
    ret = None
    if _KVARGS_ARGS in kvargs:
        ret = kvargs[_KVARGS_ARGS]
    return ret

def _gen_num_by_limit(
    limit_cfg: EvoLimitGenConfigField,
    rand: np.random.RandomState,
) -> Union[int, float]:
    ret: float = 0.
    if limit_cfg.Logspace is None:
        ret = _gen_num(
            min=limit_cfg.Min,
            max=limit_cfg.Max,
            rand=rand,
        )
    else:
        ret = _gen_log_num(
            min=limit_cfg.Min,
            max=limit_cfg.Max,
            n=limit_cfg.Logspace.N,
            power=limit_cfg.Logspace.Power,
            rand=rand,
        )

    t = limit_cfg.Type.lower()
    if t == 'int':
        return int(ret)
    elif t == 'float':
        return ret
    raise 'unknow limit gene type'

def _gen_num(
    min: Union[float, int, None],
    max: Union[float, int, None],
    rand: np.random.RandomState,
) -> float:
    if min is not None and max is not None:
        return rand.uniform(min, max)
    elif min is None and max is not None:
        return rand.uniform(0, max)
    elif min is not None and max is None:
        return rand.uniform(min)
    return rand.uniform(0)

_EPS_FLOAT = 1e-16

def _gen_log_num(
    min: Union[int, float, None],
    max: Union[int, float, None],
    n: int,
    power: int,
    rand: np.random.RandomState,
) -> float:
    if min == 0:
        min = _EPS_FLOAT
    if max == 0:
        max = _EPS_FLOAT

    # TODO : use base and power for generation random point
    if min is not None and max is not None:
        return 10**rand.uniform(np.log10(min), np.log10(max))
    elif min is None and max is not None:
        return 10**rand.uniform(np.log10(_EPS_FLOAT), np.log10(max))
    elif min is not None and max is None:
        return 10**rand.uniform(np.log10(min))
    return 10**rand.uniform(np.log10(_EPS_FLOAT))

def gen_gene(
    limit_cfg: EvoLimitGenConfigField,
    rand: np.random.RandomState,
) -> Any:
    t = limit_cfg.Type.lower()
    if t in ('int', 'float'):
        return _gen_num_by_limit(limit_cfg=limit_cfg, rand=rand)
    elif t == 'bool':
        return rand.randint(0, 2) == 1
    elif t == 'choice':
        idx = rand.randint(0, len(limit_cfg.Choice))
        return limit_cfg.Choice[idx]
    return None

def mut_gene(
    limit_cfg: EvoLimitGenConfigField,
    rand: np.random.RandomState,
    cur_gene: Union[int, float, bool, str]
) -> Union[int, float, bool, str]:
    t = limit_cfg.Type.lower()
    if t in ('int', 'float'):
        return _gen_num_by_limit(limit_cfg=limit_cfg, rand=rand)
    elif t == 'bool' and isinstance(cur_gene, bool):
        return not cur_gene
    elif t == 'choice':
        len_choice = len(limit_cfg.Choice)
        idx = rand.randint(0, len_choice)
        if limit_cfg.Choice[idx] == cur_gene:
            offset = rand.randint(0, len_choice)
            return limit_cfg.Choice[(idx + offset) % len_choice]
        return limit_cfg.Choice[idx]
    raise 'unknown limit gene type'
