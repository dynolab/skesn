import random
from struct import unpack
from types import FunctionType
from typing import Any, Dict, List, Tuple, Union
import numpy as np

import src.evo.utils as evo_utils
import deap.tools as deap_tools

import src.config as cfg

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
    args: List[cfg.KVArgConfigSection],
) -> Dict[str, Any]:
    ret = {}
    for kv in args:
        ret[kv.Key] = kv.Val
    return ret

_KVARGS_ARGS = 'args'
_KVARGS_ASYNC_MANAGER = 'async_manager'

def get_args_via_kvargs(kvargs):
    ret = None
    if _KVARGS_ARGS in kvargs:
        ret = kvargs[_KVARGS_ARGS]
    return ret

def get_via_kvargs(kvargs, name: str):
    ret = None
    if name in kvargs:
        ret = kvargs[name]
    return ret


def _gen_num_by_limit(
    gen_val: float,
    limit_cfg: cfg.EvoLimitGenConfigField,
    rand: np.random.RandomState,
) -> Union[int, float]:
    ret: float = 0.

    if limit_cfg.Mutate is not None:
        ret = _gen_num_by_method(gen_val, limit_cfg.Mutate)
    elif limit_cfg.Logspace is None:
        ret = _gen_num(
            min=limit_cfg.Min,
            max=limit_cfg.Max,
            rand=rand,
            type=limit_cfg.Type,
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

def _prepare_cfg_args(
    args: Union[List[cfg.KVArgConfigSection],None],
) -> dict:
    ret = {}
    if args is None or len(args) == 0:
        return ret

    for kv in args:
        ret[kv.Key] = kv.Val
    return ret

def _map_limit_mutate_f(
    name: str,
) -> FunctionType:
    name = name.lower()
    if name == 'gaussian':
        def _gaussian(x, mu, sigma, low, up):
            x += (-1 ** np.random.randint(1, 2)) * np.random.gauss(mu, sigma)
            return boundVaule(x, low, up)
        return _gaussian
    elif name == 'polynomial_bounded':
        def _polynomial_bounded(x, low, up, eta):
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (up - low)
            return boundVaule(x, low, up)
        return _polynomial_bounded
    raise f'unknown limit mutate method ({name})'

def _gen_num_by_method(
    gen_val: float,
    method_cfg: cfg.EvoOperatorBaseConfigField,
) -> float:
    func = _map_limit_mutate_f(method_cfg.Method)
    args = _prepare_cfg_args(method_cfg.Args)
    return func(x=gen_val, **args)

def _resolve_type_num_result(
    type: str,
    value: Union[int, float],
):
    if type.lower() == 'int':
        return int(value)
    return value

def _gen_num(
    min: Union[float, int, None],
    max: Union[float, int, None],
    rand: np.random.RandomState,
    type: str,
) -> float:
    if min is not None and max is not None:
        return _resolve_type_num_result(type, rand.uniform(min, max))
    elif min is None and max is not None:
        return _resolve_type_num_result(type, rand.uniform(0, max))
    elif min is not None and max is None:
        return _resolve_type_num_result(type, rand.uniform(min))
    return _resolve_type_num_result(type, rand.uniform(0))

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
    limit_cfg: cfg.EvoLimitGenConfigField,
    rand: np.random.RandomState,
) -> Any:
    t = limit_cfg.Type.lower()
    if t in ('int', 'float'):
        # return _gen_num_by_limit(
        #     gen_val=_gen_num(limit_cfg.Min, limit_cfg.Max, rand, limit_cfg.Type),
        #     limit_cfg=limit_cfg,
        #     rand=rand,
        # )
        return _gen_num(limit_cfg.Min, limit_cfg.Max, rand, limit_cfg.Type)
    elif t == 'bool':
        return rand.randint(0, 2) == 1
    elif t == 'choice':
        idx = rand.randint(0, len(limit_cfg.Choice))
        return limit_cfg.Choice[idx]
    return None

def boundVaule(
    value: float,
    low: float,
    up: float,
) -> float:
    if value < low:
        return low
    if value > up:
        return up
    return value

def cxGaussianBoundedGene(
    p1_gene: float,
    p2_gene: float,
    eta: float,
    low: float,
    up: float,
    rand: np.random.RandomState=np.random.RandomState,
) -> Tuple[float, float]:
    # This epsilon should probably be changed for 0 since
    # floating point arithmetic in Python is safer
    if abs(p1_gene - p2_gene) < 1e-14:
        return p1_gene, p2_gene

    beta = 0
    u = rand.rand()
    if u <= 0.5:
        beta = np.power(2 * u, 1 / (eta + 1))
    else:
        beta = np.power(0.5 / (1 - u), 1 / (eta + 1))

    x1 = boundVaule(0.5 * ((1 + beta) * p1_gene + (1 - beta) * p2_gene), low, up)
    x2 = boundVaule(0.5 * ((1 - beta) * p1_gene + (1 + beta) * p2_gene), low, up)

    return x1, x2

def cxSimulatedBinaryBoundedGene(
    p1_gene: float,
    p2_gene: float,
    eta: float,
    low: float,
    up: float,
    rand: np.random.RandomState=np.random.RandomState,
) -> Tuple[float, float]:
    # This epsilon should probably be changed for 0 since
    # floating point arithmetic in Python is safer
    if abs(p1_gene - p2_gene) < 1e-14:
        return p1_gene, p2_gene

    x1 = min(p1_gene, p2_gene)
    x2 = max(p1_gene, p2_gene)

    rand_n = rand.random()

    beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1)
    if rand_n <= 1.0 / alpha:
        beta_q = (rand_n * alpha) ** (1.0 / (eta + 1))
    else:
        beta_q = (1.0 / (2.0 - rand_n * alpha)) ** (1.0 / (eta + 1))

    ch1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

    beta = 1.0 + (2.0 * (up - x2) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1)
    if rand_n <= 1.0 / alpha:
        beta_q = (rand_n * alpha) ** (1.0 / (eta + 1))
    else:
        beta_q = (1.0 / (2.0 - rand_n * alpha)) ** (1.0 / (eta + 1))
    ch2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

    ch1 = min(max(ch1, low), up)
    ch2 = min(max(ch2, low), up)

    return (ch2, ch1) if rand.random() <= 0.5 else (ch1, ch2)

def cxRandChoiceGene(
    p1_gene: Union[int, float, bool, str],
    p2_gene: Union[int, float, bool, str],
    rand: np.random.RandomState=np.random.RandomState,
) -> Union[int, float, bool, str]:
    ch1 = p1_gene if rand.random() <= 0.5 else p2_gene
    ch2 = p1_gene if rand.random() <= 0.5 else p2_gene
    return ch1, ch2

def _map_gene_cx(
    method: str,
) -> Union[FunctionType, None]:
    if method == 'cxRandChoiceGene':
        return cxRandChoiceGene
    elif method == 'cxSimulatedBinaryBoundedGene':
        return cxSimulatedBinaryBoundedGene
    elif method == 'cxGaussianBoundedGene':
        return cxGaussianBoundedGene
    raise 'unknown cx gene method: %s' % method

def cx_gene(
    limit_cfg: cfg.EvoLimitGenConfigField,
    rand: np.random.RandomState,
    p1_gene: Union[int, float, bool, str],
    p2_gene: Union[int, float, bool, str],
) -> Union[int, float, bool, str]:
    if limit_cfg.Mate is None:
        raise 'limit_cfg doesn\'t contain mate config section'

    func = _map_gene_cx(limit_cfg.Mate.Method)
    if func is None:
        raise 'unknown gene mate method: %s' % limit_cfg.Mate.Method

    kvargs = {}
    if len(limit_cfg.Mate.Args) > 0:
        for arg in limit_cfg.Mate.Args:
            kvargs[arg.Key] = arg.Val

    ch1, ch2 = func(p1_gene, p2_gene, rand=rand, **kvargs)
    return _resolve_type_num_result(limit_cfg.Type, ch1), _resolve_type_num_result(limit_cfg.Type, ch2)

def mut_gene(
    limit_cfg: cfg.EvoLimitGenConfigField,
    rand: np.random.RandomState,
    cur_gene: Union[int, float, bool, str]
) -> Union[int, float, bool, str]:
    t = limit_cfg.Type.lower()
    if t in ('int', 'float'):
        return _resolve_type_num_result(limit_cfg.Type, _gen_num_by_limit(
            gen_val=cur_gene,
            limit_cfg=limit_cfg,
            rand=rand,
        ))
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
