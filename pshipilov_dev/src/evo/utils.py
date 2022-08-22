import pshipilov_dev.src.evo.utils as utils

from types import FunctionType
from deap import base, creator
from typing import List, Tuple, Type, Union
from deap import tools

import importlib
import os
import yaml
import random
import numpy as np

from .. import dump

import sklearn.metrics as metrics

from skesn.esn import EsnForecaster

from ..config import Config, EvoLimitGenConfigField, EvoSchemeConfigField, EvoSchemeMultiPopConfigField, KVArgConfigSection
from ..lorenz import get_lorenz_data, data_to_train, train_to_data

def normalize_name(
    name: str
) -> str:
    return name.lower().strip()

def _get_data_set(
) -> np.ndarray:
    if Config.Evaluate.Model == 'lorenz':
        return get_lorenz_data(
            Config.Models.Lorenz.Ro,
            Config.Models.Lorenz.N,
            Config.Models.Lorenz.Dt,
            Config.Models.Lorenz.RandSeed,
        )
    raise 'unknown evaluate model'

def _split_data_set(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
def wrap_esn_evaluate_f(
    esn_creator_by_ind_f,
    **evaluate_kvargs,
) -> FunctionType:
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

class DynoExtensions:
    @staticmethod
    def dynoMutGauss(individual, indpb, low, up, sigma):
        for i in range(len(individual)):
            if np.random.uniform(0, 1) < indpb:
                individual[i] = random.triangular(up, low, random.gauss(individual[i], sigma))
        return individual,


# Mapping functions

def map_metric_f():
    norm_name = normalize_name(Config.Evaluate.Metric)

    if norm_name == 'mse':
        return metrics.mean_squared_error
    raise 'unknown evaluate metric'

def map_select_f(
    select: str,
) -> FunctionType:
    if not isinstance(select, str):
        raise Exception(f'select should be a string')
    if select.startswith('selCx') and hasattr(DynoExtensions, select):
        return getattr(DynoExtensions, select)
    if not select.startswith('sel'):
        raise Exception(f'unknown select "{select}", it should start with "sel"')
    if hasattr(tools, select):
        return getattr(tools, select)
    raise Exception(f'unknown select "{select}"')

def map_mate_f(
    crossing: str,
) -> FunctionType:
    if not isinstance(crossing, str):
        raise Exception(f'crossing should be a string')
    if crossing.startswith('dynoCx') and hasattr(DynoExtensions, crossing):
        return getattr(DynoExtensions, crossing)
    if not crossing.startswith('cx'):
        raise Exception(f'unknown crossing "{crossing}", it should start with "cx"')
    if hasattr(tools, crossing):
        return getattr(tools, crossing)
    raise Exception(f'unknown crossing "{crossing}"')

def map_mutate_f(
    mutate: str,
) -> FunctionType:
    if not isinstance(mutate, str):
        raise Exception(f'mutate should be a string')
    if mutate.startswith('dynoMut') and hasattr(DynoExtensions, mutate):
        return getattr(DynoExtensions, mutate)
    if not mutate.startswith('mut'):
        raise Exception(f'unknown mutate "{mutate}", it should start with "mut"')
    if hasattr(tools, mutate):
        return getattr(tools, mutate)
    raise Exception(f'unknown mutate "{mutate}"')

def map_evaluate_f(
    metric_f,
    fit_data: np.ndarray,
    valid_data: np.ndarray,
    **evaluate_kvargs,
) -> FunctionType:
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

def dump_inds_arr(
    hromo_len: int,
    f,
    population: List[List],
    limits: List[EvoLimitGenConfigField]=[],
) -> None:
    limits_len = len(limits)
    if limits_len > 0 and limits_len != hromo_len:
        raise 'not correct gen limits length'

    dump = {}

    for i, ind in enumerate(population):
        dump_ind = [0] * hromo_len
        for j in range(hromo_len):
            if limits_len > 0:
                if limits[j].IsInt:
                    dump_ind[j] = int(ind[j])
                else:
                    dump_ind[j] = float(ind[j])
            else:
                dump_ind[j] = float(ind[j])
        dump[f'ind_{i}'] = dump_ind

    yaml.safe_dump(dump, f)

def dump_inds_multi_pop_arr(
    hromo_len: int,
    f,
    populations: List[List[List]],
    limits_pop: List[List[EvoLimitGenConfigField]]=[],
) -> None:
    limits_pop_len = len(limits_pop)
    if limits_pop_len > 0 and limits_pop_len != len(populations):
        raise 'not correct popultations limits length'

    dump = {}
    for i, population in enumerate(populations):
        if len(population) == 0:
            dump[f'pop_{i}'] = []
            continue

        limits_len = 0
        if limits_pop_len > 0 and len(limits_pop[i]) > 0:
            limits_len = len(limits_pop[i])

        if limits_len > 0 and limits_len != hromo_len:
            raise 'not correct gen limits length'

        dump_pop = [0] * len(population)
        for j, ind in enumerate(population):
            dump_ind = [0] * hromo_len
            for k in range(hromo_len):
                if limits_len > 0:
                    if limits_pop[i][k].IsInt:
                        dump_ind[k] = int(ind[k])
                    else:
                        dump_ind[k] = float(ind[k])
                else:
                    dump_ind[k] = float(ind[j])
            dump_pop[j] = dump_ind
        dump[f'pop_{i}'] = dump_pop

    yaml.safe_dump(dump, f)

def create_run_pool_dir(
    root: str,
    scheme_name: str,
) -> str:
    curr_run_pool = -1
    prefix = f'run_pool_{scheme_name.replace(" ", "_")}_'

    if not os.path.isdir(root):
        os.makedirs(root)
    else:
        _, dirs, _ = next(os.walk(root))
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_run_pool:
                    curr_run_pool = num

    ret = root
    if ord(ret[len(ret)-1]) != ord('/'):
        ret += '/'

    ret += prefix + str(curr_run_pool + 1) + '/'

    os.mkdir(ret)

    return ret

def get_or_create_last_run_pool_dir(
    root: str,
    scheme_name: str,
):
    curr_run_pool = -1
    prefix = f'run_pool_{scheme_name.replace(" ", "_")}_'

    if not os.path.isdir(root):
        os.makedirs(root)
    else:
        _, dirs, _ = next(os.walk(root))
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_run_pool:
                    curr_run_pool = num

    if curr_run_pool < 0:
        return create_run_pool_dir(root, scheme_name)

    ret = root
    if ord(ret[len(ret)-1]) != ord('/'):
        ret += '/'

    ret += prefix + str(curr_run_pool) + '/'
    return ret

def get_or_create_iter_dir(
    run_pool_dir: str,
) -> str:
    prefix = 'iter_'
    curr_iter = -1

    if not os.path.isdir(run_pool_dir):
        os.makedirs(run_pool_dir)
    else:
        _, dirs, _ = next(os.walk(run_pool_dir))
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_iter:
                    curr_iter = num

    ret = run_pool_dir
    if ord(ret[len(ret)-1]) != ord('/'):
        ret += '/'

    if curr_iter < 0:
        ret += prefix + str(0) + '/'
        os.mkdir(ret)
    else:
        ret += prefix + str(curr_iter) + '/'

    return ret

def create_iter_dir(
    run_pool_dir: str,
) -> str:
    curr_iter = -1
    prefix = 'iter_'

    if not os.path.isdir(run_pool_dir):
        os.makedirs(run_pool_dir)
    else:
        _, dirs, _ = next(os.walk(run_pool_dir))
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_iter:
                    curr_iter = num

    ret = run_pool_dir
    if ord(ret[len(ret)-1]) != ord('/'):
        ret += '/'

    ret += prefix + str(curr_iter + 1) + '/'
    os.mkdir(ret)

    return ret

def ind_creator_f(
    ind_type: Type,
    hromo_len: int,
    limits: List[EvoLimitGenConfigField]=[],
    rand: np.random.RandomState=np.random.RandomState,
) -> list:
    limits_len = len(limits)
    if limits_len > 0 and limits_len != hromo_len:
        raise 'not correct gen limits length'

    ret = [0] * hromo_len
    for i in range(hromo_len):
        if limits_len > 0:
            if limits[i].IsInt:
                ret[i] = rand.randint(limits[i].Min, limits[i].Max)
            else:
                ret[i] = rand.uniform(limits[i].Min, limits[i].Max)
        else:
            ret[i] = rand.rand()

    return ind_type(ret)

def get_evo_metric_func(
    func_name: str,
    package_name: str,
) -> FunctionType:
    module = importlib.import_module(package_name)
    if hasattr(module, func_name):
        return getattr(module, func_name)
    raise 'unknown function name'

def bind_evo_operator(
    toolbox: base.Toolbox,
    name: str,
    func: FunctionType,
    args: List[KVArgConfigSection],
    **kvargs,
) -> None:
    if len(args) > 0:
        for arg in args:
            kvargs[arg.Key] = arg.Val

    toolbox.register(name, func, **kvargs)

def get_populations_limits(
    cfg: EvoSchemeMultiPopConfigField,
) -> List[List[EvoLimitGenConfigField]]:
    ret = []
    for population in cfg.Populations:
        for _ in range(population.IncludingCount):
            ret.append(population.Limits)
    return ret

def get_evo_scheme_result_last_run_pool(
    get_result_from_file_f: FunctionType,
    ind_type: Type,
    cfg: Union[EvoSchemeMultiPopConfigField, EvoSchemeConfigField],
    root: str,
    scheme_name: str,
) -> List[List]:
    run_pool_dir = get_or_create_last_run_pool_dir(root, scheme_name)

    return get_evo_scheme_result_last_iter(
        get_result_from_file_f,
        ind_type,
        cfg,
        run_pool_dir,
    )

def get_evo_scheme_result_last_iter(
    get_result_from_file_f: FunctionType,
    ind_type: Type,
    cfg: Union[EvoSchemeMultiPopConfigField, EvoSchemeConfigField],
    runpool_dir: str,
) -> List[List]:
    iter_dir = get_or_create_iter_dir(runpool_dir)

    return get_result_from_file_f(
        ind_type,
        cfg,
        iter_dir + 'result.yaml',
    )

def get_evo_scheme_result_from_file(
    ind_type: Type,
    cfg: Union[EvoSchemeMultiPopConfigField, EvoSchemeConfigField],
    filename: str,
) -> List[List]:
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        last_population_yaml = yaml.safe_load(f)
        if last_population_yaml is None:
            raise f'the population from the file "{filename}" is None)'

        if cfg.PopulationSize != len(last_population_yaml):
            raise f'the population size from the file "{filename}" ({len(last_population_yaml)}) does not match the config ({popultaion_size})'

        ret: List[List] = [0] * cfg.PopulationSize

        if isinstance(last_population_yaml, dict):
            for i, ind in enumerate(last_population_yaml.values()):
                ret[i] = ind_type(ind)
        elif isinstance(last_population_yaml, list):
            for i, ind in enumerate(last_population_yaml):
                ret[i] = ind_type(ind)
        else:
            raise 'unknown last popultaion yaml type'

        return ret

def get_evo_scheme_multi_pop_result_from_file(
    ind_type: Type,
    cfg: Union[EvoSchemeMultiPopConfigField, EvoSchemeConfigField],
    filename: str,
) -> List[List]:
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        result = yaml.safe_load(f)
        if result is None:
            raise f'the population from the file "{filename}" is None)'
        elif len(result) != utils.get_populations_cnt(cfg):
            raise f'not correct populations length from the file "{filename}")'

        ret: List[List] = []

        k = 0
        for pop_cfg in cfg.Populations:
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                inds = result[f'pop_{k}']

                if pop_cfg.Size != len(inds):
                    raise f'the population size from the file "{filename}" ({len(result)}) does not match the config ({pop_cfg.Size})'

                if isinstance(inds, dict):
                    ret.append([ind_type(ind) for _, ind in inds.values()])
                elif isinstance(inds, list):
                    ret.append([ind_type(ind) for ind in inds])
                else:
                    raise 'unknown last popultaion yaml type'

                k += 1

        return ret

def ind_float_eq_f(ind_l: List[float], ind_r: List[float]) -> bool:
    if len(ind_l) != len(ind_r):
        return False
    EPS = 1e-6
    for i in range(len(ind_l)):
        if np.abs(ind_l[i] - ind_r[i]) > EPS:
            return False
    return True

def get_populations_cnt(
    cfg: EvoSchemeMultiPopConfigField,
) -> int:
    if cfg.Populations is None or len(cfg.Populations) == 0:
        return 0

    ret = 0
    for pop_cfg in cfg.Populations:
        if pop_cfg.IncludingCount <= 0:
            continue
        ret += pop_cfg.IncludingCount
    return ret

def get_max_population_size(
    cfg: EvoSchemeMultiPopConfigField,
) -> int:
    if cfg.Populations is None or len(cfg.Populations) == 0:
        return 0

    max = cfg.Populations[0].Size
    for pop_cfg in cfg.Populations:
        if pop_cfg.Size > max:
            max = pop_cfg.Size
    return max


def create_ind_by_list(
    list_ind: List,
    evaluate_f: FunctionType,
) -> List:
    ret = creator.Individual(list_ind)
    if not ret.fitness.valid:
        ret.fitness.values = evaluate_f(ret)
    return ret

