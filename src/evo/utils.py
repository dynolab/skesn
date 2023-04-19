import src.evo.utils as evo_utils
import src.evo.types as evo_types
import src.utils as utils
import src.config as cfg
import src.dump as dump


from src.models.abstract import Model
from src.models.lorenz import LorenzModel
from src.models.chui_moffatt import ChuiMoffattModel
from src.models.moehlis import MoehlisModel

import skesn.esn as esn

import importlib
import os
import yaml
import random
import numpy as np

from types import FunctionType
from deap import base, creator
from typing import Any, List, Tuple, Type, Union
from deap import tools
from copy import copy


import sklearn.metrics as metrics

from skesn.esn import EsnForecaster


from ..lorenz import get_lorenz_data, data_to_train, train_to_data

def normalize_name(
    name: str
) -> str:
    return name.lower().strip()

def _get_data_set(
) -> np.ndarray:
    if cfg.Config.Evaluate.Model == 'lorenz':
        return get_lorenz_data(
            cfg.Config.Models.Lorenz.Ro,
            cfg.Config.Models.Lorenz.N,
            cfg.Config.Models.Lorenz.Dt,
            cfg.Config.Models.Lorenz.RandSeed,
        )
    raise 'unknown evaluate model'

def _split_data_set(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    train_data: np.ndarray = None
    if cfg.Config.Evaluate.Opts.SparsityTrain > 0:
        train_data = data[..., :cfg.Config.Models.Lorenz.N//2:cfg.Config.Evaluate.Opts.SparsityTrain]
    else:
        train_data = data[..., :cfg.Config.Models.Lorenz.N//2]
    valid_data = data[..., cfg.Config.Models.Lorenz.N//2:]
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

    @staticmethod
    def dynoMutGeneByLimit(
        individual: List,
        indpb: float,
        limits_cfg: List[cfg.EvoLimitGenConfigField]=[],
        rand: np.random.RandomState=np.random.RandomState(),
    ) -> Tuple:
        n = len(individual)
        for i in range(n):
            if np.random.uniform(0, 1) < indpb:
                individual[i] = utils.mut_gene(
                    limits_cfg[i],
                    rand,
                    individual[i],
                )
        return individual,

    @staticmethod
    def dynoCxGeneByLimit(
        p1_ind: List,
        p2_ind: List,
        limits_cfg: List[cfg.EvoLimitGenConfigField]=[],
        rand: np.random.RandomState=np.random.RandomState(),
        # create_ind_by_list_f: FunctionType=None,
    ) -> Tuple[List, List]:
        n = len(p1_ind)
        ch1_ind = [None] * n
        ch2_ind = [None] * n
        for i in range(n):
            ch1_ind[i], ch2_ind[i] = utils.cx_gene(
                limit_cfg=limits_cfg[i],
                rand=rand,
                p1_gene=p1_ind[i],
                p2_gene=p2_ind[i],
            )

        # if create_ind_by_list_f is not None:
        #     return create_ind_by_list_f(ch1_ind), create_ind_by_list_f(ch2_ind)

        return evo_types.Individual(ch1_ind), evo_types.Individual(ch2_ind)

# Mapping functions

def map_metric_f():
    norm_name = normalize_name(cfg.Config.Evaluate.Metric)

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
    if cfg.Config.Evaluate.Steps < 0:
        raise 'bad evaluate.steps config field value, must be greater then 0'

    if cfg.Config.Evaluate.Steps > 1:
        n = valid_data.shape[1] // cfg.Config.Evaluate.Steps
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
            predict_data[i] = train_to_data(model.predict(1, True, cfg.Config.Esn.Inspect).T)
        return metric_f(valid_data, predict_data),
    return _valid_one_f

def _convert_gen_to_dump(
    limit_cfg: cfg.EvoLimitGenConfigField,
    val: Any,
) -> Any:
    t = limit_cfg.Type.lower()
    if t == 'int':
        return int(val)
    elif t == 'float':
        return float(val)
    elif t == 'bool':
        return bool(val)
    elif t == 'choice':
        return val
    return None

def dump_inds_arr(
    hromo_len: int,
    f,
    population: List[List],
    limits: List[cfg.EvoLimitGenConfigField]=[],
) -> None:
    limits_len = len(limits)
    if limits_len > 0 and limits_len != hromo_len:
        raise 'not correct gen limits length'

    dump = {}

    for i, ind in enumerate(population):
        dump_ind = [0] * hromo_len
        for j in range(hromo_len):
            if limits_len > 0:
                dump_ind[j] = _convert_gen_to_dump(limits[j], ind[j])
                continue
            dump_ind[j] = float(ind[j])
        dump[f'ind_{i}'] = dump_ind

    yaml.safe_dump(dump, f)

def dump_inds_multi_pop_arr(
    hromo_len: int,
    f,
    populations: List[List[List]],
    limits_pop: List[List[cfg.EvoLimitGenConfigField]]=[],
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
                    dump_ind[k] = _convert_gen_to_dump(limits_pop[i][k], ind[k])
                    continue
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

def get_last_iter_num(
    runpool_dir: str,
) -> int:
    if not os.path.isdir(runpool_dir):
        raise f'runpool dir dosn\'t exists (runpool_dir: {runpool_dir})'

    ret = -1

    _, dirs, _ = next(os.walk(runpool_dir))
    for dir in dirs:
        if dir.startswith(ITER_DIR_PREFIX):
            num = int(dir.split('_')[-1])
            if num > ret:
                ret = num

    return ret if ret >= 0 else None

def is_iter_dir_exists(
    runpool_dir: str,
    iter_num: int,
) -> bool:
    if not os.path.isdir(runpool_dir):
        return False

    if not isinstance(iter_num, int) or iter_num < 0:
        raise f'wrong iter_num value ({iter_num})'

    iter_dir = get_iter_dir_path(
        runpool_dir=runpool_dir,
        iter_num=iter_num,
    )

    return os.path.isdir(iter_dir)

ITER_DIR_PREFIX = 'iter'

def get_iter_dir_path(
    runpool_dir: str,
    iter_num: int,
) -> str:
    if iter_num < 0:
        raise f'can\'t create negative iter dir (iter_num: {iter_num})'

    return os.path.join(runpool_dir, f'{ITER_DIR_PREFIX}_{iter_num}')

def calculate_best_ind(
    pop: List[List],
) -> Tuple[List, float]:
    max_fitness = None
    max_ind_idx = None

    for i, ind in enumerate(pop):
        fitness = np.dot(ind.fitness.values, ind.fitness.weights)
        if max_fitness is None or max_fitness < fitness:
            max_fitness = fitness
            max_ind_idx = i

    return pop[max_ind_idx], max_fitness

def get_or_create_new_iter_dir(
    runpool_dir: str,
    iter_dir: Union[str, None],
) -> Tuple[int, str]:
    if iter_dir is None:
        return create_new_iter_dir(runpool_dir)
    return iter_dir

def create_new_iter_dir(
    runpool_dir: str,
) -> str:
    if not os.path.isdir(runpool_dir):
        iter_dir = get_iter_dir_path(
            runpool_dir=runpool_dir,
            iter_num=0,
        )
        os.makedirs(iter_dir)
        return iter_dir

    curr_iter = get_last_iter_num(
        runpool_dir=runpool_dir,
    )

    if curr_iter is None:
        curr_iter = 0
    else:
        curr_iter += 1

    iter_dir = get_iter_dir_path(
        runpool_dir=runpool_dir,
        iter_num=curr_iter,
    )

    os.mkdir(iter_dir)

    return iter_dir

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

def ind_creator_f(
    hromo_len: int,
    limits: List[cfg.EvoLimitGenConfigField]=[],
    rand: np.random.RandomState=np.random.RandomState,
) -> list:
    limits_len = len(limits)
    if limits_len > 0 and limits_len != hromo_len:
        raise 'not correct gen limits length'

    ret = [0] * hromo_len
    for i in range(hromo_len):
        if limits_len > 0:
            ret[i] = utils.gen_gene(limits[i], rand)
            continue
        ret[i] = rand.rand()

    return evo_types.Individual(ret)

def get_evo_metric_func(
    func_name: str,
    package_name: str,
) -> FunctionType:
    module = importlib.import_module(package_name)
    if hasattr(module, func_name):
        return getattr(module, func_name)
    raise 'unknown function name'

def bind_mate_operator(
    toolbox: base.Toolbox,
    cfg: cfg.EvoPopulationConfigField,
    rand: np.random.RandomState=None,
    # create_ind_by_list_f: FunctionType=None,
) -> None:
    kvargs = {}
    if cfg.Mate.Method == 'dynoCxGeneByLimit':
        kvargs['limits_cfg'] = cfg.Limits
        # kvargs['create_ind_by_list_f'] = create_ind_by_list_f
        if rand is not None:
            kvargs['rand'] = rand

    evo_utils.bind_evo_operator(
        toolbox,
        'mate',
        evo_utils.map_mate_f(cfg.Mate.Method),
        cfg.Mate.Args,
        **kvargs,
    )

def bind_mutate_operator(
    toolbox: base.Toolbox,
    hromo_len: int,
    cfg: cfg.EvoPopulationConfigField,
    rand: np.random.RandomState=None,
) -> None:
    kvargs = {}
    if cfg.Mutate.Method == 'dynoMutGeneByLimit':
        kvargs['limits_cfg'] = cfg.Limits
        if rand is not None:
            kvargs['rand'] = rand

    if cfg.Mutate.Indpb > 0:
        evo_utils.bind_evo_operator(
            toolbox,
            'mutate',
            evo_utils.map_mutate_f(cfg.Mutate.Method),
            cfg.Mutate.Args,
            indpb=cfg.Mutate.Indpb,
            **kvargs,
        )
        return

    evo_utils.bind_evo_operator(
        toolbox,
        'mutate',
        evo_utils.map_mutate_f(cfg.Mutate.Method),
        cfg.Mutate.Args,
        indpb=1/hromo_len,
        **kvargs,
    )

def bind_evo_operator(
    toolbox: base.Toolbox,
    name: str,
    func: FunctionType,
    args: List[cfg.KVArgConfigSection],
    **kvargs,
) -> None:
    if len(args) > 0:
        for arg in args:
            kvargs[arg.Key] = arg.Val

    toolbox.register(name, func, **kvargs)

def get_populations_limits(
    scheme_cfg: cfg.EvoSchemeMultiPopConfigField,
) -> List[List[cfg.EvoLimitGenConfigField]]:
    ret = []
    for population in scheme_cfg.Populations:
        for _ in range(population.IncludingCount):
            ret.append(population.Limits)
    return ret

def get_evo_scheme_result_last_run_pool(
    ind_type: Type,
    scheme_cfg: Union[cfg.EvoSchemeMultiPopConfigField, cfg.EvoSchemeConfigField],
    root: str,
    scheme_name: str,
) -> List[List]:
    run_pool_dir = get_or_create_last_run_pool_dir(root, scheme_name)

    return get_evo_scheme_result_last_iter(
        ind_type,
        scheme_cfg,
        run_pool_dir,
    )

# parse_result_f(stream) -> Any
def restore_evo_scheme_result_from_iter(
    parse_result_f: FunctionType,
    runpool_dir: str,
    iter_num: Union[int, None]=None,
) -> Any:
    if iter_num is None:
        iter_num = get_last_iter_num(
            runpool_dir=runpool_dir,
        )

    if iter_num is None:
        raise f'wrong iter num value (runpool_dir: {runpool_dir}, iter_num: {iter_num})'

    if not is_iter_dir_exists(runpool_dir, iter_num):
        raise f'iter dir isn\'t exist (runpool_dir: {runpool_dir}, iter_num: {iter_num})'

    iter_dir = get_iter_dir_path(
        runpool_dir=runpool_dir,
        iter_num=iter_num,
    )

    result_path = os.path.join(iter_dir, 'result.yaml')

    with open(result_path, 'r') as f:
        return parse_result_f(f)

def get_evo_scheme_result_last_iter(
    ind_type: Type,
    scheme_cfg: Union[cfg.EvoSchemeMultiPopConfigField, cfg.EvoSchemeConfigField],
    runpool_dir: str,
) -> List[List]:
    iter_dir = create_new_iter_dir(runpool_dir)
    if iter_dir is None or iter_dir == '':
        raise f'no iteration folders (dir: {runpool_dir})'

    if isinstance(scheme_cfg, scheme_cfg.EvoSchemeConfigField):
        return get_evo_scheme_result_from_file(
            ind_type=ind_type,
            scheme_cfg=scheme_cfg,
            filename=iter_dir+'result.yaml',
        )
    elif isinstance(scheme_cfg, scheme_cfg.EvoSchemeMultiPopConfigField):
        return get_evo_scheme_multi_pop_result_from_file(
            ind_type=ind_type,
            scheme_cfg=scheme_cfg,
            filename=iter_dir+'result.yaml',
        )

    raise 'unknown scheme cfg type'

def get_evo_scheme_result_from_file(
    ind_type: Type,
    scheme_cfg: cfg.EvoSchemeConfigField,
    filename: str,
) -> List[List]:
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        last_population_yaml = yaml.safe_load(f)
        if last_population_yaml is None:
            raise f'the population from the file "{filename}" is None)'

        if scheme_cfg.PopulationSize != len(last_population_yaml):
            raise f'the population size from the file "{filename}" ({len(last_population_yaml)}) does not match the config ({scheme_cfg.PopulationSize})'

        ret: List[List] = [0] * scheme_cfg.PopulationSize

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
    scheme_cfg: cfg.EvoSchemeMultiPopConfigField,
    filename: str,
) -> List[List]:
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        result = yaml.safe_load(f)
        if result is None:
            raise f'the population from the file "{filename}" is None)'
        elif len(result) != evo_utils.get_populations_cnt(scheme_cfg):
            raise f'not correct populations length from the file "{filename}")'

        ret: List[List] = []

        k = 0
        for pop_cfg in scheme_cfg.Populations:
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
    scheme_cfg: cfg.EvoSchemeMultiPopConfigField,
) -> int:
    if scheme_cfg.Populations is None or len(scheme_cfg.Populations) == 0:
        return 0

    ret = 0
    for pop_cfg in scheme_cfg.Populations:
        if pop_cfg.IncludingCount <= 0:
            continue
        ret += pop_cfg.IncludingCount
    return ret

def get_max_population_size(
    scheme_cfg: cfg.EvoSchemeMultiPopConfigField,
) -> int:
    if scheme_cfg.Populations is None or len(scheme_cfg.Populations) == 0:
        return 0

    max = scheme_cfg.Populations[0].Size
    for pop_cfg in scheme_cfg.Populations:
        if pop_cfg.Size > max:
            max = pop_cfg.Size
    return max

def create_ind_by_list(
    list_ind: List,
    evaluate_f: FunctionType,
) -> evo_types.Individual:
    ret = list_ind
    if not isinstance(list_ind, evo_types.Individual):
        ret = evo_types.Individual(list_ind)
    # if not ret.fitness.valid:
    #     ret.fitness.values = evaluate_f(ret)
    return ret

def create_model_by_type(
    model_type: str,
) -> Model:
    model_type = model_type.lower()
    if model_type == 'lorenz':
        return LorenzModel(cfg.Config.Models.Lorenz)
    if model_type == 'chui_moffatt':
        return ChuiMoffattModel(cfg.Config.Models.ChuiMoffat)
    if model_type == 'moehlis':
        return MoehlisModel(cfg.Config.Models.Moehlis)
    raise f'unknown model - {model_type}'


# Evo schemes utils

def _rmse(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    return np.sqrt(np.sum((expected - actual)**2)/expected.shape[0])

def _map_evaluate_metric_func(
    metric: str,
) -> FunctionType:
    metric = metric.lower()

    if metric == 'rmse':
        return _rmse
    elif metric == 'mse':
        return metrics.mean_squared_error
    elif metric == 'mae':
        metrics.median_absolute_error
    raise 'unknown metric name'

def calc_metric(
    metric: str,
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    if expected.size != actual.size:
        raise f'expected size is not equal actual (expected size: {expected.size}, actaul size: {actual.size})'

    fn = _map_evaluate_metric_func(metric)
    if fn is None:
        raise f'metric func is None (actaul: {metric})'

    return fn(expected, actual)

def get_predict_data(
    model: esn.EsnForecaster,
    evaluate_cfg: cfg.EsnEvaluateConfigField,
    data_shape: np.ndarray,
) -> np.ndarray:
    if evaluate_cfg.MaxSteps <= 0:
        return model.predict(data_shape[0])

    ret = np.ndarray(data_shape)

    i = 0
    max_i = data_shape[0]
    n = evaluate_cfg.MaxSteps
    while i < max_i:
        if i + n >= max_i:
            n = max_i - i

        predict_local = model.predict(n)
        for j in range(n):
            ret[i+j,:] = predict_local[j,:]

        i += n

    return ret

def get_fit_predict_data(
    model: esn.EsnForecaster,
    evaluate_cfg: cfg.EsnEvaluateConfigField,
    fit_data: np.ndarray,
    valid_data: np.ndarray,
) -> np.ndarray:
    if evaluate_cfg.MaxSteps <= 0:
        return model.fit_predict(fit_data, n_timesteps=valid_data.shape[0])

    model.fit(fit_data)

    ret = np.ndarray(valid_data.shape)

    i = 0
    max_i = valid_data.shape[0]
    n = evaluate_cfg.MaxSteps
    while i < max_i:
        if i + n >= max_i:
            n = max_i - i

        predict_local = model.predict(n)
        for j in range(n):
            ret[i+j,:] = predict_local[j,:]

        i += n

    return ret

_colors = [
    'green', 'red', 'blue', 'black', 'yellow', 'pink', 'gray',
]
_colors_idx = 0

def get_next_color(
    exclude: List=[],
) -> str:
    global _colors_idx

    ret = None

    while True:
        ret = _colors[_colors_idx]
        _colors_idx = (_colors_idx + 1) % len(_colors)
        if ret not in exclude:
            break

    return ret

def ind_stat(
    ind: evo_types.Individual,
) -> float:
    return np.dot(ind.fitness.values, ind.fitness.weights)
