from types import FunctionType
from matplotlib import pyplot as plt
from deap import creator, tools, base
from typing import Any, Dict, List, Tuple

import random
import logging
import numpy as np

from pshipilov_dev_run import _get_args_via_kvargs

from ..log import get_logger

from pshipilov_dev.src.utils import get_optional_arg, kv_config_arr_to_kvargs
from pshipilov_dev.src.config import EvoSchemeConfigField
from pshipilov_dev.src.evo.evo_scheme import EvoScheme

# Const

_FIG_HEIGHT_INCHES = 8
_FIG_WEIGHT_INCHES = 8

# Test 0 implementations

def _test0_wrap_ind_creator_f(
    cfg: EvoSchemeConfigField,
    ) -> None:
    def _ind_creator_f():
        ret = [0] * cfg.HromoLen
        for i in range(cfg.HromoLen):
            ret[i] = np.random.randint(2)
        return creator.Individual(ret)
    return _ind_creator_f

def _test0_wrap_evaluate_f(
    cfg: EvoSchemeConfigField,
    ) -> None:
    def _evaluate_f(ind: list) -> Tuple[int]:
        ret = 0
        for x in ind:
            ret += x
        return ret,
    return _evaluate_f

def _test0_validate_result_f(
    cfg: EvoSchemeConfigField,
    last_popultaion: list,
    ) -> Tuple[bool, List]:
    expected = cfg.HromoLen
    for ind in last_popultaion:
        actual = 0
        for x in ind:
            actual += x
        if actual == expected:
            return True, ind
    return False, None

def _test0_wrap_evo_callback(
    cfg: EvoSchemeConfigField,
    ) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(_FIG_WEIGHT_INCHES, _FIG_HEIGHT_INCHES)

    def _evo_callback(population, gen, **kvargs):
        ax.clear()

        ax.set_xlim(0, cfg.PopulationSize + 1)
        ax.set_ylim(0, cfg.HromoLen + 1)

        ax.set_xlabel('idx')
        ax.set_ylabel('fitness')

        ax.set_title(f'test #0\ngeneration = {gen}')

        points = [0] * cfg.PopulationSize
        for i in range(cfg.PopulationSize):
            points[i] = (i, population[i].fitness.values[0] * population[i].fitness.weights[0])
        ax.scatter(*zip(*points), color='green', s=2, zorder=0)

        plt.pause(0.001)

    return _evo_callback

# Test 1 implementations

def _test1_wrap_evaluate_f(
    cfg: EvoSchemeConfigField,
    ) -> None:
    def _evaluate_f(ind: list) -> Tuple[float]:
        x, y = ind
        return (x**2+y-11)**2+(x+y**2-7)**2,
    return _evaluate_f

_TEST_1_EPS = 1e-2
_TEST_1_EXPECTED = (
    (3.,2.),(-2.805118,3.131312),
    (-3.779310,-3.283186),(3.584458,-1.848126),
)

def _test1_validate_result_f(
    cfg: EvoSchemeConfigField,
    last_popultaion: list,
    ) -> Tuple[bool, List]:
    for ind in last_popultaion:
        expected_x, expected_y = ind
        for coords in _TEST_1_EXPECTED:
            actual_x, actual_y = coords
            if np.fabs(expected_x - actual_x) < _TEST_1_EPS and\
                np.fabs(expected_y - actual_y) < _TEST_1_EPS:
                return True, ind
    return False, None

def _test1_wrap_evo_callback(
    cfg: EvoSchemeConfigField,
    ) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(_FIG_WEIGHT_INCHES, _FIG_HEIGHT_INCHES)

    x_min, x_max = cfg.Limits[0].Min, cfg.Limits[0].Max
    y_min, y_max = cfg.Limits[1].Min, cfg.Limits[1].Max

    x, y = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    eval_f = _test1_wrap_evaluate_f(cfg)
    f_expected, = eval_f([x_grid, y_grid])

    def _evo_callback(population, gen, **kvargs):
        ax.clear()

        ax.set_xlim(x_min - x_min * 0.1, x_max + x_max * 0.1)
        ax.set_ylim(y_min - y_min * 0.1, y_max + y_max * 0.1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_title(f'test #1\ngeneration = {gen}')

        ax.contour(x_grid, y_grid, f_expected)
        ax.scatter(*zip(*population), color='green', s=2, zorder=0)
        ax.scatter(*zip(*_TEST_1_EXPECTED), marker='X', color='red', zorder=1)

        plt.pause(0.01)

    return _evo_callback

# Test 2 implementations

_TEST_2_EPS = 1e-2
_TEST_2_EXPECTED = ((512.,404.2319),)

def _test2_wrap_evaluate_f(
    cfg: EvoSchemeConfigField,
) -> FunctionType:
    def _evaluate_f(ind: list) -> Tuple[float]:
        x, y = ind
        return -(y+47)*np.sin(np.sqrt(np.fabs(x/2+(y+47)))-x*np.sin(np.sqrt(np.fabs(x-(y+47))))),
    return _evaluate_f

def _test2_validate_result_f(
    cfg: EvoSchemeConfigField,
    last_popultaion: list,
) -> Tuple[bool, List]:
    for ind in last_popultaion:
        expected_x, expected_y = ind
        for coords in _TEST_2_EXPECTED:
            actual_x, actual_y = coords
            if np.fabs(expected_x - actual_x) < _TEST_2_EPS and\
                np.fabs(expected_y - actual_y) < _TEST_2_EPS:
                return True, ind
    return False, None

def _test2_wrap_evo_callback(
    cfg: EvoSchemeConfigField,
    ) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(_FIG_WEIGHT_INCHES, _FIG_HEIGHT_INCHES)

    x_min, x_max = cfg.Limits[0].Min, cfg.Limits[0].Max
    y_min, y_max = cfg.Limits[1].Min, cfg.Limits[1].Max

    # x, y = np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1)
    # x_grid, y_grid = np.meshgrid(x, y)
    # eval_f = _test2_wrap_evaluate_f(cfg)
    # f_expected, = eval_f([x_grid, y_grid])

    ticks = np.linspace(-512, 512, 5)
    ticks_labels = [str(x) for x in ticks]

    def _evo_callback(population, gen, **kvargs):
        ax.clear()

        halloffame: tools.HallOfFame = kvargs.get('halloffame', None)

        radius = 100.
        _radius_iter_points(
            halloffame.items,
            _TEST_2_EXPECTED,
            radius,
            _radius_log_callback(_test_logger, 'test #2 - find point in radius {radius} ({x}, {y}), error: {r}', radius=radius),
            _radius_highlight_callback(ax, marker='X', color='green', zorder=1),
        )

        ax.scatter(*zip(*halloffame.items), marker='o', color='blue', zorder=1)
        best = halloffame.items[0]
        ax.text(-600, 650, f'best: ({best[0]};{best[1]}); value: {best.fitness.values[0]}')

        ax.set_xlim(x_min - x_min * 0.05, x_max + x_max * 0.05)
        ax.set_ylim(y_min - y_min * 0.05, y_max + y_max * 0.05)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)

        ax.set_title(f'test #2\ngeneration = {gen}')

        # ax.contour(x_grid, y_grid, f_expected, levels=5)
        ax.scatter(*zip(*population), color='green', s=2, zorder=0)
        ax.scatter(*zip(*_TEST_2_EXPECTED), marker='X', color='red', zorder=1)

        plt.pause(0.0001)

    return _evo_callback

def _test2_toolbox_setup_f(
    cfg: EvoSchemeConfigField,
) -> base.Toolbox:
    ret_toolbox = base.Toolbox()

    def _new_population_override():
        x_min, x_max = cfg.Limits[0].Min, cfg.Limits[0].Max
        y_min, y_max = cfg.Limits[1].Min, cfg.Limits[1].Max
        n = int(np.floor(np.sqrt(cfg.PopulationSize)))
        x = np.linspace(x_min, x_max, n, True)
        y = np.linspace(y_min, y_max, n, True)

        ret = [0] * cfg.PopulationSize
        ind_cnt = 0
        for i in range(n):
            for j in range(n):
                ret[ind_cnt] = creator.Individual([x[i], y[j]])
                ind_cnt += 1

        while ind_cnt < cfg.PopulationSize:
            ret[ind_cnt] = ret_toolbox.new_ind()
            ind_cnt += 1

        return ret

    ret_toolbox.register('new_population', _new_population_override)
    return ret_toolbox

# Test config params

_TEST_CFG_KEY_NAME              = 'name'
_TEST_CFG_KEY_CFG               = 'cfg'
_TEST_CFG_KEY_WRAP_EVALUATR_F   = 'wrap_evaluate_f'
_TEST_CFG_KEY_WRAP_IND_CRATOR_F = 'wrap_ind_creator_f'
_TEST_CFG_KEY_WRAP_EVO_CALLBACK = 'wrap_evo_callback'
_TEST_CFG_KEY_VALIDATE_RESULT_F = 'valitate_result_f'
_TEST_CFG_KEY_TOOLBOX_SETUP_F   = 'toolbox_setup_f'
_TEST_CFG_KEY_DISABLE           = 'disable'

# Tests main configuration

_TESTS = [
    {
        # _TEST_CFG_KEY_DISABLE: True,
        _TEST_CFG_KEY_NAME: 'one_max',
        _TEST_CFG_KEY_CFG: {
            'rand_seed': 1,
            'max_gen_num': 150,
            'population_size': 50,
            'hromo_len': 100,

            'fitness_weights': [1.0,],

            'select': {
                'method': 'selTournament',
                'args': [
                    {'key': 'tournsize','val': 3},
                ],
            },
            'mate': {
                'method': 'cxOnePoint',
                'probability': 0.9,
            },
            'mutate': {
                'method': 'mutFlipBit',
                'probability': 0.1,
               # 'indpb': 0.25,
            },

            'metrics': [
                {'name': 'max','func': 'max','package': 'numpy'},
                {'name': 'avg','func': 'mean','package': 'numpy'},
            ],
        },
        _TEST_CFG_KEY_WRAP_IND_CRATOR_F: _test0_wrap_ind_creator_f,
        _TEST_CFG_KEY_WRAP_EVALUATR_F: _test0_wrap_evaluate_f,
        _TEST_CFG_KEY_VALIDATE_RESULT_F: _test0_validate_result_f,
        _TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test0_wrap_evo_callback,
    },
    {
        # _TEST_CFG_KEY_DISABLE: True,
        _TEST_CFG_KEY_NAME: 'himmelblau',
        _TEST_CFG_KEY_CFG: {
            'rand_seed': 3,
            'max_gen_num': 150,
            'population_size': 30,
            'hromo_len': 2,
            'hall_of_fame': 5,

            'fitness_weights': [-1.0,],

            'select': {
                'method': 'selTournament',
                'args': [
                    {'key': 'tournsize','val': 3},
                ],
                # 'method': 'selRoulette',
            },
            'mate': {
                'method': 'cxSimulatedBinaryBounded',
                'probability': 0.9,
                'args': [
                    {'key':'low','val':-5},
                    {'key':'up','val':5},
                    {'key':'eta','val':20},
                ],
            },
            'mutate': {
                'method': 'mutPolynomialBounded',
                'probability': 0.1,
                'args': [
                    {'key':'low','val':-5},
                    {'key':'up','val':5},
                    {'key':'eta','val':20},
                ],
            },

            'limits': [
                {'min': -5, 'max': 5},
                {'min': -5, 'max': 5},
            ],
            'metrics': [
                {'name': 'min','func': 'min','package': 'numpy'},
                {'name': 'avg','func': 'mean','package': 'numpy'},
            ],
        },
        _TEST_CFG_KEY_WRAP_EVALUATR_F: _test1_wrap_evaluate_f,
        _TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test1_wrap_evo_callback,
        _TEST_CFG_KEY_VALIDATE_RESULT_F: _test1_validate_result_f,
    },
    {
        # _TEST_CFG_KEY_DISABLE: True,
        _TEST_CFG_KEY_NAME: 'eggholder',
        _TEST_CFG_KEY_CFG: {
            'rand_seed': 12,
            'max_gen_num': 1000,
            'population_size': 150,
            'hromo_len': 2,
            'hall_of_fame': 7,

            'fitness_weights': [-1.0,],

            'select': {
                # 'method': 'selRoulette',
                'method': 'selTournament',
                'args': [
                    {'key': 'tournsize','val': 2},
                ],
            },
            'mate': {
                'probability': 0.8,
                'method': 'cxSimulatedBinaryBounded',
                'args': [
                    {'key':'low','val':-512},
                    {'key':'up','val':512},
                    {'key':'eta','val':20},
                ],
            },
            'mutate': {
                'method': 'mutPolynomialBounded',
                'probability': 0.3,
                'args': [
                    {'key':'low','val':-512},
                    {'key':'up','val':512},
                    {'key':'eta','val':5},
                    {'key':'indpb','val':1.},
                ],
                # 'method': 'dynoMutGauss',
                # 'probability': 0.1,
                # 'args': [
                #     {'key':'low','val':-512},
                #     {'key':'up','val':512},
                #     {'key':'sigma','val':0.3},
                # ],
            },

            'limits': [
                {'min': -512, 'max': 512},
                {'min': -512, 'max': 512},
            ],
            'metrics': [
                {
                    'name':'min',
                    'func':'min',
                    'package':'numpy',
                    'plt_args': [
                        {'key':'color', 'val': 'green'},
                    ],
                },
                {'name':'avg','func':'mean','package': 'numpy'},
            ],
        },
        _TEST_CFG_KEY_WRAP_EVALUATR_F: _test2_wrap_evaluate_f,
        _TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test2_wrap_evo_callback,
        _TEST_CFG_KEY_VALIDATE_RESULT_F: _test2_validate_result_f,
        _TEST_CFG_KEY_TOOLBOX_SETUP_F: _test2_toolbox_setup_f,
    },
]

# Main test logger
_test_logger: logging.Logger = None

_DEF_TEST_DUMP_DIR = 'pshipilov_dev/dumps/tests'

def run_tests(**kvargs):
    global _test_logger
    _test_logger = get_logger(name='tests', level='debug')

    args = _get_args_via_kvargs(kvargs)
    disable_iter_graph = get_optional_arg(args, 'test_disable_iter_graph', default=False)
    disable_stat_graph = get_optional_arg(args, 'test_disable_stat_graph', default=False)
    disable_dump = get_optional_arg(args, 'test_disable_dump', default=False)
    restore_result = get_optional_arg(args, 'test_restore_result', default=False)
    tests_dumpdir = get_optional_arg(args, 'test_dump_dir', default=_DEF_TEST_DUMP_DIR)

    if not disable_iter_graph:
        plt.ion()

    active_tests = []
    metricValuesMap = {}

    for i, test in enumerate(_TESTS):
        if not _validate_test(test, i, _test_logger):
            continue

        disable = test.get(_TEST_CFG_KEY_DISABLE, False)
        if disable:
            _test_logger.info('test #%d - skip test (disabled)', i)
            continue

        active_tests.append(i)

        # Evo scheme prepare
        cfg = EvoSchemeConfigField()
        cfg.load(test[_TEST_CFG_KEY_CFG])
        name = test[_TEST_CFG_KEY_NAME] if _TEST_CFG_KEY_NAME in test else f'test_#{i}'
        ind_creator_f = test[_TEST_CFG_KEY_WRAP_IND_CRATOR_F](cfg) if test.get(_TEST_CFG_KEY_WRAP_IND_CRATOR_F, None) is not None else None
        evo_callback = test[_TEST_CFG_KEY_WRAP_EVO_CALLBACK](cfg) if not disable_iter_graph and test.get(_TEST_CFG_KEY_WRAP_EVO_CALLBACK, None) is not None else None
        toolbox = test[_TEST_CFG_KEY_TOOLBOX_SETUP_F](cfg) if test.get(_TEST_CFG_KEY_TOOLBOX_SETUP_F, None) is not None else None
        scheme = EvoScheme(
            name,
            cfg,
            test[_TEST_CFG_KEY_WRAP_EVALUATR_F](cfg),
            ind_creator_f,
            toolbox,
        )

        # Set random state
        random.seed(cfg.RandSeed)
        np.random.seed(cfg.RandSeed)

        stop_cond = None
        if _TEST_CFG_KEY_VALIDATE_RESULT_F in test:
            stop_cond = lambda population, gen, **kvargs: test[_TEST_CFG_KEY_VALIDATE_RESULT_F](cfg, population)[0]

        dumpdir = f'{tests_dumpdir}/test_{i}_{name}'

        if restore_result:
            scheme.restore_population_last_run_pool(dumpdir)

        # Action
        scheme.run(callback=evo_callback, stop_cond=stop_cond)

        if not disable_dump:
            scheme.save(dumpdir)

        if not disable_stat_graph and len(cfg.Metrics) > 0:
            logbook = scheme.get_logbook()
            metricValuesMap[i] = cfg.Metrics, logbook.select(*[metric.Name for metric in cfg.Metrics])

        # Assert
        if _TEST_CFG_KEY_VALIDATE_RESULT_F in test:
            validation_inds = scheme.get_hall_of_fame().items if cfg.HallOfFame > 0 else scheme.get_last_population()
            ok, sol = test[_TEST_CFG_KEY_VALIDATE_RESULT_F](cfg, validation_inds)
            if ok:
                _test_logger.info('test #%d - successful (one of solution: [%s])', i, ','.join(map(str, sol)))
            else:
                _test_logger.error('test #%d - wrong', i)
        else:
            _test_logger.warn('test #%d - skip result validation', i)

    if not disable_iter_graph:
        plt.ioff()

    if not disable_stat_graph and disable_dump:
        len_active_tests = len(active_tests)
        fig, axes = plt.subplots(1, len_active_tests)
        fig.suptitle('evo stats')
        if len_active_tests == 1:
            axes = axes,

        for i, n in enumerate(active_tests):
            ax = axes[i]
            ax.set_title(f'test #{n}')
            metrics = metricValuesMap.get(n, None)
            if metrics is not None:
                metric_cfgs, values = metrics
                for i in range(len(values)):
                    ax.plot(values[i], label=metric_cfgs[i].Name, **kv_config_arr_to_kvargs(metric_cfgs[i].PltArgs))
                ax.set_xlabel('generation')
                ax.set_ylabel('fitness')
                ax.legend()

    if not disable_stat_graph:
        plt.show()

# Utils

def _validate_test(test: dict, num: int, logger: logging.Logger) -> bool:
    rules = [
        # {
        #     'msg': f'skip test #{num}, validation error: len args is not in range [2,8]',
        #     'cond': lambda test: len(test) < 2 or len(test) > 8,
        # },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_TEST_CFG_KEY_CFG}"',
            'cond': lambda test: _TEST_CFG_KEY_CFG not in test,
        },
        # {
        #     'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_WRAP_IND_CRATOR_F_ARG}"',
        #     'cond': lambda test: _WRAP_IND_CRATOR_F_ARG not in test,
        # },
        {
            'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_TEST_CFG_KEY_WRAP_EVALUATR_F}"',
            'cond': lambda test: _TEST_CFG_KEY_WRAP_EVALUATR_F not in test,
        },
        # {
        #     'msg': f'skip test #{num}, validation error: test doesn\'t contain arg "{_VALIDATE_RESULT_F}"',
        #     'cond': lambda test: _VALIDATE_RESULT_F not in test,
        # },
    ]

    for rule in rules:
        if rule['cond'](test):
            logger.error(rule['msg'])
            return False
    return True

# Radius API functions: iterate 2D points and search points in radius by center points and call callbacks functions on it

def _radius_iter_points(
    points: List[Tuple[float,float]],
    centers: List[Tuple[float, float]],
    radius: float,
    *callbacks: List[FunctionType],
) -> None:
    for x, y in points:
        for c_x, c_y in centers:
            r = np.sqrt((x-c_x)**2+(y-c_y)**2)
            if r <= radius:
                for callback in callbacks:
                    callback(x=x, y=y, r=r)

def _radius_log_callback(
    logger: logging.Logger,
    pattern: str,
    **kvargs: Dict[str, Any],
) -> FunctionType:
    return lambda x, y, r: logger.info(pattern.format(x=x, y=y, r=r, **kvargs))

def _radius_highlight_callback(
    ax: plt.Axes,
    **kvargs: Dict[str, Any],
) -> FunctionType:
    if 'marker' not in kvargs:
        kvargs['marker'] = 'X'
    if 'color' not in kvargs:
        kvargs['color'] = 'green'
    if 'zorder' not in kvargs:
        kvargs['zorder'] = 1
    return lambda x, y, r: ax.scatter(x, y, **kvargs)
