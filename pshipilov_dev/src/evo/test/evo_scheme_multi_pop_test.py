from types import FunctionType
from matplotlib import pyplot as plt
from deap import creator, tools, base, algorithms
from typing import Any, Dict, List, Tuple

import pshipilov_dev.src.evo.utils as evo_utils
import pshipilov_dev.src.evo.test.utils as utils

import random
import logging
import numpy as np

from ...log import get_logger

from pshipilov_dev.src.utils import get_optional_arg, kv_config_arr_to_kvargs, get_args_via_kvargs
from pshipilov_dev.src.config import EvoSchemeMultiPopConfigField
from pshipilov_dev.src.evo.evo_scheme_multi_pop import EvoSchemeMultiPop

# Test 0 implementations

def _test0_wrap_ind_creator_f(
    cfg: EvoSchemeMultiPopConfigField,
    ) -> None:
    def _ind_creator_f():
        ret = [0] * cfg.HromoLen
        for i in range(cfg.HromoLen):
            ret[i] = np.random.randint(2)
        return evo_utils.create_ind_by_list(ret, _test0_wrap_evaluate_f(cfg))
    return _ind_creator_f

def _test0_wrap_evaluate_f(
    cfg: EvoSchemeMultiPopConfigField,
    ) -> None:
    def _evaluate_f(ind: list) -> Tuple[int]:
        ret = 0
        for x in ind:
            ret += x
        return ret,
    return _evaluate_f

def _test0_validate_result_f(
    cfg: EvoSchemeMultiPopConfigField,
    populations: List[algorithms.Popolation],
) -> Tuple[bool, List]:
    expected = cfg.HromoLen
    for population in populations:
        for ind in population.Inds:
            actual = 0
            for x in ind:
                actual += x
            if actual == expected:
                return True, ind
    return False, None

def _test0_wrap_evo_callback(
    cfg: EvoSchemeMultiPopConfigField,
    ) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(utils.DEF_FIG_WEIGHT_INCHES, utils.DEF_FIG_HEIGHT_INCHES)

    populations_cnt = evo_utils.get_populations_cnt(cfg)
    colors = [0] * populations_cnt
    for i in range(populations_cnt):
        colors[i] = utils.get_next_color()

    def _evo_callback(populations, gen, **kvargs):
        ax.clear()

        for i, population in enumerate(populations):
            ax.set_xlim(0, population.Size + 1)
            ax.set_ylim(0, cfg.HromoLen + 1)

            ax.set_xlabel('idx')
            ax.set_ylabel('fitness')

            ax.set_title(f'evo_scheme_multi_pop\ntest #0\ngeneration = {gen}')

            points = [0] * population.Size
            for j in range(population.Size):
                points[j] = (j, population.Inds[j].fitness.values[0] * population.Inds[j].fitness.weights[0])
            ax.scatter(*zip(*points), color=colors[i], s=2, zorder=0)

        plt.pause(0.001)

    return _evo_callback

# Test 1 implementations

def _test1_wrap_evaluate_f(
    cfg: EvoSchemeMultiPopConfigField,
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
    cfg: EvoSchemeMultiPopConfigField,
    popultaions: List[algorithms.Popolation],
) -> Tuple[bool, List]:
    for popultaion in popultaions:
        for ind in popultaion.Inds:
            expected_x, expected_y = ind
            for coords in _TEST_1_EXPECTED:
                actual_x, actual_y = coords
                if np.fabs(expected_x - actual_x) < _TEST_1_EPS and\
                    np.fabs(expected_y - actual_y) < _TEST_1_EPS:
                    return True, ind
    return False, None

def _test1_wrap_evo_callback(
    cfg: EvoSchemeMultiPopConfigField,
) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(utils.DEF_FIG_WEIGHT_INCHES, utils.DEF_FIG_HEIGHT_INCHES)

    # x_min, x_max = cfg.Limits[0].Min, cfg.Limits[0].Max
    # y_min, y_max = cfg.Limits[1].Min, cfg.Limits[1].Max

    x_min, x_max = cfg.Populations[0].Limits[0].Min, cfg.Populations[0].Limits[0].Max
    y_min, y_max = cfg.Populations[0].Limits[1].Min, cfg.Populations[0].Limits[1].Max

    x, y = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    x_grid, y_grid = np.meshgrid(x, y)
    eval_f = _test1_wrap_evaluate_f(cfg)
    f_expected, = eval_f([x_grid, y_grid])

    populations_cnt = evo_utils.get_populations_cnt(cfg)
    colors = [0] * populations_cnt
    for i in range(populations_cnt):
        colors[i] = utils.get_next_color()

    def _evo_callback(populations, gen, **kvargs):
        ax.clear()

        for i, population in enumerate(populations):
            ax.set_xlim(x_min - x_min * 0.1, x_max + x_max * 0.1)
            ax.set_ylim(y_min - y_min * 0.1, y_max + y_max * 0.1)

            ax.set_xlabel('x')
            ax.set_ylabel('y')

            ax.set_title(f'evo_scheme_multi_pop\ntest #1\ngeneration = {gen}')

            ax.contour(x_grid, y_grid, f_expected)
            ax.scatter(*zip(*population.Inds), color=colors[i], s=2, zorder=0)
            ax.scatter(*zip(*_TEST_1_EXPECTED), marker='X', color='red', zorder=1)

        plt.pause(0.01)

    return _evo_callback

# Test 2 implementations

_TEST_2_EPS = 1e-2
_TEST_2_EXPECTED = ((512.,404.2319),)

def _test2_wrap_evaluate_f(
    cfg: EvoSchemeMultiPopConfigField,
) -> FunctionType:
    def _evaluate_f(ind: list) -> Tuple[float]:
        x, y = ind
        return -(y+47)*np.sin(np.sqrt(np.fabs(x/2+(y+47)))-x*np.sin(np.sqrt(np.fabs(x-(y+47))))),
    return _evaluate_f

def _test2_validate_result_f(
    cfg: EvoSchemeMultiPopConfigField,
    popultaions: List[algorithms.Popolation],
) -> Tuple[bool, List]:
    for popultaion in popultaions:
        for ind in popultaion.Inds:
            expected_x, expected_y = ind
            for coords in _TEST_2_EXPECTED:
                actual_x, actual_y = coords
                if np.fabs(expected_x - actual_x) < _TEST_2_EPS and\
                    np.fabs(expected_y - actual_y) < _TEST_2_EPS:
                    return True, ind
    return False, None

def _test2_wrap_evo_callback(
    cfg: EvoSchemeMultiPopConfigField,
) -> None:
    fig, ax = plt.subplots()
    fig.set_size_inches(utils.DEF_FIG_WEIGHT_INCHES, utils.DEF_FIG_HEIGHT_INCHES)

    x_min, x_max = cfg.Populations[0].Limits[0].Min, cfg.Populations[0].Limits[0].Max
    y_min, y_max = cfg.Populations[0].Limits[1].Min, cfg.Populations[0].Limits[1].Max

    # x, y = np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1)
    # x_grid, y_grid = np.meshgrid(x, y)
    # eval_f = _test2_wrap_evaluate_f(cfg)
    # f_expected, = eval_f([x_grid, y_grid])

    ticks = np.linspace(-512, 512, 5)
    ticks_labels = [str(x) for x in ticks]

    populations_cnt = evo_utils.get_populations_cnt(cfg)
    colors = [0] * populations_cnt
    for i in range(populations_cnt):
        colors[i] = utils.get_next_color()

    def _evo_callback(populations, gen, **kvargs):
        ax.clear()

        for i, population in enumerate(populations):
            radius = 100.
            utils.radius_iter_points(
                population.HallOfFame.items,
                _TEST_2_EXPECTED,
                radius,
                utils.radius_log_callback(_test_logger, 'test #2 - find point in radius {radius} ({x}, {y}), error: {r}', radius=radius),
                utils.radius_highlight_callback(ax, marker='X', color='green', zorder=1),
            )

            ax.scatter(*zip(*population.HallOfFame.items), marker='o', color='blue', zorder=1)
            best = population.HallOfFame.items[0]
            ax.text(-600, 650, f'best: ({best[0]};{best[1]}); value: {best.fitness.values[0]}')

            ax.set_xlim(x_min - x_min * 0.05, x_max + x_max * 0.05)
            ax.set_ylim(y_min - y_min * 0.05, y_max + y_max * 0.05)

            ax.set_xlabel('x')
            ax.set_ylabel('y')

            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks_labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks_labels)

            ax.set_title(f'evo_scheme_multi_pop\ntest #2\ngeneration = {gen}')

            # ax.contour(x_grid, y_grid, f_expected, levels=5)
            ax.scatter(*zip(*population.Inds), color=colors[i], s=2, zorder=0)
            ax.scatter(*zip(*_TEST_2_EXPECTED), marker='X', color='red', zorder=1)

        plt.pause(0.0001)

    return _evo_callback

# def _test2_toolbox_setup_f(
#     cfg: EvoSchemeMultiPopConfigField,
# ) -> base.Toolbox:
#     ret_toolbox = base.Toolbox()

#     def _new_population_override():
#         x_min, x_max = cfg.Limits[0].Min, cfg.Limits[0].Max
#         y_min, y_max = cfg.Limits[1].Min, cfg.Limits[1].Max
#         n = int(np.floor(np.sqrt(cfg.PopulationSize)))
#         x = np.linspace(x_min, x_max, n, True)
#         y = np.linspace(y_min, y_max, n, True)

#         ret = [0] * cfg.PopulationSize
#         ind_cnt = 0
#         for i in range(n):
#             for j in range(n):
#                 ret[ind_cnt] = creator.Individual([x[i], y[j]])
#                 ind_cnt += 1

#         while ind_cnt < cfg.PopulationSize:
#             ret[ind_cnt] = ret_toolbox.new_ind()
#             ind_cnt += 1

#         return ret

#     ret_toolbox.register('new_population', _new_population_override)
#     return ret_toolbox

# Tests main configuration

_TESTS = [
    {
        # utils.TEST_CFG_KEY_DISABLE: True,
        utils.TEST_CFG_KEY_NAME: 'one_max',
        utils.TEST_CFG_KEY_CFG: {
            'rand_seed': 1,
            'max_gen_num': 150,

            'hromo_len': 100,
            'fitness_weights': [1.0,],

            'populations': [
                {
                    'size': 30,
                    'including_count': 5,

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
                },
            ],

            'metrics': [
                {'name': 'max','func': 'max','package': 'numpy'},
                {'name': 'avg','func': 'mean','package': 'numpy'},
            ],
        },
        utils.TEST_CFG_KEY_WRAP_IND_CRATOR_F: _test0_wrap_ind_creator_f,
        utils.TEST_CFG_KEY_WRAP_EVALUATR_F: _test0_wrap_evaluate_f,
        utils.TEST_CFG_KEY_VALIDATE_RESULT_F: _test0_validate_result_f,
        utils.TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test0_wrap_evo_callback,
    },
    {
        # utils.TEST_CFG_KEY_DISABLE: True,
        utils.TEST_CFG_KEY_NAME: 'himmelblau',
        utils.TEST_CFG_KEY_CFG: {
            'rand_seed': 3,
            'max_gen_num': 150,
            'hromo_len': 2,

            'fitness_weights': [-1.0,],

            'populations': [
                {
                    'size': 30,
                    'hall_of_fame': 3,

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
                            {'key':'eta','val':5},
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
                },
                {
                    'size': 30,
                    'hall_of_fame': 3,

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
                            {'key':'eta','val':10},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.1,
                        'args': [
                            {'key':'low','val':-5},
                            {'key':'up','val':5},
                            {'key':'eta','val':15},
                        ],
                    },

                    'limits': [
                        {'min': -5, 'max': 5},
                        {'min': -5, 'max': 5},
                    ],
                },
                {
                    'size': 30,
                    'hall_of_fame': 3,

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
                            {'key':'eta','val':15},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.1,
                        'args': [
                            {'key':'low','val':-5},
                            {'key':'up','val':5},
                            {'key':'eta','val':15},
                        ],
                    },

                    'limits': [
                        {'min': -5, 'max': 5},
                        {'min': -5, 'max': 5},
                    ],
                },
                                {
                    'size': 30,
                    'hall_of_fame': 3,

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
                            {'key':'eta','val':10},
                        ],
                    },

                    'limits': [
                        {'min': -5, 'max': 5},
                        {'min': -5, 'max': 5},
                    ],
                },
            ],

            'metrics': [
                {'name': 'min','func': 'min','package': 'numpy'},
                {'name': 'avg','func': 'mean','package': 'numpy'},
            ],
        },
        utils.TEST_CFG_KEY_WRAP_EVALUATR_F: _test1_wrap_evaluate_f,
        utils.TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test1_wrap_evo_callback,
        utils.TEST_CFG_KEY_VALIDATE_RESULT_F: _test1_validate_result_f,
    },
    {
        # utils.TEST_CFG_KEY_DISABLE: True,
        utils.TEST_CFG_KEY_NAME: 'eggholder',
        utils.TEST_CFG_KEY_CFG: {
            'rand_seed': 12,
            'max_gen_num': 500,

            'hromo_len': 2,

            'fitness_weights': [-1.0,],

            'populations': [
                {
                    'size': 50,
                    'hall_of_fame': 7,

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
                },
                {
                    'size': 50,
                    'hall_of_fame': 7,

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
                            {'key':'eta','val':15},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.3,
                        'args': [
                            {'key':'low','val':-512},
                            {'key':'up','val':512},
                            {'key':'eta','val':10},
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
                },
{
                    'size': 50,
                    'hall_of_fame': 7,

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
                            {'key':'eta','val':10},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.3,
                        'args': [
                            {'key':'low','val':-512},
                            {'key':'up','val':512},
                            {'key':'eta','val':15},
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
                },
                {
                    'size': 50,
                    'hall_of_fame': 7,

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
                            {'key':'eta','val':5},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.3,
                        'args': [
                            {'key':'low','val':-512},
                            {'key':'up','val':512},
                            {'key':'eta','val':20},
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
                },
                {
                    'size': 50,
                    'hall_of_fame': 7,

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
                            {'key':'eta','val':15},
                        ],
                    },
                    'mutate': {
                        'method': 'mutPolynomialBounded',
                        'probability': 0.3,
                        'args': [
                            {'key':'low','val':-512},
                            {'key':'up','val':512},
                            {'key':'eta','val':15},
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
                },
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
        utils.TEST_CFG_KEY_WRAP_EVALUATR_F: _test2_wrap_evaluate_f,
        utils.TEST_CFG_KEY_WRAP_EVO_CALLBACK: _test2_wrap_evo_callback,
        utils.TEST_CFG_KEY_VALIDATE_RESULT_F: _test2_validate_result_f,
        # utils.TEST_CFG_KEY_TOOLBOX_SETUP_F: _test2_toolbox_setup_f,
    },
]

# Main test logger
_test_logger: logging.Logger = None

def run_tests(**kvargs):
    global _test_logger
    _test_logger = get_logger(name='Test<evo_scheme_multi_pop>', level='debug')

    args = get_args_via_kvargs(kvargs)
    disable_iter_graph = get_optional_arg(args, 'test_disable_iter_graph', default=False)
    disable_stat_graph = get_optional_arg(args, 'test_disable_stat_graph', default=False)
    disable_dump = get_optional_arg(args, 'test_disable_dump', default=False)
    restore_result = get_optional_arg(args, 'test_restore_result', default=False)
    tests_dumpdir = get_optional_arg(args, 'test_dump_dir', default=utils.DEF_TEST_DUMP_DIR)

    if len(tests_dumpdir) == 0:
        tests_dumpdir = './'

    if ord(tests_dumpdir[len(tests_dumpdir)-1]) != ord('/'):
        tests_dumpdir += '/'

    tests_dumpdir += 'evo_scheme_multi_pop/'

    if not disable_iter_graph:
        plt.ion()

    active_tests = []
    metricValuesMap = {}

    for i, test in enumerate(_TESTS):
        if not utils.validate_test(test, i, _test_logger):
            continue

        disable = test.get(utils.TEST_CFG_KEY_DISABLE, False)
        if disable:
            _test_logger.info('test #%d - skip test (disabled)', i)
            continue

        active_tests.append(i)

        # Evo scheme prepare
        cfg = EvoSchemeMultiPopConfigField()
        cfg.load(test[utils.TEST_CFG_KEY_CFG])
        name = test[utils.TEST_CFG_KEY_NAME] if utils.TEST_CFG_KEY_NAME in test else f'test_#{i}'
        ind_creator_f = test[utils.TEST_CFG_KEY_WRAP_IND_CRATOR_F](cfg) if test.get(utils.TEST_CFG_KEY_WRAP_IND_CRATOR_F, None) is not None else None
        evo_callback = test[utils.TEST_CFG_KEY_WRAP_EVO_CALLBACK](cfg) if not disable_iter_graph and test.get(utils.TEST_CFG_KEY_WRAP_EVO_CALLBACK, None) is not None else None
        evaluate_f = test[utils.TEST_CFG_KEY_WRAP_EVALUATR_F](cfg)
        scheme = EvoSchemeMultiPop(
            name,
            cfg,
            evaluate_f,
            ind_creator_f,
        )

        dumpdir = tests_dumpdir
        if ord(dumpdir[len(dumpdir)-1]) != ord('/'):
            dumpdir += '/'
        dumpdir += f'test_{i}_{name}'

        if restore_result:
            restored_result = evo_utils.get_evo_scheme_result_last_run_pool(
                evo_utils.get_evo_scheme_multi_pop_result_from_file,
                lambda ind: evo_utils.create_ind_by_list(ind, evaluate_f),
                cfg,
                dumpdir,
                name,
            )
            if restored_result is not None:
                scheme.restore_result(restored_result)
                _test_logger.info('restored result has been set')
            else:
                _test_logger.warn('restored result has not been set')

        # Set random state
        random.seed(cfg.RandSeed)
        np.random.seed(cfg.RandSeed)

        stop_cond = None
        if utils.TEST_CFG_KEY_VALIDATE_RESULT_F in test:
            stop_cond = lambda population, gen, **kvargs: test[utils.TEST_CFG_KEY_VALIDATE_RESULT_F](cfg, population)[0]

        # Action
        scheme.run(callback=evo_callback, stop_cond=stop_cond)

        if not disable_dump:
            scheme.save(dumpdir)

        if not disable_stat_graph and len(cfg.Metrics) > 0:
            logbook = scheme.get_logbook()
            metricValuesMap[i] = cfg.Metrics, logbook.select(*[metric.Name for metric in cfg.Metrics])

        # Assert
        if utils.TEST_CFG_KEY_VALIDATE_RESULT_F in test:
            populations = scheme.get_populations()
            ok, sol = test[utils.TEST_CFG_KEY_VALIDATE_RESULT_F](cfg, populations)
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
