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
    population: List[algorithms.Popolation],
    ) -> Tuple[bool, List]:
    expected = cfg.HromoLen
    for ind in population:
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

    def _evo_callback(population, gen, **kvargs):
        ax.clear()

        pop_size = len(population)

        ax.set_xlim(0, pop_size + 1)
        ax.set_ylim(0, cfg.HromoLen + 1)

        ax.set_xlabel('idx')
        ax.set_ylabel('fitness')

        ax.set_title(f'test #0\ngeneration = {gen}')

        color = utils.get_next_color()
        points = [0] * pop_size
        for i in range(pop_size):
            points[i] = (i, population[i].fitness.values[0] * population[i].fitness.weights[0])
        ax.scatter(*zip(*points), color=color, s=2, zorder=0)

        plt.pause(0.001)

    return _evo_callback

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
                _test_logger.warn('restored result has been set')
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
            for population in populations:
                validation_inds = population.HallOfFame.items if population.HallOfFameSize > 0\
                    else population.Inds
                ok, sol = test[utils.TEST_CFG_KEY_VALIDATE_RESULT_F](cfg, validation_inds)
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
