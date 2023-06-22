from src.evo.abstract import Scheme
from src.evo.tasks.grid_search_hyper_param import GridSearchHyperParam
from src.evo.tasks.dyno_evo_esn_hyper_param import DynoEvoEsnHyperParam
from src.evo.tasks.random_search_hyper_param import RandomSearchHyperParam
from src.evo.tasks.duno_evo_esn_hyper_param_multi_pop import DynoEvoEsnHyperParamMultiPop
from src.evo.tasks.duno_evo_esn_hyper_param_multi_pop_multi_crit import DynoEvoEsnHyperParamMultiPopMultiCrit
from src.evo.tasks.dyno_evo_esn_huper_param_with_reservoir_multi_pop import DynoEvoEsnHyperParamWithReservoirMultiPop

import src.evo.test.evo_scheme_test as evo_scheme_test
import src.evo.test.evo_scheme_multi_pop_test as evo_scheme_multi_pop_test
import src.utils as utils
import src.log as log
import src.dump as dump
import src.config as cfg

from multiprocess.managers import SyncManager

import numpy as np
import argparse
import random


# Modes for running
_MODE_TESTS = 'tests'
# _MODE_GRID = 'grid'
_MODE_HYPER_PARAMS = 'hyper_param'
_MODE_HYPER_PARAMS_MULTI_POP = 'hyper_param_multi_pop'
_MODE_HYPER_PARAMS_GRID_SEARCH = 'hyper_param_grid_search'
_MODE_HYPER_PARAMS_RANDOM_SEARCH = 'hyper_param_random_search'
_MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT = 'hyper_param_multi_pop_multi_crit'
_MODE_HYPER_PARAMS_WITH_RESERVOIR_MULTI_POP = 'hyper_param_with_reservoir_multi_pop'

def run_tests(**kvargs):
    # TODO :
    evo_scheme_test.run_tests(**kvargs)
    evo_scheme_multi_pop_test.run_tests(**kvargs)

def check_restore(
    args,
    scheme: Scheme,
) -> None:
    if not hasattr(args, 'continue_dir'):
        return

    continue_dir = getattr(args, 'continue_dir', None)
    if continue_dir is None:
        return

    scheme.restore_result(continue_dir)

def run_scheme(scheme_type, scheme_cfg, **kvargs):
    args = utils.get_args_via_kvargs(kvargs)
    async_manager = utils.get_via_kvargs(kvargs, utils._KVARGS_ASYNC_MANAGER)

    scheme: Scheme = scheme_type(
        scheme_cfg=scheme_cfg,
        async_manager=async_manager,
        job_n=args.job_n,
    )

    check_restore(args, scheme)
    scheme.run()

    dump.do(evo_scheme=scheme)

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Choises

    parser.add_argument('-m',
        type=str,
        required=True,
        choices=[
            _MODE_TESTS,
            _MODE_HYPER_PARAMS,
            _MODE_HYPER_PARAMS_MULTI_POP,
            _MODE_HYPER_PARAMS_GRID_SEARCH,
            _MODE_HYPER_PARAMS_RANDOM_SEARCH,
            _MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT,
            _MODE_HYPER_PARAMS_WITH_RESERVOIR_MULTI_POP,
        ],
        help='run mode'
    )

    # Boolean flags

    parser.add_argument('--disable-config',
        action='store_true',
        help='disable loading config from config dir'
    )
    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='print all logs'
    )

    # Numbers

    parser.add_argument('--job-n',
        type=int,
        nargs='?',
        default=-1,
        help='Number of parallel processes for runing scheme'
    )

    # Paths

    parser.add_argument('-с', "--config-path",
        type=str,
        nargs='?',
        help='config path'
    )
    parser.add_argument('--log-dir',
        type=str,
        nargs='?',
        default='logs',
        help='directory for writing log files'
    )
    parser.add_argument('--dump-dir',
        type=str,
        nargs='?',
        help='directory for writing dump files'
    )
    parser.add_argument('--continue-dir',
        type=str,
        nargs='?',
        help='provide directory of itteration pull for continue calculation'
    )

    # Tests args

    parser.add_argument('--test-disable-iter-graph',
        action='store_true',
        help='disable matplotlib lib graphs on iterations for tests'
    )
    parser.add_argument('--test-disable-stat-graph',
        action='store_true',
        help='disable matplotlib lib statistics graphs for tests'
    )
    parser.add_argument('--test-restore-result',
        action='store_true',
        help='enable use last result for tests'
    )
    parser.add_argument('--test-disable-dump',
        action='store_true',
        help='disable dumping tests result'
    )
    parser.add_argument('--test-dump-dir',
        type=str,
        nargs='?',
        default='dumps/tests',
        help='directory for writing dump files for tests'
    )

    return parser

def main(manager: SyncManager):
    parser = _create_parser()
    args = parser.parse_args()

    if cfg.Config.GlobalProps.RandSeed > 0:
        # Set random state
        random.seed(cfg.Config.GlobalProps.RandSeed)
        np.random.seed(cfg.Config.GlobalProps.RandSeed)

    cfg.init(args)
    log.init(args)
    dump.init(args)

    mode = utils.get_necessary_arg(args, 'm', 'mode')

    if mode == _MODE_TESTS:
        run_tests(args=args)
    # elif mode == _MODE_GRID:
    #     run_grid(args=args)
    elif mode == _MODE_HYPER_PARAMS:
        run_scheme(DynoEvoEsnHyperParam, cfg.Config.Schemes.HyperParam, args=args, async_manager=manager)
    elif mode == _MODE_HYPER_PARAMS_MULTI_POP:
        run_scheme(DynoEvoEsnHyperParamMultiPop, cfg.Config.Schemes.HyperParamMultiPop, args=args, async_manager=manager)
    elif mode == _MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT:
        run_scheme(DynoEvoEsnHyperParamMultiPopMultiCrit, cfg.Config.Schemes.HyperParamMultiPopMultiCrit, args=args, async_manager=manager)
    elif mode == _MODE_HYPER_PARAMS_WITH_RESERVOIR_MULTI_POP:
        run_scheme(DynoEvoEsnHyperParamWithReservoirMultiPop, cfg.Config.Schemes.HyperParamWithReservoirMultiPop, args=args, async_manager=manager)
    elif mode == _MODE_HYPER_PARAMS_RANDOM_SEARCH:
        run_scheme(RandomSearchHyperParam, cfg.Config.Schemes.HyperParamRandomSearch, args=args, async_manager=manager)
    elif mode == _MODE_HYPER_PARAMS_GRID_SEARCH:
        run_scheme(GridSearchHyperParam, cfg.Config.Schemes.HyperParamGridSearch, args=args, async_manager=manager)
    else:
        raise('unknown running mode')

if __name__ == '__main__':
    # with Pool(processes=8) as pool:
    # with SyncManager() as manager:
    #     main(manager.Pool(processes=2))
    # with ThreadPool(processes=30) as pool:
    #     main(pool)
    with SyncManager() as async_manager:
        main(async_manager)

from qiskit.algorithms.optimizers import UMDA
