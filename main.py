from src.evo.tasks.duno_evo_esn_hyper_param_multi_pop_multi_crit import DynoEvoEsnHyperParamMultiPopMultiCrit
from src.evo.tasks.duno_evo_esn_hyper_param_multi_pop import DynoEvoEsnHyperParamMultiPop
from src.evo.tasks.dyno_evo_esn_hyper_param import DynoEvoEsnHyperParam
from src.evo.abstract import Scheme

import src.evo.test.evo_scheme_test as evo_scheme_test
import src.evo.test.evo_scheme_multi_pop_test as evo_scheme_multi_pop_test
import src.lorenz as lorenz
import src.utils as utils
import src.log as log
import src.dump as dump
import src.config as cfg

import skesn.esn as esn
from multiprocess.managers import SyncManager

import matplotlib.pyplot as plt
import numpy as np
import logging

import argparse
import random


# Modes for running
_MODE_TESTS = 'tests'
# _MODE_GRID = 'grid'
_MODE_HYPER_PARAMS = 'hyper_param'
_MODE_HYPER_PARAMS_MULTI_POP = 'hyper_param_multi_pop'
_MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT = 'hyper_param_multi_pop_multi_crit'

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


def run_scheme_hyper_param(**kvargs):
    args = utils.get_args_via_kvargs(kvargs)
    pool = utils.get_via_kvargs(kvargs, utils._KVARGS_POOL)

    scheme = DynoEvoEsnHyperParam(
        cfg.Config.Schemes.HyperParam,
        pool,
    )

    check_restore(args, scheme)
    scheme.run()

    dump.do(evo_scheme=scheme)

    scheme.close()

def run_scheme_hyper_param_multi_pop(**kvargs):
    args = utils.get_args_via_kvargs(kvargs)
    pool = utils.get_via_kvargs(kvargs, utils._KVARGS_POOL)

    scheme = DynoEvoEsnHyperParamMultiPop(
        cfg.Config.Schemes.HyperParamMultiPop,
        pool,
    )

    check_restore(args, scheme)
    scheme.run()

    dump.do(evo_scheme=scheme)

    scheme.close()

def run_scheme_hyper_param_multi_pop_multi_crit(**kvargs):
    args = utils.get_args_via_kvargs(kvargs)
    pool = utils.get_via_kvargs(kvargs, utils._KVARGS_POOL)

    scheme = DynoEvoEsnHyperParamMultiPopMultiCrit(
        cfg.Config.Schemes.HyperParamMultiPopMultiCrit,
        pool,
    )

    check_restore(args, scheme)
    scheme.run()

    dump.do(evo_scheme=scheme)

    scheme.close()

# def run_grid(**kvargs):
#     args = utils.get_args_via_kvargs(**kvargs)
#     best_params = grid.esn_lorenz_grid_search(args)
#     dump.do(grid_srch_best_params=best_params)

def run_test_multi(**kvargs):
    # args = get_args_via_kvargs(**kvargs)

    params = {
        'n_inputs': cfg.Config.Esn.NInputs,
        'n_reservoir': cfg.Config.Esn.NReservoir,
        'spectral_radius': cfg.Config.Esn.SpectralRadius,
        'sparsity': cfg.Config.Esn.Sparsity,
        'noise': cfg.Config.Esn.Noise,
        'lambda_r': cfg.Config.Esn.LambdaR,
        'random_state': cfg.Config.Esn.RandomState,
    }
    logging.info(f'start test ESN...  params = {params}')
    test_data = lorenz.get_lorenz_data(
        cfg.Config.Models.Lorenz.Ro,
        cfg.Config.Test.MultiStep.DataN,
        cfg.Config.Models.Lorenz.Dt,
        cfg.Config.Models.Lorenz.RandSeed,
    )
    train_data = test_data[..., :cfg.Config.Test.MultiStep.DataN//2:5]
    valid_data = test_data[..., cfg.Config.Test.MultiStep.DataN//2:]

    model = esn.EsnForecaster(**params)
    model.fit(lorenz.data_to_train(train_data).T)
    err = utils.valid_multi_f(cfg.Config.Test.MultiStep.StepN, model, valid_data)
    logging.info(f'dumping test data...')
    dump.do_np_arr(test_data=test_data)
    logging.info(f'dumping hyperparameters...')
    dump.do_var(hyperparameters=params,score={'score': float(err)})
    logging.info(f'test has been done: err = {err} (multi step testing: step_n = 5)')

    fig, axes = plt.subplots(3,figsize=(10,3))

    t_train = np.arange(0,2500)
    t_valid = np.arange(2500,5000)

    axes[0].plot(t_train, train_data[0], label='training data')
    axes[1].plot(t_train, train_data[1])
    axes[2].plot(t_train, train_data[2])

    axes[0].plot(t_valid, valid_data[0], label='valid data')
    axes[1].plot(t_valid, valid_data[1])
    axes[2].plot(t_valid, valid_data[2])

    # axes[0].plot(t_valid, predict[0], label='predicted data')
    # axes[1].plot(t_valid, predict[1])
    # axes[2].plot(t_valid, predict[2])

    fig.legend()

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Choises

    parser.add_argument('-m',
        type=str,
        required=True,
        choices=[_MODE_TESTS, _MODE_HYPER_PARAMS, _MODE_HYPER_PARAMS_MULTI_POP, _MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT],
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

def main(pool):
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
        run_scheme_hyper_param(args=args, pool=pool)
    elif mode == _MODE_HYPER_PARAMS_MULTI_POP:
        run_scheme_hyper_param_multi_pop(args=args, pool=pool)
    elif mode == _MODE_HYPER_PARAMS_MULTI_POP_MULTI_CRIT:
        run_scheme_hyper_param_multi_pop_multi_crit(args=args, pool=pool)
    else:
        raise('unknown running mode')


if __name__ == '__main__':
    # with Pool(processes=8) as pool:
    with SyncManager() as manager:
        main(manager.Pool())
