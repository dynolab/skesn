import src.evo.test.evo_scheme_test as evo_scheme_test
import src.evo.test.evo_scheme_multi_pop_test as evo_scheme_multi_pop_test
import src.grid as grid
import src.lorenz as lorenz
import src.utils as utils
import src.log as log
import src.dump as dump
import src.config as cfg

import skesn.esn as esn

import matplotlib.pyplot as plt
import numpy as np

from deap import base, algorithms

from dill import Pickler

import argparse
import random

import dill
import joblib

joblib.parallel.pickle = dill
joblib.pool.dumps = dill.dumps
joblib.pool.Pickler = Pickler

from joblib.pool import CustomizablePicklingQueue

from src.async_utils.customizable_pickler import make_methods, CustomizablePickler

CustomizablePicklingQueue._make_methods = make_methods
joblib.pool.CustomizablePickler = CustomizablePickler

from joblib import Parallel, delayed

# import pickle

# Modes for running
_MODE_TESTS = 'tests'
_MODE_TEST_MULTI = 'test_multi'
_MODE_GRID = 'grid'
_MODE_EVO_SCHEME_1 = 'evo_scheme_1'
_MODE_EVO_SCHEME_2 = 'evo_scheme_2'

def run_tests(**kvargs):
    # TODO :
    evo_scheme_test.run_tests(**kvargs)
    evo_scheme_multi_pop_test.run_tests(**kvargs)

def run_scheme1(**kvargs):
    args = utils.get_args_via_kvargs(**kvargs)
    logger = log.get_logger_via_kvargs(**kvargs)

    # scheme = Scheme_1(base.Toolbox(), args)
    # scheme.run()
    # scheme.show_plot()

    # dump.do(logger=logger, evo_scheme=scheme)

def run_scheme2(**kvargs):
    args = utils.get_args_via_kvargs(**kvargs)
    logger = log.get_logger_via_kvargs(**kvargs)

    scheme = Scheme_2(base.Toolbox(), args)
    scheme.run()
    scheme.show_plot()

    dump.do(logger=logger, evo_scheme=scheme)

def run_grid(**kvargs):
    args = utils.get_args_via_kvargs(**kvargs)
    logger = log.get_logger_via_kvargs(**kvargs)

    best_params = grid.esn_lorenz_grid_search(args)

    dump.do(logger=logger, grid_srch_best_params=best_params)

def run_test_multi(**kvargs):
    # args = get_args_via_kvargs(**kvargs)
    logger = log.get_logger_via_kvargs(**kvargs)

    params = {
        'n_inputs': cfg.Config.Esn.NInputs,
        'n_reservoir': cfg.Config.Esn.NReservoir,
        'spectral_radius': cfg.Config.Esn.SpectralRadius,
        'sparsity': cfg.Config.Esn.Sparsity,
        'noise': cfg.Config.Esn.Noise,
        'lambda_r': cfg.Config.Esn.LambdaR,
        'random_state': cfg.Config.Esn.RandomState,
    }
    logger.info(f'start test ESN...  params = {params}')
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
    logger.info(f'dumping test data...')
    dump.do_np_arr(test_data=test_data)
    logger.info(f'dumping hyperparameters...')
    dump.do_var(hyperparameters=params,score={'score': float(err)})
    logger.info(f'test has been done: err = {err} (multi step testing: step_n = 5)')

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
        choices=[_MODE_TESTS, _MODE_TEST_MULTI, _MODE_GRID, _MODE_EVO_SCHEME_1, _MODE_EVO_SCHEME_2],
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

def main():
    parser = _create_parser()
    args = parser.parse_args()

    cfg.init(args)
    log.init(args)
    dump.init(args)

    mode = utils.get_necessary_arg(args, 'm', 'mode')

    if mode == _MODE_TESTS:
        run_tests(args=args)
    elif mode == _MODE_TEST_MULTI:
        run_test_multi(args=args)
    elif mode == _MODE_GRID:
        run_grid(args=args)
    elif mode == _MODE_EVO_SCHEME_1:
        run_scheme2(args=args)
    elif mode == _MODE_EVO_SCHEME_2:
        run_scheme2(args=args)
    else:
        raise('unknown running mode')

def CustomMap(f, *iters):
    return Parallel(n_jobs=-1)(delayed(f)(*args) for args in zip(*iters))

if __name__ == '__main__':
    main()
