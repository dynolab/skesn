import sys
import logging
import argparse

import matplotlib.pyplot as plt
import numpy as np

from pshipilov_dev.src.evo.scheme_2 import Scheme_2
from pshipilov_dev.src.config import Config
from pshipilov_dev.src.grid import esn_lorenz_grid_search
from pshipilov_dev.src.lorenz import get_lorenz_data, data_to_train, train_to_data
from pshipilov_dev.src.utils import valid_multi_f

from skesn.esn import EsnForecaster

from deap import base, algorithms
# from deap import creator
# from deap import tools

# from scoop import futures

import pshipilov_dev.src.dump as dump
import pshipilov_dev.src.log as log

import random

import dill
import joblib
from dill import Pickler

joblib.parallel.pickle = dill
joblib.pool.dumps = dill.dumps
joblib.pool.Pickler = Pickler

from joblib.pool import CustomizablePicklingQueue

from pshipilov_dev.src.async_utils.customizable_pickler import make_methods, CustomizablePickler

CustomizablePicklingQueue._make_methods = make_methods
joblib.pool.CustomizablePickler = CustomizablePickler

# creator.create("FitnessESN", base.Fitness, weights=Config.Evo.Scheme_1.Weights)
# creator.create("Individual", list, fitness=creator.FitnessESN)

from joblib import Parallel, delayed

import pickle

def CustomMap(f, *iters):
    return Parallel(n_jobs=-1)(delayed(f)(*args) for args in zip(*iters))

random.seed(Config.Esn.RandomState)
log.init()

def run_scheme1():
    scheme = Scheme_2(base.Toolbox())
    scheme.run()
    scheme.show_plot()

    dump.do(scheme)

def run_scheme2():
    scheme = Scheme_2(base.Toolbox())
    scheme.run()
    scheme.show_plot()

    dump.do(scheme)

def run_grid():
    best_params = esn_lorenz_grid_search()
    dump.do(grid_srch_best_params=best_params)

def run_test_multi():
    params = {
        'n_inputs': Config.Esn.NInputs,
        'n_reservoir': Config.Esn.NReservoir,
        'spectral_radius': Config.Esn.SpectralRadius, 
        'sparsity': Config.Esn.Sparsity,
        'noise': Config.Esn.Noise,
        'lambda_r': Config.Esn.LambdaR,
        'random_state': Config.Esn.RandomState,
    }
    logging.info(f'start test ESN...  params = {params}')
    test_data = get_lorenz_data(
        Config.Models.Lorenz.Ro,
        Config.Test.MultiStep.DataN,
        Config.Models.Lorenz.Dt,
        Config.Models.Lorenz.RandSeed,
    )
    train_data = test_data[..., :Config.Test.MultiStep.DataN//2:5]
    valid_data = test_data[..., Config.Test.MultiStep.DataN//2:]

    model = EsnForecaster(**params)
    model.fit(data_to_train(train_data).T)
    err = valid_multi_f(Config.Test.MultiStep.StepN, model, valid_data)
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

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
        type=str,
        required=True,
        choices=['test_multi', 'grid', 'evo_scheme_1', 'evo_scheme_2'],
        help='run mode'
    )
    parser.add_argument('-v', '--verbose',
        action='store_false',
        help='print all logs'
    )
    parser.add_argument('--log-dir',
        type=str,
        nargs='?',
        help='directory for writing log files'
    )
    parser.add_argument('--dump-dir',
        type=str,
        nargs='?',
        help='directory for writing dump files'
    )

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    

    if args.mode == 'test_multi':
        run_test_multi()
    elif args.mode == 'grid':
        run_grid()
    elif args.mode == 'evo_scheme_1':
        run_scheme2()
    elif args.mode == 'evo_scheme_2':
        run_scheme2()
    else:
        raise('unknown running mode')

if __name__ == '__main__':
    # scheme.get_toolbox().register("map", CustomMap)
    main()
