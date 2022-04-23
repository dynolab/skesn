from typing import Tuple

from .abstract import Scheme
from .utils import wrap_scoring_f, map_crossing_f, map_select_f, map_mutate_f

from ..lorenz import get_lorenz_data, data_to_train, train_to_data
from ..config import Config

from skesn.esn import EsnForecaster

from deap import base, algorithms
from deap import creator
from deap import tools

import dill
import yaml
import random
import logging
import scipy.fftpack 
import matplotlib.pyplot as plt
import numpy as np

class Scheme_1(Scheme):
    def __init__(self, toolbox: base.Toolbox, dtype=np.float32) -> None:
        # Config setup
        self._cfg = Config.Evo.Scheme_1

        # Math setup
        self._dtype: np.dtype = dtype
        self._rand = np.random.RandomState(seed=self._cfg.Common.RandSeed)

        # Dataset setup
        self._data = get_lorenz_data(
            Config.Models.Lorenz.Ro,
            Config.Models.Lorenz.N,
            Config.Models.Lorenz.Dt,
            Config.Models.Lorenz.RandSeed,
        )
        self._train_data = self._data[..., :Config.Models.Lorenz.N//2]
        self._valid_data = self._data[..., Config.Models.Lorenz.N//2:]
        self._fit_data = data_to_train(self._train_data).T

        # DEAP setup
        creator.create("FitnessESN", base.Fitness, weights=self._cfg.Weights)
        creator.create("Individual", list, fitness=creator.FitnessESN)

        self._tool_box = toolbox
        self._tool_box.register('new_population', tools.initRepeat, list, self._new_individual, n=self._cfg.Common.PopulationSize)

        self._tool_box.register('evaluate', wrap_scoring_f(self._get_esn, self._cfg.Common.Scoring, valid_multi_n=self._cfg.Common.ValidMultiN))
        self._tool_box.register('select', map_select_f(self._cfg.Common.Select), tournsize=3)
        self._tool_box.register('mate', map_crossing_f(self._cfg.Common.Crossing))
        self._tool_box.register('mutate', map_crossing_f(self._cfg.Common.Mutation), up=10.,low=-10.,eta=20, indpb=1./Config.Esn.NReservoir)

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register('min', np.min)
        self._stats.register('agv', np.mean)

        self._hof = tools.HallOfFame(1, similar=np.allclose)


        # Result 
        self._logbook: tools.Logbook
        # self._lastPopulation: list

        # Matplotlib setup
        self._fig, self._ax = plt.subplots()

        # Postions setup
        self._positions: list = []
        self._init_positions()

    def run(self) -> None:
        logging.info('Scheme_1 is running...')

        self._lastPopulation, self._logbook = algorithms.eaSimple(
            population=self._tool_box.new_population(),
            toolbox=self._tool_box,
            cxpb=self._cfg.Common.CrossoverP,
            mutpb=self._cfg.Common.MutationP,
            ngen=self._cfg.Common.MaxGenNum,
            stats=self._stats,
            verbose=self._cfg.Common.Verbose,
            halloffame=self._hof,
        )

        logging.info('Scheme_1 has bean done')

        return lambda: self._weights_generator(self._lastPopulation[-1])

    def show_plot(self) -> None:
        minFitnessValues, avgFitnessValues = self._logbook.select('min', 'avg')
        self._ax.plot(minFitnessValues, color='green', label='min')
        self._ax.plot(avgFitnessValues, color='blue', label='avg')
        self._ax.set_xlabel('Generation')
        self._ax.set_ylabel('Error')
        self._ax.legend()
        # plt.show()

    def save(self, dirname: str) -> None:
        # Dump graph
        self._fig.savefig(f'{dirname}/graph.png', dpi=self._fig.dpi)

        # Dump weight
        with open(f'{dirname}/last_popultation.yaml', 'w') as f:
            dump_weight = {}
            for i, ind in enumerate(self._lastPopulation):
                dump_weight[f'ind_{str(i)}'] = list([float(x) for x in ind])
            yaml.safe_dump(dump_weight, f)

        # Dump positions
        with open(f'{dirname}/positions.yaml', 'w') as f:
            dump_positions = {}
            dump_positions['positions'] = list([int(x) for x in self._positions])
            yaml.safe_dump(dump_positions, f)

    # Access methods

    def get_toolbox(self) -> base.Toolbox:
        return self._tool_box

    # Internal methods

    # DEAP

    def _new_individual(self):
        # ret = np.zeros(self._cfg.M, dtype=self._dtype)
        # for i in range(self._cfg.C):
        #     ret[i] = self._rand.random()
        # return creator.Individual(scipy.fftpack.idct(ret))
        ret = np.zeros(len(self._positions), dtype=self._dtype)
        c = int(len(ret)*3/4)
        for i in range(c):
            ret[i] = float(self._rand.random() * 10)
        return creator.Individual(scipy.fftpack.idct(ret))
        # return creator.Individual(ret)

    # Othres

    def _get_esn(self, ind):
        return EsnForecaster(
            n_inputs=Config.Esn.NInputs, 
            n_reservoir=Config.Esn.NReservoir,
            spectral_radius=Config.Esn.SpectralRadius, 
            sparsity=Config.Esn.Sparsity,
            noise=Config.Esn.Noise,
            lambda_r=Config.Esn.LambdaR,
            random_state=Config.Esn.RandomState,
            w_generator=lambda: self._weights_generator(ind),
        )

    def _convert_num_to_index(self, num: int) -> Tuple:
        return num // Config.Esn.NReservoir, num % Config.Esn.NReservoir

    def _init_positions(self):
        self._positions = []
        mtx = np.abs(self._rand.rand(Config.Esn.NReservoir, Config.Esn.NReservoir) - 0.5) < 0.01
        for i, row in enumerate(mtx):
            for j, x in enumerate(row):
                if x:
                    self._positions.append(i * Config.Esn.NReservoir + j)
        logging.info('inited %d postitions', len(self._positions))

    def _weights_generator(self, ind: list):
        # TODO : investigate copy from standart weights_generator
        w_in = self._rand.rand(Config.Esn.NReservoir, Config.Esn.NInputs + 1) * 2 - 1
        w_in[self._rand.rand(*w_in.shape) < Config.Esn.Sparsity*2] = 0

        # initialize recurrent weights:
        w = np.zeros((Config.Esn.NReservoir, Config.Esn.NReservoir), dtype=self._dtype)
        for k, p in enumerate(self._positions):
            i, j = self._convert_num_to_index(p)
            if ind[k] is None or np.isnan(ind[k]):
                w[i][j] = self._dtype(0.)
            else:
                w[i][j] = self._dtype(ind[k])

        radius = np.max(np.abs(np.linalg.eigvals(w)))
        # rescale them to reach the requested spectral radius:
        w = w * (Config.Esn.SpectralRadius / radius)

        return w_in, w

# TODO :
# Temporary fucns

def _m_esn_fitness_valid_data_one_step(self: Scheme_1, ind: list) -> list:
    model = EsnForecaster(
        n_inputs=Config.Esn.NInputs, 
        n_reservoir=Config.Esn.NReservoir,
        spectral_radius=Config.Esn.SpectralRadius, 
        sparsity=Config.Esn.Sparsity,
        noise=Config.Esn.Noise,
        lambda_r=Config.Esn.LambdaR,
        random_state=Config.Esn.RandomState,
        w_generator=lambda: self._weights_generator(ind),
    )
    model.fit(self._fit_data)
    err = 0.
    max_i = len(self._valid_data[0])
    for i in range(max_i):
        predict = train_to_data(model.predict(1, True, Config.Esn.Inspect).T)
        err += (([[v] for v in self._valid_data[:,i]] - predict)**2).mean()
