from typing import Tuple

from .utils import map_crossing_f, map_select_f, wrap_scoring_f
from .abstract import Scheme
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

class HyperParamInfo:
    def __init__(self, key: str, min, max, is_int: bool=False) -> None:
        self.idx = None
        self.key = key
        self.min = min
        self.max = max
        self.is_int = is_int

class Scheme_2(Scheme):
    def __init__(self, toolbox: base.Toolbox) -> None:
        # Config setup
        self._cfg = Config.Evo.Scheme_2

        # Hyper param info setup
        self._setup_hyper_params()

        # Math setup
        self._rand = np.random.RandomState(seed=self._cfg.Common.RandSeed)

        # DEAP setup
        creator.create("FitnessESN", base.Fitness, weights=self._cfg.Weights)
        creator.create("Individual", list, fitness=creator.FitnessESN)

        self._tool_box = toolbox
        self._tool_box.register('new_population', tools.initRepeat, list, self._new_individual, n=self._cfg.Common.PopulationSize)

        self._tool_box.register('evaluate', wrap_scoring_f(self._get_esn, self._cfg.Common.Scoring, valid_multi_n=self._cfg.Common.ValidMultiN))
        self._tool_box.register('select', map_select_f(self._cfg.Common.Select), tournsize=5)
        self._tool_box.register('mate', map_crossing_f(self._cfg.Common.Crossing))
        if self._cfg.Common.MutationIndpb > 0:
            self._tool_box.register('mutate', self._mutInd, indpb=self._cfg.Common.MutationIndpb)
        else:    
            self._tool_box.register('mutate', self._mutInd, indpb=1./len(self._hyper_params))

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register('min', np.min)
        self._stats.register('agv', np.mean)

        # Result
        self._logbook: tools.Logbook
        # self._lastPopulation: list

        # Matplotlib setup
        self._fig, self._ax = plt.subplots()

    def run(self) -> None:
        logging.info('Scheme_2 is running...')

        self._lastPopulation, self._logbook = algorithms.eaSimple(
            population=self._tool_box.new_population(),
            toolbox=self._tool_box,
            cxpb=self._cfg.Common.CrossoverP,
            mutpb=self._cfg.Common.MutationP,
            ngen=self._cfg.Common.MaxGenNum,
            stats=self._stats,
            verbose=self._cfg.Common.Verbose,
        )

        logging.info('Scheme_2 has bean done')

    def show_plot(self) -> None:
        minFitnessValues, avgFitnessValues = self._logbook.select('min', 'avg')
        self._ax.plot(minFitnessValues, color='green', label='min')
        self._ax.plot(avgFitnessValues, color='blue', label='avg')
        self._ax.set_xlabel('Generation')
        self._ax.set_ylabel('Error')
        self._ax.legend()

    def save(self, dirname: str) -> None:
        # Dump graph
        self._fig.savefig(f'{dirname}/graph.png', dpi=self._fig.dpi)

        # Dump last population
        with open(f'{dirname}/last_popultation.yaml', 'w') as f:
            dump_weight = {}
            for i, ind in enumerate(self._lastPopulation):
                dump_weight[f'ind_{str(i)}'] = {param.key: ind[param.idx] for param in self._hyper_params}
            yaml.safe_dump(dump_weight, f)

    # Access methods

    def get_toolbox(self) -> base.Toolbox:
        return self._tool_box

    # Internal methods

    def _setup_hyper_params(self):
        # self._n_reservoir = HyperParamInfo('n_reservoir', self._cfg.Limits.NReservoir.Min, self._cfg.Limits.NReservoir.Max, self._cfg.Limits.NReservoir.IsInt)
        self._spectral_radius = HyperParamInfo('spectral_radius', self._cfg.Limits.SpectralRadius.Min, self._cfg.Limits.SpectralRadius.Max, self._cfg.Limits.SpectralRadius.IsInt)
        self._sparsity = HyperParamInfo('sparsity', self._cfg.Limits.Sparsity.Min, self._cfg.Limits.Sparsity.Max, self._cfg.Limits.Sparsity.IsInt)
        self._noise = HyperParamInfo('noise', self._cfg.Limits.Noise.Min, self._cfg.Limits.Noise.Max, self._cfg.Limits.Noise.IsInt)
        # self._lambda_r = HyperParamInfo('lambda_r', self._cfg.Limits.LambdaR.Min, self._cfg.Limits.LambdaR.Max, self._cfg.Limits.LambdaR.IsInt)
        self._hyper_params = (
            # self._n_reservoir, 
            self._spectral_radius, 
            self._sparsity,
            self._noise,
            # self._lambda_r,
        )
        for i, param in enumerate(self._hyper_params):
            param.idx = i

    # DEAP

    def _mutInd(self, ind: list, indpb: float):
        for i in range(len(ind)):
            if indpb < self._rand.random():
                continue
            ind[i] = self._hyper_params[i].min + self._rand.random() * (self._hyper_params[i].max - self._hyper_params[i].min)
            if self._hyper_params[i].is_int:
                ind[i] = int(ind[i])
        return ind,

    def _new_individual(self):
        ret = [0] * len(self._hyper_params)
        for param in self._hyper_params:
            ret[param.idx] = param.min + self._rand.random() * (param.max - param.min)
            if param.is_int:
                ret[param.idx] = int(ret[param.idx])
        return creator.Individual(ret)

    def _get_esn(self, ind: list) -> EsnForecaster:
        return EsnForecaster(
            n_inputs=Config.Esn.NInputs, 
            # n_reservoir=ind[self._n_reservoir.idx],
            n_reservoir=Config.Esn.NReservoir,
            spectral_radius=ind[self._spectral_radius.idx], 
            sparsity=ind[self._sparsity.idx],
            noise=ind[self._noise.idx],
            # lambda_r=ind[self._lambda_r.idx],
            lambda_r=Config.Esn.LambdaR,
            random_state=Config.Esn.RandomState,
        )

    # Othres
