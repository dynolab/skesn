from .utils import map_select_f, map_mate_f, map_mutate_f
from .abstract import Scheme
from ..log import get_logger
from ..config import EvoOperatorArg, EvoSchemeConfigField

import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from deap import base, algorithms
from deap import creator
from deap import tools

def _bind_evo_operator(
    tool_box: base.Toolbox,
    name: str,
    func,
    args: List[EvoOperatorArg],
    **kvargs,
    ):

    if len(args) > 0:
        for arg in args:
            kvargs[arg.Key] = arg.Val

    tool_box.register(name, func, **kvargs)

class EvoScheme(Scheme):
    def __init__(self,
        name: str,
        cfg: EvoSchemeConfigField,
        ind_creator_f,
        evaluate_f,
        tool_box: base.Toolbox=None,
        ) -> None:

        # Logger setup
        self._logger = get_logger(name=f'EvoScheme<{name}>')

        # Config setup
        self._name = name
        self._cfg = cfg

        # Math setup
        self._rand = np.random.RandomState(seed=self._cfg.RandSeed)

        # DEAP setup
        creator.create("Fitness", base.Fitness, weights=self._cfg.FitnessWeights)
        creator.create("Individual", list, fitness=creator.Fitness)

        self._tool_box = base.Toolbox() if tool_box is None else tool_box
        if not hasattr(self._tool_box, 'new_population'):
            self._tool_box.register('new_population', tools.initRepeat, list, ind_creator_f, n=self._cfg.PopulationSize)

        if not hasattr(self._tool_box, 'evaluate'):
            self._tool_box.register('evaluate', evaluate_f)

        if not hasattr(self._tool_box, 'select'):
            _bind_evo_operator(
                self._tool_box,
                'select',
                map_select_f(self._cfg.Select.Method),
                self._cfg.Select.Args,
            )
        if not hasattr(self._tool_box, 'mate'):
            _bind_evo_operator(
                self._tool_box,
                'mate',
                map_mate_f(self._cfg.Mate.Method),
                self._cfg.Mate.Args,
            )

        if not hasattr(self._tool_box, 'mutate'):
            if self._cfg.Mutate.Indpb > 0:
                _bind_evo_operator(
                    self._tool_box,
                    'mutate',
                    map_mutate_f(self._cfg.Mutate.Method),
                    self._cfg.Mutate.Args,
                    indpb=self._cfg.Mutate.Indpb,
                    )
            else:
                _bind_evo_operator(
                    self._tool_box,
                    'mutate',
                    map_mutate_f(self._cfg.Mutate.Method),
                    self._cfg.Mutate.Args,
                    indpb=1/self._cfg.HromoLen
                    )

        self._stats = tools.Statistics(lambda ind: ind.fitness.values)
        self._stats.register('min', np.min)
        self._stats.register('agv', np.mean)

        # Result
        self._logbook: tools.Logbook = None
        self._lastPopulation: list = None

        # Matplotlib setup
        # f._fig, self._ax = plt.subplots()

    def run(self) -> None:
        self._logger.info('Evo scheme "%s" is running...', self._name)

        self._lastPopulation, self._logbook = algorithms.eaSimple(
            population=self._tool_box.new_population(),
            toolbox=self._tool_box,
            cxpb=self._cfg.Mate.Probability,
            mutpb=self._cfg.Mutate.Probability,
            ngen=self._cfg.MaxGenNum,
            stats=self._stats,
            verbose=self._cfg.Verbose,
            # callback=lambda population, gen: print(1),
        )

        self._logger.info('Evo scheme "%s"  has bean done', self._name)

        return self._lastPopulation

    def show_plot(self) -> None:
        self._fig, self._ax = plt.subplots()
        minFitnessValues, avgFitnessValues = self._logbook.select('min', 'avg')
        self._ax.plot(minFitnessValues, color='green', label='min')
        self._ax.plot(avgFitnessValues, color='blue', label='avg')
        self._ax.set_xlabel('generation')
        self._ax.set_ylabel('error')
        self._ax.legend()

    def save(self, dirname: str) -> None:
        if not hasattr(self, '_fig'):
            return

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
