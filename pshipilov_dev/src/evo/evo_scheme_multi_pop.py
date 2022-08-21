from ctypes import util
import pshipilov_dev.src.evo.utils as utils

from .abstract import Scheme
from ..log import get_logger
from ..config import EvoSchemeMultiPopConfigField, EvoPopulationConfigField
from pshipilov_dev.src.utils import kv_config_arr_to_kvargs

import os
import yaml
import logging
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from types import FunctionType
from typing import Any, List
from deap import base, algorithms
from deap import creator
from deap import tools

class EvoSchemeMultiPop(Scheme):
    def __init__(self,
        name: str,
        cfg: EvoSchemeMultiPopConfigField,
        evaluate_f: FunctionType,
        ind_creator_f: FunctionType=None,
        ) -> None:

        # Logger setup
        self._logger: logging.Logger = get_logger(name=f'EvoSchemeMultiPop<{name}>')

        # Config setup
        self._name: str                          = name
        self._cfg:  EvoSchemeMultiPopConfigField = cfg

        # Math setup
        self._rand: np.random.RandomState = np.random.RandomState(seed=self._cfg.RandSeed)

        # DEAP setup
        self._evaluate_f = evaluate_f
        self._ind_creator_f = ind_creator_f
        if self._ind_creator_f is None:
            self._ind_creator_f = lambda: utils.create_ind_by_list(utils.ind_creator_f(
                    creator.Individual,
                    self._cfg.HromoLen,
                    pop_cfg.Limits,
                    self._rand,
                ),
                self._evaluate_f,
            )

        creator.create("Fitness", base.Fitness, weights=self._cfg.FitnessWeights)
        creator.create("Individual", list, fitness=creator.Fitness)

        self._use_restored_result: bool = False
        self._result: List[algorithms.Popolation] = []

        for pop_cfg in cfg.Populations:
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                self._result.append(_create_deap_population(
                    self._evaluate_f,
                    self._ind_creator_f,
                    self._cfg.HromoLen,
                    pop_cfg,
                ))

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._cfg.Metrics) > 0:
            self._stats = tools.Statistics(lambda ind: ind.fitness.values)
            for metric_cfg in self._cfg.Metrics:
                self._stats.register(metric_cfg.Name, utils.get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('EvoSchemeMultiPop<%s> is running...', self._name)

        self._result, self._logbook = algorithms.eaSimpleMultiPop(
            populations=self._result,
            ngen=self._cfg.MaxGenNum,
            stats=self._stats,
            verbose=self._cfg.Verbose,
            **kvargs,
        )

        self._logger.info('EvoSchemeMultiPop<%s>  has bean done', self._name)

    def restore_result(self, result: Any) -> None:
        self._use_restored_result = True
        self._result = []

        k = 0
        for pop_cfg in self._cfg.Populations:
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                pop = _create_deap_population(
                    self._evaluate_f,
                    self._ind_creator_f,
                    self._cfg.HromoLen,
                    pop_cfg,
                )
                pop.Inds = [ind if isinstance(ind, creator.Individual) else utils.create_ind_by_list(ind, self._evaluate_f) for ind in result[k]]
                k += 1
                self._result.append(pop)

    def save(self, dirname: str, **kvargs) -> str:
        run_pool_dir = utils.get_or_create_last_run_pool_dir(dirname, self._name)
        iter_dir: str = utils.get_or_create_iter_dir(run_pool_dir) if self._use_restored_result\
                else utils.create_iter_dir(run_pool_dir)

        len_metrics = len(self._cfg.Metrics)
        if not kvargs.get('disable_stat', False) and len_metrics > 0:
            # Dump stat graph
            fig, ax = plt.subplots()
            fig.suptitle(f'{self._name}\nevo stats')

            metrics = self._logbook.select(*[metric.Name for metric in self._cfg.Metrics])
            for i in range(len_metrics):
                ax.plot(metrics[i], label=self._cfg.Metrics[i].Name, **kv_config_arr_to_kvargs(self._cfg.Metrics[i].PltArgs))
            ax.set_xlabel('generation')
            ax.set_ylabel('fitness')
            ax.legend()

            fig.savefig(f'{iter_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{iter_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(self._cfg.yaml(), f)

        if not kvargs.get('disable_dump_result', False):
            # Dump last populations
            with open(f'{iter_dir}/result.yaml', 'w') as f:
                utils.dump_inds_multi_pop_arr(
                    self._cfg.HromoLen,
                    f,
                    [popultaion.Inds for popultaion in self._result],
                    utils.get_populations_limits(self._cfg),
                )

        if not kvargs.get('disable_dump_hall_of_fame', False):
            # Dump last population hall of fames
            with open(f'{iter_dir}/hall_off_fames.yaml', 'w') as f:
                utils.dump_inds_multi_pop_arr(
                    self._cfg.HromoLen,
                    f,
                    [popultaion.HallOfFame.items if popultaion.HallOfFameSize > 0 else [] for popultaion in self._result],
                    utils.get_populations_limits(self._cfg),
                )

    # Access methods

    def get_name(self) -> str:
        return self._name

    def get_logbook(self) -> tools.Logbook:
        return self._logbook

    def get_populations(self) -> List[List[List]]:
        return self._result


def _create_deap_population(
    evaluate_f: FunctionType,
    ind_creator_f: FunctionType,
    hromo_len: int,
    cfg: EvoPopulationConfigField,
) -> algorithms.Popolation:
    toolbox = base.Toolbox()

    toolbox.register('evaluate', evaluate_f)

    utils.bind_evo_operator(
        toolbox,
        'select',
        utils.map_select_f(cfg.Select.Method),
        cfg.Select.Args,
    )
    utils.bind_evo_operator(
        toolbox,
        'mate',
        utils.map_mate_f(cfg.Mate.Method),
        cfg.Mate.Args,
    )

    if cfg.Mutate.Indpb > 0:
        utils.bind_evo_operator(
            toolbox,
            'mutate',
            utils.map_mutate_f(cfg.Mutate.Method),
            cfg.Mutate.Args,
            indpb=cfg.Mutate.Indpb,
        )
    else:
        utils.bind_evo_operator(
            toolbox,
            'mutate',
            utils.map_mutate_f(cfg.Mutate.Method),
            cfg.Mutate.Args,
            indpb=1/hromo_len,
        )

    return algorithms.Popolation(
        cfg.Size,
        cfg.Mate.Probability,
        cfg.Mutate.Probability,
        cfg.HallOfFame,
        toolbox,
        ind_creator_f,
    )