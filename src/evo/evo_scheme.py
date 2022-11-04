from src.evo.graph_callback import GraphCallbackModule
import src.evo.utils as evo_utils
import src.utils as utils

from .abstract import Scheme
from ..log import get_logger
from ..config import EvoSchemeConfigField

import yaml
import numpy as np
import matplotlib.pyplot as plt

from types import FunctionType
from typing import Any, List
from deap import base, algorithms
from deap import creator
from deap import tools

class EvoScheme(Scheme):
    def __init__(self,
        name: str,
        cfg: EvoSchemeConfigField,
        evaluate_f: FunctionType,
        ind_creator_f: FunctionType=None,
        toolbox: base.Toolbox=None,
        graph_callback_module: GraphCallbackModule=None,
    ) -> None:
        self._graph_callback_module = graph_callback_module

        # Logger setup
        self._logger = get_logger(name=f'EvoScheme<{name}>')

        # Config setup
        self._name = name
        self._cfg = cfg

        # Math setup
        self._rand = np.random.RandomState(seed=self._cfg.RandSeed)

        # Monkey patch DynoExtensions
        evo_utils.DynoExtensions.LIMITS_CFG = self._cfg.Limits
        evo_utils.DynoExtensions.DEF_NP_RAND_STATE = self._rand

        # DEAP setup
        creator.create("Fitness", base.Fitness, weights=self._cfg.FitnessWeights)
        creator.create("Individual", list, fitness=creator.Fitness)

        self._toolbox: base.Toolbox() = base.Toolbox() if toolbox is None else toolbox

        self._evaluate_f = evaluate_f
        self._new_ind_f: FunctionType = ind_creator_f
        if self._new_ind_f is None:
            self._new_ind_f = lambda: evo_utils.create_ind_by_list(evo_utils.ind_creator_f(
                    creator.Individual,
                    self._cfg.HromoLen,
                    self._cfg.Limits,
                    self._rand,
                ),
                self._evaluate_f,
            )

        self._result: List = None
        self._use_restored_result = False

        if self._result is None:
            self._result = tools.initRepeat(list, self._new_ind_f, n=self._cfg.PopulationSize)

        if not hasattr(self._toolbox, 'evaluate'):
            self._toolbox.register('evaluate', evaluate_f)

        if not hasattr(self._toolbox, 'select'):
            evo_utils.bind_evo_operator(
                self._toolbox,
                'select',
                evo_utils.map_select_f(self._cfg.Select.Method),
                self._cfg.Select.Args,
            )
        if not hasattr(self._toolbox, 'mate'):
            evo_utils.bind_evo_operator(
                self._toolbox,
                'mate',
                evo_utils.map_mate_f(self._cfg.Mate.Method),
                self._cfg.Mate.Args,
            )

        if not hasattr(self._toolbox, 'mutate'):
            if self._cfg.Mutate.Indpb > 0:
                evo_utils.bind_evo_operator(
                    self._toolbox,
                    'mutate',
                    evo_utils.map_mutate_f(self._cfg.Mutate.Method),
                    self._cfg.Mutate.Args,
                    indpb=self._cfg.Mutate.Indpb,
                )
            else:
                evo_utils.bind_evo_operator(
                    self._toolbox,
                    'mutate',
                    evo_utils.map_mutate_f(self._cfg.Mutate.Method),
                    self._cfg.Mutate.Args,
                    indpb=1/self._cfg.HromoLen
                )
        self._hall_of_fame: tools.HallOfFame = None
        if self._cfg.HallOfFame > 0:
            # self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame, utils.ind_float_eq_f)
            self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame)

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._cfg.Metrics) > 0:
            self._stats = tools.Statistics(lambda ind: ind.fitness.values)
            for metric_cfg in self._cfg.Metrics:
                self._stats.register(metric_cfg.Name, evo_utils.get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('EvoScheme<%s> is running...', self._name)

        if self._graph_callback_module is not None:
            kvargs['callback'] = self._graph_callback_module.get_deap_callback()

        self._result, self._logbook = algorithms.eaSimple(
            population=self._result,
            toolbox=self._toolbox,
            cxpb=self._cfg.Mate.Probability,
            mutpb=self._cfg.Mutate.Probability,
            ngen=self._cfg.MaxGenNum,
            stats=self._stats,
            halloffame=self._hall_of_fame,
            verbose=self._cfg.Verbose,
            **kvargs,
        )

        self._logger.info('EvoScheme<%s>  has bean done', self._name)

    def restore_result(self, result: Any) -> None:
        self._use_restored_result = True
        self._result = [ind if isinstance(ind, creator.Individual) else evo_utils.create_ind_by_list(ind, self._evaluate_f) for ind in result]

    def save(self, dirname: str, **kvargs) -> str:
        # run_pool_dir = evo_utils.get_or_create_last_run_pool_dir(dirname, self._name)
        run_pool_dir = dirname
        iter_dir = evo_utils.create_iter_dir(run_pool_dir)

        len_metrics = len(self._cfg.Metrics)
        if not kvargs.get('disable_stat', False) and len_metrics > 0:
            # Dump stat graph
            fig, ax = plt.subplots()
            fig.suptitle(f'{self._name}\nevo stats')

            metrics = self._logbook.select(*[metric.Name for metric in self._cfg.Metrics])
            for i in range(len_metrics):
                ax.plot(metrics[i], label=self._cfg.Metrics[i].Name, **utils.kv_config_arr_to_kvargs(self._cfg.Metrics[i].PltArgs))
            ax.set_xlabel('generation')
            ax.set_ylabel('fitness')
            ax.legend()

            fig.savefig(f'{iter_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{iter_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(self._cfg.yaml(), f)

        if not kvargs.get('disable_dump_result', False):
            # Dump last population
            with open(f'{iter_dir}/result.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._cfg.HromoLen,
                    f,
                    self._result,
                    self._cfg.Limits,
                )

        if not kvargs.get('disable_dump_hall_of_fame', False) and self._hall_of_fame is not None:
            with open(f'{iter_dir}/hall_off_fame.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._cfg.HromoLen,
                    f,
                    self._hall_of_fame.items,
                    self._cfg.Limits,
                )

    # Access methods

    def get_name(self) -> str:
        return self._name

    def get_toolbox(self) -> base.Toolbox:
        return self._toolbox

    def get_logbook(self) -> tools.Logbook:
        return self._logbook

    def get_last_population(self) -> List[List]:
        return self._result

    def get_hall_of_fame(self) -> tools.HallOfFame:
        return self._hall_of_fame

    def get_evaluate_f(self) -> FunctionType:
        return self._evaluate_f