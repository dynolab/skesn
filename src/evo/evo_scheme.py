from src.evo.graph_callback import GraphCallbackModule
import src.evo.utils as evo_utils
import src.utils as utils

from .abstract import Scheme
from ..log import get_logger
from ..config import EvoSchemeConfigField

import yaml
import os.path
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt

from types import FunctionType
from typing import Any, List, Union
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
        # Graphics
        self._graph_callback_module = graph_callback_module

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

        self._iter_dir: str = None
        self._set_runpool_dir()
        self._result: List = None
        self._use_restored_result: bool = False

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
            evo_utils.bind_mate_operator(
                toolbox=self._toolbox,
                cfg=cfg,
                rand=self._rand,
                create_ind_by_list_f=lambda ind: evo_utils.create_ind_by_list(ind, self._evaluate_f),
            )

        if not hasattr(self._toolbox, 'mutate'):
            evo_utils.bind_mutate_operator(
                toolbox=self._toolbox,
                hromo_len=self._cfg.HromoLen,
                cfg=cfg,
                rand=self._rand,
            )

        self._hall_of_fame: tools.HallOfFame = None
        if self._cfg.HallOfFame > 0:
            # self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame, utils.ind_float_eq_f)
            self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame)

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._cfg.Metrics) > 0:
            self._stats = tools.Statistics(lambda ind: np.dot(ind.fitness.values, ind.fitness.weights))
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
            logger=self._logger,
            **kvargs,
        )

        self._best_result, self._best_result_fitness = evo_utils.calculate_best_ind(self._result)
        self._logger.info(f'add ind to best result (ind: [{str.join(",", [str(x) for x in self._best_result])}], ind_fitness: {self._best_result_fitness})')

        self._logger.info('EvoScheme<%s>  has bean done', self._name)

    def restore_result(self,
        runpool_dir: Union[str,None]=None,
        iter_num: Union[int,None]=None,
    ) -> None:
        if runpool_dir is not None:
            self._runpool_dir = os.path.normpath(runpool_dir)

        if not os.path.isdir(self._runpool_dir):
            raise f'can\'t continue calculation: runpool dir isn\'t exist ({runpool_dir})'

        if iter_num is None:
            iter_num = evo_utils.get_last_iter_num(
                runpool_dir=self._runpool_dir,
            )

        if iter_num is None:
            raise f'can\'t continue calculation: not found last iter dir ({runpool_dir})'

        self._result = evo_utils.restore_evo_scheme_result_from_iter(
            parse_result_f=self._parse_restored_result,
            runpool_dir=self._runpool_dir,
            iter_num=iter_num,
        )

        self._use_restored_result = True
        self._restored_iter = iter_num

    def _create_spec(self) -> dict:
        ret = {
            'use_restored_result': self._use_restored_result,
        }

        if self._use_restored_result:
            ret['restored_iter'] = self._restored_iter

        return ret

    def save(self, **kvargs) -> str:
        self._iter_dir = evo_utils.get_or_create_new_iter_dir(
            runpool_dir=self._runpool_dir,
            iter_dir=self._iter_dir,
        )

        if not kvargs.get('disable_dump_spec', False):
            if self._use_restored_result:
                spec = self._create_spec()
                with open(f'{self._iter_dir}/spec.yaml', 'w') as f:
                    yaml.safe_dump(spec, f)


        len_metrics = len(self._cfg.Metrics)
        if not kvargs.get('disable_dump_stat', False) and len_metrics > 0:
            # Dump stat graph
            fig, ax = plt.subplots()
            fig.suptitle(f'{self._name}\nevo stats')

            metrics = self._logbook.select(*[metric.Name for metric in self._cfg.Metrics])
            for i in range(len_metrics):
                ax.plot(metrics[i], label=self._cfg.Metrics[i].Name, **utils.kv_config_arr_to_kvargs(self._cfg.Metrics[i].PltArgs))
            ax.set_xlabel('generation')
            ax.set_ylabel('fitness')
            ax.legend()

            fig.savefig(f'{self._iter_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{self._iter_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(self._cfg.yaml(), f)

        if not kvargs.get('disable_dump_result', False):
            # Dump last population
            with open(f'{self._iter_dir}/result.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._cfg.HromoLen,
                    f,
                    self._result,
                    self._cfg.Limits,
                )

        if not kvargs.get('disable_dump_hall_of_fame', False) and self._hall_of_fame is not None:
            with open(f'{self._iter_dir}/hall_off_fame.yaml', 'w') as f:
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

    # Private methods

    def _set_runpool_dir(self) -> None:
        dump_dir = self._cfg.DumpDir
        if dump_dir is None or\
            len(dump_dir) == 0:
            dump_dir = pathlib.Path().resolve()

        today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self._runpool_dir = os.path.join(dump_dir, f'runpool_{today}')

    def _parse_restored_result(self,
        stream,
    ) -> Any:
        last_population_yaml = yaml.safe_load(stream)
        if last_population_yaml is None:
            raise 'restored population is None'

        if self._cfg.PopulationSize != len(last_population_yaml):
            raise f'restored population size isn\'t equal config size (restored: {len(last_population_yaml)}), config: {self._cfg.PopulationSize})'

        ret: List[List] = [0] * self._cfg.PopulationSize

        if isinstance(last_population_yaml, dict):
            for i, ind in enumerate(last_population_yaml.values()):
                ret[i] = creator.Individual(ind)
        elif isinstance(last_population_yaml, list):
            for i, ind in enumerate(last_population_yaml):
                ret[i] = creator.Individual(ind)
        else:
            raise 'unknown last popultaion yaml type'

        return ret
