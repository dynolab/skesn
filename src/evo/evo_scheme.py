import src.evo.utils as evo_utils
import src.evo.types as evo_types
import src.utils as utils
import src.config as config
import src.log as log
import src.dump as dump

from src.evo.abstract import Scheme
from src.evo.graph_callback import GraphCallbackModule

import yaml
import shutil
import os.path
import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.pool import Pool

from types import FunctionType
from typing import Any, List, Union
from deap import base, algorithms
from deap import tools


class EvoScheme(Scheme):
    def __init__(self,
        name: str,
        cfg: config.EvoSchemeConfigField,
        evaluator: FunctionType,
        toolbox: base.Toolbox=None,
        graph_callback_module: GraphCallbackModule=None,
        pool: Pool=None,
    ) -> None:
        # Graphics
        self._graph_callback_module = graph_callback_module

        # Logger setup
        # self._logger = log.get_logger(name=f'EvoScheme<{name}>')
        self._logger = log.logging.root

        # Config setup
        self._name = name
        self._evo_cfg = cfg

        # Math setup
        self._rand = np.random.RandomState(seed=self._evo_cfg.RandSeed)

        # DEAP setup
        # creator.create("Fitness", base.Fitness, weights=self._cfg.FitnessWeights)
        # creator.create("Individual", list, fitness=creator.Fitness)
        evo_types.Fitness.patch_weights(self._evo_cfg.FitnessWeights)

        self._toolbox: base.Toolbox() = base.Toolbox() if toolbox is None else toolbox

        if cfg.MandatoryNewNum > 0:
            self._toolbox.register('mandatory_new_num', lambda: cfg.MandatoryNewNum)

        if pool is not None:
            self._toolbox.register('map', pool.map)

        self._evaluator = evaluator
        self._evo_ind_creator = lambda: evo_utils.ind_creator_f(
            self._evo_cfg.HromoLen,
            self._evo_cfg.Limits,
            self._rand,
        )

        self._iter_dir: str = None
        self._set_runpool_dir()
        self._result: List = None
        self._use_restored_result: bool = False

        if self._result is None:
            self._result = tools.initRepeat(list, self._evo_ind_creator, n=self._evo_cfg.PopulationSize)

        if not hasattr(self._toolbox, 'evaluate'):
            self._toolbox.register('evaluate', evaluator)

        if not hasattr(self._toolbox, 'select'):
            evo_utils.bind_evo_operator(
                self._toolbox,
                'select',
                evo_utils.map_select_f(self._evo_cfg.Select.Method),
                self._evo_cfg.Select.Args,
            )
        if not hasattr(self._toolbox, 'mate'):
            evo_utils.bind_mate_operator(
                toolbox=self._toolbox,
                cfg=cfg,
                rand=self._rand,
                # create_ind_by_list_f=self._create_ind_by_list_f,
            )

        if not hasattr(self._toolbox, 'mutate'):
            evo_utils.bind_mutate_operator(
                toolbox=self._toolbox,
                hromo_len=self._evo_cfg.HromoLen,
                cfg=cfg,
                rand=self._rand,
            )

        self._hall_of_fame: tools.HallOfFame = None
        if self._evo_cfg.HallOfFame > 0:
            # self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame, utils.ind_float_eq_f)
            self._hall_of_fame = tools.HallOfFame(self._evo_cfg.HallOfFame)

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._evo_cfg.Metrics) > 0:
            self._stats = tools.Statistics(evo_utils.ind_stat)
            for metric_cfg in self._evo_cfg.Metrics:
                self._stats.register(metric_cfg.Name, evo_utils.get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('EvoScheme<%s> is running...', self._name)

        if self._graph_callback_module is not None:
            kvargs['callback'] = self._graph_callback_module.get_deap_callback()

        self._result, self._logbook = algorithms.eaSimple(
            population=self._result,
            toolbox=self._toolbox,
            cxpb=self._evo_cfg.Mate.Probability,
            mutpb=self._evo_cfg.Mutate.Probability,
            ngen=self._evo_cfg.MaxGenNum,
            stats=self._stats,
            halloffame=self._hall_of_fame,
            verbose=self._evo_cfg.Verbose,
            logger=self._logger,
            ind_creator_f=self._evo_ind_creator,
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

        self._logger.info(f'dump evo scheme... (dir: %s)', self._iter_dir)

        if not kvargs.get('disable_dump_spec', False):
            if self._use_restored_result:
                spec = self._create_spec()
                with open(f'{self._iter_dir}/spec.yaml', 'w') as f:
                    yaml.safe_dump(spec, f)

        len_metrics = len(self._evo_cfg.Metrics)
        if not kvargs.get('disable_dump_stat', False) and len_metrics > 0:
            # Dump stat graph
            fig, ax = plt.subplots()
            fig.suptitle(f'{self._name}\nevo stats')

            metrics = self._logbook.select(*[metric.Name for metric in self._evo_cfg.Metrics])
            for i in range(len_metrics):
                ax.plot(metrics[i], label=self._evo_cfg.Metrics[i].Name, **utils.kv_config_arr_to_kvargs(self._evo_cfg.Metrics[i].PltArgs))
            ax.set_xlabel('generation')
            ax.set_ylabel('fitness')
            ax.legend()

            fig.savefig(f'{self._iter_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{self._iter_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(config.Config.yaml(), f)

        if not kvargs.get('disable_dump_logs', False):
            # Dump logs
            logfile = log.get_logfile()
            if os.path.isfile(logfile):
                shutil.copyfile(logfile, f'{self._iter_dir}/log')

        if not kvargs.get('disable_dump_result', False):
            # Dump last population
            with open(f'{self._iter_dir}/result.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._evo_cfg.HromoLen,
                    f,
                    self._result,
                    self._evo_cfg.Limits,
                )

        if not kvargs.get('disable_dump_hall_of_fame', False) and self._hall_of_fame is not None:
            with open(f'{self._iter_dir}/hall_off_fame.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._evo_cfg.HromoLen,
                    f,
                    self._hall_of_fame.items,
                    self._evo_cfg.Limits,
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
        return self._evaluator

    # Private methods

    def _set_runpool_dir(self) -> None:
        dump_dir = self._evo_cfg.DumpDir
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

        if self._evo_cfg.PopulationSize != len(last_population_yaml):
            raise f'restored population size isn\'t equal config size (restored: {len(last_population_yaml)}), config: {self._evo_cfg.PopulationSize})'

        ret: List[List] = [0] * self._evo_cfg.PopulationSize

        if isinstance(last_population_yaml, dict):
            for i, ind in enumerate(last_population_yaml.values()):
                ret[i] = evo_types.Individual(ind)
        elif isinstance(last_population_yaml, list):
            for i, ind in enumerate(last_population_yaml):
                ret[i] = evo_types.Individual(ind)
        else:
            raise 'unknown last popultaion yaml type'

        return ret
