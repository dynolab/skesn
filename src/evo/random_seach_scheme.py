import src.evo.utils as evo_utils
import src.evo.types as evo_types
import src.utils as utils
import src.config as config
import src.log as log

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


class RandomSearchScheme(Scheme):
    def __init__(self,
        name: str,
        cfg: config.RandomSearchSchemeSchemeConfigField,
        evaluator: FunctionType,
        toolbox: base.Toolbox=None,
        graph_callback_module: GraphCallbackModule=None,
        pool: Pool=None,
    ) -> None:
        # Graphics
        self._graph_callback_module = graph_callback_module

        # Logger setup
        self._logger = log.get_logger(name=f'RandomSearchScheme<{name}>')

        # Config setup
        self._name = name
        self._cfg = cfg

        # Math setup
        self._rand = np.random.RandomState(seed=self._cfg.RandSeed)

        # DEAP setup
        evo_types.Fitness.patch_weights(self._cfg.FitnessWeights)

        self._toolbox: base.Toolbox() = base.Toolbox() if toolbox is None else toolbox

        if pool is not None:
            self._toolbox.register('map', pool.map)

        self._evaluator = evaluator
        self._ind_creator = lambda: evo_utils.ind_creator_f(
            self._cfg.HromoLen,
            self._cfg.Limits,
            self._rand,
        )

        self._set_runpool_dir()
        self._result: List = None
        self._use_restored_result: bool = False

        if self._result is None:
            self._result = ()

        if not hasattr(self._toolbox, 'generate'):
            self._toolbox.register('generate', tools.initRepeat, list, self._ind_creator, self._cfg.PopulationSize)

        if not hasattr(self._toolbox, 'evaluate'):
            self._toolbox.register('evaluate', evaluator)

        if not hasattr(self._toolbox, 'update'):
            self._toolbox.register('update', lambda _: 'dummy')

        self._hall_of_fame: tools.HallOfFame = None
        if self._cfg.HallOfFame > 0:
            # self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame, utils.ind_float_eq_f)
            self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame)

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._cfg.Metrics) > 0:
            self._stats = tools.Statistics(evo_utils.ind_stat)
            for metric_cfg in self._cfg.Metrics:
                self._stats.register(metric_cfg.Name, evo_utils.get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('RandomSearchScheme<%s> is running...', self._name)

        # if self._graph_callback_module is not None:
        #     kvargs['callback'] = self._graph_callback_module

        self._result, self._logbook = algorithms.eaGenerateUpdate(
            toolbox=self._toolbox,
            ngen=self._cfg.MaxGenNum,
            halloffame=self._hall_of_fame,
            stats=self._stats,
            verbose=self._cfg.Verbose,
            logger=self._logger,
            # **kvargs,
        )

        self._best_result, self._best_result_fitness = evo_utils.calculate_best_ind(self._result)
        self._logger.info(f'add ind to best result (ind: [{str.join(",", [str(x) for x in self._best_result])}], ind_fitness: {self._best_result_fitness})')

        self._logger.info('RandomSearchScheme<%s>  has bean done', self._name)

    def restore_result(self,
        runpool_dir: Union[str,None]=None,
        iter_num: Union[int,None]=None,
    ) -> None:
        raise 'not implemented'

    def _create_spec(self) -> dict:
        return {}

    def save(self, **kvargs) -> str:
        self._logger.info(f'dump evo scheme... (dir: %s)', self._runpool_dir)

        if not os.path.isdir(self._runpool_dir):
            os.makedirs(self._runpool_dir)

        if not kvargs.get('disable_dump_spec', False):
            if self._use_restored_result:
                spec = self._create_spec()
                with open(f'{self._runpool_dir}/spec.yaml', 'w') as f:
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

            fig.savefig(f'{self._runpool_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{self._runpool_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(config.Config.yaml(), f)

        if not kvargs.get('disable_dump_logs', False):
            # Dump logs
            logfile = log.get_logfile()
            if os.path.isfile(logfile):
                shutil.copyfile(logfile, f'{self._runpool_dir}/log')

        if not kvargs.get('disable_dump_result', False):
            # Dump last population
            with open(f'{self._runpool_dir}/result.yaml', 'w') as f:
                evo_utils.dump_inds_arr(
                    self._cfg.HromoLen,
                    f,
                    self._result,
                    self._cfg.Limits,
                )

        if not kvargs.get('disable_dump_hall_of_fame', False) and self._hall_of_fame is not None:
            with open(f'{self._runpool_dir}/hall_off_fame.yaml', 'w') as f:
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
        return self._evaluator

    # Private methods

    def _set_runpool_dir(self) -> None:
        dump_dir = self._cfg.DumpDir
        if dump_dir is None or\
            len(dump_dir) == 0:
            dump_dir = pathlib.Path().resolve()

        today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self._runpool_dir = os.path.join(dump_dir, f'runpool_random_search_{today}')
