import datetime
import pathlib
import shutil
import src.log as log
import src.evo.utils as evo_utils
import src.evo.types as evo_types

from src.evo.abstract import Scheme
from src.log import get_logger
from src.config import EvoSchemeMultiPopConfigField, EvoPopulationConfigField
from src.utils import kv_config_arr_to_kvargs
from src.evo.graph_callback import GraphCallbackModule

import yaml
import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt

from multiprocess.pool import Pool
from types import FunctionType
from typing import Any, List, Union
from deap import base, algorithms
from deap import tools

class EvoSchemeMultiPop(Scheme):
    def __init__(self,
        name: str,
        evo_cfg: EvoSchemeMultiPopConfigField,
        evaluator: FunctionType,
        graph_callback_module: GraphCallbackModule=None,
        pool: Pool=None,
    ) -> None:
        # Graphics
        self._graph_callback_module = graph_callback_module

        # Logger setup
        self._logger: logging.Logger = get_logger(name=f'EvoSchemeMultiPop<{name}>')

        # Config setup
        self._name:    str                          = name
        self._evo_cfg: EvoSchemeMultiPopConfigField = evo_cfg

        # Math setup
        self._rand: np.random.RandomState = np.random.RandomState(seed=self._evo_cfg.RandSeed)

        # Async setup
        self._pool: Pool = pool

        # DEAP setup
        evo_types.Fitness.patch_weights(self._evo_cfg.FitnessWeights)
        self._pop_len = len(evo_cfg.Populations)
        self._evaluator = evaluator
        self._ind_creators = [None] * self._pop_len
        for i in range(self._pop_len):
            self._ind_creators[i] = lambda: evo_utils.ind_creator_f(
                self._evo_cfg.HromoLen,
                self._evo_cfg.Populations[i].Limits,
                self._rand,
            )

        # Set up iter dir
        self._iter_dir: str = None
        self._set_runpool_dir()
        self._use_restored_result: bool = False
        self._result: List[algorithms.Popolation] = []
        self._best_result: List[evo_types.Individual] = []

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._evo_cfg.Metrics) > 0:
            self._stats = tools.Statistics(evo_utils.ind_stat)
            for metric_cfg in self._evo_cfg.Metrics:
                self._stats.register(metric_cfg.Name, evo_utils.get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('EvoSchemeMultiPop<%s> is running...', self._name)

        if self._graph_callback_module is not None:
            kvargs['callback'] = self._graph_callback_module.get_deap_callback()

        if len(self._result) == 0:
            self._set_result()

        self._result, self._logbook = algorithms.eaSimpleMultiPop(
            populations=self._result,
            ngen=self._evo_cfg.MaxGenNum,
            stats=self._stats,
            pbcltex=0.5,
            cltex_f=cltex,
            verbose=self._evo_cfg.Verbose,
            logger=self._logger,
            # pool=self._pool,
            **kvargs,
        )

        for pop in self._result:
            pop_best_ind = None
            pop_best_fitness = None

            if pop.HallOfFame is not None:
                pop_best_ind, pop_best_fitness = evo_utils.calculate_best_ind(pop.HallOfFame)
            else:
                pop_best_ind, pop_best_fitness = evo_utils.calculate_best_ind(pop.Inds)

            self._best_result.append(pop_best_ind)
            self._logger.info(f'add ind to best result (ind: [{str.join(",", [str(x) for x in pop_best_ind])}], ind_fitness: {pop_best_fitness})')

        self._logger.info('EvoSchemeMultiPop<%s>  has bean done', self._name)

    # def restore_result(self, result: Any) -> None:
    #     self._use_restored_result = True
    #     self._result = []

    #     k = 0
    #     for pop_cfg in self._cfg.Populations:
    #         if pop_cfg.IncludingCount <= 0:
    #             continue

    #         for _ in range(pop_cfg.IncludingCount):
    #             pop = _create_deap_population(
    #                 self._evaluate_f,
    #                 self._ind_creator_f,
    #                 self._cfg.HromoLen,
    #                 pop_cfg,
    #                 self._rand,
    #             )
    #             pop.Inds = [ind if isinstance(ind, evo_types.Individual) else evo_utils.create_ind_by_list(ind, self._evaluate_f) for ind in result[k]]
    #             k += 1
    #             self._result.append(pop)

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

        restored_result = evo_utils.restore_evo_scheme_result_from_iter(
            parse_result_f=self._parse_restored_result,
            runpool_dir=self._runpool_dir,
            iter_num=iter_num,
        )

        if len(self._result) > 0:
            self._result.clear()

        k = 0
        for pop_cfg in self._evo_cfg.Populations:
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                pop = _create_deap_population(
                    inds=restored_result[k],
                    evaluate_f=self._evaluator,
                    hromo_len=self._evo_cfg.HromoLen,
                    cfg=pop_cfg,
                    rand=self._rand,
                    pool=self._pool,
                )

                k += 1

                self._result.append(pop)

        self._use_restored_result = True
        self._restored_iter = iter_num

    def save(self,
        **kvargs,
    ) -> str:
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
                ax.plot(metrics[i], label=self._evo_cfg.Metrics[i].Name, **kv_config_arr_to_kvargs(self._evo_cfg.Metrics[i].PltArgs))
            ax.set_xlabel('generation')
            ax.set_ylabel('fitness')
            ax.legend()

            fig.savefig(f'{self._iter_dir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_dump_logs', False):
            # Dump logs
            logfile = log.get_logfile()
            if os.path.isfile(logfile):
                shutil.copyfile(logfile, f'{self._iter_dir}/log')

        if not kvargs.get('disable_dump_cfg', False):
            # Dump config
            with open(f'{self._iter_dir}/config.yaml', 'w') as f:
                yaml.safe_dump(self._evo_cfg.yaml(), f)

        if not kvargs.get('disable_dump_result', False):
            # Dump last populations
            with open(f'{self._iter_dir}/result.yaml', 'w') as f:
                evo_utils.dump_inds_multi_pop_arr(
                    self._evo_cfg.HromoLen,
                    f,
                    [popultaion.Inds for popultaion in self._result],
                    evo_utils.get_populations_limits(self._evo_cfg),
                )

        if not kvargs.get('disable_dump_hall_of_fame', False):
            # Dump last population hall of fames
            with open(f'{self._iter_dir}/hall_off_fames.yaml', 'w') as f:
                evo_utils.dump_inds_multi_pop_arr(
                    self._evo_cfg.HromoLen,
                    f,
                    [popultaion.HallOfFame.items if popultaion.HallOfFameSize > 0 else [] for popultaion in self._result],
                    evo_utils.get_populations_limits(self._evo_cfg),
                )

    def get_evaluate_f(self) -> FunctionType:
        return self._evaluator

    # Access methods

    def get_name(self) -> str:
        return self._name

    def get_logbook(self) -> tools.Logbook:
        return self._logbook

    def get_populations(self) -> List[List[List]]:
        return self._result

    # Private methods

    def _set_result(self) -> None:
        self._result = [None] * self._pop_len
        for i, pop_cfg in enumerate(self._evo_cfg.Populations):
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                pop = [self._ind_creators[i]() for _ in range(pop_cfg.Size)]
                self._result[i] = _create_deap_population(
                    inds=pop,
                    evaluate_f=self._evaluator,
                    hromo_len=self._evo_cfg.HromoLen,
                    cfg=pop_cfg,
                    rand=self._rand,
                    pool=self._pool,
                )

    def _create_spec(self) -> dict:
        ret = {
            'use_restored_result': self._use_restored_result,
        }

        if self._use_restored_result:
            ret['restored_iter'] = self._restored_iter

        return ret

    def _parse_restored_result(self,
        stream,
    ) -> Any:
        result = yaml.safe_load(stream)
        if result is None:
            raise 'restored result is None'

        populations_cnt = evo_utils.get_populations_cnt(self._evo_cfg)

        if len(result) != populations_cnt:
            raise f'restored popultations size isn\'t equal config (restored: {len(result)}, config: {populations_cnt})'

        ret: List[List] = []

        k = 0
        for pop_cfg in self._evo_cfg.Populations:
            if pop_cfg.IncludingCount <= 0:
                continue

            for _ in range(pop_cfg.IncludingCount):
                pop_key = f'pop_{k}'
                if pop_key not in result:
                    raise f'restored popultations doesn\'t contain "{pop_key}"'

                inds = result[pop_key]

                if pop_cfg.Size != len(inds):
                    raise f'restored popultation size isn\'t equal config (restored: {len(inds)}, config: {pop_cfg.Size})'

                if isinstance(inds, dict):
                    # ret.append([evo_utils.create_ind_by_list(ind, self._evaluator) for _, ind in inds.values()])
                    ret.append([evo_types.Individual(ind) for _, ind in inds.values()])
                elif isinstance(inds, list):
                    # ret.append([evo_utils.create_ind_by_list(ind, self._evaluator) for ind in inds])
                    ret.append([evo_types.Individual(ind) for ind in inds])
                else:
                    raise 'unknown last popultaion yaml type'

                k += 1

        return ret

    def _set_runpool_dir(self) -> None:
        dump_dir = self._evo_cfg.DumpDir
        if dump_dir is None or\
            len(dump_dir) == 0:
            dump_dir = pathlib.Path().resolve()

        today = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self._runpool_dir = os.path.join(dump_dir, f'runpool_{today}')

def cltex(
    populations: List[algorithms.Popolation],
) -> None:
    base_hall_of_fame = tools.HallOfFame(3)
    for population in populations:
        if population.HallOfFameSize > 0:
            base_hall_of_fame.update(population.HallOfFame.items)
            continue

    for population in populations:
        population.Inds.sort(key=_cltex_key)
        for i, best in enumerate(base_hall_of_fame):
            population.Inds[i] = best

def _cltex_key(
    ind: List,
) -> float:
    return np.dot(ind.fitness.values, ind.fitness.weights)

def _create_deap_population(
    inds: List,
    evaluate_f: FunctionType,
    hromo_len: int,
    cfg: EvoPopulationConfigField,
    rand: np.random.RandomState,
    pool: Pool=None,
) -> algorithms.Popolation:
    toolbox = base.Toolbox()

    if pool is not None:
        toolbox.register('map', pool.map)

    toolbox.register('evaluate', evaluate_f)

    evo_utils.bind_evo_operator(
        toolbox,
        'select',
        evo_utils.map_select_f(cfg.Select.Method),
        cfg.Select.Args,
    )

    evo_utils.bind_mate_operator(
        toolbox=toolbox,
        cfg=cfg,
        rand=rand,
    )
    evo_utils.bind_mutate_operator(
        toolbox=toolbox,
        hromo_len=hromo_len,
        cfg=cfg,
        rand=rand,
    )

    inds=[ evo_types.Individual(ind) for ind in inds ]
    return algorithms.Popolation(
        inds=inds,
        cxpb=cfg.Mate.Probability,
        mutpb=cfg.Mutate.Probability,
        hall_of_fame_size=cfg.HallOfFame,
        toolbox=toolbox,
    )