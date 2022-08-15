import os
from pshipilov_dev.src.utils import kv_config_arr_to_kvargs
from .utils import map_select_f, map_mate_f, map_mutate_f
from .abstract import Scheme
from ..log import get_logger
from ..config import KVArgConfigSection, EvoSchemeConfigField

import yaml
import importlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from types import FunctionType
from typing import List
from deap import base, algorithms
from deap import creator
from deap import tools

def _ind_float_eq(ind_l: List[float], ind_r: List[float]) -> bool:
    if len(ind_l) != len(ind_r):
        return False
    EPS = 1e-6
    for i in range(len(ind_l)):
        if np.abs(ind_l[i] - ind_r[i]) > EPS:
            return False
    return True

class EvoScheme(Scheme):
    def __init__(self,
        name: str,
        cfg: EvoSchemeConfigField,
        evaluate_f: FunctionType,
        ind_creator_f: FunctionType=None,
        tool_box: base.Toolbox=None,
        last_poputaion: List[List]=None,
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

        if ind_creator_f is None:
            self._tool_box.register('new_ind', self._ind_creator_def_f)
        else:
            self._tool_box.register('new_ind', ind_creator_f)

        self._last_population: List = last_poputaion
        self._first_run = True
        if self._last_population is not None:
            self._tool_box.register('new_population', lambda: self._last_population)
            self._first_run = False
        elif not hasattr(self._tool_box, 'new_population'):
            self._tool_box.register('new_population', tools.initRepeat, list, self._tool_box.new_ind, n=self._cfg.PopulationSize)

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
        self._hall_of_fame = None
        if self._cfg.HallOfFame > 0:
            self._hall_of_fame = tools.HallOfFame(self._cfg.HallOfFame, _ind_float_eq)

        # Evo stats
        self._logbook: tools.Logbook = None
        self._stats: tools.Statistics = None
        if len(self._cfg.Metrics) > 0:
            self._stats = tools.Statistics(lambda ind: ind.fitness.values)
            for metric_cfg in self._cfg.Metrics:
                self._stats.register(metric_cfg.Name, _get_evo_metric_func(metric_cfg.Func, metric_cfg.Package))

    # Inherited methods

    def run(self, **kvargs) -> None:
        self._logger.info('Evo scheme "%s" is running...', self._name)

        self._last_population, self._logbook = algorithms.eaSimple(
            population=self._tool_box.new_population(),
            toolbox=self._tool_box,
            cxpb=self._cfg.Mate.Probability,
            mutpb=self._cfg.Mutate.Probability,
            ngen=self._cfg.MaxGenNum,
            stats=self._stats,
            halloffame=self._hall_of_fame,
            verbose=self._cfg.Verbose,
            **kvargs,
        )

        self._logger.info('Evo scheme "%s"  has bean done', self._name)

    def save(self, dirname: str, **kvargs) -> str:
        dumpdir = self._create_dump_dir(dirname)

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

            fig.savefig(f'{dumpdir}/stat_graph.png', dpi=fig.dpi)

        if not kvargs.get('disable_cfg', False):
            # Dump config
            with open(f'{dumpdir}/config.yaml', 'w') as f:
                yaml.safe_dump(self._cfg.yaml(), f)

        if not kvargs.get('disable_last_population', False):
            # Dump last population
            with open(f'{dumpdir}/last_population.yaml', 'w') as f:
                self._dump_ind_arr(f, self._last_population)

        if not kvargs.get('disable_hall_of_fame', False) and self._hall_of_fame is not None:
            with open(f'{dumpdir}/hall_off_fame.yaml', 'w') as f:
                self._dump_ind_arr(f, self._hall_of_fame.items)

        return dumpdir

    # Access methods

    def get_toolbox(self) -> base.Toolbox:
        return self._tool_box

    def get_logbook(self) -> tools.Logbook:
        return self._logbook

    def get_last_population(self) -> List[List]:
        return self._last_population

    def get_hall_of_fame(self) -> tools.HallOfFame:
        return self._hall_of_fame

    # Public methods

    def restore_population_last_run_pool(self,
        root: str,
    ) -> None:
        _, dirs, _ = next(os.walk(root))
        curr_run_pool = -1
        prefix = f'run_pool_{self._name.replace(" ", "")}_'
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_run_pool:
                    curr_run_pool = num

        if curr_run_pool < 0:
            raise 'run pools not found'

        self.restore_population_last_iter(f'{root}/{prefix}{curr_run_pool}')

    def restore_population_last_iter(self,
        runpool_dir: str,
    ) -> None:
        _, dirs, _ = next(os.walk(runpool_dir))
        curr_iter = -1
        prefix = 'iter_'
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_iter:
                    curr_iter = num

        if curr_iter < 0:
            raise 'run pool is empty'

        self.restore_population_from_file(f'{runpool_dir}/{prefix}{curr_iter}/last_population.yaml')

    def restore_population_from_file(self,
        filename: str,
    ) -> None:
        with open(filename, 'r') as f:
            last_population_yaml = yaml.safe_load(f)
            if last_population_yaml is None:
                raise f'the population from the file "{filename}" is None)'

            if self._cfg.PopulationSize != len(last_population_yaml):
                raise f'the population size from the file "{filename}" ({len(last_population_yaml)}) does not match the config ({self._cfg.PopulationSize})'

            if self._last_population is None:
                self._last_population = [0] * self._cfg.PopulationSize
            self._tool_box.register('new_population', lambda: self._last_population)

            if isinstance(last_population_yaml, dict):
                for i, ind in enumerate(last_population_yaml.values()):
                    self._last_population[i] = creator.Individual(ind)
            elif isinstance(last_population_yaml, list):
                for i, ind in enumerate(last_population_yaml):
                    self._last_population[i] = creator.Individual(ind)
            else:
                raise 'unknown last popultaion yaml type'

        self._first_run = False

    # Internal methods

    def _dump_ind_arr(self, f, inds: List) -> None:
        dump = {}
        for i, ind in enumerate(inds):
            dump_ind = [0] * self._cfg.HromoLen
            allow_limits = len(self._cfg.Limits) == self._cfg.HromoLen
            for j in range(self._cfg.HromoLen):
                if allow_limits:
                    if self._cfg.Limits[j].IsInt:
                        dump_ind[j] = int(ind[j])
                    else:
                        dump_ind[j] = float(ind[j])
                else:
                    dump_ind[j] = float(ind[j])

            dump[f'ind_{i}'] = dump_ind
        yaml.safe_dump(dump, f)

    def _ind_creator_def_f(self) -> list:
        ret = [0] * self._cfg.HromoLen
        for i in range(self._cfg.HromoLen):
            if len(self._cfg.Limits) > 0:
                if self._cfg.Limits[i].IsInt:
                    ret[i] = self._rand.randint(self._cfg.Limits[i].Min, self._cfg.Limits[i].Max)
                else:
                    ret[i] = self._rand.uniform(self._cfg.Limits[i].Min, self._cfg.Limits[i].Max)
            else:
                ret[i] = self._rand.rand()
        return creator.Individual(ret)

    def _create_dump_dir(self, root: str) -> str:
        curr_run_pool = -1
        prefix = f'run_pool_{self._name.replace(" ", "")}_'

        if not os.path.isdir(root):
            os.makedirs(root)
        else:
            _, dirs, _ = next(os.walk(root))
            for dir in dirs:
                if dir.startswith(prefix):
                    num = int(dir.split('_')[-1])
                    if num > curr_run_pool:
                        curr_run_pool = num

        if self._first_run:
            curr_run_pool += 1

        ret = root
        if ret[len(ret)-1] != ord('/'):
            ret += '/'

        ret += prefix + str(curr_run_pool) + '/'

        if not os.path.isdir(ret):
            os.makedirs(ret)

        _, dirs, _ = next(os.walk(ret))
        curr_iter = -1
        prefix = 'iter_'
        for dir in dirs:
            if dir.startswith(prefix):
                num = int(dir.split('_')[-1])
                if num > curr_iter:
                    curr_iter = num

        ret += prefix + str(curr_iter + 1)
        os.mkdir(ret)

        return ret


def _get_evo_metric_func(
    func_name: str,
    package_name: str,
) -> FunctionType:
    module = importlib.import_module(package_name)
    if hasattr(module, func_name):
        return getattr(module, func_name)
    raise 'unknown function name'

def _bind_evo_operator(
    tool_box: base.Toolbox,
    name: str,
    func: FunctionType,
    args: List[KVArgConfigSection],
    **kvargs,
    ):

    if len(args) > 0:
        for arg in args:
            kvargs[arg.Key] = arg.Val

    tool_box.register(name, func, **kvargs)
