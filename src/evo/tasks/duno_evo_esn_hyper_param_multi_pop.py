import matplotlib.pyplot as plt

from typing import List
from multiprocess.managers import SyncManager

from src.evo.graph_callback import GraphCallbackModule

import src.evo.evo_esn_scheme_multi_pop as evo_esn_scheme_multi_pop

import src.config as cfg
import src.evo.types as evo_types


class DynoEvoEsnHyperParamMultiPop(evo_esn_scheme_multi_pop.EvoEsnSchemeMultiPop):
    def __init__(self,
        scheme_cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig,
        async_manager: SyncManager=None,
    ) -> None:
        self._scheme_cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig = scheme_cfg

        # graph_callback_module = self._create_graph_callback_module()
        graph_callback_module = None

        super().__init__(
            name='hyper_param',
            evo_cfg=self._scheme_cfg.Evo,
            esn_cfg=self._scheme_cfg.Esn,
            evaluate_cfg=self._scheme_cfg.Evaluate,
            esn_creator=evo_types.HyperParamEsnCreatorByInd(self._scheme_cfg.Esn),
            graph_callback_module=graph_callback_module,
            async_manager=async_manager,
        )

    def _create_graph_callback_module(self) -> GraphCallbackModule:
        ret = GraphCallbackModule(1)
        ret.regist_callback(self._graph_callback)

        ax = ret.get_axes()
        ax.set_title('error per generation graph')
        ax.set_xlabel('gen')
        ax.set_xlim((0, self._scheme_cfg.Evo.MaxGenNum + 1))
        ax.set_ylabel(self._scheme_cfg.Evaluate.Metric)

        return ret

    def _graph_callback(self,
        i,
        j,
        fig,
        ax,
        **kvargs,
    ) -> None:
        items = []
        if 'halloffame' in kvargs:
            items = kvargs['halloffame'].items
        if len(items) == 0 and 'population' in kvargs:
            items = kvargs['population']
        gen = 0
        if 'gen' in kvargs:
            gen = kvargs['gen']
        points = [(gen, item.fitness.values[0]) for item in items]
        ax.scatter(*zip(*points), marker='o', color='blue', zorder=1)
