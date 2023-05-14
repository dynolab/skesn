from typing import List, Union
from multiprocess.managers import SyncManager

from src.evo.graph_callback import GraphCallbackModule
from src.evo.esn_data_holder import EsnMultiDataHolder

import src.evo.evo_esn_scheme_multi_pop_multi_crit as evo_esn_scheme_multi_pop_multi_crit

import src.config as cfg
import src.evo.utils as evo_utils


class DynoEvoEsnHyperParamMultiPopMultiCrit(evo_esn_scheme_multi_pop_multi_crit.EvoEsnSchemeMultiPopMultiCrit):
    def __init__(self,
        scheme_cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig,
        asyn_manager: SyncManager=None,
        job_n: int=-1,
    ) -> None:
        self._cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig = scheme_cfg

        # graph_callback_module = self._create_graph_callback_module()

        super().__init__(
            name='hyper_param',
            evo_cfg=self._cfg.Evo,
            esn_cfg=self._cfg.Esn,
            evaluate_cfg=self._cfg.Evaluate,
            esn_creator_by_ind_f=self._esn_creator_by_ind_f,
            job_n=job_n,
        )

    # def run(self, **kvargs) -> None:
    #     super().run(**kvargs)
    #     plt.show()

    def _create_graph_callback_module(self) -> GraphCallbackModule:
        ret = GraphCallbackModule(1)
        ret.regist_callback(self._graph_callback)

        ax = ret.get_axes()
        ax.set_title('error per generation graph')
        ax.set_xlabel('gen')
        ax.set_xlim((0, self._cfg.Evo.MaxGenNum + 1))
        ax.set_ylabel(self._cfg.Evaluate.Metric)

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
