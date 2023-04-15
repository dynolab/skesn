import matplotlib.pyplot as plt

from typing import Any, List

from skesn.esn import EsnForecaster

from src.evo.esn_data_holder import EsnDataHolder
from src.evo.graph_callback import GraphCallbackModule
from src.evo.evo_esn_scheme import EvoEsnScheme

import src.config as scheme_cfg
import src.evo.types as evo_types

from multiprocess.managers import SyncManager

SPECTRAL_RADIUS_IDX = 0
SPARSITY_IDX = 1
LAMBDA_R_IDX = 2
REGULIRIZATION_IDX = 3
USE_ADDITIVE_NOISE_WHEN_FORECASTING_IDX = 4
USE_BIAS_IDX = 5

class DynoEvoEsnHyperParam(EvoEsnScheme):
    def __init__(self,
        scheme_cfg: scheme_cfg.DynoEvoEsnHyperParamConfig,
        async_manager: SyncManager=None,
    ) -> None:
        self._scheme_cfg = scheme_cfg

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

    def run(self, **kvargs) -> None:
        super().run(**kvargs)
        plt.show()

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
