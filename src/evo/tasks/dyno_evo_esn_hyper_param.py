import matplotlib.pyplot as plt

from typing import List

from skesn.esn import EsnForecaster

from src.evo.graph_callback import GraphCallbackModule
from src.evo.evo_esn_scheme import EvoEsnScheme

import src.config as cfg

from multiprocess.pool import Pool

SPECTRAL_RADIUS_IDX = 0
SPARSITY_IDX = 1
LAMBDA_R_IDX = 2
REGULIRIZATION_IDX = 3
USE_ADDITIVE_NOISE_WHEN_FORECASTING_IDX = 4
USE_BIAS_IDX = 5

class DynoEvoEsnHyperParam(EvoEsnScheme):
    def __init__(self,
        cfg: cfg.DynoEvoEsnHyperParamConfig,
        pool: Pool=None,
    ) -> None:
        self._cfg = cfg

        # graph_callback_module = self._create_graph_callback_module()

        graph_callback_module = None

        super().__init__(
            name='hyper_param',
            evo_cfg=self._cfg.Evo,
            esn_cfg=self._cfg.Esn,
            evaluate_cfg=self._cfg.Evaluate,
            esn_creator_by_ind_f=self._esn_creator_by_ind_f,
            graph_callback_module=graph_callback_module,
            pool=pool,
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

    def _esn_creator_by_ind_f(self,
        ind: List,
    ) -> EsnForecaster:
        return EsnForecaster(
            n_reservoir=self._esn_cfg.NReservoir,
            spectral_radius=ind[SPECTRAL_RADIUS_IDX],
            sparsity=ind[SPARSITY_IDX],
            regularization=ind[REGULIRIZATION_IDX],
            lambda_r=ind[LAMBDA_R_IDX],
            in_activation=self._esn_cfg.InActivation,
            out_activation=self._esn_cfg.OutActivation,
            use_additive_noise_when_forecasting=ind[USE_ADDITIVE_NOISE_WHEN_FORECASTING_IDX],
            random_state=self._esn_cfg.RandomState,
            use_bias=ind[USE_BIAS_IDX],
        )
