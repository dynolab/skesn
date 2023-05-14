import matplotlib.pyplot as plt

from typing import List, Union
from multiprocess.managers import SyncManager

from src.evo.graph_callback import GraphCallbackModule

import src.evo.evo_esn_scheme_multi_pop as evo_esn_scheme_multi_pop

import src.config as cfg
import src.evo.types as evo_types


class DynoEvoEsnHyperParamWithReservoirMultiPop(evo_esn_scheme_multi_pop.EvoEsnSchemeMultiPop):
    def __init__(self,
        scheme_cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig,
        async_manager: SyncManager=None,
        job_n: int=-1,
    ) -> None:
        self._scheme_cfg: cfg.DynoEvoEsnHyperParamMultiPopConfig = scheme_cfg

        super().__init__(
            name='hyper_param',
            evo_cfg=self._scheme_cfg.Evo,
            esn_cfg=self._scheme_cfg.Esn,
            evaluate_cfg=self._scheme_cfg.Evaluate,
            esn_creator=evo_types.HyperParamWithReservoirEsnCreatorByInd(self._scheme_cfg.Esn),
            graph_callback_module=None,
            async_manager=async_manager,
            job_n=job_n,
        )
