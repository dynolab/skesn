
import src.config as cfg
import src.evo.utils as evo_utils
import src.evo.types as evo_types

import matplotlib.pyplot as plt

from multiprocess.managers import SyncManager

from types import FunctionType
from typing import Any, Dict, List, Tuple, Union

import skesn.esn as esn

import numpy as np

from src.evo.graph_callback import GraphCallbackModule
from src.evo.evo_scheme import EvoScheme
from src.evo.esn_data_holder import EsnDataHolder


class EvoEsnScheme(EvoScheme):
    def __init__(self,
        name: str,
        evo_cfg: cfg.EvoSchemeConfigField,
        esn_cfg: cfg.EsnConfigField,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        esn_creator: FunctionType,
        graph_callback_module: GraphCallbackModule=None,
        async_manager: SyncManager=None,
        job_n: Union[None, int]=None,
    ) -> None:
        # Init configs
        self._esn_cfg: cfg.EsnConfigField = esn_cfg
        self._evaluate_cfg: cfg.EsnEvaluateConfigField = evaluate_cfg

        self._esn_creator: FunctionType = esn_creator

        # Init evaluate data
        self._data_holder = EsnDataHolder(self._evaluate_cfg)

        self._fit_data: np.ndarray = self._data_holder.FitData
        self._valid_data: np.ndarray = self._data_holder.ValidData

        pool = None
        if async_manager is not None and job_n > 0:
            pool = async_manager.Pool(
                processes=job_n,
                initializer=evo_types.esn_pool_init,
                initargs=(esn_creator, self._evaluate_cfg, self._data_holder),
            )

        super().__init__(
            name=name,
            cfg=evo_cfg,
            evaluator=evo_types.EsnEvaluator(
                self._evaluate_cfg,
                esn_creator,
                self._fit_data,
                self._valid_data,
            ),
            graph_callback_module=graph_callback_module,
            pool=pool,
        )

    def save(self,
        **kvargs,
    ) -> str:
        self._iter_dir = evo_utils.get_or_create_new_iter_dir(
            runpool_dir=self._runpool_dir,
            iter_dir=self._iter_dir,
        )

        if not kvargs.get('disable_dump_graph_best', False):
            dim = self._data_holder.Model.get_dim()

            fig, axes = plt.subplots(dim, figsize=(16,10))

            fit_len = len(self._fit_data)
            valid_len = len(self._valid_data)

            for i in range(dim):
                # show fit data
                if self._evaluate_cfg.FitStep > 0:
                    axes[i].plot([x * self._evaluate_cfg.FitStep for x in range(fit_len)], self._fit_data[:,i],
                        color='blue',linestyle='dashed',linewidth=1,label='fit_data',
                    )

                    axes[i].plot([x for x in range(fit_len * self._evaluate_cfg.FitStep, fit_len * self._evaluate_cfg.FitStep + valid_len)], self._valid_data[:,i],
                        color='blue',linewidth=1,label='valid_data',
                    )
                else:
                    axes[i].plot([x for x in range(fit_len)], self._fit_data[:,i],
                        color='blue',linestyle='dashed',linewidth=1,label='fit_data',
                    )

                    # show valid data
                    axes[i].plot([x for x in range(fit_len, fit_len + valid_len)], self._valid_data[:,i],
                        color='blue',linewidth=1,label='valid_data',
                    )


            fig.text(0, 0, f'best ind: [{str.join(",", [str(x) for x in self._best_result])}], best fitness: {self._best_result_fitness}')

            model: esn.EsnForecaster = self._esn_creator(self._best_result)
            model.fit(self._fit_data)

            ind_color = evo_utils.get_next_color(exclude=['blue'])

            predict_data = evo_utils.get_predict_data(
                model,
                self._evaluate_cfg,
                self._valid_data.shape,
            )

            for i in range(dim):
                if self._evaluate_cfg.FitStep > 0:
                    axes[i].plot([x for x in range(fit_len * self._evaluate_cfg.FitStep, fit_len * self._evaluate_cfg.FitStep + valid_len)], predict_data[:,i],
                        color=ind_color,linewidth=1,label=f'best_predict_data',
                    )
                else:
                    axes[i].plot([x for x in range(fit_len, fit_len + valid_len)], predict_data[:,i],
                        color=ind_color,linewidth=1,label=f'best_predict_data',
                    )

                # setup labels
                axes[i].set_xlabel('time')
                axes[i].set_ylabel(f'dim_{i}')
                axes[i].legend(loc='upper left')

            fig.savefig(f'{self._iter_dir}/graph_best.png', dpi=fig.dpi)

        super().save(**kvargs)

    def close(self) -> None:
        'dummy'
        # self._shm_fit.close()
        # self._shm_valid.close()
