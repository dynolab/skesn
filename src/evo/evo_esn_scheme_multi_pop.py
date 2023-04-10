
import logging

import yaml
import src.dump as dump
import src.config as cfg
import src.evo.utils as evo_utils

from src.evo.graph_callback import GraphCallbackModule
from src.evo.evo_scheme_multi_pop import EvoSchemeMultiPop
from src.evo.esn_data_holder import EsnDataHolder

import numpy as np
import skesn.esn as esn
import matplotlib.pyplot as plt

from types import FunctionType
from typing import List, Tuple

class EvoEsnSchemeMultiPop(EvoSchemeMultiPop):
    def __init__(self,
        name: str,
        evo_cfg: cfg.EvoSchemeMultiPopConfigField,
        esn_cfg: cfg.EsnConfigField,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        data_holder: EsnDataHolder,
        esn_creator_by_ind_f: FunctionType,
        ind_creator_f: FunctionType=None,
        graph_callback_module: GraphCallbackModule=None,
    ) -> None:
        # Init configs
        self._esn_cfg: cfg.EsnConfigField = esn_cfg
        self._evo_cfg: cfg.EvoSchemeMultiPopConfigField = evo_cfg
        self._evaluate_cfg: cfg.EsnEvaluateConfigField = evaluate_cfg

        # Init graphics
        self._graph_callback_module: GraphCallbackModule = graph_callback_module

        # Init callbacks
        self._esn_creator_by_ind_f: FunctionType = esn_creator_by_ind_f

        # Init train data
        self._data_holder: EsnDataHolder = data_holder
        self._fit_data = self._data_holder.FitData
        self._valid_data = self._data_holder.ValidData

        super().__init__(
            name=name,
            evo_cfg=self._evo_cfg,
            evaluate_f=self._evaluate_esn,
            ind_creator_f=ind_creator_f,
            graph_callback_module=graph_callback_module,
        )

    def save(self,
        **kvargs,
    ) -> str:
        self._iter_dir = evo_utils.get_or_create_new_iter_dir(
            runpool_dir=self._runpool_dir,
            iter_dir=self._iter_dir,
        )

        # if not kvargs.get('disable_dump_model', False):
        #     dump.do_np_arr(
        #         dump_dir=self._iter_dir,
        #         name='fit_data.yaml',
        #         data=self._fit_data,
        #     )
        #     dump.do_np_arr(
        #         dump_dir=self._iter_dir,
        #         name='valid_data.yaml',
        #         data=self._valid_data,
        #     )

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


            best_ind, best_ind_fitness = evo_utils.calculate_best_ind(self._best_result)
            fig.text(0, 0, f'best ind: [{str.join(",", [str(x) for x in best_ind])}], best fitness: {best_ind_fitness}')

            model: esn.EsnForecaster = self._esn_creator_by_ind_f(best_ind)
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

            # for i, best_ind in enumerate(self._best_result):
            #     model: esn.EsnForecaster = self._esn_creator_by_ind_f(best_ind)
            #     model.fit(self._fit_data)

            #     ind_color = evo_utils.get_next_color(exclude=['blue'])

            #     predict_data = evo_utils.get_predict_data(
            #         model,
            #         self._evaluate_cfg,
            #         self._valid_data.shape,
            #     )

            #     for j in range(dim):
            #         axes[j].plot([x for x in range(fit_len, fit_len + valid_len)], predict_data[:,j],
            #         color=ind_color,linewidth=1,label=f'predict_data_{i}',
            #     )

            # for i in range(dim):
            #     # setup labels
            #     axes[i].set_xlabel('time')
            #     axes[i].set_ylabel(f'dim_{i}')
            #     axes[i].legend(loc='upper left')

            fig.savefig(f'{self._iter_dir}/graph_best.png', dpi=fig.dpi)

        super().save(**kvargs)

    def _evaluate_esn(self,
        ind: List,
    ) -> Tuple[float]:
        if ind.fitness.valid:
            return ind.fitness.values

        model: esn.EsnForecaster = self._esn_creator_by_ind_f(ind)
        model.fit(self._fit_data)

        predict_data = evo_utils.get_predict_data(
            model,
            self._evaluate_cfg,
            self._valid_data.shape,
        )

        return evo_utils.calc_metric(
            self._evaluate_cfg.Metric,
            self._valid_data,
            predict_data,
        ),

