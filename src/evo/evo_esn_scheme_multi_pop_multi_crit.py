
import logging

import yaml
import src.dump as dump
import src.config as cfg
import src.evo.utils as evo_utils
import src.evo.types as evo_types
import src.config as cfg


from src.evo.graph_callback import GraphCallbackModule
from src.evo.evo_scheme_multi_pop import EvoSchemeMultiPop
from src.evo.esn_data_holder import EsnMultiDataHolder

import numpy as np
import skesn.esn as esn
import matplotlib.pyplot as plt

from types import FunctionType
from typing import List, Tuple

class EvoEsnSchemeMultiPopMultiCrit(EvoSchemeMultiPop):
    def __init__(self,
        name: str,
        evo_cfg: cfg.EvoSchemeMultiPopConfigField,
        esn_cfg: cfg.EsnConfigField,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        data_holder: EsnMultiDataHolder,
        esn_creator_by_ind_f: FunctionType,
        ind_creator_f: FunctionType=None,
        graph_callback_module: GraphCallbackModule=None,
    ) -> None:
        # Init configs
        self._esn_cfg: cfg.EsnConfigField = esn_cfg
        self._evaluate_cfg: cfg.EsnEvaluateConfigField = evaluate_cfg

        # Init graphics
        self._graph_callback_module: GraphCallbackModule = graph_callback_module

        # Init callbacks
        self._esn_creator_by_ind_f: FunctionType = esn_creator_by_ind_f

        # TODO :
        old_lorenz_seed = cfg.Config.Models.Lorenz.RandSeed
        models_cnt = len(evo_cfg.FitnessWeights)
        models = [None] * models_cnt
        for i in range(models_cnt):
            models[i] = evo_utils.create_model_by_type(self._evaluate_cfg.Model)
            cfg.Config.Models.Lorenz.RandSeed += 1
        cfg.Config.Models.Lorenz.RandSeed = old_lorenz_seed

        # Init train data
        self._data_holder = EsnMultiDataHolder(
            models,
            self._evaluate_cfg.SplitN,
            self._evaluate_cfg.FitStep,
            self._evaluate_cfg.Normalize,
        )
        crit_cnt = len(self._data_holder.Models)
        self._fit_datas = [self._data_holder.FitDataByN(i) for i in range(crit_cnt)]
        self._valid_datas = [self._data_holder.ValidDataByN(i) for i in range(crit_cnt)]

        super().__init__(
            name=name,
            evo_cfg=self._evo_cfg,
            evaluator=evo_types.EsnEvaluator(
                self._evaluate_cfg,
                ind_creator_f
            ),
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
            best_ind, _ = evo_utils.calculate_best_ind(self._best_result)

            esn_model: esn.EsnForecaster = self._esn_creator(best_ind)
            # esn_model.fit(self._fit_datas[0])

            ind_color = evo_utils.get_next_color(exclude=['blue'])

            for i, model in enumerate(self._data_holder.Models):
                dim = model.get_dim()

                fig, axes = plt.subplots(dim, figsize=(16,10))

                fit_len = len(self._fit_datas[i])
                valid_len = len(self._valid_datas[i])

                for j in range(dim):
                    # show fit data
                    if self._evaluate_cfg.FitStep > 0:
                        axes[j].plot([x for x in range(0, fit_len, self._evaluate_cfg.FitStep)], self._fit_datas[i][:,j],
                            color='blue',linestyle='dashed',linewidth=1,label='fit_data',
                        )
                    else:
                        axes[j].plot([x for x in range(fit_len)], self._fit_datas[i][:,j],
                            color='blue',linestyle='dashed',linewidth=1,label='fit_data',
                        )

                    # show valid data
                    axes[j].plot([x for x in range(fit_len, fit_len + valid_len)], self._valid_datas[i][:,j],
                        color='blue',linewidth=1,label='valid_data',
                    )

                predict_data = evo_utils.get_fit_predict_data(
                    esn_model,
                    self._evaluate_cfg,
                    self._fit_datas[i],
                    self._valid_datas[i],
                )

                crit_val = evo_utils.calc_metric(
                    self._evaluate_cfg.Metric,
                    self._valid_datas[i],
                    predict_data,
                )

                fig.text(0, 0, f'best ind: [{str.join(",", [str(x) for x in best_ind])}], best fitness: {crit_val * self._evo_cfg.FitnessWeights[i]}')

                for j in range(dim):
                    axes[j].plot([x for x in range(fit_len, fit_len + valid_len)], predict_data[:,j],
                        color=ind_color,linewidth=1,label=f'best_predict_data',
                    )

                    # setup labels
                    axes[j].set_xlabel('time')
                    axes[j].set_ylabel(f'dim_{j}')
                    axes[j].legend(loc='upper left')

                fig.savefig(f'{self._iter_dir}/graph_best_{i}.png', dpi=fig.dpi)

        super().save(**kvargs)
