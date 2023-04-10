from src.evo.graph_callback import GraphCallbackModule

import src.config as cfg
import src.evo.utils as evo_utils
import src.evo.types as evo_types

import matplotlib.pyplot as plt

from multiprocess.pool import Pool
from multiprocess.shared_memory import SharedMemory

from types import FunctionType
from typing import Any, Dict, List, Tuple, Union

import skesn.esn as esn

import numpy as np

from src.evo.evo_scheme import EvoScheme
from src.evo.esn_data_holder import EsnDataHolder


class EvoEsnScheme(EvoScheme):
    def __init__(self,
        name: str,
        evo_cfg: cfg.EvoSchemeConfigField,
        esn_cfg: cfg.EsnConfigField,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        esn_creator_by_ind_f: FunctionType,
        ind_creator_f: FunctionType=None,
        graph_callback_module: GraphCallbackModule=None,
        pool: Pool=None,
    ) -> None:
        # Init configs
        self._esn_cfg: cfg.EsnConfigField = esn_cfg
        self._evo_cfg: cfg.EvoSchemeConfigField = evo_cfg
        self._evaluate_cfg: cfg.EsnEvaluateConfigField = evaluate_cfg

        # Init graphics
        self._graph_callback_module: GraphCallbackModule = graph_callback_module

        # Init callback
        self._esn_creator_by_ind_f: FunctionType = esn_creator_by_ind_f

        # Init evaluate data
        self._model = evo_utils.create_model_by_type(evaluate_cfg.Model)
        self._data_holder: EsnDataHolder = EsnDataHolder(
            self._model,
            evaluate_cfg.SplitN,
            evaluate_cfg.FitStep,
            evaluate_cfg.Normalize,
        )

        self._fit_data: np.ndarray = self._data_holder.FitData
        self._shm_fit = SharedMemory(create=True, size=self._fit_data.nbytes)
        shm_fit_data = np.ndarray(self._fit_data.shape, self._fit_data.dtype, buffer=self._shm_fit.buf)
        np.copyto(shm_fit_data, self._fit_data)

        self._valid_data: np.ndarray = self._data_holder.ValidData
        self._shm_valid = SharedMemory(create=True, size=self._valid_data.nbytes)
        shm_valid_data = np.ndarray(self._valid_data.shape, self._valid_data.dtype, buffer=self._shm_valid.buf)
        np.copyto(shm_valid_data, self._valid_data)

        super().__init__(
            name=name,
            cfg=self._evo_cfg,
            evaluate_f=EvaluatorEsn(
                evaluate_cfg,
                esn_creator_by_ind_f,
                ShmArrayProvider(self._fit_data.nbytes, self._fit_data.dtype, self._fit_data.shape, self._shm_fit),
                ShmArrayProvider(self._valid_data.nbytes, self._valid_data.dtype, self._valid_data.shape, self._shm_valid)
            ),
            ind_creator_f=ind_creator_f,
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

            model: esn.EsnForecaster = self._esn_creator_by_ind_f(self._best_result)
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
        self._shm_fit.close()
        self._shm_valid.close()


class ShmArrayProvider(object):
    def __init__(self, nbytes: int, dtype, shape, shm: SharedMemory):
        self._shape = shape
        self._nbytes = nbytes
        self._dtype = dtype
        self._shm = shm

    @property
    def nbytes(self) -> int: return self._nbytes

    @property
    def shape(self) -> Union[int, Tuple[int]]: return self._shape

    @property
    def dtype(self): return self._dtype

    @property
    def shm(self): return self._shm


class EvaluatorEsn(object):
    def __init__(self,
            cfg: cfg.EsnEvaluateConfigField,
            esn_creator: FunctionType,
            shm_valid_data: ShmArrayProvider,
            shm_fit_data: ShmArrayProvider,
    ) -> None:
        self._evaluate_cfg = cfg
        self._esn_creator = esn_creator
        self._shm_valid_data = shm_valid_data
        self._shm_fit_data = shm_fit_data

        self._fit_data = self.fit_data
        self._valid_data = self.valid_data

    @property
    def valid_data(self) -> np.ndarray: return np.ndarray(self._shm_valid_data.shape, dtype=self._shm_valid_data.dtype, buffer=self._shm_valid_data.shm.buf)

    @property
    def fit_data(self) -> np.ndarray: return np.ndarray(self._shm_fit_data.shape, dtype=self._shm_fit_data.dtype, buffer=self._shm_fit_data.shm.buf)

    def __call__(self,
        ind: evo_types.Individual,
    ) -> Tuple[float]:
        if ind.fitness.valid:
            return ind.fitness.values

        model: esn.EsnForecaster = self._esn_creator(ind)
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

    def __getstate__(self) -> Dict[int, Any]:
        return {
            '_evaluate_cfg': self._evaluate_cfg,
            '_esn_creator': self._esn_creator,
            '_shm_valid_data': self._shm_valid_data,
            '_shm_fit_data': self._shm_fit_data,
        }

    def __setstate__(self, state: Dict[int, Any]) -> None:
        self._evaluate_cfg = state['_evaluate_cfg']
        self._esn_creator = state['_esn_creator']
        self._shm_valid_data = state['_shm_valid_data']
        self._shm_fit_data = state['_shm_fit_data']

        self._valid_data = self.valid_data
        self._fit_data = self.fit_data
