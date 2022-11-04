from src.evo.graph_callback import GraphCallbackModule
from src.models.abstract import Model
import src.config as cfg

from types import FunctionType
from typing import List, Tuple

import numpy as np
import skesn.esn as esn
import sklearn.metrics as metrics

from src.evo.evo_scheme import EvoScheme

class EsnDataHolder:
    @property
    def Model(self) -> Model: return self._model

    @property
    def FitData(self) -> np.ndarray: return self._fit_data

    @property
    def ValidData(self) -> np.ndarray: return self._valid_data

    def __init__(self,
        model: Model,
    ) -> None:
        self._model: Model = model
        self._setup_data_by_model()

    def _setup_data_by_model(self) -> None:
        data = self._model.gen_data()
        splitN = data.shape[0] // 2
        self._fit_data: np.ndarray = data[:splitN,]
        self._valid_data: np.ndarray = data[splitN:,:]

class EvoEsnScheme(EvoScheme):
    def __init__(self,
        name: str,
        evo_cfg: cfg.EvoSchemeConfigField,
        esn_cfg: cfg.EsnConfigField,
        evaluate_cfg: cfg.EsnEvaluateConfigField,
        data_holder: EsnDataHolder,
        esn_creator_by_ind_f: FunctionType,
        ind_creator_f: FunctionType=None,
        graph_callback_module: GraphCallbackModule=None,
    ) -> None:
        self._esn_cfg: cfg.EsnConfigField = esn_cfg
        self._evo_cfg: cfg.EvoSchemeConfigField = evo_cfg
        self._evaluate_cfg: cfg.EsnEvaluateConfigField = evaluate_cfg

        self._graph_callback_module: GraphCallbackModule = graph_callback_module
        self._data_holder: EsnDataHolder = data_holder
        self._esn_creator_by_ind_f: FunctionType = esn_creator_by_ind_f

        super().__init__(
            name=name,
            cfg=self._evo_cfg,
            evaluate_f=self._evaluate_esn,
            ind_creator_f=ind_creator_f,
            graph_callback_module=graph_callback_module,
        )

    def _evaluate_esn(self,
        ind: List,
    ) -> Tuple[float]:
        model: esn.EsnForecaster = self._esn_creator_by_ind_f(ind)
        fit_data = self._data_holder.Model.normalize(
            self._data_holder.FitData,
        )
        model.fit(fit_data)

        valid_data = self._data_holder.Model.normalize(
            self._data_holder.ValidData,
        )
        predict_data = np.ndarray(valid_data.shape)

        if self._evaluate_cfg.MaxSteps <= 0:
            return _calc_metric(
                self._evaluate_cfg.Metric,
                valid_data,
                model.predict(valid_data.shape[0]),
            ),

        i = 0
        max_i = valid_data.shape[0]
        n = self._evaluate_cfg.MaxSteps
        while i < max_i:
            if i + n >= max_i:
                n = max_i - i

            predict_local = model.predict(n)
            for j in range(n):
                predict_data[i+j,:] = predict_local[j,:]

            i += self._evaluate_cfg.MaxSteps

        return _calc_metric(
            self._evaluate_cfg.Metric,
            valid_data,
            predict_data,
        ),

def _rmse(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    return np.sqrt(np.sum((expected - actual)**2)/expected.shape[0])

def _map_evaluate_metric_func(
    metric: str,
) -> FunctionType:
    metric = metric.lower()

    if metric == 'rmse':
        return _rmse
    elif metric == 'mse':
        return metrics.mean_squared_error
    elif metric == 'mae':
        metrics.median_absolute_error
    raise 'unknown metric name'

def _calc_metric(
    metric: str,
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    if expected.size != actual.size:
        raise f'expected size is not equal actual (expected size: {expected.size}, actaul size: {actual.size})'

    fn = _map_evaluate_metric_func(metric)
    if fn is None:
        raise f'metric func is None (actaul: {metric})'

    return fn(expected, actual)
