from cmath import e
from types import FunctionType
from typing import Any, Dict, Iterable, List, Tuple
from deap import base

import numpy as np

import skesn.esn as esn
from src.evo.esn_data_holder import EsnDataHolder

import src.config as cfg
import src.evo.utils as evo_utils

class Fitness(base.Fitness):
    def __init__(self, values: Iterable[float]=()):
        super().__init__(values)

    @staticmethod
    def patch_weights(weights: Iterable[float]):
        Fitness.weights = weights

class Individual(list):
    def __init__(self, iterable):
        self.fitness = Fitness()
        super().__init__(iterable)


evaluate_cfg = None
esn_creator = None
fit_data = None
valid_data = None

def esn_pool_init(
    creator: FunctionType,
    eval_cfg: cfg.EsnEvaluateConfigField,
    data_holder: EsnDataHolder,
) -> None:
    global evaluate_cfg, esn_creator, fit_data, valid_data
    evaluate_cfg = eval_cfg
    esn_creator = creator
    fit_data = data_holder.FitData
    valid_data = data_holder.ValidData


class EsnEvaluator(object):
    def __init__(self,
            _evaluate_cfg: cfg.EsnEvaluateConfigField=None,
            _esn_creator: FunctionType=None,
            _fit_data: np.ndarray=None,
            _valid_data: np.ndarray=None,
    ) -> None:
        global evaluate_cfg, esn_creator, fit_data, valid_data
        self._evaluate_cfg = evaluate_cfg = _evaluate_cfg
        self._esn_creator = esn_creator = _esn_creator

        self._fit_data = fit_data = _fit_data
        self._valid_data = fit_data = _valid_data

    def __call__(self,
        ind: Individual,
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
        return {}

    def __setstate__(self, _: Dict[int, Any]) -> None:
        self._evaluate_cfg = evaluate_cfg
        self._esn_creator = esn_creator

        self._fit_data = fit_data
        self._valid_data = valid_data


class HyperParamEsnCreatorByInd(object):
    SPECTRAL_RADIUS_IDX = 0
    SPARSITY_IDX = 1
    LAMBDA_R_IDX = 2
    REGULIRIZATION_IDX = 3
    USE_ADDITIVE_NOISE_WHEN_FORECASTING_IDX = 4
    USE_BIAS_IDX = 5

    def __init__(self, esn_cfg: cfg.EsnConfigField) -> None:
        self._cfg = esn_cfg

    def __call__(self, ind: Individual) -> esn.EsnForecaster:
        return esn.EsnForecaster(
            n_reservoir=self._cfg.NReservoir,
            spectral_radius=ind[HyperParamEsnCreatorByInd.SPECTRAL_RADIUS_IDX],
            sparsity=ind[HyperParamEsnCreatorByInd.SPARSITY_IDX],
            regularization=ind[HyperParamEsnCreatorByInd.REGULIRIZATION_IDX],
            lambda_r=ind[HyperParamEsnCreatorByInd.LAMBDA_R_IDX],
            in_activation=self._cfg.InActivation,
            out_activation=self._cfg.OutActivation,
            use_additive_noise_when_forecasting=ind[HyperParamEsnCreatorByInd.USE_ADDITIVE_NOISE_WHEN_FORECASTING_IDX],
            random_state=self._cfg.RandomState,
            use_bias=ind[HyperParamEsnCreatorByInd.USE_BIAS_IDX],
        )
