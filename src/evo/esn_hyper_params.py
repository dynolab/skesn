import utils
import config

from src.evo.evo_esn_scheme import EvoEsnScheme

from skesn.esn import EsnForecaster
from deap import base
from deap import creator

class HyperParamInfo:
    def __init__(self, key: str, min, max, is_int: bool=False) -> None:
        self.idx = None
        self.key = key
        self.min = min
        self.max = max
        self.is_int = is_int

class EvoSchemeEsnHyperParams(EvoEsnScheme):
    def __init__(self,
        name: str,
        cfg: config.EvoSchemeConfigField,
        esn_cfg: config.EsnConfigField,
        ) -> None:

        super().__init__(
            name,
            cfg,
            esn_cfg,
            utils.wrap_esn_evaluate_f(self._get_esn),
        )

    # Internal methods

    def _setup_hyper_params(self):
        # self._n_reservoir = HyperParamInfo('n_reservoir', self._cfg.Limits.NReservoir.Min, self._cfg.Limits.NReservoir.Max, self._cfg.Limits.NReservoir.IsInt)
        self._spectral_radius = HyperParamInfo('spectral_radius', self._cfg.Limits.SpectralRadius.Min, self._cfg.Limits.SpectralRadius.Max, self._cfg.Limits.SpectralRadius.IsInt)
        self._sparsity = HyperParamInfo('sparsity', self._cfg.Limits.Sparsity.Min, self._cfg.Limits.Sparsity.Max, self._cfg.Limits.Sparsity.IsInt)
        self._noise = HyperParamInfo('noise', self._cfg.Limits.Noise.Min, self._cfg.Limits.Noise.Max, self._cfg.Limits.Noise.IsInt)
        # self._lambda_r = HyperParamInfo('lambda_r', self._cfg.Limits.LambdaR.Min, self._cfg.Limits.LambdaR.Max, self._cfg.Limits.LambdaR.IsInt)
        self._hyper_params = (
            # self._n_reservoir,
            self._spectral_radius,
            self._sparsity,
            self._noise,
            # self._lambda_r,
        )
        for i, param in enumerate(self._hyper_params):
            param.idx = i

    # DEAP

    def _mutInd(self, ind: list, indpb: float):
        for i in range(len(ind)):
            if indpb < self._rand.random():
                continue

            if self._hyper_params[i].is_int:
                ind[i] = self._rand.randint(self._hyper_params[i].min, self._hyper_params[i].max)
            else:
                ind[i] = self._rand.uniform(self._hyper_params[i].min, self._hyper_params[i].max)
        return ind,

    def _get_esn(self, ind: list) -> EsnForecaster:
        return EsnForecaster(
            n_inputs=self._esn_cfg.NInputs,
            lambda_r=self._esn_cfg.LambdaR,
            random_state=self._esn_cfg.RandomState,
            n_reservoir=self._esn_cfg.NReservoir,
            lambda_r=self._esn_cfg.LambdaR,
            # n_reservoir=ind[self._n_reservoir.idx],
            spectral_radius=ind[self._spectral_radius.idx],
            sparsity=ind[self._sparsity.idx],
            noise=ind[self._noise.idx],
            # lambda_r=ind[self._lambda_r.idx],
        )
