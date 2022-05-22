from typing import Tuple
import tqdm
import numpy as np

from skesn.esn import EsnForecaster

from .config import Config
from .evo.utils import map_evaluate_f

_lorenz_cfg = Config.Grid.ParamsSet.Lorenz
LORENZ_PARAMS_SET: dict = {
    # 'n_reservoir':     np.linspace(_lorenz_cfg.NReservoir.Start, _lorenz_cfg.NReservoir.Stop, _lorenz_cfg.NReservoir.Num, dtype=np.int32),
    'spectral_radius': np.linspace(_lorenz_cfg.SpectralRadius.Start, _lorenz_cfg.SpectralRadius.Stop, _lorenz_cfg.SpectralRadius.Num),
    'sparsity':        np.linspace(_lorenz_cfg.Sparsity.Start, _lorenz_cfg.Sparsity.Stop, _lorenz_cfg.Sparsity.Num),
    'noise':           np.linspace(_lorenz_cfg.Noise.Start, _lorenz_cfg.Noise.Stop, _lorenz_cfg.Noise.Num),
    # 'lambda_r':        np.logspace(_lorenz_cfg.LambdaR.Start, _lorenz_cfg.LambdaR.Stop, _lorenz_cfg.LambdaR.Num),
}

def _ens_lenrenz_grid_search_impl(args, params_set: dict, score_f) -> dict:
    best_params = {
        # 'n_reservoir': int(params_set['n_reservoir'][0]),
        'spectral_radius': float(params_set['spectral_radius'][0]),
        'sparsity': float(params_set['sparsity'][0]),
        'noise': float(params_set['noise'][0]),
        # 'lambda_r': float(params_set['lambda_r'][0]),
    }
    best_score = 1e10

    max_iters = 0
    pbar = None
    if Config.Grid.Verbose:
         # len(params_set['n_reservoir']) * \
        max_iters = len(params_set['spectral_radius']) * \
            len(params_set['sparsity']) * \
            len(params_set['noise']) # * \
            # len(params_set['lambda_r'])
        pbar = tqdm.tqdm(total=max_iters,position=0, leave=True)

    # for n_reservoir in params_set['n_reservoir']:
    #     for spectral_radius in params_set['spectral_radius']:
    #         for sparsity in params_set['sparsity']:
    #             for noise in params_set['noise']:
    #                 for lambda_r in params_set['lambda_r']:
    for spectral_radius in params_set['spectral_radius']:
        for sparsity in params_set['sparsity']:
            for noise in params_set['noise']:
                model = EsnForecaster(
                    Config.Esn.NInputs,
                    n_reservoir=Config.Esn.NReservoir,
                    spectral_radius=spectral_radius,
                    sparsity=sparsity,
                    noise=noise,
                    lambda_r=Config.Esn.LambdaR,
                    random_state=Config.Esn.RandomState,
                )
                score, = score_f(model)
                if score < best_score:
                    best_score = score
                    best_params = {
                        # 'n_reservoir': int(n_reservoir),
                        'spectral_radius': float(spectral_radius),
                        'sparsity': float(sparsity),
                        'noise': float(noise),
                        # 'lambda_r': float(lambda_r),
                    }
                if Config.Grid.Verbose:
                    pbar.update(1)

    if Config.Grid.Verbose:
        pbar.close()

    return {
        'params': best_params,
        'score': float(best_score),
    }

def esn_lorenz_grid_search(args):
    return _ens_lenrenz_grid_search_impl(
        args,
        LORENZ_PARAMS_SET,
        map_evaluate_f(Config.Grid.Scoring, Config.Grid.ValidMultiN),
    )
