from skesn.esn import EsnForecaster

import numpy as np

def rmse(esn: EsnForecaster, true_data: np.ndarray, steps: int = 1) -> np.ndarray:
    esn.fit()
    pass

def mape(esn: EsnForecaster, true_data: np.ndarray, steps: int = 1) -> np.ndarray:
    pass

def mase(esn: EsnForecaster, true_data: np.ndarray, steps: int = 1) -> np.ndarray:
    pass
