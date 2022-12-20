import numpy as np
from scipy.interpolate import interp1d

class ToNormalConverter():
    def _reset(self):
        if hasattr(self, "norm_to_uni_"):
            del self.norm_to_uni_
            del self.uni_to_norm_
            del self.data_to_uni_
            del self.uni_to_data_

    def _get_FV(self, D, count=100):
        Cd = D.size
        Vd = np.linspace(np.min(D), np.max(D), count-2)
        Fd = np.array([np.count_nonzero(D < v)/Cd for v in Vd])

        Vd = np.concatenate(([-1000], Vd, [1000]))
        Fd = np.concatenate(([0], Fd, [1]))

        return Vd, Fd

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        G = np.random.normal(0, 1, (5000,))
        D = X - X.min()
        Vd, Fd = self._get_FV(G, 1000)
        self.norm_to_uni_ = interp1d(Vd, Fd, "linear")
        self.uni_to_norm_ = interp1d(Fd, Vd, "linear")
        Vd, Fd = self._get_FV(D, 1000)
        self.data_to_uni_ = interp1d(Vd, Fd, "linear")
        self.uni_to_data_ = interp1d(Fd, Vd, "linear")

        return self
        
    def transform(self, X):
        X = X.clip(self.data_to_uni_.x[0], self.data_to_uni_.x[-1])
        X = self.data_to_uni_(X)
        X = X.clip(self.uni_to_norm_.x[1], self.uni_to_norm_.x[-1])
        X = self.uni_to_norm_(X)
        # X = self.uni_to_norm_(self.data_to_uni_(X))
        return X
        
    def inverse_transform(self, X):
        X = X.clip(self.norm_to_uni_.x[0], self.norm_to_uni_.x[-1])
        X = self.norm_to_uni_(X)
        X = X.clip(self.uni_to_data_.x[0], self.uni_to_data_.x[-1])
        X = self.uni_to_data_(X)
        # X = self.uni_to_data_(self.norm_to_uni_(X))
        return X
