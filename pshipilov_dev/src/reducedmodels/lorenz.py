import numpy as np

from .dynamical_systems import DynamicalSystem


class LorenzModel(DynamicalSystem):
    def __init__(self, Pr, Ra, beta):
        self.Pr = Pr
        self.Ra = Ra
        self.beta = beta
        super().__init__(3)

    def f(self, u):
        f_ = np.zeros(3)
        f_[0] = self.Pr * (u[1] - u[0])
        f_[1] = self.Ra * u[0] - u[0] * u[2] - u[1]
        f_[2] = u[0] * u[1] - self.beta * u[2]
        return f_


class Lorenz96Model(DynamicalSystem):
    def __init__(self, dim, forcing):
        """
        :param dim: dimension of the model
        :param forcing: forcing term
        """
        self.forcing = forcing
        super().__init__(dim)

    def f(self, u):
        f_ = np.zeros((self.dim,))
        f_[0] = (u[1] - u[self.dim - 2]) * u[self.dim - 1] - u[0] + self.forcing
        f_[1] = (u[2] - u[self.dim - 1]) * u[0] - u[1] + self.forcing
        f_[self.dim - 1] = (u[0] - u[self.dim - 3]) * u[self.dim - 2] - u[self.dim - 1] + self.forcing
        for i in range(2, self.dim - 1):
            f_[i] = (u[i + 1] - u[i - 2]) * u[i - 1] - u[i] + self.forcing
        return f_
