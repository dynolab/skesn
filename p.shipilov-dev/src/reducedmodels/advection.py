import numpy as np

from .dynamical_systems import DynamicalSystem


class LinearAdvectionModel(DynamicalSystem):
    def __init__(self, x_dim, delta_x, U):
        """
        This class sets up the linear advection model:
        du/dt = -U * du/dx
        Periodic boundary conditions are assumed.

        :param x_dim: number of grid points in the x-direction
        :param delta_x: grid point spacing
        :param U: advection speed
        """
        self.x_dim = x_dim
        self.delta_x = delta_x
        self.U = U
        super().__init__(self.x_dim)

    def f(self, u):
        f_ = np.zeros(self.dim)
        for i in range(self.x_dim):
            f_[i] = - self.U/self.delta_x * (u[i] - u[i-1])
        return f_


class NonlinearAdvectionModel(DynamicalSystem):
    def __init__(self, x_dim, delta_x):
        """
        This class sets up the linear advection model:
        du/dt = -u * du/dx
        Periodic boundary conditions are assumed.

        :param x_dim: number of grid points in the x-direction
        :param delta_x: grid point spacing
        :param U: advection speed
        """
        self.x_dim = x_dim
        self.delta_x = delta_x
        super().__init__(self.x_dim)

    def f(self, u):
        f_ = np.zeros(self.dim)
        for i in range(self.x_dim):
            f_[i] = - u[i]/self.delta_x * (u[i] - u[i-1])
        return f_
