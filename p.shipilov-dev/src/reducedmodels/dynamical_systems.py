from abc import ABC, abstractmethod

class DynamicalSystem(ABC):
    def __init__(self, dim):
        """
        This class is an interface for any dynamical system described as
        du/dt = f(u)
        or
        u_{n+1} = f(u_n)
        :param dim: dimension of the model
        """
        self.dim = dim

    @abstractmethod
    def f(self, x):
        raise NotImplementedError('Must be implemented')
