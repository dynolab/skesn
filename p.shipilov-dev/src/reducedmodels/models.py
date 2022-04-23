from abc import ABC, abstractmethod

import numpy as np

class DynamicalSystem(ABC):
    def __init__(self, dim):
        """
        This class is an interface for any dynamical system described as
        \frac{du}{dt} = f(u)
        or
        u_{n+1} = f(u_n)
        :param dim: dimension of the model
        """
        self.dim = dim

    @abstractmethod
    def f(self, x):
        raise NotImplementedError('Must be implemented')


class LinearAdvectionModel(DynamicalSystem):
    def __init__(self, x_dim, delta_x, U):
        """
        This class sets up the linear advection model:
        \frac{du}{dt} = -U * \frac{du}{dx}
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
        \frac{du}{dt} = -u * \frac{du}{dx}
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


class MoehlisFaisstEckhardtModel(DynamicalSystem):
    def __init__(self, Re, L_x, L_z):
        """
        This class sets up the shear-flow model from Moehlis et al. 2004. Namely, it is the right-hand side of the
        amplitude equations

        :param Re: Reynolds number
        :param L_x: domain wavelength in the x-direction
        :param L_y: domain wavelength in the y-direction
        """
        self.Re = float(Re)
        self.L_x = float(L_x)
        self.L_z = float(L_z)
        self.alpha = 2.*np.pi/L_x
        self.beta = np.pi/2.
        self.gamma = 2.*np.pi/L_z
        self.N_8 = 2.*np.sqrt(2) / np.sqrt((self.alpha**2 + self.gamma**2) * (4.*self.alpha**2 + 4.*self.gamma**2 + np.pi**2))
        self.k_ag = np.sqrt(self.alpha**2 + self.gamma**2)
        self.k_bg = np.sqrt(self.beta**2 + self.gamma**2)
        self.k_abg = np.sqrt(self.alpha**2 + self.beta**2 + self.gamma**2)
        super().__init__(9)

    def f(self, u):
        """
        In the paper, amplitudes, denoted a_1, a_2, etc., are enumerated from 1 to 9. Here they are denoted u[0], u[1],
        etc. and enumerated from 0 to 8.
        """
        f_ = np.zeros((self.dim,))
        f_[0] =   self.beta**2/self.Re \
                - self.beta**2/self.Re * u[0]\
                - np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[5] * u[7] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_bg * u[1] * u[2]

        f_[1] = - (4.*self.beta**2/3. + self.gamma**2)/self.Re * u[1] \
                + (5*np.sqrt(2))/(3*np.sqrt(3))*self.gamma**2/self.k_ag * u[3] * u[5] \
                - self.gamma**2/(np.sqrt(6)*self.k_ag) * u[4] * u[6] \
                - self.alpha*self.beta*self.gamma/(np.sqrt(6)*self.k_ag*self.k_abg) * u[4] * u[7] \
                - np.sqrt(3./2)*self.beta*self.gamma/self.k_bg * u[0] * u[2] \
                - np.sqrt(3./2)*self.beta*self.gamma/self.k_bg * u[2] * u[8]

        f_[2] = - (self.beta**2 + self.gamma**2)/self.Re * u[2] \
                + (2./np.sqrt(6.)) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * (u[3]*u[6] + u[4]*u[5]) \
                + (self.beta**2*(3*self.alpha**2 + self.gamma**2) - 3*self.gamma**2*(self.alpha**2 + self.gamma**2))/(np.sqrt(6)*self.k_ag*self.k_abg*self.k_bg) * u[3]*u[7]

        f_[3] = - (3*self.alpha**2 + 4*self.beta**2)/(3*self.Re) * u[3] \
                - self.alpha/np.sqrt(6) * u[0]*u[4] \
                - 10./(3.*np.sqrt(6)) * self.alpha**2/self.k_ag * u[1]*u[5] \
                - np.sqrt(3./2) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[6] \
                - np.sqrt(3./2) * self.alpha**2*self.beta**2/(self.k_abg*self.k_ag*self.k_bg) * u[2]*u[7] \
                - self.alpha/np.sqrt(6.) * u[4]*u[8]

        f_[4] = - (self.alpha**2 + self.beta**2)/self.Re * u[4] \
                + self.alpha/np.sqrt(6) * u[0]*u[3] \
                + self.alpha**2/(np.sqrt(6)*self.k_ag) * u[1]*u[6] \
                - self.alpha*self.beta*self.gamma/(np.sqrt(6)*self.k_abg*self.k_ag) * u[1]*u[7] \
                + self.alpha/np.sqrt(6) * u[3]*u[8] \
                + 2./np.sqrt(6.) * self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[5]

        f_[5] = - (3*self.alpha**2 + 4*self.beta**2 + 3*self.gamma**2)/(3*self.Re) * u[5] \
                + self.alpha/np.sqrt(6) * u[0]*u[6] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[0]*u[7] \
                + 10./(3.*np.sqrt(6.)) * (self.alpha**2 - self.gamma**2)/self.k_ag * u[1]*u[3] \
                - 2.*np.sqrt(2./3)*self.alpha*self.beta*self.gamma/(self.k_ag*self.k_bg) * u[2]*u[4] \
                + self.alpha/np.sqrt(6) * u[6]*u[8] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[7]*u[8]

        f_[6] = - (self.alpha**2 + self.beta**2 + self.gamma**2)/self.Re * u[6] \
                - self.alpha/np.sqrt(6) * (u[0]*u[5] + u[5]*u[8]) \
                + 1./np.sqrt(6) * (self.gamma**2-self.alpha**2)/self.k_ag * u[1]*u[4] \
                + 1./np.sqrt(6) * (self.alpha*self.beta*self.gamma)/(self.k_ag*self.k_bg) * u[2]*u[3]

        f_[7] = - (self.alpha**2 + self.beta**2 + self.gamma**2)/self.Re * u[7] \
                + 2./np.sqrt(6.) * (self.alpha*self.beta*self.gamma)/(self.k_abg*self.k_ag) * u[1]*u[4] \
                + self.gamma**2*(3*self.alpha**2 - self.beta**2 + 3*self.gamma**2)/(np.sqrt(6)*self.k_ag*self.k_abg*self.k_bg) * u[2]*u[3]

        f_[8] = - 9*self.beta**2/self.Re * u[8] \
                + np.sqrt(3./2) * self.beta*self.gamma/self.k_bg * u[1]*u[2] \
                - np.sqrt(3./2) * self.beta*self.gamma/self.k_abg * u[5]*u[7]

        return f_

    def modes(self, x, y, z):
        """
        In the paper, modes are enumerated from 1 to 9. Here from 0 to 8.

        :return:
        """
        u = np.zeros((self.dim, 3))
        cos_pi_y = np.cos(np.pi * self.gamma / 2.)
        sin_pi_y = np.sin(np.pi * self.gamma / 2.)
        cos_g_z = np.cos(self.gamma * z)
        sin_g_z = np.sin(self.gamma * z)
        cos_a_x = np.cos(self.alpha * x)
        sin_a_x = np.sin(self.alpha * x)
        u[0, :] = np.array([np.sqrt(2)*sin_pi_y, 0, 0])
        u[1, :] = np.array([4./np.sqrt(3)*cos_pi_y**2 * cos_g_z, 0, 0])
        u[2, :] = 2./np.sqrt(4.*self.gamma**2 + np.pi**2) * np.array([0, 2.*self.gamma*cos_pi_y * cos_g_z, np.pi * sin_pi_y * sin_g_z])
        u[3, :] = np.array([0, 0, 4./np.sqrt(3)*cos_a_x * cos_pi_y**2])
        u[4, :] = np.array([0, 0, 2.*sin_a_x * sin_pi_y])
        u[5, :] = 4.* np.sqrt(2)/np.sqrt(3.*(self.alpha**2 + self.gamma**2)) * np.array([-self.gamma*cos_a_x*cos_pi_y**2*sin_g_z, 0, self.alpha*sin_a_x*cos_pi_y**2*cos_g_z])
        u[6, :] = 2.* np.sqrt(2)/np.sqrt(self.alpha**2 + self.gamma**2) * np.array([self.gamma*sin_a_x*sin_pi_y*sin_g_z, 0, self.alpha*cos_a_x*sin_pi_y*cos_g_z])
        u[7, :] = self.N_8 * np.array([np.pi*sin_a_x*sin_pi_y*sin_g_z, 2.*(self.alpha**2 + self.gamma**2)*cos_a_x*cos_pi_y*sin_g_z, -np.pi*self.gamma*cos_a_x*sin_pi_y*cos_g_z])
        u[8, :] = np.array([np.sqrt(2)*np.sin(3.*np.pi*y/2.), 0, 0])


class BarkleyPipeModel(DynamicalSystem):
    def __init__(self, x_dim, delta_x, r, zeta, D, U_0, U_bar, delta, epsilon_1, epsilon_2, sigma):
        """
        This class sets up the Barkley's pipe model as described in Barkley, 2016.
        Periodic boundary conditions are assumed.

        :param x_dim:
        :param delta_x:
        :param r:
        :param zeta:
        :param D:
        :param U_0:
        :param U_bar:
        :param delta:
        :param epsilon_1:
        :param epsilon_2:
        :param sigma:
        """
        self.x_dim = x_dim
        self.delta_x = delta_x
        self.r = r
        self.zeta = zeta
        self.D = D
        self.U_0 = U_0
        self.U_bar = U_bar
        self.delta = delta
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.sigma = sigma
        super().__init__(2*self.x_dim)

    def f(self, u):
        f_ = np.zeros(self.dim)
        for i in range(self.x_dim):
            f_[i] = - 1./self.delta_x * (self._u(u, i) - self.zeta) * (self._q(u, i) - self._q(u, i-1)) \
                    + self._f_local(u, i) \
                    + self.D / self.delta_x**2 * (self._q(u, i-1) - 2*self._q(u, i) + self._q(u, i+1))
            f_[self.x_dim + i] = - 1./self.delta_x * self._u(u, i) * (self._u(u, i) - self._u(u, i-1)) \
                               + self._g_local(u, i)
        return f_

    def _q(self, u, i):
        return u[i]

    def _u(self, u, i):
        return u[self.x_dim + i]

    def _f_local(self, u, i):
        return self._q(u, i)*(self.r + self._u(u, i) - self.U_0 - (self.r + self.delta)*(self._q(u, i) - 1.)**2)

    def _g_local(self, u, i):
        return self.epsilon_1 * (self.U_0 - self._u(u, i)) \
             + self.epsilon_2 * (self.U_bar - self._u(u, i)) * self._q(u, i)


def euler_timestepping(model: DynamicalSystem, ic, delta_t, n_steps, time_skip=1, space_skip=1):
    timeseries = np.zeros((int(n_steps//time_skip) + 1, int(model.dim//space_skip)))
    cur_state = ic
    for k in range(n_steps):
        if k % time_skip == 0:
            timeseries[int(k//time_skip), :] = cur_state[::space_skip]
        if k % int(n_steps//10) == 0:
            print('Step {} out of {}'.format(k, n_steps))
        cur_state = cur_state + delta_t * model.f(cur_state)
        #timeseries[k + 1, :] = timeseries[k, :] + delta_t * model.f(timeseries[k, :])
    return timeseries


def rk4_timestepping(model: DynamicalSystem, ic, delta_t, n_steps, time_skip=1, space_skip=1, debug=True, stop_condition=lambda state: False):
    timeseries = np.zeros((int(n_steps//time_skip) + 1, int(model.dim//space_skip)))
    cur_state = ic
    for k in range(n_steps):
        if k % time_skip == 0:
            timeseries[int(k//time_skip), :] = cur_state[::space_skip]
        if (k % int(n_steps//10) == 0) and debug:
            print('Step {} out of {}'.format(k, n_steps))
        k_1 = delta_t*model.f(cur_state)
        k_2 = delta_t*model.f(cur_state + k_1/2.)
        k_3 = delta_t*model.f(cur_state + k_2/2.)
        k_4 = delta_t*model.f(cur_state + k_3)
        cur_state = cur_state + 1/6. * (k_1 + 2*k_2 + 2*k_3 + k_4)
        if stop_condition(cur_state):
            return timeseries[:int(k//time_skip) + 1, :]
    return timeseries


def make_input_output_for_training_out_of_timeseries(timeseries, skip=1):
    input = timeseries[:-1, ::skip]
    output = timeseries[1:, ::skip]
    return input, output


def split_into_training_and_test_sets(input, output, chunk_number, ):
    n_training = int(round((1. - test_fraction) * input.shape[0]))
    n_test = int(round(test_fraction * input.shape[0]))
    print(test_fraction)
    input_training = input[0:n_training]
    output_training = output[0:n_training]
    input_test = input[n_training:n_training + n_test]
    output_test = output[n_training:n_training + n_test]
    return input_training, output_training, input_test, output_test


#def dtLz (lst, Pr, Ra, beta ):
#
#    out = np.zeros(3)
#
#    out[0] = Pr * ( lst[1] - lst[0] )
#    out[1] = Ra * lst[0] - lst[0] * lst[2] - lst[1]
#    out[2] = lst[0] * lst[1] - beta * lst[2]
#
#    return out


#def model (nsave, totD, dt, Pr, Ra, beta, ic):
#
#    rI = np.zeros( (nsave, totD) )
#    rO = np.zeros( (nsave, totD) )
#
#    ans = ic
#    for k in range (nsave):
#
#        rI[k] = ans
#        md = dtLz (ans, Pr, Ra, beta)
#        ans = ans + dt * md
#        rO[k] = ans
#
#    return rI, rO
