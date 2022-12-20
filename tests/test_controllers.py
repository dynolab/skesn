import unittest

import numpy as np

from skesn.esn import EsnForecaster, update_modes
from skesn.weight_generators import optimal_weights_generator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import isclose
from skesn.esn_controllers import HomotopyController, InjectedController

def _lorenz(x_0, dt, t_final):
    sigma_ = 10.
    beta_ = 8./3.
    rho_ = 28.

    def rhs(x):
        f_ = np.zeros(3)
        f_[0] = sigma_ * (x[1] - x[0])
        f_[1] = rho_ * x[0] - x[0] * x[2] - x[1]
        f_[2] = x[0] * x[1] - beta_ * x[2]
        return f_

    times = np.arange(0, t_final, dt)
    ts = np.zeros((len(times), 3))
    ts[0, :] = x_0
    cur_x = x_0
    dt_integr = 10**(-3)
    n_timesteps = int(np.ceil(dt / dt_integr))
    dt_integr = dt / n_timesteps
    for i in range(1, n_timesteps*len(times)):
        cur_x = cur_x + dt_integr * rhs(cur_x)
        saved_time_i = i*dt_integr / dt
        if isclose(saved_time_i, np.round(saved_time_i)):
            saved_time_i = int(np.round(i*dt_integr / dt))
            ts[saved_time_i, :] = cur_x
    return ts, times

class ControlResFunctionalCheck(unittest.TestCase):
    def test_none(self):
        data = np.random.rand(2, 10, 3)
        controls = np.random.rand(2, 10, 2)

        model = EsnForecaster()
        model.fit(data)
        
        model.fit(data, controls)
        self.assertTrue(True)

    def test_injection(self):
        print("Injection testing...")
        
        data = np.zeros((2, 300, 1))
        controls = np.zeros((2, 100, 1))
        t = np.linspace(0, 100, 300)
        for i in range(2):
            data[i, :, 0] = 0.5 - i + np.cos(t)/3.
            controls[i] = (0.5 - i)

        model = EsnForecaster(
            n_reservoir=50,
            spectral_radius=0.7,
            sparsity=0.1,
            regularization='l2',
            lambda_r=1e-3,
            in_activation='tanh',
            random_state=0,
            controller=InjectedController()
        )

        model.fit(data[:, :100], controls, inspect = True, initialization_strategy = optimal_weights_generator(
            verbose = 1,
            range_generator=np.linspace,
            steps = 600,
            hidden_std = 0.3,
            find_optimal_input = True,
            thinning_step = 5,
        ))
        
        plt.figure(figsize=(8,4))
        plt.xlabel("time")
        plt.ylabel("value")

        for i in range(2):
            plt.plot(t[:101], data[i, :101, 0], color="blue")
            plt.plot(t[100:], data[i, 100:, 0], color="gray", alpha = 0.5)
            plt.plot(t[:100], controls[i, :, 0] / 0.5 / 2, color="gray", alpha = 0.5)
        test_controls = np.clip(np.tanh(np.linspace(-2, 6, 200)) * 1.1, -1, 1) * 0.5
        plt.plot(t[100:], test_controls, color="green")

        model.update(data[:, :100], controls, mode=update_modes.refit)
        output = model.predict(200,test_controls,inspect=False)

        plt.plot(t[100:], output[:, 0], alpha=0.8, color="red")
        plt.legend()

        plt.show()
        
        self.assertTrue(True)

    def test_homotopy_ext(self):
        print("Homotopy extrapolation testing...")

        data = np.zeros((3, 600, 1))
        controls = np.ones((2, 480, 1))
        controls[0] *= 0
        t = np.linspace(0, 250, 600)

        data[0, :, 0] =  0.5 + np.cos(t)/3.
        data[1, :, 0] =  0.0 + np.cos(t)/3.
        data[2, :, 0] = -0.5 + np.cos(t)/3.

        model = EsnForecaster(
            n_reservoir=50,
            spectral_radius=0.9,
            sparsity=0.2,
            regularization='l2',
            lambda_r=5e-1,
            in_activation='tanh',
            random_state=0,
            controller=HomotopyController(False, 0)
        )

        model.fit(data[:2, :120][::-1], controls[::-1], inspect = True, 
        initialization_strategy = optimal_weights_generator(
            verbose = 1,
            range_generator=np.linspace,
            steps = 500,
            hidden_std = 0.5,
            find_optimal_input = False,
            thinning_step = 10,
        ))

        M = np.clip(0.5 + np.tanh(np.linspace(-4, 14, 360))*0.6, 0, 1) * 2

        output3 = model.predict(360, M)

        plt.figure(figsize=(8,4))

        plt.plot(t[:-120], data[:, :-120, 0].T,color="gray", alpha=0.5)
        plt.plot(t[:121], controls[:,:121,0].T / 2,color="orange")
        plt.plot(t[:121], data[0,:121,0],color="blue", label="train")
        plt.plot(t[:121], data[1,:121,0],color="blue")
        plt.plot(t[120:-120], 0.5-M/2,color="orange", label="control")
        plt.plot(t[120:-120], output3[:, 0],color="green", label="predict homotopy")

        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.ylim(-1.25, 0.9)
        
        plt.show()
        
        self.assertTrue(True)

    def test_homotopy_int(self):
        print("Homotopy interpolation testing...")

        data = np.zeros((3, 600, 1))
        controls = np.ones((2, 480, 1))
        controls[0] *= 0
        t = np.linspace(0, 250, 600)

        data[0, :, 0] =  0.5 + np.cos(t)/3.
        data[1, :, 0] =  0.0 + np.cos(t)/3.
        data[2, :, 0] = -0.5 + np.cos(t)/3.

        model = EsnForecaster(
            n_reservoir=50,
            spectral_radius=0.9,
            sparsity=0.2,
            regularization='l2',
            lambda_r=5e-4,
            in_activation='tanh',
            random_state=0,
            controller=HomotopyController(False, 0)
        )

        model.fit(data[[2, 0], :120], controls[::-1], inspect = True, 
        initialization_strategy = optimal_weights_generator(
            verbose = 1,
            range_generator=np.linspace,
            steps = 800,
            hidden_std = 0.5,
            find_optimal_input = False,
            thinning_step = 5,
        ))

        M = np.clip(0.5 + np.tanh(np.linspace(-4, 14, 360))*0.6, 0, 1) * 0.5

        output3 = model.predict(360, M)

        plt.figure(figsize=(8,4))

        plt.plot(t[:-120], data[:, :-120, 0].T,color="gray", alpha=0.5)
        plt.plot(t[:121], controls[:,:121,0].T-0.5,color="orange")
        plt.plot(t[:121], data[0,:121,0],color="blue", label="train")
        plt.plot(t[:121], data[2,:121,0],color="blue")
        plt.plot(t[120:-120], 0.5-M,color="orange", label="control")
        plt.plot(t[120:-120], output3[:, 0],color="green", label="predict homotopy")

        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.ylim(-1.25, 0.9)
        
        plt.show()
        
        self.assertTrue(True)
        
    def test_homotopy_tl_ext(self):
        print("Homotopy extrapolation testing...")

        data = np.zeros((3, 400, 1))
        controls = np.ones((2, 101, 1))
        controls[0] *= 0
        t = np.linspace(0, 200, 400)

        data[0, :, 0] =  0.5 + np.cos(t)/3.
        data[1, :, 0] =  0.0 + np.cos(t)/3.
        data[2, :, 0] = -0.5 + np.cos(t)/3.

        model = EsnForecaster(
            n_reservoir=50,
            spectral_radius=0.9,
            sparsity=0.2,
            regularization='l2',
            lambda_r=1e-3,
            in_activation='tanh',
            random_state=0,
            controller=HomotopyController(True, 0, 2e-1)
        )

        model.fit(data[:2, :100][::-1], controls[::-1, :100], inspect = True, 
        initialization_strategy = optimal_weights_generator(
            verbose = 1,
            range_generator=np.linspace,
            steps = 500,
            hidden_std = 0.5,
            find_optimal_input = True,
            thinning_step = 10,
        ))

        M = np.clip(0.5 + np.tanh(np.linspace(-4, 14, 300))*0.6, 0, 1) * 2

        output3 = model.predict(300, M)

        plt.figure(figsize=(8,4))

        plt.plot(t, data[:, :, 0].T,color="gray", alpha=0.5)
        plt.plot(t[:101], controls[:,:101,0].T / 2,color="orange")
        plt.plot(t[:101], data[0,:101,0],color="blue", label="train")
        plt.plot(t[:101], data[1,:101,0],color="blue")
        plt.plot(t[100:], 0.5-M/2,color="orange", label="control")
        plt.plot(t[100:], output3[:, 0],color="green", label="predict homotopy")

        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.ylim(-1.25, 0.9)
        
        plt.show()
        
        self.assertTrue(True)


    def test_homotopy_tl_int(self):
        print("Homotopy interpolation testing...")

        data = np.zeros((3, 400, 1))
        controls = np.ones((2, 101, 1))
        controls[0] *= 0
        t = np.linspace(0, 200, 400)

        data[0, :, 0] =  0.5 + np.cos(t)/3.
        data[1, :, 0] =  0.0 + np.cos(t)/3.
        data[2, :, 0] = -0.5 + np.cos(t)/3.

        model = EsnForecaster(
            n_reservoir=50,
            spectral_radius=0.9,
            sparsity=0.2,
            regularization='l2',
            lambda_r=1e-4,
            in_activation='tanh',
            random_state=0,
            controller=HomotopyController(True, 0, 1e-1)
        )

        model.fit(data[[0, 2], :100][::-1], controls[::-1, :100], inspect = True, 
        initialization_strategy = optimal_weights_generator(
            verbose = 1,
            range_generator=np.linspace,
            steps = 500,
            hidden_std = 0.5,
            find_optimal_input = True,
            thinning_step = 10,
        ))

        M = np.clip(0.5 + np.tanh(np.linspace(-4, 14, 300))*0.6, 0, 1) * 0.5

        output3 = model.predict(300, M)

        plt.figure(figsize=(8,4))

        plt.plot(t, data[:, :, 0].T,color="gray", alpha=0.5)
        plt.plot(t[:101], controls[:,:101,0].T + 0.5,color="orange")
        plt.plot(t[:101], data[0,:101,0],color="blue", label="train")
        plt.plot(t[:101], data[2,:101,0],color="blue")
        plt.plot(t[100:], 0.5-M,color="orange", label="control")
        plt.plot(t[100:], output3[:, 0],color="green", label="predict homotopy")

        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.ylim(-1.25, 0.9)
        
        plt.show()
        
        self.assertTrue(True)