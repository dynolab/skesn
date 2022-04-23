import numpy as np

def get_lorenz_data(ro, N, dt, seed):
    np.random.seed(seed)
    Y = np.zeros((3,N))
    Y[:,0] = np.random.random(3)

    for i in range(1,N):
        [x,y,z] = Y[:,i-1]    
        Y[0,i] = x + 10*(y-x)*dt
        Y[1,i] = y + (x*(ro-z)-y)*dt
        Y[2,i] = z + (x*y-8*z/3.)*dt

    return Y

# Нормализация данных
# В модели Лоренца значения - двузначные числа
# Но ESN желательно обучать на числах ~0-1

def data_to_train(data):
    return (data - [[0],[0],[25]])/[[30],[30],[30]]

def train_to_data(data):
    return data*[[30],[30],[30]] + [[0],[0],[25]]
