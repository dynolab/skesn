import argparse
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # Choises

    parser.add_argument('-m',
        type=str,
        required=True,
        choices=[_MODE_TESTS, _MODE_HYPER_PARAMS, _MODE_HYPER_PARAMS_MULTI_POP],
        help='run mode'
    )

    # Boolean flags

    parser.add_argument('--disable-config',
        action='store_true',
        help='disable loading config from config dir'
    )
    parser.add_argument('-v', '--verbose',
        action='store_true',
        help='print all logs'
    )

    # Paths

    parser.add_argument('-с', "--config-path",
        type=str,
        nargs='?',
        help='config path'
    )
    parser.add_argument('--log-dir',
        type=str,
        nargs='?',
        default='logs',
        help='directory for writing log files'
    )
    parser.add_argument('--dump-dir',
        type=str,
        nargs='?',
        help='directory for writing dump files'
    )
    parser.add_argument('--continue-dir',
        type=str,
        nargs='?',
        help='provide directory of itteration pull for continue calculation'
    )

    # Tests args

    parser.add_argument('--test-disable-iter-graph',
        action='store_true',
        help='disable matplotlib lib graphs on iterations for tests'
    )
    parser.add_argument('--test-disable-stat-graph',
        action='store_true',
        help='disable matplotlib lib statistics graphs for tests'
    )
    parser.add_argument('--test-restore-result',
        action='store_true',
        help='enable use last result for tests'
    )
    parser.add_argument('--test-disable-dump',
        action='store_true',
        help='disable dumping tests result'
    )
    parser.add_argument('--test-dump-dir',
        type=str,
        nargs='?',
        default='dumps/tests',
        help='directory for writing dump files for tests'
    )

    return parser

def cx_gaussian_bounded_gene(
    p1_gene: float,
    p2_gene: float,
    eta: float,
    low: float,
    up: float,
) -> Tuple[float, float]:
    # This epsilon should probably be changed for 0 since
    # floating point arithmetic in Python is safer
    if abs(p1_gene - p2_gene) < 1e-14:
        return p1_gene, p2_gene

    beta = 0
    u = np.random.random()
    if u <= 0.5:
        beta = np.power(2 * u, 1 / (eta + 1))
    else:
        beta = np.power(0.5 / (1 - u), 1 / (eta + 1))

    x1 = bound_vaule(0.5 * ((1 + beta) * p1_gene + (1 - beta) * p2_gene), low, up)
    x2 = bound_vaule(0.5 * ((1 - beta) * p1_gene + (1 + beta) * p2_gene), low, up)

    return x1, x2

def bound_vaule(
    value: float,
    low: float,
    up: float,
) -> float:
    if value < low:
        return low
    if value > up:
        return up
    return value

def polynomial_bounded(x, low, up, eta):
    delta_1 = (x - low) / (up - low)
    delta_2 = (up - x) / (up - low)
    rand = np.random.random()
    mut_pow = 1.0 / (eta + 1.)

    if rand < 0.5:
        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
        delta_q = val ** mut_pow - 1.0
    else:
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
        delta_q = 1.0 - val ** mut_pow

    x = x + delta_q * (up - low)
    return bound_vaule(x, low, up)

def mut():
    fig, ax = plt.subplots(1)
    ticks = [i for i in range(N)]
    for eta in ETAS:
        vals = [polynomial_bounded(X, MIN, MAX, eta) for _ in range(N)]
        ax.scatter(ticks, vals, label=f'eta={eta}')
    ax.set_xlabel('Номер пробы')
    ax.set_ylabel('Значение мутировавшего гена')
    fig.legend()
    fig.savefig('visualize_mut.png')

P1 = -0.25
P2 = 0.25

def cx():
    fig, ax = plt.subplots(1)
    ticks = [i for i in range(N)]
    for eta in ETAS:
        i = 0
        vals = [0] * N
        while i < N:
            vals[i], vals[i+1] = cx_gaussian_bounded_gene(P1, P2, eta, MIN, MAX)
            i += 2
        ax.scatter(ticks, vals, label=f'eta={eta}')
    ax.set_xlabel('Номер пробы')
    ax.set_ylabel('Значение гена потомка')
    fig.legend()
    fig.savefig('visualize_cx.png')


N = 10000
X = 0
MIN = -1
MAX = 1
ETAS = [2, 5, 10, 15, 20, 30]

def main():
    mut()
    cx()

if __name__ == '__main__':
    main()
