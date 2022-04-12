from cpp_bindings import solve, prod_F, get_payoffs
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool

n_players = 2
A = 10.0
alpha = 0.5
B = 10.0
beta = 0.5
theta = 0.0

r = 0.01
d = 1.0

MAX_ITER = 200
SOLVER_TOL = 0.001


def _r_d_helper(args):
    r_, d_ = args
    ones = np.ones(n_players)
    K = solve(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        d_ * ones,
        r_ * ones,
        MAX_ITER,
        SOLVER_TOL
    )
    s, _ = prod_F(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        K[:, 0].copy(),
        K[:, 1].copy()
    )
    return (s / (1 + s)).prod()


def vary_r_d(min_r, max_r, r_steps, min_d, max_d, d_steps, plot=True):
    R = np.linspace(min_r, max_r, r_steps)
    D = np.linspace(min_d, max_d, d_steps)
    with Pool(min(cpu_count(), r_steps * d_steps)) as pool:
        safety_vals = pool.map(_r_d_helper, [(r_, d_) for r_ in R for d_ in D])
    grid = np.array(safety_vals).reshape((r_steps, d_steps))
    if plot:
        plt.imshow(grid)
        plt.colorbar()
        plt.xticks(ticks=range(d_steps), labels=[f'{d_:.3f}' for d_ in D])
        plt.xlabel('d')
        plt.yticks(ticks=range(r_steps), labels=[f'{r_:.3f}' for r_ in R])
        plt.ylabel('r')
        plt.show()
    return grid


vary_r_d(0.01, 0.02, 4, 0.5, 1.0, 3)
