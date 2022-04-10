import numpy as np
# import matplotlib.pyplot as plt
# import sys
from multiprocessing import Pool
from cpp_bindings import solve

A = 1.0
alpha = 0.5
B = 1.0
beta = 0.5
theta = 0.0

d = 0.1
r = 0.06

MAX_ITERS = 100
EXIT_TOL = 1e-4


# SCENARIO 1
# In homogeneous setting, see what happens when disaster cost increases
def _scenario_1_helper(args):
    n_players, A_, alpha_, B_, beta_, theta_, d_, = args
    return solve(
        n_players,
        A_,
        alpha_,
        B_,
        beta_,
        theta_,
        d_,
        r,
        MAX_ITERS,
        EXIT_TOL
    )

def run_scenario_1(n_players=2, max_d=0.5, n_steps=20, plot=True):
    ones = np.ones(n_players, dtype=np.float64)
    D = np.linspace(0.0, max_d, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_1_helper, [
            (
                n_players,
                A*ones,
                alpha*ones,
                B*ones,
                beta*ones,
                theta*ones,
                d_*ones
            ) for d_ in D
        ])
    return strats

run_scenario_1()