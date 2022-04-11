import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from cpp_bindings import solve, prod_F, get_payoffs


A = 10.0
alpha = 0.5
B = 10.0
beta = 0.5
theta = 0.0

d = 1.0

MAX_ITER = 200
SOLVER_TOL = 0.001

def _multiproc_helper(args):
    n_players, A_, alpha_, B_, beta_, theta_, d_, r_ = args
    return solve(
        n_players,
        A_,
        alpha_,
        B_,
        beta_,
        theta_,
        d_,
        r_,
        MAX_ITER,
        SOLVER_TOL
    )

# SCENARIO 1
# In homogeneous setting, see what happens when factor cost increases
def run_scenario_1(
    n_players=2,
    min_r=0.05,
    max_r=0.10,
    n_steps=20,
    init_guess_mu=-2.0,
    plot=True,
    plotname='fcost',
    A=A,
    B=B,
    alpha=alpha,
    beta=beta,
    theta=theta,
    d=d
):
    ones = np.ones(n_players, dtype=np.float64)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_multiproc_helper, [
            (
                n_players,
                A * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                d * ones,
                r_
            ) for r_ in R
        ])
    strats = np.array(strats)
    if plot:
        s_p = np.array([
            prod_F(
                n_players,
                A * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                strat[:, 0].copy(),
                strat[:, 1].copy()
            )
            for strat in strats
        ])
        s, p = s_p[..., 0], s_p[..., 1]
        # plot performance
        plt.plot(R, p.mean(axis=-1))
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_performance.png')
        plt.clf()
        # plot safety
        plt.plot(R, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoffs = np.array([
            get_payoffs(
                n_players,
                A * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                d * ones,
                r_,
                strat[:, 0].copy(),
                strat[:, 1].copy()
            )
            for strat, r_ in zip(strats, R)
        ])
        plt.plot(R, payoffs.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_payoff.png')
        plt.clf()
    return strats

run_scenario_1()
