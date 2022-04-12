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

MIN_R = 0.05
MAX_R = 0.15

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
    min_r=MIN_R,
    max_r=MAX_R,
    n_steps=20,
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

# run_scenario_1()


def run_scenario_1_multiple(
    n_players=2,
    min_r=MIN_R,
    max_r=MAX_R,
    n_steps=20,
    plot=True,
    plotname='fcost',
    As=[A],
    Bs=[B],
    alphas=[alpha],
    betas=[beta],
    thetas=[theta],
    ds=[d],
    labels=None
):
    strat_list = [
        run_scenario_1(
            n_players, min_r, max_r, n_steps,
            plot=False, plotname=plotname,
            A=A_, B=B_, alpha=alpha_, beta=beta_, theta=theta_, d=d_
        )
        for A_, B_, alpha_, beta_, theta_, d_ in zip(As, Bs, alphas, betas, thetas, ds)
    ]
    if plot:
        if labels is None:
            labels = [str(i) for i in range(n_steps)]
        R = np.linspace(min_r, max_r, n_steps)
        ones = np.ones(n_players, dtype=np.float64)
        S, P = [], []
        payoffs_list = []
        for A_, B_, alpha_, beta_, theta_, d_, strats in zip(As, Bs, alphas, betas, thetas, ds, strat_list):
            s_p = np.array([
                prod_F(
                    n_players,
                    A_ * ones,
                    alpha_ * ones,
                    B_ * ones,
                    beta_ * ones,
                    theta_ * ones,
                    strat[:, 0].copy(),
                    strat[:, 1].copy()
                )
                for strat in strats
            ])
            S.append(s_p[:, 0])
            P.append(s_p[:, 1])
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
            payoffs_list.append(payoffs)
        # plot performance
        for p, label in zip(P, labels):
            plt.plot(R, p.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_performance.png')
        plt.clf()
        # plot safety
        for s, label in zip(S, labels):
            plt.plot(R, s.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_safety.png')
        plt.clf()
        # plot total safety
        for s, label in zip(S, labels):
            probas = s / (1 + s)
            total_proba = probas.prod(axis=-1)
            plt.plot(R, total_proba, label=label)
        plt.legend()
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_total_safety.png')
        plt.clf()
        # plot payoffs
        for payoffs, label in zip(payoffs_list, labels):
            plt.plot(R, payoffs.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_payoff.png')
        plt.clf()
    return strat_list

# vary theta
# run_scenario_1_multiple(
#     As=[A]*3,
#     Bs=[B]*3,
#     alphas=[alpha]*3,
#     betas=[beta]*3,
#     thetas=[0.0, 0.15, 0.3],
#     ds=[d]*3,
#     labels=['θ = 0.0', 'θ = 0.15', 'θ = 0.3'],
#     plotname='0'
# )

# vary d with theta = 0
# run_scenario_1_multiple(
#     As=[A]*4,
#     Bs=[B]*4,
#     alphas=[alpha]*4,
#     betas=[beta]*4,
#     thetas=[0.0]*4,
#     ds=[0.1, 0.5, 1.0, 1.5],
#     labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
#     plotname='1'
# )

# vary d with theta = 0.3
# run_scenario_1_multiple(
#     As=[A]*4,
#     Bs=[B]*4,
#     alphas=[alpha]*4,
#     betas=[beta]*4,
#     thetas=[0.3]*4,
#     ds=[0.1, 0.5, 1.0, 1.5],
#     labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
#     plotname='2'
# )

# vary d with higher productivity, theta = 0
# run_scenario_1_multiple(
#     As=[20.0]*4,
#     Bs=[20.0]*4,
#     alphas=[alpha]*4,
#     betas=[beta]*4,
#     thetas=[0.0]*4,
#     ds=[0.1, 0.5, 1.0, 1.5],
#     labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
#     plotname='3'
# )
