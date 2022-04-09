import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
from simple_model import ProdFunc, Problem, simple_CSF, simple_CSF_deriv


# Define default parameter values

A = 10.0
alpha = 0.5
B = 10.0
beta = 0.5
theta = 0.0

d = 1.0

MAX_ITER = 200


# SCENARIO 1
# In homogeneous setting, see what happens when factor cost increases
def _scenario_1_helper(args):
    n_players, r_, d_, prodFunc, init_guess_mu = args
    problem = Problem(np.ones(n_players) * d_, r_, simple_CSF, simple_CSF_deriv, prodFunc)
    return problem.solve(T=MAX_ITER, init_guess_mu=init_guess_mu)[-1]

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
    prodFunc = ProdFunc(A*ones, alpha*ones, B*ones, beta*ones, theta*ones)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_1_helper, [(n_players, r_, d, prodFunc, init_guess_mu) for r_ in R])
    strats = np.array(strats)
    if plot:
        s, p = prodFunc.F(strats[..., 0], strats[..., 1])
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
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = Problem(ones * d, r_, simple_CSF, simple_CSF_deriv, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    j,
                    strats[i, :, 0],
                    strats[i, :, 1]
                )
        plt.plot(R, payoff.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_payoff.png')
        plt.clf()
    return strats


def run_scenario_1_multiple(
    n_players=2,
    min_r=0.05,
    max_r=0.10,
    n_steps=20,
    init_guess_mu=-2.0,
    plot=True,
    plotname='fcost',
    As=[A],
    Bs=[B],
    alphas=[alpha],
    betas=[beta],
    thetas=[theta],
    ds=[d]
):
    strat_list = []
    strat_list = [
        run_scenario_1(
            n_players, min_r, max_r, n_steps, init_guess_mu,
            plot=False, plotname=plotname,
            A=A_, B=B_, alpha=alpha_, beta=beta_, theta=theta_, d=d_
        )
        for A_, B_, alpha_, beta_, theta_, d_ in zip(As, Bs, alphas, betas, thetas, ds)
    ]
    if plot:
        R = np.linspace(min_r, max_r, n_steps)
        ones = np.ones(n_players, dtype=np.float64)
        prodFuncs = [
            ProdFunc(ones*A_, ones*alpha_, ones*B_, ones*beta_, ones*theta_)
            for A_, B_, alpha_, beta_, theta_ in zip(As, Bs, alphas, betas, thetas)
        ]
        S, P = [], []
        for i, prodFunc in enumerate(prodFuncs):
            s, p = prodFunc.F(strat_list[i][..., 0], strat_list[i][..., 1])
            S.append(s)
            P.append(p)
        # plot performance
        for i, p in enumerate(P):
            plt.plot(R, p.mean(axis=-1), label=f'i = {i}')
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_performance.png')
        plt.clf()
        # plot safety
        for i, s in enumerate(S):
            plt.plot(R, s.mean(axis=-1), label=f'i = {i}')
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_safety.png')
        plt.clf()
        # plot total safety
        for i, s in enumerate(S):
            probas = s / (1 + s)
            total_proba = probas.prod(axis=-1)
            plt.plot(R, total_proba, label=f'i = {i}')
        plt.legend()
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_total_safety.png')
        plt.clf()
        # plot payoffs
        for i, prodFunc in enumerate(prodFuncs):
            payoff = np.empty_like(p)
            for j, r_ in enumerate(R):
                problem = Problem(ones * ds[i], r_, simple_CSF, simple_CSF_deriv, prodFunc)
                for k in range(n_players):
                    payoff[j, k] = problem.net_payoff(
                        k,
                        strat_list[i][j, :, 0],
                        strat_list[i][j, :, 1]
                    )
            plt.plot(R, payoff.mean(axis=-1), label=f'i = {i}')
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_payoff.png')
        plt.clf()
    return strat_list


# run_scenario_1(n_steps=20)

# theta = 0.25
# run_scenario_1(n_steps=20, plotname='fcosttheta0_1')

# run_scenario_1_multiple(
#     As=[10.0, 11.0, 12.0],
#     Bs=[10.0, 11.0, 12.0],
#     alphas=[alpha]*3,
#     betas=[beta]*3,
#     thetas=[theta]*3,
#     ds=[d]*3
# )

run_scenario_1_multiple(
    As=[A]*3,
    Bs=[B]*3,
    alphas=[alpha]*3,
    betas=[beta]*3,
    thetas=[0.0, 0.2, 0.4]*3,
    ds=[d]*3
)


# SCENARIO 2
# With variation in A param, see what happens when factor cost increases

def run_scenario_2(
    As=np.array([9.0, 11.0]),
    min_r=0.05,
    max_r=0.15,
    n_steps=20,
    init_guess_mu=0.0,
    plot=True,
    plotname='fcost'
):
    n_players = len(As)
    ones = np.ones(n_players, dtype=np.float64)
    prodFunc = ProdFunc(As, alpha*ones, B*ones, beta*ones, theta*ones)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        # can use same helper
        strats = pool.map(_scenario_1_helper, [(n_players, r_, prodFunc, init_guess_mu) for r_ in R])
    strats = np.array(strats)
    if plot:
        s, p = prodFunc.F(strats[..., 0], strats[..., 1])
        # plot performance
        for i in range(n_players):
            plt.plot(R, p[..., i], label=f'A = {As[i]}')
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc2_performance.png')
        plt.clf()
        # plot safety
        for i in range(n_players):
            plt.plot(R, s[..., i], label=f'A = {As[i]}')
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc2_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc2_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = Problem(ones * d, r_, simple_CSF, simple_CSF_deriv, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    j,
                    strats[i, :, 0],
                    strats[i, :, 1]
                )
        for i in range(n_players):
            plt.plot(R, payoff[..., i], label=f'A = {As[i]}')
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc2_payoff.png')
        plt.clf()
    return strats


# run_scenario_2()



# SCENARIO 3
# With variation in B param, see what happens when factor cost increases

def run_scenario_3(
    Bs=np.array([9.0, 11.0]),
    min_r=0.05,
    max_r=0.15,
    n_steps=20,
    init_guess_mu=0.0,
    plot=True,
    plotname='fcost'
):
    n_players = len(Bs)
    ones = np.ones(n_players, dtype=np.float64)
    prodFunc = ProdFunc(A*ones, alpha*ones, Bs, beta*ones, theta*ones)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        # can use same helper
        strats = pool.map(_scenario_1_helper, [(n_players, r_, prodFunc, init_guess_mu) for r_ in R])
    strats = np.array(strats)
    if plot:
        s, p = prodFunc.F(strats[..., 0], strats[..., 1])
        # plot performance
        for i in range(n_players):
            plt.plot(R, p[..., i], label=f'B = {Bs[i]}')
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc3_performance.png')
        plt.clf()
        # plot safety
        for i in range(n_players):
            plt.plot(R, s[..., i], label=f'B = {Bs[i]}')
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc3_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc3_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = Problem(ones * d, r_, simple_CSF, simple_CSF_deriv, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    j,
                    strats[i, :, 0],
                    strats[i, :, 1]
                )
        for i in range(n_players):
            plt.plot(R, payoff[..., i], label=f'B = {Bs[i]}')
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc3_payoff.png')
        plt.clf()
    return strats

# run_scenario_3()



# SCENARIO 4
# With variation in A and B params, see what happens when factor cost increases

def run_scenario_4(
    As=np.array([9.0, 11.0]),
    Bs=np.array([9.0, 11.0]),
    min_r=0.05,
    max_r=0.15,
    n_steps=20,
    init_guess_mu=0.0,
    plot=True,
    plotname='fcost'
):
    n_players = len(As)
    assert(n_players == len(Bs))
    ones = np.ones(n_players, dtype=np.float64)
    prodFunc = ProdFunc(As, alpha*ones, Bs, beta*ones, theta*ones)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        # can use same helper
        strats = pool.map(_scenario_1_helper, [(n_players, r_, prodFunc, init_guess_mu) for r_ in R])
    strats = np.array(strats)
    if plot:
        s, p = prodFunc.F(strats[..., 0], strats[..., 1])
        # plot performance
        for i in range(n_players):
            plt.plot(R, p[..., i], label=f'A = {As[i]}, B = {Bs[i]}')
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc4_performance.png')
        plt.clf()
        # plot safety
        for i in range(n_players):
            plt.plot(R, s[..., i], label=f'A = {As[i]}, B = {Bs[i]}')
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc4_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc4_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = Problem(ones * d, r_, simple_CSF, simple_CSF_deriv, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    j,
                    strats[i, :, 0],
                    strats[i, :, 1]
                )
        for i in range(n_players):
            plt.plot(R, payoff[..., i], label=f'A = {As[i]}, B = {Bs[i]}')
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc4_payoff.png')
        plt.clf()
    return strats


# run_scenario_4()
