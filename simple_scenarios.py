import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
from simple_model import ProdFunc, Problem, simple_CSF, simple_CSF_deriv

A = 1.0
alpha = 0.5
B = 1.0
beta = 0.5
theta = 0.0

d = 0.1
r = 0.06


# SCENARIO 1
# In homogeneous setting, see what happens when disaster cost increases
def _scenario_1_helper(args):
    n_players, d_, prodFunc = args
    problem = Problem(np.ones(n_players) * d_, r, simple_CSF, simple_CSF_deriv, prodFunc)
    return problem.solve()[-1]

def run_scenario_1(n_players=2, max_d=0.5, n_steps=20, plot=True):
    ones = np.ones(n_players, dtype=np.float64)
    prodFunc = ProdFunc(A*ones, alpha*ones, B*ones, beta*ones, theta*ones)
    D = np.linspace(0.0, max_d, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_1_helper, [(n_players, d_, prodFunc) for d_ in D])
    strats = np.array(strats)
    if plot:
        s, p = prodFunc.F(strats[..., 0], strats[..., 1])
        # plot performance
        plt.plot(D, p.mean(axis=-1), label='p')
        plt.ylabel('performance')
        plt.xlabel('disaster cost')
        plt.savefig('plots/simple_sc1_performance.png')
        plt.clf()
        # plot safety
        plt.plot(D, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel('disaster cost')
        plt.savefig('plots/simple_sc1_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(D, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('disaster cost')
        plt.savefig('plots/simple_sc1_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, d_ in enumerate(D):
            problem = Problem(ones * d_, r, simple_CSF, simple_CSF_deriv, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    j,
                    strats[i, :, 0],
                    strats[i, :, 1]
                )
        plt.plot(D, payoff.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('disaster cost')
        plt.savefig('plots/simple_sc1_payoff.png')
        plt.clf()
    return strats

# run_scenario_1()



# SCENARIO 2
# In homogeneous setting, vary number of participants

def _scenario_2_helper(i):
    ones = np.ones(i, dtype=np.float64)
    prodFunc = ProdFunc(A*ones, alpha*ones, B*ones, beta*ones, theta*ones)
    problem = Problem(ones * d, r, simple_CSF, simple_CSF_deriv, prodFunc)
    strat = problem.solve()[-1]
    # also calculate safety and performance
    s, p = prodFunc.F(strat[:, 0], strat[:, 1])
    payoff = np.empty_like(p)
    for j in range(i):
        payoff[j] = problem.net_payoff(j, strat[:, 0], strat[:, 1])
    return strat, p, s, payoff

def run_scenario_2(start_at=2, n_steps=9, plot=True):
    n_players = start_at + np.arange(n_steps, dtype=int)
    with Pool(n_steps) as pool:
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_2_helper, n_players)))

    if plot:
        x_axis = sum(([i] * i for i in n_players), [])
        flat_p = sum((list(x) for x in p), [])
        flat_s = sum((list(x) for x in s), [])
        flat_payoff = sum((list(x) for x in payoff), [])
        # plot performance
        plt.scatter(x_axis, flat_p, label='p', alpha=0.5)
        plt.ylabel('performance')
        plt.xlabel('number of players')
        plt.savefig('plots/simple_sc2_performance.png')
        plt.clf()
        # plot safety
        plt.scatter(x_axis, flat_s, label='s', alpha=0.5)
        plt.ylabel('safety')
        plt.xlabel('number of players')
        plt.savefig('plots/simple_sc2_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = [s_ / (1 + s_) for s_ in s]
        total_proba = [pr.prod() for pr in probas]
        plt.plot(n_players, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('number of players')
        plt.savefig('plots/simple_sc2_total_safety.png')
        plt.clf()
        # plot net payoff
        plt.scatter(x_axis, flat_payoff, alpha=0.5)
        plt.ylabel('net payoff')
        plt.xlabel('B')
        plt.savefig('plots/simple_sc2_payoff.png')
        plt.clf()
    return strats

run_scenario_2()
