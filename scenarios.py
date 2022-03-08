import numpy as np
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
from model import VecProdFunc, HomogeneousProdFunc, MultiAgent, SimpleProblem


A = 10.0
a = 0.2
rho = 0.5
mu = 0.6
B = 10.0
b = 0.8
sigma = 0.5
nu = 0.6
theta = 0.0

d = 0.1
r = 0.02
w = 0.03


# SCENARIO 1
# In homogeneous setting, see what happens when disaster cost increases
def _scenario_1_helper(args):
    n_players, d_, prodFunc = args
    problem = SimpleProblem(np.ones(n_players) * d_, r, w, prodFunc)
    return problem.solve(multicore=False)[-1]

def run_scenario_1(n_players, max_d=0.5, n_steps=20, plot=True):
    prodFunc = HomogeneousProdFunc(n_players, A, a, rho, mu, B, b, sigma, nu, theta)
    D = np.linspace(0.0, max_d, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_1_helper, [(n_players, d_, prodFunc) for d_ in D])
    strats = np.array(strats)
    # for i, d in enumerate(np.linspace(0.0, 1.0, n_steps)):
    #     problem = SimpleProblem(np.ones(n_players) * d, 0.02, 0.03, prodFunc)
    #     strat = problem.solve()[-1]
    #     assert((strat.std(axis=0) / np.abs(strat.mean(axis=0))).max() < tol, 'Heterogeneous strategies detected')
    #     strats[i] = strat.mean(axis=0)
    if plot:
        p = prodFunc.P(strats[..., 1], strats[..., 3])
        s = prodFunc.S(strats[..., 0], strats[..., 2], p)
        plt.scatter(np.repeat(D, n_players), p.flatten(), label='p', alpha=1/n_players)
        plt.scatter(np.repeat(D, n_players), s.flatten(), label='s', alpha=1/n_players)
        plt.legend()
        plt.show()
    return strats

# run_scenario_1(4, n_steps=20, plot=True)



# SCENARIO 2
# In homogeneous setting, vary number of participants

def _scenario_2_helper(i):
    prodFunc = HomogeneousProdFunc(i, A, a, rho, mu, B, b, sigma, nu, theta)
    problem = SimpleProblem(np.ones(i) * d, r, w, prodFunc)
    strat = problem.solve(multicore=False)[-1]
    # also calculate safety and performance
    p = prodFunc.P(strat[:, 1], strat[:, 3])
    s = prodFunc.S(strat[:, 0], strat[:, 2], p)
    payoff = np.empty_like(p)
    for j in range(i):
        payoff[j] = problem.net_payoff(strat[:, 0], strat[:, 1], strat[:, 2], strat[:, 3], j)
    return strat, p, s, payoff

def run_scenario_2(start_at=2, n_steps=10, plot=True):
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
        plt.savefig('plots/sc2_performance.png')
        plt.clf()
        # plot safety
        plt.scatter(x_axis, flat_s, label='s', alpha=0.5)
        plt.ylabel('safety')
        plt.xlabel('number of players')
        plt.savefig('plots/sc2_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = [s_ / (1 + s_) for s_ in s]
        total_proba = [pr.prod() for pr in probas]
        plt.plot(n_players, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('number of players')
        plt.savefig('plots/sc2_total_safety.png')
        plt.clf()
        # plot net payoff
        plt.scatter(x_axis, flat_payoff, alpha=0.5)
        plt.ylabel('net payoff')
        plt.xlabel('B')
        plt.savefig('plots/sc2_payoff.png')
        plt.clf()
    return strats

# run_scenario_2(n_steps=9, plot=True)



# SCENARIO 3
# Homogeneous, performance productivity increases
def _scenario_3_helper(args):
    n_players, B_ = args
    prodFunc = HomogeneousProdFunc(n_players, A, a, rho, mu, B_, b, sigma, nu, theta)
    problem = SimpleProblem(d*np.ones(n_players), r, w, prodFunc)
    strat = problem.solve(multicore=False)[-1]
    p = prodFunc.P(strat[:, 1], strat[:, 3])
    s = prodFunc.S(strat[:, 0], strat[:, 2], p)
    payoff = np.empty_like(p)
    for i in range(n_players):
        payoff[i] = problem.net_payoff(strat[:, 0], strat[:, 1], strat[:, 2], strat[:, 3], i)
    return strat, p, s, payoff

def run_scenario_3(n_players, min_B=5.0, max_B=15.0, n_steps=20, plot=True):
    Bs = np.linspace(min_B, max_B, n_steps)
    with Pool(n_steps) as pool:
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_3_helper, [(n_players, B_) for B_ in Bs])))
    strats, p, s, payoff = np.array(strats), np.array(p), np.array(s), np.array(payoff)

    if plot:
        # plot performance
        plt.scatter(np.repeat(Bs, n_players), p.flatten(), label='p', alpha=0.5)
        plt.ylabel('performance')
        plt.xlabel('B')
        plt.savefig('plots/sc3_performance.png')
        plt.clf()
        # plot safety
        plt.scatter(np.repeat(Bs, n_players), s.flatten(), label='s', alpha=0.5)
        plt.ylabel('safety')
        plt.xlabel('B')
        plt.savefig('plots/sc3_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(Bs, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('B')
        plt.savefig('plots/sc3_total_safety.png')
        plt.clf()
        # plot net payoff
        plt.scatter(np.repeat(Bs, n_players), payoff.flatten(), alpha=0.5)
        plt.ylabel('net payoff')
        plt.xlabel('B')
        plt.savefig('plots/sc3_payoff.png')
        plt.clf()
    return strats, p, s, payoff

# run_scenario_3(2)



# SCENARIO 4
# Vary homogeneity of performance productivity

def _scenario_4_helper(args):
    n_players, B_ = args
    ones = np.ones(n_players)
    prodFunc = VecProdFunc(A*ones, a*ones, rho*ones, mu*ones, B_, b*ones, sigma*ones, nu*ones, theta*ones)
    problem = SimpleProblem(d*ones, r, w, prodFunc)
    strat = problem.solve(multicore=False)[-1]
    p = prodFunc.P(strat[:, 1], strat[:, 3])
    s = prodFunc.S(strat[:, 0], strat[:, 2], p)
    payoff = np.empty_like(p)
    for i in range(n_players):
        payoff[i] = problem.net_payoff(strat[:, 0], strat[:, 1], strat[:, 2], strat[:, 3], i)
    return strat, p, s, payoff

def run_scenario_4(n_players, avg_B=10.0, widening=0.1, n_steps=20, plot=True):
    Bs = [np.linspace(avg_B-widening*i, avg_B+widening*i, n_players) for i in range(n_steps)]
    with Pool(n_steps) as pool:
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_4_helper, [(n_players, B_) for B_ in Bs])))
    strats, p, s, payoff = np.array(strats), np.array(p), np.array(s), np.array(payoff)

    if plot:
        # x_axis = sum(([widening*i]*n_players for i in range(n_steps)), [])
        colors = (
            np.linspace(0.0, 1.0, n_players) * np.tile(np.array([-1.0, 0.0, 1.0]), (n_players, 1)).T
            + np.tile(np.array([1.0, 0.0, 0.0]), (n_players, 1)).T
        ).T
        x_axis = 2 * np.arange(n_steps) * widening
        for i in range(n_players):
            plt.plot(x_axis, p[:, i], color=colors[i])
        plt.ylabel('performance')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4_performance.png')
        plt.clf()
        for i in range(n_players):
            plt.plot(x_axis, s[:, i], color=colors[i])
        plt.ylabel('safety')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot([widening*i for i in range(n_steps)], total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4_total_safety.png')
        plt.clf()
        # plot payoffs
        for i in range(n_players):
            plt.plot(x_axis, payoff[:, i], color=colors[i])
        plt.ylabel('net payoff')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4_payoff.png')
        plt.clf()

    return strats, p, s, payoff

# run_scenario_4(n_players=2, widening=0.4)
