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

def run_scenario_1(n_players=2, max_d=0.5, n_steps=20, plot=True):
    prodFunc = HomogeneousProdFunc(n_players, A, a, rho, mu, B, b, sigma, nu, theta)
    D = np.linspace(0.0, max_d, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_1_helper, [(n_players, d_, prodFunc) for d_ in D])
    strats = np.array(strats)
    if plot:
        p = prodFunc.P(strats[..., 1], strats[..., 3])
        s = prodFunc.S(strats[..., 0], strats[..., 2], p)
        # plot performance
        plt.plot(D, p.mean(axis=-1), label='p')
        plt.ylabel('performance')
        plt.xlabel('disaster cost')
        plt.savefig('plots/sc1_performance.png')
        plt.clf()
        # plot safety
        plt.plot(D, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel('disaster cost')
        plt.savefig('plots/sc1_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(D, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('disaster cost')
        plt.savefig('plots/sc1_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, d_ in enumerate(D):
            problem = SimpleProblem(np.ones(n_players) * d_, r, w, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    strats[i, :, 0],
                    strats[i, :, 1],
                    strats[i, :, 2],
                    strats[i, :, 3],
                    j
                )
        plt.plot(D, payoff.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('disaster cost')
        plt.savefig('plots/sc1_payoff.png')
        plt.clf()
    return strats

# run_scenario_1()



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

def run_scenario_3(n_players=2, min_B=5.0, max_B=15.0, n_steps=20, plot=True):
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

# run_scenario_3()



# SCENARIO 4a
# Vary homogeneity of performance productivity

def _scenario_4a_helper(args):
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

def run_scenario_4a(n_players=2, avg_B=10.0, max_B_width=18.0, n_steps=20, plot=True):
    B_widths = max_B_width * np.linspace(0.0, 1.0, n_steps)
    Bs = np.array([
        np.linspace(
            avg_B - width / 2,
            avg_B + width / 2,
            n_players
        ) for width in B_widths
    ])
    with Pool(n_steps) as pool:
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_4a_helper, [(n_players, B_) for B_ in Bs])))
    strats, p, s, payoff = np.array(strats), np.array(p), np.array(s), np.array(payoff)

    if plot:
        colors = (
            np.linspace(0.0, 1.0, n_players) * np.tile(np.array([-1.0, 0.0, 1.0]), (n_players, 1)).T
            + np.tile(np.array([1.0, 0.0, 0.0]), (n_players, 1)).T
        ).T
        for i in range(n_players):
            plt.plot(B_widths, p[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('performance')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4a_performance.png')
        plt.clf()
        for i in range(n_players):
            plt.plot(B_widths, s[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('safety')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4a_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(B_widths, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4a_total_safety.png')
        plt.clf()
        # plot payoffs
        for i in range(n_players):
            plt.plot(B_widths, payoff[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4a_payoff.png')
        plt.clf()

    return strats, p, s, payoff

# run_scenario_4a()



# SCENARIO 4b
# Vary homogeneity of performance AND safety productivity (in same direction)

def _scenario_4b_helper(args):
    n_players, A_, B_ = args
    ones = np.ones(n_players)
    prodFunc = VecProdFunc(A_, a*ones, rho*ones, mu*ones, B_, b*ones, sigma*ones, nu*ones, theta*ones)
    problem = SimpleProblem(d*ones, r, w, prodFunc)
    strat = problem.solve(multicore=False)[-1]
    p = prodFunc.P(strat[:, 1], strat[:, 3])
    s = prodFunc.S(strat[:, 0], strat[:, 2], p)
    payoff = np.empty_like(p)
    for i in range(n_players):
        payoff[i] = problem.net_payoff(strat[:, 0], strat[:, 1], strat[:, 2], strat[:, 3], i)
    return strat, p, s, payoff

def run_scenario_4b(n_players=2, avg_A=10.0, max_A_width=18.0, avg_B=10.0, max_B_width=18.0, n_steps=20, plot=True):
    A_widths = max_A_width * np.linspace(0.0, 1.0, n_steps)
    B_widths = max_B_width * np.linspace(0.0, 1.0, n_steps)
    As = np.array([
        np.linspace(
            avg_A - width / 2,
            avg_A + width / 2,
            n_players
        ) for width in A_widths
    ])
    Bs = np.array([
        np.linspace(
            avg_B - width / 2,
            avg_B + width / 2,
            n_players
        ) for width in B_widths
    ])
    with Pool(n_steps) as pool:
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_4b_helper, [(n_players, A_, B_) for A_, B_ in zip(As, Bs)])))
    strats, p, s, payoff = np.array(strats), np.array(p), np.array(s), np.array(payoff)

    if plot:
        # TODO: update x axis labels to reflect that A is also increasing
        colors = (
            np.linspace(0.0, 1.0, n_players) * np.tile(np.array([-1.0, 0.0, 1.0]), (n_players, 1)).T
            + np.tile(np.array([1.0, 0.0, 0.0]), (n_players, 1)).T
        ).T
        for i in range(n_players):
            plt.plot(B_widths, p[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('performance')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4b_performance.png')
        plt.clf()
        for i in range(n_players):
            plt.plot(B_widths, s[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('safety')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4b_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(B_widths, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4b_total_safety.png')
        plt.clf()
        # plot payoffs
        for i in range(n_players):
            plt.plot(B_widths, payoff[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4b_payoff.png')
        plt.clf()

    return strats, p, s, payoff

# run_scenario_4b()



# SCENARIO 4c
# Vary homogeneity of performance AND safety productivity (in OPPOSITE directions)
# i.e., inequality increases, but A and B go in opposite directions for a given player

def run_scenario_4c(n_players=2, avg_A=10.0, max_A_width=18.0, avg_B=10.0, max_B_width=18.0, n_steps=20, plot=True):
    A_widths = max_A_width * np.linspace(0.0, 1.0, n_steps)
    B_widths = max_B_width * np.linspace(0.0, 1.0, n_steps)
    As = np.array([
        np.flip(np.linspace(
            avg_A - width / 2,
            avg_A + width / 2,
            n_players
        )) for width in A_widths
    ])
    Bs = np.array([
        np.linspace(
            avg_B - width / 2,
            avg_B + width / 2,
            n_players
        ) for width in B_widths
    ])
    with Pool(n_steps) as pool:
        # can re-use helper from 4b
        strats, p, s, payoff = tuple(zip(*pool.map(_scenario_4b_helper, [(n_players, A_, B_) for A_, B_ in zip(As, Bs)])))
    strats, p, s, payoff = np.array(strats), np.array(p), np.array(s), np.array(payoff)

    if plot:
        # TODO: Update x axis labels to actual changes in A and B
        # red means safety up, performance down; blue means performance up, safety down
        colors = (
            np.linspace(0.0, 1.0, n_players) * np.tile(np.array([-1.0, 0.0, 1.0]), (n_players, 1)).T
            + np.tile(np.array([1.0, 0.0, 0.0]), (n_players, 1)).T
        ).T
        for i in range(n_players):
            plt.plot(B_widths, p[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('performance')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4c_performance.png')
        plt.clf()
        for i in range(n_players):
            plt.plot(B_widths, s[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('safety')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4c_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(B_widths, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4c_total_safety.png')
        plt.clf()
        # plot payoffs
        for i in range(n_players):
            plt.plot(B_widths, payoff[:, i], color=colors[i], label=f'B={Bs[:,i]}')
        # plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('Distance between highest and lowest B values')
        plt.savefig(f'plots/sc4c_payoff.png')
        plt.clf()

    return strats, p, s, payoff

run_scenario_4c()



# SCENARIO 5a
# In homogeneous setting, vary r
def _scenario_5a_helper(args):
    n_players, r_, prodFunc = args
    problem = SimpleProblem(np.ones(n_players) * d, r_, w, prodFunc)
    return problem.solve(multicore=False)[-1]

def run_scenario_5a(n_players=2, min_r=0.01, max_r=0.05, n_steps=20, plot=True):
    prodFunc = HomogeneousProdFunc(n_players, A, a, rho, mu, B, b, sigma, nu, theta)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_5a_helper, [(n_players, r_, prodFunc) for r_ in R])
    strats = np.array(strats)
    if plot:
        p = prodFunc.P(strats[..., 1], strats[..., 3])
        s = prodFunc.S(strats[..., 0], strats[..., 2], p)
        # plot performance
        plt.plot(R, p.mean(axis=-1), label='p')
        plt.ylabel('performance')
        plt.xlabel('r')
        plt.savefig('plots/sc5a_performance.png')
        plt.clf()
        # plot safety
        plt.plot(R, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel('r')
        plt.savefig('plots/sc5a_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('r')
        plt.savefig('plots/sc5a_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = SimpleProblem(np.ones(n_players) * d, r_, w, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    strats[i, :, 0],
                    strats[i, :, 1],
                    strats[i, :, 2],
                    strats[i, :, 3],
                    j
                )
        plt.plot(R, payoff.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('r')
        plt.savefig('plots/sc5a_payoff.png')
        plt.clf()
    return strats

# run_scenario_5a()



# SCENARIO 5b
# In homogeneous setting, vary r
def _scenario_5b_helper(args):
    n_players, r_, prodFunc = args
    problem = SimpleProblem(np.ones(n_players) * d, r_, w, prodFunc)
    return problem.solve(multicore=False)[-1]

def run_scenario_5b(n_players=2, min_r=0.01, max_r=0.05, n_steps=20, plot=True):
    prodFunc = HomogeneousProdFunc(n_players, A, a, rho, mu, B, b, sigma, nu, theta)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(n_steps) as pool:
        strats = pool.map(_scenario_5b_helper, [(n_players, r_, prodFunc) for r_ in R])
    strats = np.array(strats)
    if plot:
        p = prodFunc.P(strats[..., 1], strats[..., 3])
        s = prodFunc.S(strats[..., 0], strats[..., 2], p)
        # plot performance
        plt.plot(R, p.mean(axis=-1), label='p')
        plt.ylabel('performance')
        plt.xlabel('r')
        plt.savefig('plots/sc5b_performance.png')
        plt.clf()
        # plot safety
        plt.plot(R, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel('r')
        plt.savefig('plots/sc5b_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(R, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('r')
        plt.savefig('plots/sc5b_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoff = np.empty_like(p)
        for i, r_ in enumerate(R):
            problem = SimpleProblem(np.ones(n_players) * d, r_, w, prodFunc)
            for j in range(n_players):
                payoff[i, j] = problem.net_payoff(
                    strats[i, :, 0],
                    strats[i, :, 1],
                    strats[i, :, 2],
                    strats[i, :, 3],
                    j
                )
        plt.plot(R, payoff.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel('r')
        plt.savefig('plots/sc5b_payoff.png')
        plt.clf()
    return strats

# run_scenario_5b()
