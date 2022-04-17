import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
from typing import Callable

# set active directory to location of this file
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from simple_model import ProdFunc, CSF, Problem
# check if cpp backend is available
if os.path.exists('build/libpybindings.so'):
    BACKEND = 'cpp'
    from cpp_bindings import solve, prod_F, get_payoffs
else:
    BACKEND = 'python'

print(f'Using {BACKEND} backend')


VEC_PARAM_NAMES = ['A', 'alpha', 'B', 'beta', 'theta', 'd', 'r']


def _python_multiproc_helper(args):
    (
        n_players,
        A, alpha, B, beta, theta,
        d, r,
        W, L, a_w, a_l,
        max_iters, exit_tol, nlp_max_iters, nlp_exit_tol
    ) = args
    prodFunc = ProdFunc(A, alpha, B, beta, theta)
    csf = CSF(W, L, a_w, a_l)
    problem = Problem(d, r, prodFunc, csf)
    return problem.solve(
        max_iters, iter_tol=exit_tol,
        solver_max_iters=nlp_max_iters, solver_tol=nlp_exit_tol
    )[-1]


def _cpp_multiproc_helper(args):
    (
        n_players,
        A, alpha, B, beta, theta,
        d, r,
        W, L, a_w, a_l,
        max_iters, exit_tol, nlp_max_iters, nlp_exit_tol
    ) = args
    return solve(
        n_players,
        A,
        alpha,
        B,
        beta,
        theta,
        d,
        r,
        W=W,
        L=L,
        a_w=a_w,
        a_l=a_l,
        max_iters=max_iters,
        exit_tol=exit_tol,
        ipopt_max_iters=nlp_max_iters,
        ipopt_tol=nlp_exit_tol
    )


class Scenario:

    def __init__(
        self,
        n_players: int,
        # params for production functions
        A: np.ndarray,  # safety productivity
        alpha: np.ndarray,  # safety returns to scale
        B: np.ndarray,  # performance productivity
        beta: np.ndarray,  # performance returns to scale
        theta: np.ndarray,  # safety scaling factor (higher theta -> more p makes s more expensive)
        # params for player objectives
        d: np.ndarray,  # cost of disaster
        r: np.ndarray,  # factor cost
        # params for CSF
        W: float = 1.0,  # reward if winner
        L: float = 0.0,  # reward if loser
        a_w: float = 0.0,  # reward per p if winner
        a_l: float = 0.0,  # reward per p if loser
        # params for solver
        max_iters: int = 500,
        exit_tol: float = 1e-3,  # stop iterating if players' strategies change by less than this in an iteration
        nlp_max_iters: int = 500,
        nlp_exit_tol: float = 1e-3,
        # which params are we changing?
        varying_param: str = 'r',  # By default we look at changes in r
        # vary_secondary: str = None  # not necessary to provide a secondary varying param
    ):
        # save params to this object's memory
        self.n_players = n_players
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta
        self.theta = theta
        self.d = d
        self.r = r
        self.W = W
        self.L = L
        self.a_w = a_w
        self.a_l = a_l
        self.max_iters = max_iters
        self.exit_tol = exit_tol
        self.nlp_max_iters = nlp_max_iters
        self.nlp_exit_tol = nlp_exit_tol
        self.varying_param = varying_param
        # self.vary_secondary = vary_secondary
        self.n_steps = 0
        # self.n_steps_secondary = 0
        # make sure the vector parameters are the right sizes
        for param_name in VEC_PARAM_NAMES:
            param = getattr(self, param_name)
            if varying_param == param_name:
                assert param.ndim == 1, "Primary varying param is expected to be a 1d numpy array"
                
                self.n_steps = len(param)
            # elif vary_secondary == param_name:
            #     assert(
            #         param.ndim == 2 and param.shape[1] == n_players,
            #         "Secondary varying param is expected to be 2d numpy array; second dimension should match number of players"
            #     )
            #     self.n_steps_secondary = param.shape[0]
            else:
                assert param.ndim == 1 and len(param) == n_players, "Length of param should match number of players"
    
    def _solver_helper(self, _multiproc_helper: Callable, param_dict: dict):
        with Pool(min(cpu_count(), self.n_steps)) as pool:
            strats = pool.map(
                _multiproc_helper,
                [
                    (
                        self.n_players,
                        A_,
                        alpha_,
                        B_,
                        beta_,
                        theta_,
                        d_,
                        r_,
                        self.W,
                        self.L,
                        self.a_w,
                        self.a_l,
                        self.max_iters,
                        self.exit_tol,
                        self.nlp_max_iters,
                        self.nlp_exit_tol
                    )
                    for A_, alpha_, B_, beta_, theta_, d_, r_ in zip(
                        param_dict['A'],
                        param_dict['alpha'],
                        param_dict['B'],
                        param_dict['beta'],
                        param_dict['theta'],
                        param_dict['d'],
                        param_dict['r']
                    )
                ]
            )
        return strats
    
    def _plot_helper(self, s: np.ndarray, p: np.ndarray, payoffs: np.ndarray, plotname: str, labels: list = None):
        xvar = getattr(self, self.varying_param)
        # plot performance
        if labels is None:
            plt.plot(xvar, p.mean(axis=-1))
        else:
            for i in range(self.n_players):
                plt.plot(xvar, p[:, i], label=labels[i])
            plt.legend()
        plt.ylabel('performance')
        plt.xlabel(self.varying_param)
        plt.savefig(f'plots/{plotname}_performance.png')
        plt.clf()
        if labels is None:
            plt.plot(xvar, s.mean(axis=-1))
        else:
            for i in range(self.n_players):
                plt.plot(xvar, s[:, i], label=labels[i])
            plt.legend()
        plt.ylabel('safety')
        plt.xlabel(self.varying_param)
        plt.savefig(f'plots/{plotname}_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(xvar, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel(self.varying_param)
        plt.savefig(f'plots/{plotname}_total_safety.png')
        plt.clf()
        # plot net payoffs
        if labels is None:
            plt.plot(xvar, payoffs.mean(axis=-1))
        else:
            for i in range(self.n_players):
                plt.plot(xvar, payoffs[:, i], label=labels[i])
            plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel(self.varying_param)
        plt.savefig(f'plots/{plotname}_payoff.png')
        plt.clf()
        
    def _solve_cpp(self, param_dict: dict, plot: bool, plotname: str, labels: list = None):
        assert BACKEND == 'cpp', 'You must have cpp backend available to use this method'
        strats = self._solver_helper(_cpp_multiproc_helper, param_dict)
        if plot:
            # get s and p for each strategy
            s_p = np.array([
                prod_F(
                    self.n_players,
                    strat[:, 0].copy(),
                    strat[:, 1].copy(),
                    A_,
                    alpha_,
                    B_,
                    beta_,
                    theta_
                )
                for strat, A_, alpha_, B_, beta_, theta_ in zip(
                    strats,
                    param_dict['A'],
                    param_dict['alpha'],
                    param_dict['B'],
                    param_dict['beta'],
                    param_dict['theta']
                )
            ])
            s, p = s_p[:, 0, :], s_p[:, 1, :]
            payoffs = np.array([
                get_payoffs(
                    self.n_players,
                    strat[:, 0].copy(),
                    strat[:, 1].copy(),
                    A_,
                    alpha_,
                    B_,
                    beta_,
                    theta_,
                    d_,
                    r_,
                    W=self.W,
                    L=self.L,
                    a_w=self.a_w,
                    a_l=self.a_w,
                )
                for strat, A_, alpha_, B_, beta_, theta_, d_, r_ in zip(
                    strats,
                    param_dict['A'],
                    param_dict['alpha'],
                    param_dict['B'],
                    param_dict['beta'],
                    param_dict['theta'],
                    param_dict['d'],
                    param_dict['r']
                )
            ])
            self._plot_helper(s, p, payoffs, plotname, labels)
        return strats
            

    def _solve_python(self, param_dict: dict, plot: bool, plotname: str, labels: list = None):
        strats = self._solver_helper(_python_multiproc_helper, param_dict)
        if plot:
            # get s and p for each strategy
            prodFuncs = [
                ProdFunc(A_, alpha_, B_, beta_, theta_)
                for A_, alpha_, B_, beta_, theta_ in zip(
                    param_dict['A'],
                    param_dict['alpha'],
                    param_dict['B'],
                    param_dict['beta'],
                    param_dict['theta']
                )
            ]
            s_p = np.array([
                prodFunc.F(strat[:, 0], strat[:, 1])
                for prodFunc, strat in zip(prodFuncs, strats)
            ])
            s, p = s_p[:, 0, :], s_p[:, 1, :]
            # get payoffs for each strategy
            problems = [
                Problem(
                    param_dict['d'][i],
                    param_dict['r'][i],
                    prodFunc,
                    CSF(self.W, self.L, self.a_w, self.a_l)
                )
                for i, prodFunc in enumerate(prodFuncs)
            ]
            payoffs = np.array([
                problem.all_net_payoffs(
                    strat[:, 0], strat[:, 1]
                )
                for strat, problem in zip(strats, problems)
            ])
            self._plot_helper(s, p, payoffs, plotname, labels)
        return strats

    
    def solve(self, plot: bool = True, plotname: str = 'scenario', labels: list = None):
        if labels is not None:
            assert len(labels) == self.n_players, "Length of labels should match number of players"
        # build dict of params to solve over
        param_dict = {
            param_name:
            np.tile(
                getattr(self, param_name),
                (self.n_steps, 1)
            ) if param_name != self.varying_param
            else np.tile(
                getattr(self, param_name),
                (self.n_players, 1)
            ).T.copy()  # copy so it remains contiguous in memory
            for param_name in VEC_PARAM_NAMES
        }
        if BACKEND == 'cpp':
            return self._solve_cpp(param_dict, plot, plotname, labels)
        elif BACKEND == 'python':
            return self._solve_python(param_dict, plot, plotname, labels)



if __name__ == '__main__':
    # Run some examples

    # Example 0: What happens if we increase r (factor cost) in a case where everyone is identical
    scenario = Scenario(
        n_players = 2,
        A = np.array([10., 10.]),
        alpha = np.array([0.5, 0.5]),
        B = np.array([10., 10.]),
        beta = np.array([0.5, 0.5]),
        theta = np.array([0., 0.]),
        d = np.array([1., 1.]),
        r = np.linspace(0.02, 0.04, 20),
        varying_param='r'  # We need to specify which parameter we're varying; this will be the x-axis on resulting plots
    )
    scenario.solve(
        plotname='example0',
        labels = None  # We don't provide labels for each player, since they're all the same
    )

    # Example 1: What happens if we increase B in a case where one player has a higher A than the other?
    scenario = Scenario(
        n_players = 2,
        A = np.array([5., 10.]),
        alpha = np.array([0.5, 0.5]),
        B = np.linspace(10., 20., 20),
        beta = np.array([0.5, 0.5]),
        theta = np.array([0., 0.]),
        d = np.array([1., 1.]),
        r = np.array([0.04, 0.04]),
        varying_param='B'
    )
    scenario.solve(
        plotname='example1',
        labels=['A=5', 'A=10']  # We provide labels for each player here since the players have different params
    )

    # Example 2: What if r increases when we have 3 players of varying productivities?
    scenario = Scenario(
        n_players = 3,
        # notice that all parameters here (except the one we vary) should be arrays with length == n_players
        A = np.array([5., 10., 15.]),
        alpha = np.array([0.5, 0.5, 0.5]),
        B = np.array([5., 10., 15.]),
        beta = np.array([0.5, 0.5, 0.5]),
        theta = np.array([0., 0., 0.]),
        d = np.array([1., 1., 1.]),
        r = np.linspace(0.02, 0.04, 20),
        varying_param='r'
    )
    scenario.solve(
        plotname='example2',
        labels=['weak player', 'medium player', 'strong player']
    )
