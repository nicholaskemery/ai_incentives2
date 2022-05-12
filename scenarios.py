import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, cpu_count
from typing import Callable, Tuple

# set active directory to location of this file
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from simple_model import ProdFunc, CSF, Problem, HybridProblem, MixedProblem

# check if cpp backend is available
if os.path.exists('build/libpybindings.so'):
    CPP_AVAILABLE = True
    try:
        from cpp_bindings import solve, prod_F, get_payoffs
    except OSError:
        CPP_AVAILABLE = False
else:
    CPP_AVAILABLE = False

# Note on backends:
# Python backend can actually be faster than cpp backend on a lot of problems,
# so you may want to disable cpp backend even if it is available.
# Cpp backend is more likely to be better on more complicated problems.
# Also note that the same tolerance params will probably yield less precise
# solutions on the cpp backend, relative to the python backend
# (i.e., for comparable precision on the cpp backend, increase nlp_exit_tol).

DEFAULT_MAX_ITERS = 50
DEFAULT_EXIT_TOL = 1e-6
DEFAULT_NLP_MAX_ITERS = 500
DEFAULT_NLP_ITER_TOL = 1e-6

PLOT_FIGSIZE = (15, 12)

VEC_PARAM_NAMES = ['A', 'alpha', 'B', 'beta', 'theta', 'd', 'r']

def _roots_multiproc_helper(args):
    (
        _,
        A, alpha, B, beta, theta,
        d, r,
        w, l, a_w, a_l,
        _, _, _, _
    ) = args
    prodFunc = ProdFunc(A, alpha, B, beta, theta)
    csf = CSF(w, l, a_w, a_l)
    result = Problem(d, r, prodFunc, csf).solve()
    return (result.results, result.s, result.p, result.payoffs)


def _hybrid_multiproc_helper(args):
    (
        _,
        A, alpha, B, beta, theta,
        d, r,
        w, l, a_w, a_l,
        max_iters, exit_tol, nlp_max_iters, nlp_exit_tol
    ) = args
    prodFunc = ProdFunc(A, alpha, B, beta, theta)
    csf = CSF(w, l, a_w, a_l)
    result = HybridProblem(d, r, prodFunc, csf).solve(
        iter_max_iters = max_iters, iter_tol = exit_tol,
        solver_tol = nlp_exit_tol, solver_max_iters = nlp_max_iters
    )
    return result.unpack()


def _python_multiproc_helper(args):
    (
        _,
        A, alpha, B, beta, theta,
        d, r,
        w, l, a_w, a_l,
        max_iters, exit_tol, nlp_max_iters, nlp_exit_tol
    ) = args
    prodFunc = ProdFunc(A, alpha, B, beta, theta)
    csf = CSF(w, l, a_w, a_l)
    problem = MixedProblem(d, r, prodFunc, csf)
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


def get_colors(
    n: int,
    color_1: Tuple[float, float, float] = (0., 0.5, 1.), 
    color_2: Tuple[float, float, float] = (1., 0., 0.)
):
    lambda_ = np.linspace(0., 1., n).reshape((-1, 1))
    return (1 - lambda_) * np.array(color_1) + lambda_ * np.array(color_2)


def _multivar_plot_helper(
    labels: list,
    yvar_list: list,
    ax: plt.Axes,
    xvar: np.ndarray, xvar_label: str,
    yvar_label: str,
    colors: np.ndarray,
    combine: bool = True
):
    if labels is None:
        for yvar in yvar_list:
            ax.plot(xvar, yvar.mean(axis=-1))
    else:
        for yvar, label, color in zip(yvar_list, labels, colors):
            if yvar.ndim == 1:
                ax.plot(xvar, yvar, label=label, color=color)
            elif combine:
                ax.plot(xvar, yvar.mean(axis=-1), label=label, color=color)
            else:
                ax.plot(xvar, yvar[:, 0], label=label, color=color)
                ax.plot(xvar, yvar[:, 1:], color=color)
        ax.legend()
    ax.set_ylabel(yvar_label)
    ax.set_xlabel(xvar_label)



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
        w: float = 1.0,  # reward if winner
        l: float = 0.0,  # reward if loser
        a_w: float = 0.0,  # reward per p if winner
        a_l: float = 0.0,  # reward per p if loser
        # params for solver (ignored if using roots method)
        max_iters: int = DEFAULT_MAX_ITERS,
        exit_tol: float = DEFAULT_EXIT_TOL,  # stop iterating if players' strategies change by less than this in an iteration
        nlp_max_iters: int = DEFAULT_MAX_ITERS,
        nlp_exit_tol: float = DEFAULT_EXIT_TOL,
        # which params are we changing?
        varying_param: str = 'r',  # By default we look at changes in r
        secondary_varying_param: str = None,  # not necessary to provide a secondary varying param
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
        self.w = w
        self.l = l
        self.a_w = a_w
        self.a_l = a_l
        self.max_iters = max_iters
        self.exit_tol = exit_tol
        self.nlp_max_iters = nlp_max_iters
        self.nlp_exit_tol = nlp_exit_tol
        self.varying_param = varying_param
        self.secondary_varying_param = secondary_varying_param
        self.n_steps = 0
        self.n_steps_secondary = 0
        # make sure the vector parameters are the right sizes
        for param_name in VEC_PARAM_NAMES:
            param = getattr(self, param_name)
            if varying_param == param_name:
                assert param.ndim == 1, "Primary varying param is expected to be a 1d numpy array"
                
                self.n_steps = len(param)
            elif secondary_varying_param == param_name:
                assert param.ndim == 2 and param.shape[1] == n_players, \
                    "Secondary varying param is expected to be 2d numpy array; second dimension should match number of players"
                self.n_steps_secondary = param.shape[0]
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
                        self.w,
                        self.l,
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
    
    def _plot_helper(
        self,
        s: np.ndarray,
        p: np.ndarray,
        payoffs: np.ndarray,
        plotname: str,
        labels: list = None,
        title: str = None,
        logscale: bool = False
    ):
        xvar = getattr(self, self.varying_param)
        fig, axs = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
        # plot performance
        if labels is None:
            axs[0, 0].plot(xvar, p.mean(axis=-1), logscale)
        else:
            for i in range(self.n_players):
                axs[0, 0].plot(xvar, p[:, i], label=labels[i])
            axs[0, 0].legend()
        axs[0, 0].set_ylabel('performance')
        axs[0, 0].set_xlabel(self.varying_param)
        # plt.xlabel(self.varying_param)
        # plt.savefig(f'plots/{plotname}_performance.png')
        # plt.clf()
        # plot safety
        if labels is None:
            axs[0, 1].plot(xvar, s.mean(axis=-1))
        else:
            for i in range(self.n_players):
                axs[0, 1].plot(xvar, s[:, i], label=labels[i])
            axs[0, 1].legend()
        axs[0, 1].set_ylabel('safety')
        axs[0, 1].set_xlabel(self.varying_param)
        # plt.xlabel(self.varying_param)
        # plt.savefig(f'plots/{plotname}_safety.png')
        # plt.clf()
        if logscale:
            axs[0, 0].semilogy()
            axs[0, 1].semilogy()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        axs[1, 0].plot(xvar, total_proba)
        axs[1, 0].set_ylabel('Proba of safe outcome')
        axs[1, 0].set_xlabel(self.varying_param)
        # plt.xlabel(self.varying_param)
        # plt.savefig(f'plots/{plotname}_total_safety.png')
        # plt.clf()
        # plot net payoffs
        if labels is None:
            axs[1, 1].plot(xvar, payoffs.mean(axis=-1))
        else:
            for i in range(self.n_players):
                axs[1, 1].plot(xvar, payoffs[:, i], label=labels[i])
            axs[1, 1].legend()
        axs[1, 1].set_ylabel('net payoff')
        axs[1, 1].set_xlabel(self.varying_param)

        if title is not None:
            fig.suptitle(title)
        plt.savefig(f'plots/{plotname}.png')
        plt.clf()
        
    def _solve_cpp(
        self,
        param_dict: dict,
        plot: bool,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False
    ):
        strats = self._solver_helper(_cpp_multiproc_helper, param_dict)
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
                W=self.w,
                L=self.l,
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
        if plot:
            self._plot_helper(s, p, payoffs, plotname, labels, title, logscale)
        return strats, s, p, payoffs
            

    def _solve_python(
        self,
        param_dict: dict,
        plot: bool,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False
    ):
        strats = self._solver_helper(_python_multiproc_helper, param_dict)
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
            MixedProblem(
                param_dict['d'][i],
                param_dict['r'][i],
                prodFunc,
                CSF(self.w, self.l, self.a_w, self.a_l)
            )
            for i, prodFunc in enumerate(prodFuncs)
        ]
        payoffs = np.array([
            problem.all_net_payoffs(
                strat[:, 0], strat[:, 1]
            )
            for strat, problem in zip(strats, problems)
        ])
        if plot:
            self._plot_helper(s, p, payoffs, plotname, labels, title, logscale)
        return strats, s, p, payoffs
    
    def _solve_roots(
        self,
        param_dict: dict,
        plot: bool,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False
    ):
        strats, s, p, payoffs = tuple(
            np.array(x) for x in zip(*self._solver_helper(_roots_multiproc_helper, param_dict))
        )
        if plot:
            self._plot_helper(s, p, payoffs, plotname, labels, title, logscale)
        return strats, s, p, payoffs
    
    def _solve_hybrid(
        self,
        param_dict: dict,
        plot: bool,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False
    ):
        strats, s, p, payoffs = tuple(
            np.array(x) for x in zip(*self._solver_helper(_hybrid_multiproc_helper, param_dict))
        )
        if plot:
            self._plot_helper(s, p, payoffs, plotname, labels, title, logscale)
        return strats, s, p, payoffs
    
    def solve_with_secondary_variation(
        self,
        plot: bool = True,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False,
        method: str = 'roots'
    ):
        if labels is not None:
            assert len(labels) == self.n_steps_secondary, "Length of labels should match number of secondary variations"
        param_dicts = [
            {
                param_name:
                np.tile(
                    getattr(self, param_name),
                    (self.n_players, 1)
                ).T.copy()
                if param_name == self.varying_param
                else
                np.tile(
                    secondary_variation,
                    (self.n_steps, 1)
                )
                if param_name == self.secondary_varying_param
                else
                np.tile(
                    getattr(self, param_name),
                    (self.n_steps, 1)
                )
                for param_name in VEC_PARAM_NAMES
            }
            for secondary_variation in getattr(self, self.secondary_varying_param)
        ]
        solver = self._solve_cpp if method == 'cpp' else self._solve_python if method == 'python' else self._solve_roots
        strats_list, s_list, p_list, payoffs_list = tuple(zip(*[
            solver(param_dict, plot = False) for param_dict in param_dicts
        ]))
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
            xvar = getattr(self, self.varying_param)
            colors = get_colors(self.n_steps_secondary)
            combine = all((
                np.isclose(s.T, s[:, 0], rtol=0.01).all() and np.isclose(p.T, p[:, 0], rtol=0.01).all()
                for s, p in zip(s_list, p_list)
            ))
            # plot performance
            _multivar_plot_helper(
                labels,
                p_list,
                axs[0, 0],
                xvar, self.varying_param,
                'performance',
                colors,
                combine
            )
            # plot safety
            _multivar_plot_helper(
                labels,
                s_list,
                axs[0, 1],
                xvar, self.varying_param,
                'safety',
                colors,
                combine
            )
            # set performance and safety plots to log scale
            if logscale:
                axs[0, 0].semilogy()
                axs[0, 1].semilogy()
            # plot total disaster proba
            total_proba_list = [(s / (1 + s)).prod(axis=-1) for s in s_list]
            _multivar_plot_helper(
                labels,
                total_proba_list,
                axs[1, 0],
                xvar, self.varying_param,
                'proba of safe outcome',
                colors
            )
            # plot net payoffs
            _multivar_plot_helper(
                labels,
                payoffs_list,
                axs[1, 1],
                xvar, self.varying_param,
                'net payoff',
                colors,
                combine
            )

            if title is not None:
                fig.suptitle(title)
            plt.savefig(f'plots/{plotname}.png')
            plt.clf()
        return tuple(np.array(x) for x in (strats_list, s_list, p_list, payoffs_list))
    
    def solve(
        self,
        plot: bool = True,
        plotname: str = 'scenario',
        labels: list = None,
        title: str = None,
        logscale: bool = False,
        method: str = 'hybrid' # other options are 'roots', 'python' and 'cpp'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # returns strats, s, p, payoffs
        if self.n_steps_secondary != 0:
            return self.solve_with_secondary_variation(plot, plotname, labels, title, logscale, method=method)

        if labels is not None:
            assert len(labels) == self.n_players, "Length of labels should match number of players"
        # build dict of params to solve over
        param_dict = {
            param_name:
            np.tile(
                getattr(self, param_name),
                (self.n_steps, 1)
            ) 
            if param_name != self.varying_param
            else
            np.tile(
                getattr(self, param_name),
                (self.n_players, 1)
            ).T.copy()  # copy so it remains contiguous in memory
            for param_name in VEC_PARAM_NAMES
        }
        solver = self._solve_cpp if method == 'cpp' \
            else self._solve_python if method == 'python' \
                else self._solve_roots if method == 'roots' \
                    else self._solve_hybrid
        return solver(param_dict, plot, plotname, labels, title, logscale)


if __name__ == '__main__':
    # Run some examples
    method = 'hybrid'

    # Example 0: What happens if we increase r (factor cost) in a case where everyone is identical
    scenario = Scenario(
        n_players = 2,
        A = np.array([10., 10.]),
        alpha = np.array([0.5, 0.5]),
        B = np.array([10., 10.]),
        beta = np.array([0.5, 0.5]),
        theta = np.array([0.25, 0.25]),
        d = np.array([1., 1.]),
        r = np.linspace(0.02, 0.04, 20),
        varying_param = 'r'  # We need to specify which parameter we're varying; this will be the x-axis on resulting plots
    )
    scenario.solve(
        plotname = 'example0',
        labels = None,  # We don't provide labels for each player, since they're all the same
        title = 'Outcomes with r increasing, homogeneous players',
        method = method
    )

    # Example 1: What happens if we increase B in a case where one player has a higher A than the other?
    scenario = Scenario(
        n_players = 2,
        A = np.array([5., 10.]),
        alpha = np.array([0.5, 0.5]),
        B = np.linspace(10., 20., 20),
        beta = np.array([0.5, 0.5]),
        theta = np.array([0.25, 0.25]),
        d = np.array([1., 1.]),
        r = np.array([0.04, 0.04]),
        varying_param = 'B'
    )
    scenario.solve(
        plotname = 'example1',
        labels = ['A=5', 'A=10'],  # We provide labels for each player here since the players have different params
        title = 'Outcomes with B increasing, heterogeneous players',
        method = method
    )

    # Example 2: What if r increases when we have 3 players of varying productivities?
    scenario = Scenario(
        n_players = 3,
        # notice that all parameters here (except the one we vary) should be arrays with length == n_players
        A = np.array([5., 10., 15.]),
        alpha = np.array([0.5, 0.5, 0.5]),
        B = np.array([5., 10., 15.]),
        beta = np.array([0.5, 0.5, 0.5]),
        theta = np.array([0.25, 0.25, 0.25]),
        d = np.array([1., 1., 1.]),
        r = np.linspace(0.02, 0.04, 20),
        varying_param = 'r'
    )
    scenario.solve(
        plotname = 'example2',
        labels = ['weak player', 'medium player', 'strong player'],
        title = 'Outcomes with three players of varying productivity (A and B)',
        method = method
    )

    # Example 3: Change two things at once (note: all other params should be homogeneous in this case)
    scenario = Scenario(
        n_players = 2,
        A = np.array([10., 10.]),
        alpha = np.array([0.5, 0.5]),
        B = np.array([
            [10., 10.],
            [20., 20.],
            [30., 30.]
        ]),
        beta = np.array([0.5, 0.5]),
        theta = np.array([0.25, 0.25]),
        d = np.array([1., 1.]),
        r = np.linspace(0.02, 0.04, 20),
        varying_param = 'r',
        secondary_varying_param = 'B'
    )
    scenario.solve(
        plotname = 'example3',
        labels = ['B=10', 'B=20', 'B=30'],
        title = 'Outcomes with varying r and B',
        method = method
    )
