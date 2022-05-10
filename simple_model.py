import numpy as np
# import pygambit as pg
from scipy import optimize
from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')



SOLVER_TOL = 1e-3
SOLVER_MAX_ITERS = 500
DEFAULT_WINDOW = 1


class ProdFunc:

    def __init__(self, A, alpha, B, beta, theta):
        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta
        self.theta = theta

        assert(len(self.A) == len(self.alpha) == len(self.B) == len(self.beta) == len(self.theta))
        self.n = len(self.A)
    
    def F_single_i(self, i, Ks, Kp):
        """
        Determine safety & performance for a given person (index i) at various levels of Ks and Kp

        i is an integer index
        Ks is a np array of positive floats
        Kp is a np array of positive floats (same length as Ks)

        returns s, p; both are np arrays of same type and shape as Ks & Kp
        """
        p = self.B[i] * Kp ** self.beta[i]
        s = self.A[i] * Ks ** self.alpha[i] * p ** -self.theta[i]
        return s, p
    
    def F(self, Ks, Kp):
        """
        Determine safety & performance for all persons at given levels of Ks and Kp

        Ks and Kp are 1d np arrays of positive floats
        Ks & Kp should both have length/size of last dimension equal to self.n

        returns s, p; both are np arrays of same type and shape as Ks & Kp
        """
        p = self.B * Kp ** self.beta
        s = self.A * Ks ** self.alpha * p ** -self.theta
        return s, p

    def get_jac(self, i):
        """
        returns jacobian matrix function for self.F_single_i
        """
        def jac(x):
            s, p = self.F_single_i(i, x[..., 0], x[..., 1])
            # s_mult is ds/dKs if theta = 0
            s_mult = self.A[i] * self.alpha[i] * (s / self.A[i]) ** (1 - 1 / self.alpha[i])
            # p_mult is dp/dKp
            p_mult = self.B[i] * self.beta[i] * (p / self.B[i]) ** (1 - 1 / self.beta[i])
            return np.array([
                # partial derivatives of s
                [
                    s_mult * p ** -self.theta[i],  # w.r.t. Ks
                    -self.theta[i] * s * p**(-self.theta[i] - 1) * p_mult  # w.r.t. Kp
                ],
                # partial derivatives of p
                [
                    0.0,  # w.r.t. Ks
                    p_mult  # w.r.t. Kp
                ]
            ])
        return jac



class CSF:

    def __init__(self, w: float = 1.0, l: float = 0.0, a_w: float = 0.0, a_l: float = 0.0):
        """
        w is reward for winner
        l is reward for loser(s)
        a_w is reward per unit of p for winner
        a_l is reward per unit of p for loser

        win proba is p[i] / sum(p)
        """
        self.w = w
        self.l = l
        self.a_w = a_w
        self.a_l = a_l
    
    def reward(self, i: int, p: np.ndarray):
        win_proba = p[..., i] / p.sum(axis=-1)
        return (
            (self.w + p[..., i] * self.a_w) * win_proba
            + (self.l + p[..., i] * self.a_l) * (1 - win_proba)
        )
    
    def all_rewards(self, p: np.ndarray):
        win_probas = p / p.sum(axis=-1)
        return (
            (self.w + p * self.a_w) * win_probas
            + (self.l + p * self.a_l) * (1 - win_probas)
        )
    
    def reward_deriv(self, i: int, p: np.ndarray):
        sum_ = p.sum(axis=-1)
        win_proba = p[..., i] / sum_
        win_proba_deriv = (sum_ - p[..., i]) / sum_**2
        return (
            self.a_l + (self.a_w - self.a_l) * win_proba
            + (self.w - self.l + (self.a_w - self.a_l) * p[..., i]) * win_proba_deriv
        )
    
    def all_reward_derivs(self, p: np.ndarray):
        sum_ = p.sum(axis=-1)
        win_probas = p / sum_
        win_proba_derivs = (sum_ - p) / sum_**2
        return (
            self.a_l + (self.a_w - self.a_l) * win_probas
            + (self.w - self.l + (self.a_w - self.a_l) * p) * win_proba_derivs
        )



@dataclass
class SolverResult:
    success: bool
    results: np.ndarray
    s: np.ndarray
    p: np.ndarray
    payoffs: np.ndarray

def get_index(solverResult: SolverResult, i: int) -> SolverResult:
    return SolverResult(
        solverResult.success,
        solverResult.results[i],
        solverResult.s[i],
        solverResult.p[i],
        solverResult.payoffs[i]
    )

def join_results(solverResults) -> SolverResult:
    return SolverResult(
        all((r.success for r in solverResults)),
        np.stack((r.results for r in solverResults)),
        np.stack((r.s for r in solverResults)),
        np.stack((r.p for r in solverResults)),
        np.stack((r.payoffs for r in solverResults))
    )


class Problem:

    def __init__(self, d: np.ndarray, r: np.ndarray, prodFunc: ProdFunc, csf: CSF = CSF()):
        """
        d and r are np arrays of length n
        """
        self.d = d
        self.r = r
        self.prodFunc = prodFunc
        self.csf = csf

        self.n = len(d)
        assert(prodFunc.n == self.n == len(r))

    def net_payoff(self, i: int, Ks: np.ndarray, Kp: np.ndarray):
        """
        Gets payoff for player i when strategies are represented by Ks and Kp vectors
        i is an integer index
        Ks and Kp are np arrays with length == self.n
        outputs an ndarray (if ndim of Ks and Kp > 1) or a float
        """
        s, p = self.prodFunc.F(Ks, Kp)
        proba = (s / (1 + s)).prod(axis=-1)
        return proba * self.csf.reward(i, p) - (1 - proba) * self.d[i] - self.r[i] * (Ks[..., i] + Kp[..., i])
    
    def all_net_payoffs(self, Ks: np.ndarray, Kp: np.ndarray):
        """Basically just runs self.net_payoff for all the i"""
        s, p = self.prodFunc.F(Ks, Kp)
        proba = (s / (1 + s)).prod(axis=-1)
        if p.ndim == 1:
            return proba * self.csf.all_rewards(p) - (1 - proba) * self.d - self.r * (Ks + Kp)
        elif p.ndim == 2:
            return np.stack(
                [
                    proba_ * self.csf.all_rewards(p_) - (1 - proba_) * self.d - self.r * (Ks_ + Kp_)
                    for proba_, p_, Ks_, Kp_ in zip(proba, p, Ks, Kp)
                ]
            )
        else:
            raise ValueError('Inputs must be 1- or 2-dimensional')
    
    def get_jac(self):
        prod_jacs = [self.prodFunc.get_jac(i) for i in range(self.n)]
        def jac(x):
            """
            x is an n x 2 array of strategies;
            returns an n x 2 jacobian matrix of the payoff function
            """
            s, p = self.prodFunc.F(x[..., 0], x[..., 1])
            probas = s / (1 + s)
            proba = probas.prod(axis=-1)
            proba_mult = proba / (s * (1 + s))
            prod_jac_ = np.array([prod_jac(x[..., i, :]) for i, prod_jac in enumerate(prod_jacs)])
            s_ks = prod_jac_[..., 0, 0]
            s_kp = prod_jac_[..., 0, 1]
            p_ks = prod_jac_[..., 1, 0]
            p_kp = prod_jac_[..., 1, 1]
            proba_ks = proba_mult * s_ks
            proba_kp = proba_mult * s_kp
            R_ = self.csf.all_rewards(p)
            R_deriv_ = self.csf.all_reward_derivs(p)
            return np.array([
                proba_ks * (R_ + self.d) + proba * R_deriv_ * p_ks - self.r,
                proba_kp * (R_ + self.d) + proba * R_deriv_ * p_kp - self.r
            ])
        return jac
    
    def _null_result(self) -> SolverResult:
        return SolverResult(
            False,
            np.ones((self.n, 2)) * np.nan,
            np.ones(self.n) * np.nan,
            np.ones(self.n) * np.nan,
            np.ones(self.n) * np.nan
        )
    
    def _resolve_multiple_solutions(
        self,
        result: SolverResult,
        prefer_best_avg: bool = False
    ) -> SolverResult:
        if result.results.shape[0] == 1:
            return get_index(result, 0)
        # try and see if one solution is best for all players
        argmaxes = np.argmax(result.payoffs, axis=0)  # shape = self.n
        best = argmaxes[0]
        if any(x != best for x in argmaxes[1:]):
            # Note: Maybe possible to use pygambit to disambiguate solutions here?
            if prefer_best_avg:
                print('Warning: Multiple potential solutions found; there may be more than one equilibrium!')
                best = np.argmax(np.mean(result.payoffs, axis=1))
            else:
                # just return no result
                return self._null_result()
        return get_index(result, best)

    def _get_unique_results_with_roots_method(self, init_guesses: list) -> SolverResult:
        jac = self.get_jac()
        # in nash equilibrium, all elements of the jacobian should be 0
        # try at multiple initial guesses to be more confident we're not finding local optimum
        results = np.empty((len(init_guesses), self.n, 2))
        successes = np.zeros(len(init_guesses), dtype=bool)
        for i, init_guess in enumerate(init_guesses):
            res = optimize.root(
                fun = lambda x: jac(x.reshape(self.n, 2)).flatten(),
                x0 = init_guess * np.ones((self.n, 2)),
                # x0 = np.exp(np.random.randn(self.n, 2)),
                method = 'lm',
                options = {
                    # this is twice as much as the default maxiter
                    # note that I've found that if it doesn't converge very quickly, it probably won't at all
                    'maxiter': 200 * (self.n+1)
                }
            )
            if res.success and np.all(res.x >= 0):
                successes[i] = True
                results[i] = res.x.reshape((self.n, 2))
        if not any(successes):
            print('Solver failed to converge from the given initial guesses')
            return self._null_result()
        # otherwise, calculate other values and figure out which of the solutions are stable
        results = results[[i for i, s in enumerate(successes) if s]]
        results = results[np.unique(results.astype(np.float16), axis=0, return_index=True)[1]]  # remove results that are approximate duplicates
        p, s = self.prodFunc.F(results[..., 0], results[..., 1])
        payoffs = self.all_net_payoffs(results[..., 0], results[..., 1])  # shape = len(results) x self.n
        # reject solutions with payoffs worse than worse case scenario of doing nothing
        # (players can always get at least -d if they just produce nothing)
        good_idxs = [all(x>= -self.d) for x in payoffs]
        if not any(good_idxs):
            return self._null_result()
        results = results[good_idxs]
        s = s[good_idxs]
        p = p[good_idxs]
        payoffs = payoffs[good_idxs]
        return SolverResult(True, results, s, p, payoffs)

    def solve(self, init_guesses: list = [10.0**i for i in range(-3, 4)]) -> SolverResult:
        solverResult = self._get_unique_results_with_roots_method(init_guesses)
        if not solverResult.success:
            return solverResult
        return self._resolve_multiple_solutions(solverResult)



class HybridProblem(Problem):

    def get_func(self, i: int, last_strats: np.ndarray):
        def func(x):
            last_strats[i, :] = x
            return -self.net_payoff(i, last_strats[:, 0], last_strats[:, 1]).sum()
        return func
    
    def get_jac_single_i(self, i: int, last_strats: np.ndarray):
        prod_jac = self.prodFunc.get_jac(i)
        def jac(x):
            last_strats[i, :] = x
            s, p = self.prodFunc.F(last_strats[:, 0], last_strats[:, 1])
            probas = s / (1 + s)
            proba = probas.prod()
            proba_mult = proba / (s[i] * (1 + s[i]))
            prod_jac_ = prod_jac(last_strats[i, :])
            s_ks = prod_jac_[0, 0]
            s_kp = prod_jac_[0, 1]
            p_ks = prod_jac_[1, 0] # == 0
            p_kp = prod_jac_[1, 1]
            proba_ks = proba_mult * s_ks
            proba_kp = proba_mult * s_kp
            R_ = self.csf.reward(i, p)
            R_deriv_ = self.csf.reward_deriv(i, p)
            return -np.array([
                proba_ks * (R_ + self.d[i]) + proba * R_deriv_ * p_ks - self.r[i],
                proba_kp * (R_ + self.d[i]) + proba * R_deriv_ * p_kp - self.r[i]
            ])
        return jac
    
    def _solve_single_as_iter(self, strats: np.ndarray, solver_tol: float, solver_max_iters: int):
        # get everyone's best response to given strats
        for i in range(self.n):
            res = optimize.minimize(
                self.get_func(i, strats),
                jac=self.get_jac_single_i(i, strats),
                x0=strats[i, :],
                method='trust-constr',
                bounds=optimize.Bounds([0.0, 0.0], [np.inf, np.inf]),
                options={
                    'xtol': solver_tol,
                    'maxiter': solver_max_iters
                }
            )
            strats[i, :] = res.x
        return strats
    
    def _solve_as_iter(
        self,
        strats: np.ndarray,
        iter_tol: float,
        iter_max_iters: int,
        solver_tol: float,
        solver_max_iters: int
    ):
        for t in range(1, iter_max_iters):
            new_strats = self._solve_single_as_iter(
                strats,
                solver_tol=solver_tol,
                solver_max_iters=solver_max_iters
            )
            if np.abs((new_strats - strats) / strats).max() < iter_tol:
                # print(f'Exited on iteration {t}')
                s, p = self.prodFunc.F(new_strats[:, 0], new_strats[:, 1])
                payoffs = self.all_net_payoffs(new_strats[:, 0], new_strats[:, 1])
                return SolverResult(True, new_strats, s, p, payoffs)
            strats = new_strats
        # print(f'Reached max iterations ({solver_max_iters})')
        s, p = self.prodFunc.F(strats[:, 0], strats[:, 1])
        payoffs = self.all_net_payoffs(strats[:, 0], strats[:, 1])
        # Signal not successful since it doesn't seem to have converged
        return SolverResult(False, strats, s, p, payoffs)

    def solve(
        self,
        init_guesses: list = [10.0**i for i in range(-5, 6)],
        iter_tol: float = 1e-3,
        iter_max_iters: int = 10,
        solver_tol: float = SOLVER_TOL,
        solver_max_iters: int = SOLVER_MAX_ITERS
    ):
        # start by looking for solutions with roots method
        solverResult = self._get_unique_results_with_roots_method(init_guesses)
        if not solverResult.success:
            return solverResult
        good_results = []
        for strats in solverResult.results:
            # iterate a bit on suggested solutions to get them to settle down
            iter_result = self._solve_as_iter(
                strats, iter_tol, iter_max_iters, solver_tol, solver_max_iters
            )
            if iter_result.success:
                good_results.append(iter_result)
            # if np.abs((new_strats - strats) / strats).max() < comparison_tol:
            #     good_idxs.append(i)
        return self._resolve_multiple_solutions(join_results(good_results))



class MixedProblem(Problem):

    def __init__(self, d: np.ndarray, r: np.ndarray, prodFunc: ProdFunc, csf: CSF = CSF()):
        super().__init__(d, r, prodFunc, csf)
        self.hist = np.empty((0, self.n, 2))
    
    def get_func(self, i: int):
        assert(self.hist.shape[0] > 0)
        def func(x):
            self.hist[:, i] = np.repeat(x.reshape(1, -1), self.hist.shape[0], axis=0)
            return -self.net_payoff(i, self.hist[..., 0], self.hist[..., 1]).sum()
        return func
    
    def get_jac(self, i: int):
        assert(self.hist.shape[0] > 0)
        prod_jac = self.prodFunc.get_jac(i)
        def jac(x):
            self.hist[:, i, :] = np.repeat(x.reshape(1, -1), self.hist.shape[0], axis=0)
            s, p = self.prodFunc.F(self.hist[..., 0], self.hist[..., 1])
            probas = s / (1 + s)
            proba = probas.prod(axis=-1)
            proba_mult = proba / (s[:, i] * (1 + s[:, i]))
            prod_jac_ = prod_jac(self.hist[:, i, :])
            s_ks = prod_jac_[0, 0]
            s_kp = prod_jac_[0, 1]
            p_ks = prod_jac_[1, 0] # == 0
            p_kp = prod_jac_[1, 1]
            proba_ks = proba_mult * s_ks
            proba_kp = proba_mult * s_kp
            R_ = self.csf.reward(i, p)
            R_deriv_ = self.csf.reward_deriv(i, p)
            return -np.array([
                proba_ks * (R_ + self.d[i]) + proba * R_deriv_ * p_ks - self.r[i],
                proba_kp * (R_ + self.d[i]) + proba * R_deriv_ * p_kp - self.r[i]
            ]).sum(axis=1)
        return jac

    def solve_single(
        self,
        history: np.ndarray,
        verbose: int = 1,
        solver_tol: float = SOLVER_TOL,
        solver_max_iters: int = SOLVER_MAX_ITERS
    ):
        """
        history should be 3d np.array
        dim 0 is iteration
        dim 1 is agent
        dim 2 is factor (s, p)
        """
        assert(self.n == history.shape[1])
        self.hist = history.copy()
        strategies = np.empty((self.n, 2))
        for i in range(self.n):
            res = optimize.minimize(
                self.get_func(i),
                jac=self.get_jac(i),
                x0=history[-1, i, :],
                method='trust-constr',
                bounds=optimize.Bounds([0.0, 0.0], [np.inf, np.inf]),
                options={
                    'xtol': solver_tol,
                    'maxiter': solver_max_iters,
                    'verbose': verbose
                }
            )
            strategies[i, :] = res.x
        return strategies
    
    def solve(
        self,
        T: int = 100,
        window: int = DEFAULT_WINDOW,
        iter_tol: float = 1e-3,
        verbose: int = 0,
        init_guess: float = 1.0, 
        solver_tol: float = SOLVER_TOL,
        solver_max_iters: int = SOLVER_MAX_ITERS
    ):
        history = np.empty((T, self.n, 2))
        history[0, :, :] = np.ones((self.n, 2)) * init_guess
        for t in range(1, T):
            history[t, :, :] = self.solve_single(
                history[max(0, t-window):t, :, :],
                verbose=verbose,
                solver_tol=solver_tol,
                solver_max_iters=solver_max_iters
            )
            if np.abs((history[t, :, :] - history[t-1, :, :]) / history[t-1, :, :]).max() < iter_tol:
                print(f'Exited on iteration {t}')
                return history[:t+1, :, :]
        print('Reached max iterations')
        return history



if __name__ == '__main__':
    # run test
    from time import time
    n = 2
    ones = np.ones(n, dtype=np.float64)
    prodFunc = ProdFunc(1 * ones, 0.5 * ones, 1 * ones, 0.5 * ones, 0.0 * ones)

    print('\nSearching for solution with Problem solver...')
    problem = Problem(0.1 * ones, 0.06 * ones, prodFunc)
    start_time = time()
    sol = problem.solve()
    end_time = time()
    print(f'Found solution in {end_time - start_time:.2f} seconds:')
    print(sol)

    print('\nSearching for solution with MixedProblem solver...')
    problem = MixedProblem(0.1 * ones, 0.06 * ones, prodFunc)
    start_time = time()
    hist = problem.solve()[-1]
    end_time = time()
    print(f'Found solution in {end_time - start_time:.2f} seconds:')
    print(hist)
