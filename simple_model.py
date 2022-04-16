import warnings

import numpy as np
from scipy.optimize import Bounds, minimize

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

    def __init__(self, W: float = 1.0, L: float = 0.0, a_w: float = 0.0, a_l: float = 0.0):
        """
        W is reward for winner
        L is reward for loser(s)
        a_w is reward per unit of p for winner
        a_l is reward per unit of p for loser

        win proba is p[i] / sum(p)
        """
        self.W = W
        self.L = L
        self.a_w = a_w
        self.a_l = a_l
    
    def reward(self, i: int, p: np.ndarray):
        win_proba = p[..., i] / p.sum(axis=-1)
        return (
            (self.W + p[..., i] * self.a_w) * win_proba
            + (self.L + p[..., i] * self.a_l) * (1 - win_proba)
        )
    
    def reward_deriv(self, i: int, p: np.ndarray):
        sum_ = p.sum(axis=-1)
        win_proba = p[..., i] / sum_
        win_proba_deriv = (sum_ - p[..., i]) / sum_**2
        return (
            self.a_l + (self.a_w - self.a_l) * win_proba
            + (self.W - self.L + (self.a_w - self.a_l) * p[..., i]) * win_proba_deriv
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

        self.hist = np.empty((0, self.n, 2))

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
        return np.array([
            proba * self.csf.reward(i, p) - (1 - proba) * self.d[i] - self.r[i] * (Ks[..., i] + Kp[..., i])
            for i in range(self.n)
        ])
    
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
            res = minimize(
                self.get_func(i),
                jac=self.get_jac(i),
                x0=history[-1, i, :],
                method='trust-constr',
                bounds=Bounds([0.0, 0.0], [np.inf, np.inf]),
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
    n = 2
    ones = np.ones(n, dtype=np.float64)
    prodFunc = ProdFunc(1 * ones, 0.5 * ones, 1 * ones, 0.5 * ones, 0.0 * ones)
    problem = Problem(0.1 * ones, 0.06 * ones, prodFunc)
    hist = problem.solve()
    print(hist)
