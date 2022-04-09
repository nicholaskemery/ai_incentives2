import numpy as np
from scipy.optimize import minimize, Bounds

import warnings
warnings.filterwarnings('ignore')

SOLVER_TOL = 1e-8
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


class Problem:

    def __init__(self, d, r, R, R_deriv, prodFunc):
        """
        d is np array of length n
        r is a single float value
        R is the reward function
        R_deriv is the derivative of R
        prodFunc is of class ProdFunc
        """
        self.d = d
        self.r = r
        self.R = R
        self.R_deriv = R_deriv
        self.prodFunc = prodFunc

        self.n = len(d)
        assert(prodFunc.n == self.n)

    def net_payoff(self, i, Ks, Kp):
        """
        Gets payoff for player i when strategies are represented by Ks and Kp vectors
        i is an integer index
        Ks and Kp are np arrays with length == self.n
        """
        s, p = self.prodFunc.F(Ks, Kp)
        proba = (s / (1 + s)).prod(axis=-1)
        return proba * self.R(i, p) - (1 - proba) * self.d[i] - self.r * (Ks[..., i] + Kp[..., i])
    
    def get_func(self, i, history):
        hist = history.copy()
        def func(x):
            hist[:, i] = np.repeat(x.reshape(1, -1), hist.shape[0], axis=0)
            return -self.net_payoff(i, hist[..., 0], hist[..., 1]).sum()
        return func
    
    def get_jac(self, i, history):
        prod_jac = self.prodFunc.get_jac(i)
        hist = history.copy()
        def jac(x):
            # print('x', x)
            hist[:, i, :] = np.repeat(x.reshape(1, -1), hist.shape[0], axis=0)
            s, p = self.prodFunc.F(hist[..., 0], hist[..., 1])
            # print('s, p', s, p)
            probas = s / (1 + s)
            # print('probas', probas)
            proba = probas.prod(axis=-1)
            # print('proba', proba)
            proba_mult = proba / (s[:, i] * (1 + s[:, i]))
            # print('proba_mult', proba_mult)
            prod_jac_ = prod_jac(hist[:, i, :])
            s_ks = prod_jac_[0, 0]
            s_kp = prod_jac_[0, 1]
            p_ks = prod_jac_[1, 0] # == 0
            p_kp = prod_jac_[1, 1]
            proba_ks = proba_mult * s_ks
            # print('proba_ks', proba_ks)
            proba_kp = proba_mult * s_kp
            # print('proba_kp', proba_kp)
            R = self.R(i, p)
            # print('R', R)
            R_deriv = self.R_deriv(i, p)
            # print('R_deriv', R_deriv)
            return -np.array([
                proba_ks * (R + self.d[i]) + proba * R_deriv * p_ks - self.r,
                proba_kp * (R + self.d[i]) + proba * R_deriv * p_kp - self.r
            ]).sum(axis=1)
        return jac

    def solve_single(self, history, verbose=1):
        """
        history should be 3d np.array
        dim 0 is iteration
        dim 1 is agent
        dim 2 is factor (s, p)
        """
        assert(self.n == history.shape[1])
        strategies = np.empty((self.n, 2))
        for i in range(self.n):
            res = minimize(
                self.get_func(i, history),
                jac=self.get_jac(i, history),
                x0=history[-1, i, :],
                method='trust-constr',
                bounds=Bounds([0.0, 0.0], [np.inf, np.inf]),
                options={
                    'xtol': SOLVER_TOL,
                    'verbose': verbose
                }
            )
            strategies[i, :] = res.x
        return strategies
    
    def solve(self, T=100, window=DEFAULT_WINDOW, iter_tol=1e-3, verbose=0, init_guess_mu=0.0):
        history = np.empty((T, self.n, 2))
        # initialize strategy guesses with log normal dist
        history[0, :, :] = np.exp(np.random.randn(self.n, 2)*0.2 + init_guess_mu)
        for t in range(1, T):
            history[t, :, :] = self.solve_single(
                history[max(0, t-window):t, :, :],
                verbose=verbose
            )
            if np.abs((history[t, :, :] - history[t-1, :, :]) / history[t-1, :, :]).max() < iter_tol:
                print(f'Exited on iteration {t}')
                return history[:t+1, :, :]
        print('Reached max iterations')
        return history

def simple_CSF(i, p):
    return p[..., i] / p.sum(axis=-1)

def simple_CSF_deriv(i, p):
    sum_ = p.sum(axis=-1)
    return (sum_ - p[..., i]) / sum_**2


if __name__ == '__main__':
    # run test
    n = 2
    ones = np.ones(n, dtype=np.float64)
    prodFunc = ProdFunc(1 * ones, 0.5 * ones, 1 * ones, 0.5 * ones, 0.0 * ones)
    problem = Problem(0.1 * ones, 0.06, simple_CSF, simple_CSF_deriv, prodFunc)
    hist = problem.solve()
    print(hist)
