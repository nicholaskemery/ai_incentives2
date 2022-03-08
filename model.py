import numpy as np
from scipy.optimize import minimize, Bounds
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')


class VecProdFunc:

    def __init__(self, A, a, rho, mu, B, b, sigma, nu, theta):
        """All args here and in all following functions expected to be np arrays of same length"""
        self.A = A
        self.a = a
        self.rho = rho
        self.mu = mu
        self.B = B
        self.b = b
        self.sigma = sigma
        self.nu = nu
        self.theta = theta
    
    def S_single_i(self, Ks, Ls, p, i):
        return self.A[i] * (self.a[i] * Ks**self.rho[i] + (1-self.a[i]) * Ls**self.rho[i])**(self.mu[i] / self.rho[i]) * p**(-self.theta[i])
    
    def S(self, Ks, Ls, p):
        return self.A * (self.a * Ks**self.rho + (1-self.a) * Ls**self.rho)**(self.mu / self.rho) * p**(-self.theta)
    
    def P_single_i(self, Kp, Lp, i):
        return self.B[i] * (self.b[i] * Kp**self.sigma[i] + (1-self.b[i]) * Lp**self.sigma[i])**(self.nu[i] / self.sigma[i])
    
    def P(self, Kp, Lp):
        return self.B * (self.b * Kp**self.sigma + (1-self.b) * Lp**self.sigma)**(self.nu/self.sigma)

    def get_jac(self, i):
        def jac(x):
            p = self.P_single_i(x[..., 1], x[..., 3], i)
            s = self.S_single_i(x[..., 0], x[..., 2], p, i)
            A = self.A[i]
            a = self.a[i]
            mu = self.mu[i]
            rho = self.rho[i]
            theta = self.theta[i]
            B = self.B[i]
            b = self.b[i]
            sigma = self.sigma[i]
            nu = self.nu[i]
            S_mult = A * mu * (s / A)**((mu - rho) / mu) * p**(-theta)
            P_mult = B * nu * (p / B)**((nu - sigma) / nu)
            dPdK = b * x[..., 1]**(sigma - 1) * P_mult
            dPdL = (1-b) * x[..., 3]**(sigma - 1) * P_mult
            return np.array([
                [
                    a * x[..., 0]**(rho-1) * S_mult,
                    s * (-theta) * p**(-theta-1) * dPdK,
                    (1-a) * x[..., 2]**(rho-1) * S_mult,
                    s * (-theta) * p**(-theta-1) * dPdL
                ],
                [
                    np.zeros_like(dPdK),
                    dPdK,
                    np.zeros_like(dPdL),
                    dPdL
                ]
            ])
        return jac


def multiprocessing_helper(args):
    problem, history, i, verbose = args
    res = minimize(
        problem.get_func(history, i),
        x0=history[-1, i, :],
        method='trust-constr',
        jac=problem.get_jac(history, i),
        bounds=Bounds([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
        options={
            'xtol': 1e-5,
            'verbose': verbose
        }
    )
    return res.x


class MultiAgent:

    def __init__(self, d, r, w, R, R_deriv, prodFunc):
        """
        r and w should be scalars
        Here d should be iterable of length n
        (one entry for each person in the system)
        R and prodFunc should be functions mapping R^n -> R^n
        """
        self.d = d
        self.r = r
        self.w = w
        self.R = R
        self.R_deriv = R_deriv
        self.prodFunc = prodFunc

        self.n = len(d)

    def net_payoff(self, Ks, Kp, Ls, Lp, i):
        p = self.prodFunc.P(Kp, Lp)
        s = self.prodFunc.S(Ks, Ls, p)
        proba = (s / (1 + s)).prod(axis=-1)
        return proba * self.R(p, i) - (1 - proba) * self.d[i] - self.r * (Ks[..., i] + Kp[..., i]) - self.w * (Ls[..., i] + Lp[..., i])

    def get_func(self, history, i):
        hist = history.copy()
        def func(x):
            hist[:, i] = np.repeat(x.reshape(1, -1), hist.shape[0], axis=0)
            return -self.net_payoff(hist[..., 0], hist[..., 1], hist[..., 2], hist[..., 3], i).sum()
        return func
    
    def get_jac(self, history, i):
        prod_jac = self.prodFunc.get_jac(i)
        hist = history.copy()
        def jac(x):
            hist[:, i, :] = np.repeat(x.reshape(1, -1), hist.shape[0], axis=0)
            p = self.prodFunc.P(hist[:, :, 1], hist[:, :, 3])
            s = self.prodFunc.S(hist[:, :, 0], hist[:, :, 2], p)
            probas = s / (1 + s)
            proba = probas.prod(axis=-1)
            proba_no_i = proba / probas[:, i]
            prod_jac_ = prod_jac(hist[:, i, :])
            s_ks = prod_jac_[0, 0]
            s_kp = prod_jac_[0, 1]
            s_ls = prod_jac_[0, 2]
            s_lp = prod_jac_[0, 3]
            proba_ks = proba_no_i * s_ks / (1 + s[:, i])**2
            proba_kp = proba_no_i * s_kp / (1 + s[:, i])**2
            proba_ls = proba_no_i * s_ls / (1 + s[:, i])**2
            proba_lp = proba_no_i * s_lp / (1 + s[:, i])**2
            p_ks = prod_jac_[1, 0]  # == 0
            p_kp = prod_jac_[1, 1]
            p_ls = prod_jac_[1, 2]  # == 0
            p_lp = prod_jac_[1, 3]
            R = self.R(p, i)
            R_deriv = self.R_deriv(p, i)
            return -np.array([
                proba_ks * (R + self.d[i]) + proba * R_deriv * p_ks - self.r,
                proba_kp * (R + self.d[i]) + proba * R_deriv * p_kp - self.r,
                proba_ls * (R + self.d[i]) + proba * R_deriv * p_ls - self.w,
                proba_lp * (R + self.d[i]) + proba * R_deriv * p_lp - self.w
            ]).sum(axis=1)
        return jac
    
    def solve_single_multicore(self, history, verbose=1):
        # at each iter, we figure out response for each player that maximizes average score over other players' past responses
        # history should be t x n x 4, where t is number of previous trials, and n is number of players (== self.n)
        assert(self.n == history.shape[1])
        with Pool(self.n) as pool:
            strategies = pool.map(multiprocessing_helper, [(self, history, i, verbose) for i in range(self.n)])
        return np.array(strategies)
    
    def solve_single(self, history, verbose=1):
        # at each iter, we figure out response for each player that maximizes average score over other players' past responses
        # history should be t x n x 4, where t is number of previous trials, and n is number of players (== self.n)
        assert(self.n == history.shape[1])
        strategies = np.empty(history.shape[1:])
        for i in range(self.n):
            res = minimize(
                self.get_func(history, i),
                x0=history[-1, i, :],
                method='trust-constr',
                jac=self.get_jac(history, i),
                bounds=Bounds([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
                options={
                    'xtol': 1e-5,
                    'verbose': verbose
                }
            )
            strategies[i, :] = res.x
        return strategies


    def solve(self, T=100, window=10, tol=1e-3, multicore=False):
        history = np.empty((T, self.n, 4))
        history[0, :, :] = np.exp(np.random.randn(self.n, 4))
        for t in range(1, T):
            solver_func = self.solve_single_multicore if multicore else self.solve_single
            history[t, :, :] = solver_func(history[max(0, t-window):t, :, :], verbose=0)
            if np.abs((history[t, :, :] - history[t-1, :, :]) / history[t-1, :, :]).max() < tol:
                print(f'Exited on iteration {t}')
                return history[:t+1, :, :]
        return history
    


def simple_CSF(p, i):
    return p[..., i] / p.sum(axis=-1)

def simple_CSF_deriv(p, i):
    # dim = len(p)
    sum_ = p.sum(axis=-1)
    # return (-np.tile(p, (dim, 1)).T + np.eye(dim) * sum_) / sum_**2
    return (sum_ - p[..., i]) / sum_**2



class HomogeneousProdFunc(VecProdFunc):

    def __init__(self, n, A, a, rho, mu, B, b, sigma, nu, theta):
        self.A = np.ones(n) * A
        self.a = np.ones(n) * a
        self.rho = np.ones(n) * rho
        self.mu = np.ones(n) * mu
        self.B = np.ones(n) * B
        self.b = np.ones(n) * b
        self.sigma = np.ones(n) * sigma
        self.nu = np.ones(n) * nu
        self.theta = np.ones(n) * theta


class SimpleProblem(MultiAgent):

    def __init__(self, d, r, w, prodFunc):
        super().__init__(d, r, w, simple_CSF, simple_CSF_deriv, prodFunc)



if __name__ == '__main__':
    # run test
    n = 100
    prodFunc = HomogeneousProdFunc(n, 1.0, 0.2, 0.5, 0.6, 1.0, 0.8, 0.5, 0.6, 0.0)
    problem = SimpleProblem(np.ones(n) * 0.1, 0.02, 0.03, prodFunc)
    print(problem.solve()[-3:])
