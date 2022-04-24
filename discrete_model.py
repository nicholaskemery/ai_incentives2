import numpy as np
import matplotlib.pyplot as plt

from simple_model import CSF


class Game:

    def __init__(self, pi: np.ndarray, sigma: np.ndarray, d: np.ndarray, csf: CSF = CSF()):
        self.pi = pi
        self.sigma = sigma
        self.d = d
        self.csf = csf
        assert len(pi) == len(sigma) == len(d)
        self.n_players = len(pi)

    def get_payoffs(self, x: np.ndarray):
        """x is vector of strategies of len n_players, should have dtype bool (or equivalent)
        True corresponds to risky strategy, False to safe strategy
        """
        # print('s', self.sigma * x + (1-x))
        safe_proba = (self.sigma * x + (1-x)).prod()
        # print('p', self.pi * (1-x) + x)
        rewards = self.csf.all_rewards(self.pi * (1 - x) + x)
        return safe_proba * rewards - (1 - safe_proba) * self.d


class SymmetricTwoPlayerGame(Game):
    def __init__(self, pi: float, sigma: float, d: float = 0.0, csf: CSF = CSF()):
        ones = np.ones(2)
        super().__init__(pi * ones, sigma * ones, d * ones, csf)
        self.payoff_matrix = self._get_payoff_matrix()
    
    def _get_payoff_matrix(self):
        return np.array([
            [self.get_payoffs(np.array([False, False])), self.get_payoffs(np.array([False, True]))],
            [self.get_payoffs(np.array([True, False])), self.get_payoffs(np.array([True, True]))]
        ])
    
    def find_nash_eqs(self):
        eqs = []
        # check for pure strategy safe eq
        if self.payoff_matrix[0, 0, 0] > self.payoff_matrix[1, 0, 0]:
            eqs.append((0.0, 0.0))
        # check for pure strategy risky eq
        if self.payoff_matrix[1, 1, 0] > self.payoff_matrix[0, 1, 0]:
            eqs.append((1.0, 1.0))
        # check for mixed strategy eq
        q0 = (self.pi - 2*self.sigma + 1 + 2*self.d*(self.pi + 1)*(1 - self.sigma)) / ((2*self.d + 1) * (self.pi + 1) * (1-self.sigma)**2)
        if 0 < q0[0] < 1:
            eqs.append(tuple(q0))
        return eqs
    
    def nash_payoffs(self, ps=None):
        if ps is None:
            ps = self.find_nash_eqs()
        return [
            p[0] * p[1] * self.payoff_matrix[1, 1, 0]
            + p[0]*(1-p[1]) * self.payoff_matrix[1, 0, 0]
            + (1 - p[0]) * p[1] * self.payoff_matrix[0, 1, 0]
            + (1-p[0]) * (1-p[1]) * self.payoff_matrix[0, 0, 0]
            for p in ps
        ]


if __name__ == '__main__':
    n = 50
    pi = 0.25
    sigmas = np.linspace(0.0, 1.0, n)
    mixed_ps = np.ones(n) * np.nan
    safe_payoffs = np.ones(n) * np.nan
    mixed_payoffs = np.ones(n) * np.nan
    risky_payoffs = np.ones(n) * np.nan
    for i, sigma in enumerate(sigmas):
        game = SymmetricTwoPlayerGame(pi, sigma, 0.5, CSF(a_w=0.1, a_l=0.1))
        ps = game.find_nash_eqs()
        payoffs = game.nash_payoffs(ps)
        for (p0, _), payoff in zip(ps, payoffs):
            if p0 == 1.0:
                risky_payoffs[i] = payoff
            elif p0 == 0.0:
                safe_payoffs[i] = payoff
            else:
                mixed_ps[i] = p0
                mixed_payoffs[i] = payoff
    plt.plot(sigmas, mixed_ps)
    plt.show()
    plt.plot(sigmas, safe_payoffs, label='safe')
    plt.plot(sigmas, mixed_payoffs, label='mixed')
    plt.plot(sigmas, risky_payoffs, label='risky')
    plt.legend()
    plt.show()
