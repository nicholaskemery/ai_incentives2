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
    
    def nash_safeties(self, ps=None):
        if ps is None:
            ps = self.find_nash_eqs()
        return [
            (1 - p[0]) * (1 - p[1])
            + (p[0] * (1 - p[1]) + (1 - p[0]) * p[1]) * self.sigma[0]
            + p[0] * p[1] * self.sigma[0]**2
            for p in ps
        ]


def _plot_helper(
    xvar,
    safe_payoffs, mixed_payoffs, risky_payoffs,
    safe_safeties, mixed_safeties, risky_safeties,
    title, x_axis_label
):
    fig, axs = plt.subplots(2, sharex=True, figsize=(8, 10))
    axs[0].plot(xvar, safe_payoffs, label='safe')
    axs[0].plot(xvar, mixed_payoffs, label='mixed')
    axs[0].plot(xvar, risky_payoffs, label='risky')
    axs[0].set_ylabel('payoff')
    axs[0].legend()
    axs[1].plot(xvar, safe_safeties, label='safe')
    axs[1].plot(xvar, mixed_safeties, label='mixed')
    axs[1].plot(xvar, risky_safeties, label='risky')
    axs[1].set_ylabel('safety')
    axs[1].legend()
    plt.xlabel(x_axis_label)
    fig.suptitle(title)
    plt.show()


def plot_two_player_varying_sigma(pi: float, sigmas: np.ndarray, d: float, csf: CSF = CSF()):
    safe_payoffs = np.ones_like(sigmas) * np.nan
    mixed_payoffs = np.ones_like(sigmas) * np.nan
    risky_payoffs = np.ones_like(sigmas) * np.nan
    safe_safeties = np.ones_like(sigmas) * np.nan
    mixed_safeties = np.ones_like(sigmas) * np.nan
    risky_safeties = np.ones_like(sigmas) * np.nan
    for i, sigma in enumerate(sigmas):
        game = SymmetricTwoPlayerGame(pi, sigma, d, csf)
        ps = game.find_nash_eqs()
        payoffs = game.nash_payoffs(ps)
        safeties = game.nash_safeties(ps)
        for (p0, _), payoff, safety in zip(ps, payoffs, safeties):
            if p0 == 1.0:
                risky_payoffs[i] = payoff
                risky_safeties[i] = safety
            elif p0 == 0.0:
                safe_payoffs[i] = payoff
                safe_safeties[i] = safety
            else:
                mixed_payoffs[i] = payoff
                mixed_safeties[i] = safety
    _plot_helper(
        sigmas,
        safe_payoffs, mixed_payoffs, risky_payoffs,
        safe_safeties, mixed_safeties, risky_safeties,
        title=f'Outcomes with π = {pi}', x_axis_label='σ'
    )


def plot_two_player_varying_pi(pis: np.ndarray, sigma: float, d: float, csf: CSF = CSF()):
    safe_payoffs = np.ones_like(pis) * np.nan
    mixed_payoffs = np.ones_like(pis) * np.nan
    risky_payoffs = np.ones_like(pis) * np.nan
    safe_safeties = np.ones_like(pis) * np.nan
    mixed_safeties = np.ones_like(pis) * np.nan
    risky_safeties = np.ones_like(pis) * np.nan
    for i, pi in enumerate(pis):
        game = SymmetricTwoPlayerGame(pi, sigma, d, csf)
        ps = game.find_nash_eqs()
        payoffs = game.nash_payoffs(ps)
        safeties = game.nash_safeties(ps)
        for (p0, _), payoff, safety in zip(ps, payoffs, safeties):
            if p0 == 1.0:
                risky_payoffs[i] = payoff
                risky_safeties[i] = safety
            elif p0 == 0.0:
                safe_payoffs[i] = payoff
                safe_safeties[i] = safety
            else:
                mixed_payoffs[i] = payoff
                mixed_safeties[i] = safety
    _plot_helper(
        pis,
        safe_payoffs, mixed_payoffs, risky_payoffs,
        safe_safeties, mixed_safeties, risky_safeties,
        title=f'Outcomes with σ = {sigma}', x_axis_label='π'
    )


if __name__ == '__main__':
    pi = 0.25
    sigmas = np.linspace(0.0, 1.0, 200)
    plot_two_player_varying_sigma(pi, sigmas, 0.2, CSF(a_w=0.2, a_l=0.2))
    pis = np.linspace(0.0, 1.0, 200)
    sigma = 0.75
    plot_two_player_varying_pi(pis, sigma, 0.2, CSF(a_w=0.2, a_l=0.2))
