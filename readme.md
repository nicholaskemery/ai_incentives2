The code in this repository is meant to find Nash equilibria for the following model:

We assume that $n$ players produce safety, $s$, and performance, $p$, as
$$s_i = A_i K_{s,i}^{\alpha_i} p_i^{-\theta_i}, \quad p_i = B_i K_{p,i}^{\beta_i}.$$
for $i = 1, \dots, n$. The $K$ are inputs chosen by the players, and all other variables are fixed parameters.

In a Nash equilibrium, each player $i$ chooses $K_{s,i}$ and $K_{p,i}$ to maximize the payoff
$$\pi_i := \left( \prod_{j=1}^n \frac{s_j}{1+s_j} \right) \rho_i(p) - \left( 1 - \prod_{j=1}^n \frac{s_j}{1+s_j} \right) d_i - r_i(K_{i,s} + K_{i,p}),$$
subject to the other players' choices of $K_s$ and $K_p$. Here $\rho_i(p)$ is a contest success function (the expected payoff for player $i$ given a safe outcome and a vector of performances $p$), and $d_i$ is the damage incurred by player $i$ in the event of an unsafe outcome.

The easiest way to use this code is via the API in `scenarios.py`: just run `python3 scenarios.py` from this directory to run some example scenarios, or import the `Scenario` class and run your own scenarios.
