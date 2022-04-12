from cpp_bindings import solve, prod_F, get_payoffs, solve_variable_r
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool

n_players = 2
A = 10.0
alpha = 0.5
B = 10.0
beta = 0.5
theta = 0.2

r = 0.01
d = 1.0

MAX_ITER = 200
EXIT_TOL = 0.001
IFOPT_TOL = 0.001


def _r_d_helper(args):
    r_, d_ = args
    ones = np.ones(n_players)
    K = solve(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        d_ * ones,
        r_ * ones,
        MAX_ITER,
        EXIT_TOL,
        IFOPT_TOL
    )
    s, _ = prod_F(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        K[:, 0].copy(),
        K[:, 1].copy()
    )
    return (s / (1 + s)).prod()


def vary_r_d(min_r, max_r, r_steps, min_d, max_d, d_steps, plot=True, plotname='varying_r_d'):
    R = np.linspace(min_r, max_r, r_steps)
    D = np.linspace(min_d, max_d, d_steps)
    with Pool(min(cpu_count(), r_steps * d_steps)) as pool:
        safety_vals = pool.map(_r_d_helper, [(r_, d_) for r_ in R for d_ in D])
    grid = np.array(safety_vals).reshape((r_steps, d_steps))
    if plot:
        plt.imshow(np.flip(grid, axis=0))
        plt.colorbar()
        dstep = max(1, d_steps // 5)
        plt.xticks(ticks=np.arange(d_steps, step=dstep), labels=[f'{d_:.3f}' for d_ in D[::dstep]])
        plt.xlabel('d')
        rstep = max(1, r_steps // 5)
        plt.yticks(ticks=np.arange(r_steps, step=rstep), labels=[f'{r_:.3f}' for r_ in reversed(R[::rstep])])
        plt.ylabel('r')
        plt.savefig(f'plots/{plotname}.png')
        plt.clf()
    return grid



def _rfunc_helper(args):
    r0_, c_ = args
    ones = np.ones(n_players)
    K = solve_variable_r(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        d * ones,
        r0_,
        c_,
        MAX_ITER,
        EXIT_TOL,
        IFOPT_TOL
    )
    s, _ = prod_F(
        n_players,
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones,
        K[:, 0].copy(),
        K[:, 1].copy()
    )
    return (s / (1 + s)).prod()

def vary_rfunc(min_r0, max_r0, r0_steps, min_c, max_c, c_steps, plot=True, plotname='varying_rfunc'):
    R0 = np.linspace(min_r0, max_r0, r0_steps)
    C = np.linspace(min_c, max_c, c_steps)
    with Pool(min(cpu_count(), r0_steps * c_steps)) as pool:
        safety_vals = pool.map(_rfunc_helper, [(r0_, c_) for r0_ in R0 for c_ in C])
    grid = np.array(safety_vals).reshape((r0_steps, c_steps))
    if plot:
        # TODO: add plots for revenue and other outcomes
        plt.imshow(np.flip(grid, axis=0))
        plt.colorbar()
        cstep = max(1, c_steps // 5)
        plt.xticks(ticks=np.arange(c_steps, step=cstep), labels=[f'{c_:.3f}' for c_ in C[::cstep]])
        plt.xlabel('c')
        r0step = max(1, r0_steps // 5)
        plt.yticks(ticks=np.arange(r0_steps, step=r0step), labels=[f'{r0_:.3f}' for r0_ in reversed(R0[::r0step])])
        plt.ylabel('$r_0$')
        plt.savefig(f'plots/{plotname}.png')
        plt.clf()
    return grid



if __name__ == '__main__':
    # for t in [0, 0.1, 0.2]:
    #     theta = t
    #     vary_r_d(0.01, 0.03, 21, 0.5, 1.5, 21, plotname=f'varying_r_d_theta{t}')
    # # note theta = 0.2 after this

    theta = 0.2
    vary_rfunc(0.025, 0.05, 20, 0.01, 0.05, 20)
