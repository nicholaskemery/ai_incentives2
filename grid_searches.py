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

MAX_ITERS = 200
EXIT_TOL = 0.001
IPOPT_TOL = 0.005
IPOPT_MAX_ITERS = 200



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
        r_,
        max_iters=MAX_ITERS,
        exit_tol=EXIT_TOL,
        ipopt_max_iters=IPOPT_MAX_ITERS,
        ipopt_tol=IPOPT_TOL
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
        max_iters=MAX_ITERS,
        exit_tol=EXIT_TOL,
        ipopt_max_iters=IPOPT_MAX_ITERS,
        ipopt_tol=IPOPT_TOL
    )
    s, p = prod_F(
        n_players,
        K[:, 0].copy(),
        K[:, 1].copy(),
        A * ones,
        alpha * ones,
        B * ones,
        beta * ones,
        theta * ones
    )
    total_safety = (s / (1 + s)).prod()
    average_performance = p.mean()
    revenue = (r0_ * np.exp(-c_ * s) * K.T).sum()
    return total_safety, average_performance, revenue

def _rfunc_plot(grid, R0, C, r0_steps, c_steps, plot_title, filename):
    plt.imshow(np.flip(grid, axis=0))
    plt.colorbar()
    plt.title(plot_title)
    cstep = max(1, c_steps // 5)
    plt.xticks(ticks=np.arange(c_steps, step=cstep), labels=[f'{c_:.3f}' for c_ in C[::cstep]])
    plt.xlabel('c')
    r0step = max(1, r0_steps // 5)
    plt.yticks(
        ticks=list(reversed(r0_steps-np.arange(r0_steps, step=r0step)-1)),
        labels=[f'{r0_:.3f}' for r0_ in reversed(R0[::r0step])]
    )
    plt.ylabel('$r_0$')
    plt.savefig(filename)
    plt.clf()

def vary_rfunc(min_r0, max_r0, r0_steps, min_c, max_c, c_steps, plot=True, plotname='varying_rfunc'):
    R0 = np.linspace(min_r0, max_r0, r0_steps)
    C = np.linspace(min_c, max_c, c_steps)
    with Pool(min(cpu_count(), r0_steps * c_steps)) as pool:
        out_vals = pool.map(_rfunc_helper, [(r0_, c_) for r0_ in R0 for c_ in C])
    safety, performance, revenue = tuple(
        np.array(grid).reshape((r0_steps, c_steps))
        for grid in zip(*out_vals)
    )
    if plot:
        _rfunc_plot(safety, R0, C, r0_steps, c_steps, 'Probability of safe outcome', f'plots/{plotname}_total_safety.png')
        _rfunc_plot(performance, R0, C, r0_steps, c_steps, 'Average performance', f'plots/{plotname}_avg_performance.png')
        _rfunc_plot(revenue, R0, C, r0_steps, c_steps, 'Total rental revenue', f'plots/{plotname}_revenue.png')
    return safety, performance, revenue



if __name__ == '__main__':
    # for t in [0, 0.1, 0.2]:
    #     theta = t
    #     vary_r_d(0.01, 0.03, 21, 0.5, 1.5, 21, plotname=f'varying_r_d_theta{t}')
    # # note theta = 0.2 after this

    theta = 0.2

    # A = 20.0
    # B = 20.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB20_theta02')

    A = 10.0
    B = 10.0
    vary_rfunc(0.001, 0.03, 20, 0.01, 0.15, 20, plotname='varying_rfunc_AB10theta02')

    # A = 5.0
    # B = 5.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB5theta02')

    # A = 1.0
    # B = 1.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB1theta02')


    theta = 0.5

    # A = 20.0
    # B = 20.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB20_theta05')

    A = 10.0
    B = 10.0
    vary_rfunc(0.001, 0.03, 20, 0.01, 0.15, 20, plotname='varying_rfunc_AB10_theta05')

    # A = 5.0
    # B = 5.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB5_theta05')

    # A = 1.0
    # B = 1.0
    # vary_rfunc(0.01, 0.1, 20, 0.01, 0.1, 20, plotname='varying_rfunc_AB1_theta05')