import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from cpp_bindings import solve, prod_F, get_payoffs


A = 10.0
alpha = 0.5
B = 10.0
beta = 0.5
theta = 0.0

d = 1.0

W = 1.0
L = 0.0
a_w = 0.0
a_l = 0.0

MIN_R = 0.025
MAX_R = 0.075

MAX_ITERS = 500
EXIT_TOL = 0.001
IPOPT_MAX_ITERS = 500
IPOPT_EXIT_TOL = 0.001


def _multiproc_helper(args):
    n_players, A_, alpha_, B_, beta_, theta_, d_, r_ = args
    return solve(
        n_players,
        A_,
        alpha_,
        B_,
        beta_,
        theta_,
        d_,
        r_,
        W=W,
        L=L,
        a_w=a_w,
        a_l=a_l,
        max_iters=MAX_ITERS,
        exit_tol=EXIT_TOL,
        ipopt_max_iters=IPOPT_MAX_ITERS,
        ipopt_tol=IPOPT_EXIT_TOL
    )

def _plot_helper(
    n_players, R, strats,
    As, alphas, Bs, betas, thetas, ds,
    plotname, scenario_name,
    labels=None
):
    s_p = np.array([
        prod_F(
            n_players,
            strat[:, 0].copy(),
            strat[:, 1].copy(),
            As,
            alphas,
            Bs,
            betas,
            thetas
        )
        for strat in strats
    ])
    s, p = s_p[:, 0, :], s_p[:, 1, :]
    # plot performance
    if labels is None:
        plt.plot(R, p.mean(axis=-1))
    else:
        for i in range(n_players):
            plt.plot(R, p[:, i], label=labels[i])
        plt.legend()
    plt.ylabel('performance')
    plt.xlabel('factor cost')
    plt.savefig(f'plots/{plotname}_{scenario_name}_performance.png')
    plt.clf()
    # plot safety
    if labels is None:
        plt.plot(R, s.mean(axis=-1), label='s')
    else:
        for i in range(n_players):
            plt.plot(R, s[:, i], label=labels[i])
        plt.legend()
    plt.ylabel('safety')
    plt.xlabel('factor cost')
    plt.savefig(f'plots/{plotname}_{scenario_name}_safety.png')
    plt.clf()
    # plot total disaster proba
    probas = s / (1 + s)
    total_proba = probas.prod(axis=-1)
    plt.plot(R, total_proba)
    plt.ylabel('Proba of safe outcome')
    plt.xlabel('factor cost')
    plt.savefig(f'plots/{plotname}_{scenario_name}_total_safety.png')
    plt.clf()
    # plot net payoffs
    payoffs = np.array([
        get_payoffs(
            n_players,
            strat[:, 0].copy(),
            strat[:, 1].copy(),
            As,
            alphas,
            Bs,
            betas,
            thetas,
            ds,
            r_,
            W=W,
            L=L,
            a_w=a_w,
            a_l=a_l,
        )
        for strat, r_ in zip(strats, R)
    ])
    if labels is None:
        plt.plot(R, payoffs.mean(axis=-1))
    else:
        for i in range(n_players):
            plt.plot(R, payoffs[:, i], label=labels[i])
        plt.legend()
    plt.ylabel('net payoff')
    plt.xlabel('factor cost')
    plt.savefig(f'plots/{plotname}_{scenario_name}_payoff.png')
    plt.clf()


# SCENARIO 1
# In homogeneous setting, see what happens when factor cost increases
def run_scenario_1(
    n_players=2,
    min_r=MIN_R,
    max_r=MAX_R,
    n_steps=20,
    plot=True,
    plotname='fcost',
    A=A,
    B=B,
    alpha=alpha,
    beta=beta,
    theta=theta,
    d=d
):
    ones = np.ones(n_players, dtype=np.float64)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(min(cpu_count(), n_steps)) as pool:
        strats = pool.map(_multiproc_helper, [
            (
                n_players,
                A * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                d * ones,
                r_
            ) for r_ in R
        ])
    strats = np.array(strats)
    if plot:
        _plot_helper(
            n_players, R, strats,
            A * ones, alpha * ones,
            B * ones, beta * ones,
            theta * ones, d * ones,
            plotname, 'sc1'
        )
    return strats



def run_scenario_1_multiple(
    n_players=2,
    min_r=MIN_R,
    max_r=MAX_R,
    n_steps=20,
    plot=True,
    plotname='fcost',
    As=[A],
    Bs=[B],
    alphas=[alpha],
    betas=[beta],
    thetas=[theta],
    ds=[d],
    labels=None
):
    strat_list = [
        run_scenario_1(
            n_players, min_r, max_r, n_steps,
            plot=False, plotname=plotname,
            A=A_, B=B_, alpha=alpha_, beta=beta_, theta=theta_, d=d_
        )
        for A_, B_, alpha_, beta_, theta_, d_ in zip(As, Bs, alphas, betas, thetas, ds)
    ]
    if plot:
        if labels is None:
            labels = [str(i) for i in range(n_steps)]
        R = np.linspace(min_r, max_r, n_steps)
        ones = np.ones(n_players, dtype=np.float64)
        S, P = [], []
        payoffs_list = []
        for A_, B_, alpha_, beta_, theta_, d_, strats in zip(As, Bs, alphas, betas, thetas, ds, strat_list):
            s_p = np.array([
                prod_F(
                    n_players,
                    strat[:, 0].copy(),
                    strat[:, 1].copy(),
                    A_ * ones,
                    alpha_ * ones,
                    B_ * ones,
                    beta_ * ones,
                    theta_ * ones
                )
                for strat in strats
            ])
            S.append(s_p[:, 0, :])
            P.append(s_p[:, 1, :])
            payoffs = np.array([
                get_payoffs(
                    n_players,
                    strat[:, 0].copy(),
                    strat[:, 1].copy(),
                    A_ * ones,
                    alpha * ones,
                    B_ * ones,
                    beta_ * ones,
                    theta_ * ones,
                    d_ * ones,
                    r_,
                    W=W,
                    L=L,
                    a_w=a_w,
                    a_l=a_l
                )
                for strat, r_ in zip(strats, R)
            ])
            payoffs_list.append(payoffs)
        # plot performance
        for p, label in zip(P, labels):
            plt.plot(R, p.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('performance')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_performance.png')
        plt.clf()
        # plot safety
        for s, label in zip(S, labels):
            plt.plot(R, s.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('safety')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_safety.png')
        plt.clf()
        # plot total safety
        for s, label in zip(S, labels):
            probas = s / (1 + s)
            total_proba = probas.prod(axis=-1)
            plt.plot(R, total_proba, label=label)
        plt.legend()
        plt.ylabel('Proba of safe outcome')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_total_safety.png')
        plt.clf()
        # plot payoffs
        for payoffs, label in zip(payoffs_list, labels):
            plt.plot(R, payoffs.mean(axis=-1), label=label)
        plt.legend()
        plt.ylabel('net payoff')
        plt.xlabel('factor cost')
        plt.savefig(f'plots/{plotname}_sc1_combined_payoff.png')
        plt.clf()
    return strat_list

# vary theta
# run_scenario_1_multiple(
#     As=[A]*3,
#     Bs=[B]*3,
#     alphas=[alpha]*3,
#     betas=[beta]*3,
#     thetas=[0.0, 0.15, 0.3],
#     ds=[d]*3,
#     labels=['θ = 0.0', 'θ = 0.15', 'θ = 0.3'],
#     plotname='0'
# )

# vary d with theta = 0
run_scenario_1_multiple(
    As=[A]*4,
    Bs=[B]*4,
    alphas=[alpha]*4,
    betas=[beta]*4,
    thetas=[0.0]*4,
    ds=[0.1, 0.5, 1.0, 1.5],
    labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
    plotname='varyd_theta0'
)

# vary d with theta = 0.25
run_scenario_1_multiple(
    As=[A]*4,
    Bs=[B]*4,
    alphas=[alpha]*4,
    betas=[beta]*4,
    thetas=[0.3]*4,
    ds=[0.1, 0.5, 1.0, 1.5],
    labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
    plotname='varyd_theta025'
)

# vary d with higher productivity, theta = 0
# run_scenario_1_multiple(
#     As=[20.0]*4,
#     Bs=[20.0]*4,
#     alphas=[alpha]*4,
#     betas=[beta]*4,
#     thetas=[0.0]*4,
#     ds=[0.1, 0.5, 1.0, 1.5],
#     labels=['d = 0.0', 'd = 0.5', 'd = 1.0', 'd = 1.5'],
#     plotname='3'
# )


# SCENARIO 2
# In heterogeneous setting (varying A and/or B), see what happens when factor cost increases
def run_scenario_2(
    n_players=2,
    As=np.array([8.0, 12.0]),
    Bs=np.array([8.0, 12.0]),
    min_r=MIN_R,
    max_r=MAX_R,
    n_steps=20,
    plot=True,
    plotname='fcost',
    alpha=alpha,
    beta=beta,
    theta=theta,
    d=d
):
    ones = np.ones(n_players, dtype=np.float64)
    R = np.linspace(min_r, max_r, n_steps)
    with Pool(min(cpu_count(), n_steps)) as pool:
        strats = pool.map(_multiproc_helper, [
            (
                n_players,
                As,
                alpha * ones,
                Bs,
                beta * ones,
                theta * ones,
                d * ones,
                r_
            ) for r_ in R
        ])
    strats = np.array(strats)
    if plot:
        _plot_helper(
            n_players, R, strats,
            As, alpha * ones,
            Bs, beta * ones,
            theta * ones, d * ones,
            plotname, 'sc2',
            labels=[f'A={A_}, B={B_}' for A_, B_ in zip(As, Bs)]
        )
    return strats


# SCENARIO 3
# In homogeneous setting, A increases (r is constant)
def run_scenario_3(
    n_players=2,
    min_A=6.0,
    max_A=14.0,
    n_steps=20,
    plot=True,
    plotname='fcost',
    B=B,
    alpha=alpha,
    beta=beta,
    theta=theta,
    d=d,
    r=MAX_R
):
    ones = np.ones(n_players, dtype=np.float64)
    A_ = np.linspace(min_A, max_A, n_steps)
    with Pool(min(cpu_count(), n_steps)) as pool:
        strats = pool.map(_multiproc_helper, [
            (
                n_players,
                a_ * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                d * ones,
                r
            ) for a_ in A_
        ])
    strats = np.array(strats)
    if plot:
        scenario_name = 'sc3'
        xlabel = 'safety productivity'
        s_p = np.array([
            prod_F(
                n_players,
                strat[:, 0].copy(),
                strat[:, 1].copy(),
                a_ * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones
            )
            for a_, strat in zip(A_, strats)
        ])
        s, p = s_p[:, 0, :], s_p[:, 1, :]
        # plot performance
        plt.plot(A_, p.mean(axis=-1))
        plt.ylabel('performance')
        plt.xlabel(xlabel)
        plt.savefig(f'plots/{plotname}_{scenario_name}_performance.png')
        plt.clf()
        # plot safety
        plt.plot(A_, s.mean(axis=-1), label='s')
        plt.ylabel('safety')
        plt.xlabel(xlabel)
        plt.savefig(f'plots/{plotname}_{scenario_name}_safety.png')
        plt.clf()
        # plot total disaster proba
        probas = s / (1 + s)
        total_proba = probas.prod(axis=-1)
        plt.plot(A_, total_proba)
        plt.ylabel('Proba of safe outcome')
        plt.xlabel(xlabel)
        plt.savefig(f'plots/{plotname}_{scenario_name}_total_safety.png')
        plt.clf()
        # plot net payoffs
        payoffs = np.array([
            get_payoffs(
                n_players,
                strat[:, 0].copy(),
                strat[:, 1].copy(),
                a_ * ones,
                alpha * ones,
                B * ones,
                beta * ones,
                theta * ones,
                d * ones,
                r,
                W=W,
                L=L,
                a_w=a_w,
                a_l=a_l,
            )
            for strat, a_ in zip(strats, A_)
        ])
        plt.plot(A_, payoffs.mean(axis=-1))
        plt.ylabel('net payoff')
        plt.xlabel(xlabel)
        plt.savefig(f'plots/{plotname}_{scenario_name}_payoff.png')
        plt.clf()
    return strats


# run_scenario_1()
# run_scenario_1_multiple(
#     plotname='fcost_varying_A',
#     As=[8.0, 10.0, 12.0],
#     Bs=[B]*3, alphas=[alpha]*3, betas=[beta]*3, thetas=[theta]*3, ds=[d]*3,
#     labels=["A=8", "A=10", "A=12"]
# )
# run_scenario_1_multiple(
#     plotname='fcost_varying_B',
#     As=[A]*3,
#     Bs=[8.0, 10.0, 12.0],
#     alphas=[alpha]*3, betas=[beta]*3, thetas=[theta]*3, ds=[d]*3,
#     labels=["B=8", "B=10", "B=12"]
# )
# run_scenario_2()
# run_scenario_2(Bs=np.array([10., 10.]), plotname='fcost_sameB')
# run_scenario_2(As=np.array([10., 10.]), plotname='fcost_sameA')
# run_scenario_2(As=np.array([6.0, 14.0]), Bs=np.array([10.0, 10.0]), plotname='fcost_bighet_A')
# run_scenario_3(r=0.05)

# # Some trials with different CSF
# a_w = 0.05
# b_w = 0.05
# run_scenario_1_multiple(
#     plotname='fcost_varying_A_awbw005',
#     As=[8.0, 10.0, 12.0],
#     Bs=[B]*3, alphas=[alpha]*3, betas=[beta]*3, thetas=[theta]*3, ds=[d]*3,
#     labels=["A=8", "A=10", "A=12"]
# )
# run_scenario_1_multiple(
#     plotname='fcost_varying_B_awbw005',
#     As=[A]*3,
#     Bs=[8.0, 10.0, 12.0],
#     alphas=[alpha]*3, betas=[beta]*3, thetas=[theta]*3, ds=[d]*3,
#     labels=["B=8", "B=10", "B=12"]
# )
# run_scenario_2(plotname='fcost_awbw005')
# run_scenario_2(Bs=np.array([10., 10.]), plotname='fcost_sameB_awbw005')
# run_scenario_2(As=np.array([10., 10.]), plotname='fcost_sameA_awbw005')

# theta = 0.25
# run_scenario_1_multiple(
#     plotname='fcost_varying_B_awbw005_theta025',
#     As=[A]*3,
#     Bs=[8.0, 10.0, 12.0],
#     alphas=[alpha]*3, betas=[beta]*3, thetas=[theta]*3, ds=[d]*3,
#     labels=["B=8", "B=10", "B=12"]
# )
