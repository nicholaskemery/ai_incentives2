#include <cmath>
#include "problem.hpp"


double R(int i, const Eigen::ArrayXd& p) {
    return p(i) / p.sum();
}

double R_deriv(int i, const Eigen::ArrayXd& p) {
    double sum_ = p.sum();
    return (sum_ - p(i)) / (sum_ * sum_);
}


Problem::Problem(
    Eigen::ArrayXd d,
    double r,
    ProdFunc prodFunc
) : d(d), r(r), prodFunc(prodFunc) {}

double Problem::net_payoff(
    int i,
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    return proba * R(i, p) - (1 - proba) * d(i)  - r * (Ks(i) + Kp(i));
}


Objective::Objective(
    const Problem& problem,
    int i,
    // copy of last_strat is intentional
    Eigen::ArrayXXd last_strat
) : problem(problem), i(i), last_strat(last_strat), prodJac(problem.prodFunc, i) {}

double Objective::f(const Eigen::Array2d& x) {
    last_strat.row(i) = x;
    return -problem.net_payoff(
        i, last_strat.col(0), last_strat.col(1)
    );
}

Eigen::Array2d Objective::jac(const Eigen::Array2d& x) {
    auto [s, p] = problem.prodFunc.f(
        last_strat.col(0),
        last_strat.col(1)
    );
    Eigen::ArrayXd probas = s / (1 + s);
    double proba = probas.prod();
    double proba_mult = (proba / probas(i)) / std::pow((1 + s(i)), 2);

    Eigen::Array22d prod_jac = prodJac.get(x);
    double proba_ks = proba_mult * prod_jac(0, 0);
    double proba_kp = proba_mult * prod_jac(0, 1);

    double R_ = R(i, p);
    double R_deriv_ = R_deriv(i, p);

    return Eigen::Array2d(
        -(
            proba_ks * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1,0) - problem.r
        ),
        -(
            proba_kp * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1, 1) - problem.r
        )
    );
}

