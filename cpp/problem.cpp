#include <cmath>
#include "problem.hpp"

#include<iostream>

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
    // copy of history is intentional
    std::vector<Eigen::ArrayXXd> history
) : problem(problem), i(i), history(history) {}

double Objective::get(const Eigen::Array2d& x) {
    double out = 0.0;
    for (int j = 0; j < history.size(); j++) {
        history[j].row(i) = x;
        out -= problem.net_payoff(
            i, history[j].col(0), history[j].col(1)
        );
    }
    return out;
}


ProblemJac::ProblemJac(
    const Problem& problem,
    int i,
    std::vector<Eigen::ArrayXXd> history
) : problem(problem), i(i), history(history), prodJac(problem.prodFunc, i) {}

Eigen::Array2d ProblemJac::get(const Eigen::Array2d& x) {
    Eigen::Array2d out(0.0, 0.0);
    for (int j = 0; j < history.size(); j++) {

        auto [s, p] = problem.prodFunc.f(
            history[j].col(0),
            history[j].col(1)
        );
        Eigen::ArrayXd probas = s / (1 + s);
        double proba = probas.prod();
        double proba_mult = (proba / probas(i)) / std::pow((1 + s(i)), 2);

        Eigen::Array22d prod_jac = prodJac.get(x);
        double proba_ks = proba_mult * prod_jac(0, 0);
        double proba_kp = proba_mult * prod_jac(0, 1);

        double R_ = R(i, p);
        double R_deriv_ = R_deriv(i, p);

        out(0) -= proba_ks * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1,0) - problem.r;
        out(1) -= proba_kp * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1, 1) - problem.r;
    }
    return out;
}

