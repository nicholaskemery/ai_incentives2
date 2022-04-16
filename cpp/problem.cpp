#include <cmath>
#include "problem.hpp"


CSF::CSF() : W(1.0), L(0.0), a_w(0.0), a_l(0.0) {}

CSF::CSF(
    double W,
    double L,
    double a_w,
    double a_l
) : W(W), L(L), a_w(a_w), a_l(a_l) {}

double CSF::reward(int i, const Eigen::ArrayXd& p) const {
    double win_proba = p(i) / p.sum();
    return (W + a_w * p(i)) * win_proba + (L + a_l * p(i)) * (1 - win_proba);
}

double CSF::reward_deriv(int i, const Eigen::ArrayXd& p) const {
    double sum_ = p.sum();
    double win_proba = p(i) / sum_;
    double win_proba_deriv = (sum_ - p(i)) / (sum_ * sum_);
    return a_l + (a_w - a_l) * win_proba + (W - L + (a_w - a_l) * p(i)) * win_proba_deriv;
}


Problem::Problem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc
) : Problem(d, prodFunc, CSF()) {}

Problem::Problem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc,
    CSF csf
) : d(d), prodFunc(prodFunc), csf(csf), n_players(d.size()) {
    assert(n_players == prodFunc.n_players);
}

double Problem::net_payoff(
    int i,
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    return proba * csf.reward(i, p) - (1 - proba) * d(i)  - r(i, s(i)) * (Ks(i) + Kp(i));
}

Eigen::ArrayXd Problem::get_all_net_payoffs(
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    std::vector<double> payoffs;
    payoffs.reserve(n_players);
    for (int i = 0; i < n_players; i++) {
        payoffs.push_back(proba * csf.reward(i, p) - (1 - proba) * d(i) - r(i, s(i)) * (Ks(i) + Kp(i)));
    }
    return Eigen::Map<Eigen::ArrayXd>(payoffs.data(), n_players);
}


Objective::Objective(
    const Problem& problem,
    int i,
    Eigen::ArrayX2d last_strat
) : problem(problem), i(i), last_strat(last_strat) {}

double Objective::f(const Eigen::Array2d& x) {
    last_strat.row(i) = x;
    return -problem.net_payoff(
        i, last_strat.col(0), last_strat.col(1)
    );
}

Eigen::Array2d Objective::jac(const Eigen::Array2d& x) {
    last_strat.row(i) = x;
    auto [s, p] = problem.prodFunc.f(
        last_strat.col(0),
        last_strat.col(1)
    );
    Eigen::ArrayXd probas = s / (1 + s);
    double proba = probas.prod();
    double proba_mult = proba / (s(i) * (1 + s(i)));

    Eigen::Array22d prod_jac = problem.prodFunc.jac_single_i(i, x);
    // dsdKs = prod_jac(0, 0)
    // dsdKp = prod_jac(0, 1)
    // dpdKs = prod_jac(1, 0)
    // dpdKp = prod_jac(1, 1)
    double proba_ks = proba_mult * prod_jac(0, 0);
    double proba_kp = proba_mult * prod_jac(0, 1);

    double R_ = problem.csf.reward(i, p);
    double R_deriv_ = problem.csf.reward_deriv(i, p);

    double r = problem.r(i, s(i));
    double drds = problem.drds(i, s(i));
    double drKsdKs = r - drds * prod_jac(0, 0) * x(0);
    double drKpdKp = r - drds * prod_jac(0, 1) * x(1);

    return Eigen::Array2d(
        -(
            proba_ks * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1,0) - drKsdKs
        ),
        -(
            proba_kp * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1, 1) - drKpdKp
        )
    );
}


ConstantRProblem::ConstantRProblem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc,
    Eigen::ArrayXd r_
) : ConstantRProblem(d, prodFunc, r_, CSF()) {}

ConstantRProblem::ConstantRProblem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc,
    Eigen::ArrayXd r_,
    CSF csf
) : Problem(d, prodFunc, csf), r_(r_) {
    assert(r_.size() == n_players);
}

double ConstantRProblem::r(int i, double s) const {
    return r_(i);
}

double ConstantRProblem::drds(int i, double s) const {
    return 0;
}


DecayingExpRProblem::DecayingExpRProblem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc,
    double r0,
    double c
) : Problem(d, prodFunc), r0(r0), c(c) {}

double DecayingExpRProblem::r(int i, double s) const {
    return r0 * std::exp(-c * s);
}

double DecayingExpRProblem::drds(int i, double s) const {
    return -c * r0 * std::exp(-c * s);
}


