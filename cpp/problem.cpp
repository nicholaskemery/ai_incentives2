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
    Eigen::ArrayXd r,
    ProdFunc prodFunc
) : d(d), r(r), prodFunc(prodFunc), n_players(d.size()) {
    assert(n_players == prodFunc.n_players && n_players == r.size());
}

double Problem::net_payoff(
    int i,
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    return proba * R(i, p) - (1 - proba) * d(i)  - r(i) * (Ks(i) + Kp(i));
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
        payoffs.push_back(proba * R(i, p) - (1 - proba) * d(i) - r(i) * (Ks(i) + Kp(i)));
    }
    return Eigen::Map<Eigen::ArrayXd>(payoffs.data(), n_players);
}


BaseObjective::BaseObjective(
    int i,
    Eigen::ArrayX2d last_strat
) : i(i), last_strat(last_strat) {}


Objective::Objective(
    const Problem& problem,
    int i,
    // copy of last_strat is intentional
    Eigen::ArrayX2d last_strat
) : BaseObjective(i, last_strat), problem(problem) {}

double Objective::f(const Eigen::Array2d& x) {
    last_strat.row(i) = x;
    return -problem.net_payoff(
        i, last_strat.col(0), last_strat.col(1)
    );
}

Eigen::Array2d Objective::jac(const Eigen::Array2d& x) {
    // std::cout << "x: " << x << '\n';
    last_strat.row(i) = x;
    auto [s, p] = problem.prodFunc.f(
        last_strat.col(0),
        last_strat.col(1)
    );
    // std::cout << "s: " << s << '\n';
    // std::cout << "p: " << p << '\n';
    Eigen::ArrayXd probas = s / (1 + s);
    // std::cout << "probas: " << probas << '\n';
    double proba = probas.prod();
    // std::cout << "proba: " << proba << '\n';
    double proba_mult = proba / (s(i) * (1 + s(i)));
    // std::cout << "proba_mult: " << proba_mult << '\n';

    Eigen::Array22d prod_jac = problem.prodFunc.jac_single_i(i, x);
    // std::cout << "prod_jac: " << prod_jac << '\n';
    double proba_ks = proba_mult * prod_jac(0, 0);
    // std::cout << "proba_ks: " << proba_ks << '\n';
    double proba_kp = proba_mult * prod_jac(0, 1);
    // std::cout << "proba_kp: " << proba_kp << '\n';

    double R_ = R(i, p);
    // std::cout << "R: " << R_ << '\n';
    double R_deriv_ = R_deriv(i, p);
    // std::cout << "R_deriv: " << R_deriv_ << '\n';

    return Eigen::Array2d(
        -(
            proba_ks * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1,0) - problem.r(i)
        ),
        -(
            proba_kp * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1, 1) - problem.r(i)
        )
    );
}


VariableRProblem::VariableRProblem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc
) : d(d), prodFunc(prodFunc), n_players(d.size()) {
    assert(n_players == prodFunc.n_players);
}

double VariableRProblem::net_payoff(
    int i,
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    return proba * R(i, p) - (1 - proba) * d(i)  - r(s(i)) * (Ks(i) + Kp(i));
}

Eigen::ArrayXd VariableRProblem::get_all_net_payoffs(
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    std::vector<double> payoffs;
    payoffs.reserve(n_players);
    for (int i = 0; i < n_players; i++) {
        payoffs.push_back(proba * R(i, p) - (1 - proba) * d(i) - r(s(i)) * (Ks(i) + Kp(i)));
    }
    return Eigen::Map<Eigen::ArrayXd>(payoffs.data(), n_players);
}


DecayingExpRProblem::DecayingExpRProblem(
    Eigen::ArrayXd d,
    ProdFunc prodFunc,
    double r0,
    double c
) : VariableRProblem(d, prodFunc), r0(r0), c(c) {}

double DecayingExpRProblem::r(double s) const {
    return r0 * std::exp(-c * s);
}

double DecayingExpRProblem::drds(double s) const {
    return -c * r0 * std::exp(-c * s);
}


VariableRObjective::VariableRObjective(
    const VariableRProblem* problem,
    int i,
    Eigen::ArrayX2d last_strat
) : BaseObjective(i, last_strat), problem(problem) {}

double VariableRObjective::f(const Eigen::Array2d& x) {
    // exactly the same as vanilla Objective::f
    last_strat.row(i) = x;
    return -problem->net_payoff(
        i, last_strat.col(0), last_strat.col(1)
    );
}

Eigen::Array2d VariableRObjective::jac(const Eigen::Array2d& x) {
    last_strat.row(i) = x;
    auto [s, p] = problem->prodFunc.f(
        last_strat.col(0),
        last_strat.col(1)
    );
    Eigen::ArrayXd probas = s / (1 + s);
    double proba = probas.prod();
    double proba_mult = proba / (s(i) * (1 + s(i)));

    Eigen::Array22d prod_jac = problem->prodFunc.jac_single_i(i, x);
    // dsdKs = prod_jac(0, 0)
    // dsdKp = prod_jac(0, 1)
    // dpdKs = prod_jac(1, 0)
    // dpdKp = prod_jac(1, 1)
    double proba_ks = proba_mult * prod_jac(0, 0);
    double proba_kp = proba_mult * prod_jac(0, 1);

    double R_ = R(i, p);
    double R_deriv_ = R_deriv(i, p);

    double r = problem->r(s(i));
    double drds = problem->drds(s(i));
    double drKsdKs = r - drds * prod_jac(0, 0) * x(0);
    double drKpdKp = r - drds * prod_jac(0, 1) * x(1);

    return Eigen::Array2d(
        -(
            proba_ks * (R_ + problem->d(i)) + proba * R_deriv_ * prod_jac(1,0) - drKsdKs
        ),
        -(
            proba_kp * (R_ + problem->d(i)) + proba * R_deriv_ * prod_jac(1, 1) - drKpdKp
        )
    );
}
