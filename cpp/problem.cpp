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
) : d(d), r(r), prodFunc(prodFunc) {
    n_players = d.size();
    assert(n_players == prodFunc.n_players);
}

double Problem::net_payoff(
    int i,
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    return proba * R(i, p) - (1 - proba) * d(i)  - r * (Ks(i) + Kp(i));
}

Eigen::ArrayXd Problem::get_all_net_payoffs(
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) {
    auto [s, p] = prodFunc.f(Ks, Kp);
    double proba = (s / (1 + s)).prod();
    std::vector<double> payoffs;
    payoffs.reserve(n_players);
    for (int i = 0; i < n_players; i++) {
        payoffs.push_back(proba * R(i, p) - (1 - proba) * d(i) - r * (Ks(i) + Kp(i)));
    }
    return Eigen::Map<Eigen::ArrayXd>(payoffs.data(), n_players);
}


Objective::Objective(
    const Problem& problem,
    int i,
    // copy of last_strat is intentional
    Eigen::ArrayX2d last_strat
) : problem(problem), i(i), last_strat(last_strat) {}

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
            proba_ks * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1,0) - problem.r
        ),
        -(
            proba_kp * (R_ + problem.d(i)) + proba * R_deriv_ * prod_jac(1, 1) - problem.r
        )
    );
}

