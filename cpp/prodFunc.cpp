#include <cmath>
#include "prodFunc.hpp"


ProdFunc::ProdFunc(
    Eigen::ArrayXd A,
    Eigen::ArrayXd alpha,
    Eigen::ArrayXd B,
    Eigen::ArrayXd beta,
    Eigen::ArrayXd theta
) : A(A), alpha(alpha), B(B), beta(beta), theta(theta) {
    n_players = A.size();
    assert(
        n_players == alpha.size()
        && n_players == B.size()
        && n_players == beta.size()
        && n_players == theta.size()
    );
}

std::tuple<double, double> ProdFunc::f_single_i(
    int i,
    double Ks,
    double Kp
) const {
    double p = B(i) * std::pow(Kp, beta(i));
    double s = A(i) * std::pow(Ks, alpha(i)) * std::pow(p, -theta(i));
    return {s, p};
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> ProdFunc::f(
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    Eigen::ArrayXd p = B * Kp.pow(beta);
    Eigen::ArrayXd s = A * Ks.pow(alpha) * p.pow(-theta);
    return {s, p};
}

Eigen::Array22d ProdFunc::jac_single_i(
    int i,
    const Eigen::ArrayXd& x
) const {
    auto [s, p] = f_single_i(i, x(0), x(1));
    double s_mult = A(i) * alpha(i) * std::pow(s / A(i), 1.0 - 1.0 / alpha(i));
    double p_mult = B(i) * beta(i) * std::pow(p / B(i), 1.0 - 1.0 / beta(i));
    double out00 = s_mult * std::pow(p, -theta(i));
    double out01 = -theta(i) * s * std::pow(p, -theta(i) - 1.0) * p_mult;
    Eigen::Array22d out;
    out << out00, out01,
           0.0,   p_mult;
    return out;
}
