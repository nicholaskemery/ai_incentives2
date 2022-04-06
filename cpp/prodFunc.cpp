#include <cmath>
#include "prodFunc.hpp"

#include <iostream>

ProdFuncJac::ProdFuncJac(
    const ProdFunc& prodFunc,
    int i
) : prodFunc(prodFunc), i(i) {}

Eigen::Array22d ProdFuncJac::get(
    const Eigen::Array2d& x
) const {
    auto [p, s] = prodFunc.f_single_i(i, x(0), x(1));
    double s_mult = prodFunc.A(i) * prodFunc.alpha(i) * std::pow(s / prodFunc.A(i), 1 - 1 / prodFunc.alpha(i));
    double p_mult = prodFunc.B(i) * prodFunc.beta(i) * std::pow(p / prodFunc.B(i), 1 - 1 / prodFunc.beta(i));
    double out00 = s_mult * std::pow(p, -prodFunc.theta(i));
    double out01 = -prodFunc.theta(i) * s * std::pow(p, -prodFunc.theta(i) - 1) * p_mult;
    Eigen::Array22d out;
    out << out00, out01,
           0.0,   p_mult;
    return out;
}


ProdFunc::ProdFunc(
    Eigen::ArrayXd A,
    Eigen::ArrayXd alpha,
    Eigen::ArrayXd B,
    Eigen::ArrayXd beta,
    Eigen::ArrayXd theta
) : A(A), alpha(alpha), B(B), beta(beta), theta(theta) {}

std::tuple<double, double> ProdFunc::f_single_i(
    int i,
    double Ks,
    double Kp
) const {
    double p = B(i) * std::pow(Kp, beta(i));
    double s = A(i) * std::pow(Ks, alpha(i)) * std::pow(p, -theta(i));
    return {p, s};
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> ProdFunc::f(
    const Eigen::ArrayXd& Ks,
    const Eigen::ArrayXd& Kp
) const {
    Eigen::ArrayXd p = B * Kp.pow(beta);
    Eigen::ArrayXd s = A * Ks.pow(alpha) * p.pow(-theta);
    return {p, s};
}
