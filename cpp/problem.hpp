#pragma once

#include <Eigen/Eigen>
#include <utility>
#include <cassert>

#include "prodFunc.hpp"


class Problem {
public:
    Problem(
        Eigen::ArrayXd d,
        double r,
        ProdFunc prodFunc
    );

    double net_payoff(
        int i,
        const Eigen::ArrayXd& Ks,
        const Eigen::ArrayXd& Kp
    ) const;

    Eigen::ArrayXd get_all_net_payoffs(
        const Eigen::ArrayXd& Ks,
        const Eigen::ArrayXd& Kp
    );

    Eigen::ArrayXd d;
    double r;
    ProdFunc prodFunc;
    int n_players;
};


class Objective {
public:
    Objective(
        const Problem& problem,
        int i,
        Eigen::ArrayX2d last_strat
    );

    double f(const Eigen::Array2d& x);
    Eigen::Array2d jac(const Eigen::Array2d& x);

    Eigen::ArrayX2d last_strat;
    const int i;
    const Problem& problem;
};
