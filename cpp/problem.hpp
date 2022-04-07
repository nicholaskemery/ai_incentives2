#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <utility>

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

    Eigen::ArrayXXd solve_single(
        const Eigen::ArrayXXd& last_strat
    ) {
        // do something
    }

    Eigen::ArrayXd d;
    double r;
    ProdFunc prodFunc;
};


class Objective {
public:
    Objective(
        const Problem& problem,
        int i,
        Eigen::ArrayXXd last_strat
    );

    double f(const Eigen::Array2d& x);
    Eigen::Array2d jac(const Eigen::Array2d& x);

    Eigen::ArrayXXd last_strat;
    const int i;
    const Problem& problem;
    const ProdFuncJac prodJac;
};
