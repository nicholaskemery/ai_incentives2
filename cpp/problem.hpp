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
        const std::vector<Eigen::ArrayXXd> history
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
        std::vector<Eigen::ArrayXXd> history
    );

    double get(const Eigen::Array2d& x);

    std::vector<Eigen::ArrayXXd> history;
    const int i;
    const Problem& problem;
};


class ProblemJac {
public:
    ProblemJac(
        const Problem& problem,
        int i,
        std::vector<Eigen::ArrayXXd> history
    );

    Eigen::Array2d get(const Eigen::Array2d& x);

    std::vector<Eigen::ArrayXXd> history;
    const int i;
    const Problem& problem;
    const ProdFuncJac prodJac;
};
