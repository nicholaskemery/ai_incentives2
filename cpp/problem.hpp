#pragma once

#include <Eigen/Eigen>
#include <utility>
#include <cassert>

#include "prodFunc.hpp"


class CSF {
public:

    CSF();

    CSF(
        double W,
        double L,
        double a_w,
        double a_l
    );

    double reward(int i, const Eigen::ArrayXd& p) const;

    double reward_deriv(int i, const Eigen::ArrayXd& p) const;

    const double W;
    const double L;
    const double a_w;
    const double a_l;
};


class Problem {
public:
// here r is allowed to vary for players as a function of s
    Problem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc
    );

    Problem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc,
        CSF csf
    );

    double net_payoff(
        int i,
        const Eigen::ArrayXd& Ks,
        const Eigen::ArrayXd& Kp
    ) const;

    Eigen::ArrayXd get_all_net_payoffs(
        const Eigen::ArrayXd& Ks,
        const Eigen::ArrayXd& Kp
    ) const;

    virtual double r(int i, double s) const = 0;
    virtual double drds(int i, double s) const = 0;

    const Eigen::ArrayXd d;
    ProdFunc prodFunc;
    const int n_players;
    const CSF csf;
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

    const Problem& problem;
    const int i;
    Eigen::ArrayX2d last_strat;
};


class ConstantRProblem : public Problem {
public:
    ConstantRProblem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc,
        Eigen::ArrayXd r_
    );

    ConstantRProblem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc,
        Eigen::ArrayXd r_,
        CSF csf
    );

    virtual double r(int i, double s) const;
    virtual double drds(int i, double s) const;

    const Eigen::ArrayXd r_;
};


class DecayingExpRProblem : public Problem {
    // r = r0 exp(-c*s)
public:
    DecayingExpRProblem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc,
        double r0,
        double c
    );

    virtual double r(int i, double s) const;
    virtual double drds(int i, double s) const;

    const double r0;
    const double c;
};
