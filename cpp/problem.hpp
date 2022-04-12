#pragma once

#include <Eigen/Eigen>
#include <utility>
#include <cassert>

#include "prodFunc.hpp"

class Problem {
public:
    Problem(
        Eigen::ArrayXd d,
        Eigen::ArrayXd r,
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
    ) const;

    const Eigen::ArrayXd d;
    const Eigen::ArrayXd r;
    ProdFunc prodFunc;
    const int n_players;
};


class BaseObjective {
public:
    BaseObjective(
        int i,
        Eigen::ArrayX2d last_strat
    );
    virtual double f(const Eigen::Array2d& x) = 0;
    virtual Eigen::Array2d jac(const Eigen::Array2d& x) = 0;

    Eigen::ArrayX2d last_strat;
    const int i;
};


class Objective : public BaseObjective {
public:
    Objective(
        const Problem& problem,
        int i,
        Eigen::ArrayX2d last_strat
    );

    virtual double f(const Eigen::Array2d& x) override;
    virtual Eigen::Array2d jac(const Eigen::Array2d& x) override;

    const Problem& problem;
};


class VariableRProblem {
public:
// here r is allowed to vary for players as a function of s
    VariableRProblem(
        Eigen::ArrayXd d,
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
    ) const;

    virtual double r(double s) const = 0;
    virtual double drds(double s) const = 0;

    const Eigen::ArrayXd d;
    ProdFunc prodFunc;
    const int n_players;
};


class DecayingExpRProblem : public VariableRProblem {
    // r = r0 exp(-c*s)
public:
    DecayingExpRProblem(
        Eigen::ArrayXd d,
        ProdFunc prodFunc,
        double r0,
        double c
    );

    virtual double r(double s) const;
    virtual double drds(double s) const;

    const double r0;
    const double c;
};


class VariableRObjective : public BaseObjective {
public:
    VariableRObjective(
        const VariableRProblem* problem,
        int i,
        Eigen::ArrayX2d last_strat
    );

    virtual double f(const Eigen::Array2d& x) override;
    virtual Eigen::Array2d jac(const Eigen::Array2d& x) override;

    const VariableRProblem* const problem;
};
