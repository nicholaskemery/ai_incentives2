#include <Eigen/Eigen>
#include "pybindings.hpp"


ProdFunc build_ProdFunc(
    int n_persons,
    double* A,
    double* alpha,
    double* B,
    double* beta,
    double* theta
) {
    return ProdFunc(
        Eigen::Map<Eigen::ArrayXd>(A, n_persons),
        Eigen::Map<Eigen::ArrayXd>(alpha, n_persons),
        Eigen::Map<Eigen::ArrayXd>(B, n_persons),
        Eigen::Map<Eigen::ArrayXd>(beta, n_persons),
        Eigen::Map<Eigen::ArrayXd>(theta, n_persons)
    );
}


void run(
    int n_persons,
    double* A,
    double* alpha,
    double* B,
    double* beta,
    double* theta,
    double* d,
    double* r,
    int max_iters,
    double exit_tol,
    double* result
) {
    ProdFunc prodFunc = build_ProdFunc(n_persons, A, alpha, B, beta, theta);
    Eigen::ArrayXd d_ = Eigen::Map<Eigen::ArrayXd>(d, n_persons);
    Problem problem(
        Eigen::Map<Eigen::ArrayXd>(d, n_persons),
        Eigen::Map<Eigen::ArrayXd>(r, n_persons),
        prodFunc
    );

    Eigen::ArrayX2d solution = solve(problem, max_iters, exit_tol);
    Eigen::Map<Eigen::ArrayX2d>(result, solution.rows(), solution.cols()) = solution;
}


void prod_F(
    int n_persons,
    double* A,
    double* alpha,
    double* B,
    double* beta,
    double* theta,
    double* Ks,
    double* Kp,
    double* s_out,
    double* p_out
) {
    auto [s, p] = build_ProdFunc(
        n_persons, A, alpha, B, beta, theta
    ).f(
        Eigen::Map<Eigen::ArrayXd>(Ks, n_persons),
        Eigen::Map<Eigen::ArrayXd>(Kp, n_persons)
    );
    Eigen::Map<Eigen::ArrayXd>(s_out, n_persons) = s;
    Eigen::Map<Eigen::ArrayXd>(p_out, n_persons) = p;
}


void get_payoffs(
    int n_persons,
    double* A,
    double* alpha,
    double* B,
    double* beta,
    double* theta,
    double* d,
    double* r,
    double* Ks,
    double* Kp,
    double* payoffs_out
) {
    ProdFunc prodFunc = build_ProdFunc(n_persons, A, alpha, B, beta, theta);
    Eigen::ArrayXd Ks_ = Eigen::Map<Eigen::ArrayXd>(Ks, n_persons);
    Eigen::ArrayXd Kp_ = Eigen::Map<Eigen::ArrayXd>(Kp, n_persons);
    Eigen::ArrayXd payoffs = Problem(
        Eigen::Map<Eigen::ArrayXd>(d, n_persons),
        Eigen::Map<Eigen::ArrayXd>(r, n_persons),
        prodFunc
    ).get_all_net_payoffs(
        Ks_, Kp_
    );
    Eigen::Map<Eigen::ArrayXd>(payoffs_out, n_persons) = payoffs;
}
