#include <Eigen/Eigen>
#include <vector>
#include "pybindings.hpp"


void run(
    int n_persons,
    double* A,
    double* alpha,
    double* B,
    double* beta,
    double* theta,
    double* d,
    double r,
    int max_iters,
    double exit_tol,
    double* result
) {
    Eigen::ArrayXd d_ = Eigen::Map<Eigen::ArrayXd>(d, n_persons);
    ProdFunc prodFunc(
        Eigen::Map<Eigen::ArrayXd>(A, n_persons),
        Eigen::Map<Eigen::ArrayXd>(alpha, n_persons),
        Eigen::Map<Eigen::ArrayXd>(B, n_persons),
        Eigen::Map<Eigen::ArrayXd>(beta, n_persons),
        Eigen::Map<Eigen::ArrayXd>(theta, n_persons)
    );
    Problem problem(
        Eigen::Map<Eigen::ArrayXd>(d, n_persons),
        r,
        prodFunc
    );

    Eigen::ArrayX2d solution = solve(problem, max_iters, exit_tol);
    Eigen::Map<Eigen::ArrayX2d>(result, solution.rows(), solution.cols()) = solution;
}
