#include <iostream>
#include "prodFunc.hpp"
#include "problem.hpp"

int main() {
    ProdFunc prodFunc(
        Eigen::Array3d(1.0, 1.0, 1.0),
        Eigen::Array3d(0.5, 0.5, 0.5),
        Eigen::Array3d(1.0, 1.0, 1.0),
        Eigen::Array3d(0.5, 0.5, 0.5),
        Eigen::Array3d(0.1, 0.2, 0.3)
    );

    Problem problem(Eigen::Array3d(1.0, 1.0, 1.0), 0.01, prodFunc);

    std::vector<Eigen::ArrayXXd> history;
    Eigen::Array<double, 3, 2> hist0;
    hist0 << 1.0, 1.0,
             2.0, 2.0,
             3.0, 3.0;
    history.push_back(hist0);
    

    ProblemJac jac(problem, 0, history);

    std::cout << jac.get(Eigen::Array2d(1.0, 1.5)) << '\n';

    return 0;
}