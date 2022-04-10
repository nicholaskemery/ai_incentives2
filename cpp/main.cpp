#include <iostream>
#include "prodFunc.hpp"
#include "problem.hpp"
#include "solve.hpp"

int main() {
    ProdFunc prodFunc(
        Eigen::Array2d(1.0, 1.0),
        Eigen::Array2d(0.5, 0.5),
        Eigen::Array2d(1.0, 1.0),
        Eigen::Array2d(0.5, 0.5),
        Eigen::Array2d(0.0, 0.0)
    );

    std::vector<Problem> problems = {
        Problem(Eigen::Array2d(1.0, 1.0), 0.01, prodFunc),
        Problem(Eigen::Array2d(0.5, 0.5), 0.01, prodFunc)
    };

    Eigen::Array<double, 2, 2> last_strat;
    last_strat << 1.0, 1.0,
                  1.0, 1.0;
    

    std::vector<Eigen::ArrayX2d> solutions;
    for (const Problem& problem : problems) {
        solutions.push_back(
            solve(
                problem,
                100,
                0.001,
                last_strat
            )
        );
    }
     
    for (const auto& s : solutions) {
        std::cout << s << '\n';
    }

    return 0;
}