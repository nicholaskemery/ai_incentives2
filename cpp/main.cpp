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

    Problem problem(Eigen::Array2d(1.0, 1.0), 0.01, prodFunc);

    Eigen::Array<double, 2, 2> last_strat;
    last_strat << 1.0, 1.0,
                  0.1, 0.1;
    

    Objective obj(problem, 0, last_strat);

    Eigen::Array2d x(1.0, 1.5);

    auto [s, p] = prodFunc.f_single_i(0, x(0), x(1));
    std::cout << "prodfunc: " << s << ' ' << p << '\n';

    std::cout << "f: " << obj.f(x) << '\n';
    std::cout << "jac: " << obj.jac(x) << '\n';

    auto varSet = std::make_shared<VarSet>("vars", 2);
    auto objective = std::make_shared<IfoptObjective>("obj", "vars", obj);
    IfoptProblem ifoptProblem(varSet, objective);

    std::cout << ifoptProblem.solve() << '\n';

    return 0;
}