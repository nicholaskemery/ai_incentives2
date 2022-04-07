#include "solve.hpp"

VarSet::VarSet(
    const std::string& name,
    int numVars
) : ifopt::VariableSet(numVars, name),
    vars(Eigen::VectorXd::Constant(numVars, 1.0)),
    bounds(std::vector<ifopt::Bounds>(numVars, ifopt::BoundGreaterZero))
{}

VarSet::VarSet(
    const std::string& name,
    int numVars,
    Eigen::VectorXd initVals
) : ifopt::VariableSet(numVars, name),
    vars(initVals),
    bounds(std::vector<ifopt::Bounds>(numVars, ifopt::BoundGreaterZero))
{}

VarSet::VarSet(
    const std::string& name,
    int numVars,
    Eigen::VectorXd initVals,
    std::vector<ifopt::Bounds> bounds
) : ifopt::VariableSet(numVars, name),
    vars(initVals),
    bounds(bounds)
{}

IfoptObjective::IfoptObjective(
    const std::string& name,
    const std::string& varName,
    Objective& objectiveFunc
) : ifopt::CostTerm(name),
    varName(varName),
    objectiveFunc(objectiveFunc)
{}

double IfoptObjective::GetCost() const {
    Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
    // notice we return the negative objectiveFunc since we're trying to maximize objective
    return objectiveFunc.f(x);
}

void IfoptObjective::FillJacobianBlock(
    std::string var_set,
    SparseJacobian& jac
) const {
    if (var_set == varName) {
        Eigen::ArrayXd x = GetVariables()->GetComponent(varName)->GetValues().array();
        Eigen::Array2d jac_ = objectiveFunc.jac(x);
        jac.coeffRef(0, 0) = jac_(0);
        jac.coeffRef(0, 1) = jac_(1);
    }
}

void IfoptObjective::FillJacobianBlock(SparseJacobian& jac) const {
    return FillJacobianBlock(varName, jac);
}


void configure_to_default_solver(std::shared_ptr<ifopt::IpoptSolver> solver) {
    // use MUMPS as linear solver
    // if you have the HSL solvers, you should use those instead
    solver->SetOption("linear_solver", "mumps");
    // require jacobians to be pre-provided
    solver->SetOption("jacobian_approximation", "exact");
    solver->SetOption("print_level", 3);

    // solver->SetOption("tol", 0.1);
    // solver->SetOption("max_iter", 100000);
}


IfoptProblem::IfoptProblem(
    std::shared_ptr<VarSet> varSet,
    std::shared_ptr<IfoptObjective> objective
) {
    problem.AddVariableSet(varSet);
    problem.AddCostSet(objective);
    configure_to_default_solver(solver);
}

Eigen::ArrayXd IfoptProblem::solve() {
    solver->Solve(problem);
    return problem.GetOptVariables()->GetValues().array();
}

void IfoptProblem::changeSolver(std::shared_ptr<ifopt::IpoptSolver> newSolver) {
    solver = newSolver;
}
