#include "newton.h"

NewtonSolver::NewtonSolver(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    const std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector &rhs_functions)
    : Solver(dt, history, interactions, rhs_functions)
{
}
