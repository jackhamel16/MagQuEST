#include "euler.h"

EulerIntegrator::EulerIntegrator(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> &rhs_functions)
    : dt(dt), history(history), rhs_functions(std::move(rhs_functions))
{
}

void EulerIntegrator::solve(const int step) const
{
  rhs_functions->evaluate(step);
  for(unsigned int num = 0; num < history->array.size(); ++num) {
    history->array[step][num][0] =
        history->array[step][num][1] * dt + history->array[step][num][0];
  }
}
