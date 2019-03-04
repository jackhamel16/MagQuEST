#include "euler.h"

EulerIntegrator::EulerIntegrator(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> &rhs_functions)
    : time_idx_ubound(history->array.index_bases()[1] +
                      history->array.shape()[1]),
      dt(dt),
      history(history),
      rhs_functions(std::move(rhs_functions))
{
}

void EulerIntegrator::solve() const
{
  for(int step = 1; step < time_idx_ubound; ++step) {
    solve_step(step);
  }
}

void EulerIntegrator::solve_step(const int step) const
{
  rhs_functions->evaluate(step - 1);
  for(int src = 0; src < static_cast<int>(history->array.shape()[0]); ++src) {
    history->array[src][step][0] = history->array[src][step - 1][1] * dt +
                                   history->array[src][step - 1][0];
  }
}
