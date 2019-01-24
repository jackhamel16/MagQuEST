#include "solver.h"

Solver::Solver(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    const std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector &rhs_functions)
    : dt(dt),
      history(history),
      interactions(interactions),
      rhs_functions(std::move(rhs_functions)),
      time_idx_ubound(history->array.index_bases()[1] +
                      history->array.shape()[1]),
      num_solutions(history->array.shape()[0])
{
}
void Solver::solve() {
  for(int step = 1; step < time_idx_ubound; ++step) {
    solve_step(step);
  }
}

