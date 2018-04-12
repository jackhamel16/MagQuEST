#include "newton.h"

NewtonSolver::NewtonSolver(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    const std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector &rhs_functions)
    : Solver(dt, history, interactions, rhs_functions)
{
}

void NewtonSolver::solve_step(int step)
{
  auto pulse_interactions_past = interactions[0]->evaluate(step - 1);
  auto history_interactions_past = interactions[1]->evaluate(step - 1);
  auto self_interactions_past = interactions[2]->evaluate(step - 1);

  for(int sol = 0; sol < num_solutions; ++sol) {
    Eigen::Vector3d f_past =
        (history->array[sol][step - 1][0] - history->array[sol][step - 2][0]) /
            dt +
        rhs_functions[sol](history->array[sol][step - 1][0],
                           pulse_interactions_past[sol] +
                               history_interactions_past[sol] +
                               self_interactions_past[sol]);
    // Guess M
    history->array[sol][step][0] =
        history->array[sol][step - 1][0] + Eigen::Vector3d(1, 1, 1);
    // The addition is to prevent singularities

    int max_iter = 10;
    for(int iter = 0; iter < max_iter; ++iter) {
      // need to be recomputed for particles within 1dt of each other each loop
      auto pulse_interactions = interactions[0]->evaluate(step);
      auto history_interactions = interactions[1]->evaluate(step);
      auto self_interactions = interactions[2]->evaluate(step);

      Eigen::Vector3d f =
          (history->array[sol][step][0] - history->array[sol][step - 1][0]) /
              dt +
          rhs_functions[sol](history->array[sol][step][0],
                             pulse_interactions[sol] +
                                 history_interactions[sol] +
                                 self_interactions[sol]);

      Eigen::Vector3d df = (f - f_past) / dt;
      for(int i = 0; i < 3; ++i) {
        history->array[sol][step][0][i] =
            history->array[sol][step][0][i] - f[i] / df[i];
      }
    }
  }
}
