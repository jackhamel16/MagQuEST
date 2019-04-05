#include "newton.h"

NewtonSolver::NewtonSolver(
    const double dt,
    double max_iter,
    const std::shared_ptr<Integrator::History<vec3d>> &history,
    const std::shared_ptr<Integrator::History<vec3d>> &delta_history,
    std::vector<std::shared_ptr<Interaction>> interactions,
    std::vector<std::shared_ptr<Interaction>> delta_interactions,
    rhs_func_vector &rhs_functions,
    jacobian_matvec_func_vector &jacobian_matvec_funcs)
    : Solver(dt, history, interactions, rhs_functions),
      max_iter(max_iter),
      delta_history(delta_history),
      delta_interactions(delta_interactions),
      jacobian_matvec_funcs(jacobian_matvec_funcs)
{
}

vec3d newton_rhs(vec3d mag,
                 vec3d mag_past,
                 vec3d field,
                 int sol,
                 int step_size,
                 rhs_func_vector &rhs_functions)
{
  // may have issues with not recomputing fields when mag is changed
  return (mag - mag_past) / step_size -
         rhs_functions[sol](mag,
                            field);  // may need to negate rhs_functions
}

void NewtonSolver::solve_step(int step)
{
  double tol = 1e-8;

  // Guess solutions (bad guess right now; hold magnitude constant)
  for(int sol = 0; sol < num_solutions; ++sol) {
    history->array[sol][step][0] =
        history->array[sol][step - 1][0] + vec3d(1, 1, 1) * 1e-12;
    delta_history->array[sol][step][0] =
        delta_history
            ->array[sol][step - 1]
                   [0];  // Make sure setting up delta_history accounts for this
  }

  for(int iter = 0; iter < max_iter; ++iter) {
    auto pulse_interactions = interactions[0]->evaluate(step);
    auto history_interactions = interactions[1]->evaluate(step);
    auto self_interactions = interactions[2]->evaluate(step);
    auto delta_history_interactions = delta_interactions[1]->evaluate(step);
    auto delta_self_interactions = delta_interactions[2]->evaluate(step);

    for(int sol = 0; sol < num_solutions; ++sol) {
      vec3d mag_fields = pulse_interactions[sol] + history_interactions[sol] +
                         self_interactions[sol];
      vec3d delta_fields =
          delta_history_interactions[sol] + delta_self_interactions[sol];
      vec3d jacobian_matvec = jacobian_matvec_funcs[sol](
          history->array[sol][step][0], delta_history->array[sol][step][0],
          mag_fields, delta_fields);

      vec3d newton_func = newton_rhs(history->array[sol][step][0],
                                     history->array[sol][step - 1][0],
                                     mag_fields, sol, dt, rhs_functions);

      delta_history->array[sol][step][0] = dt * jacobian_matvec + newton_func;

      history->array[sol][step][0] =
          history->array[sol][step][0] + delta_history->array[sol][step][0];
    }
  }
}
