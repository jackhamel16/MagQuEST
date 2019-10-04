#include "jfnk.h"

extern double gmres_time;
extern double interaction_time;
extern double gmres_setup_time;

JFNKSolver::JFNKSolver(
    const double dt,
    int max_iter,
    const std::shared_ptr<Integrator::History<vec3d>> &history,
    const std::shared_ptr<Integrator::History<vec3d>> &delta_history,
    std::vector<std::shared_ptr<Interaction>> interactions,
    std::vector<std::shared_ptr<Interaction>> delta_interactions,
    rhs_func_vector &rhs_functions,
    jacobian_matvec_func_vector &matvec_funcs)
    : Solver(dt, history, interactions, rhs_functions),
      max_iter(max_iter),
      delta_history(delta_history),
      delta_interactions(delta_interactions),
      matvec_funcs(matvec_funcs)
{
}

vec3d JFNKSolver::residual_rhs(vec3d mag,
                   vec3d mag_past,
                   vec3d field,
                   int sol,
                   double step_size,
                   rhs_func_vector &rhs_functions)
{
  return mag_past - mag + step_size * rhs_functions[sol](mag, field);
}

void JFNKSolver::solve_step(int step)
{
  // GMRES Parameters
  Eigen::Matrix<double, 3, 3> H;
  int m = 1;  // num of restarts
  int gmres_iters = 1000;
  double gmres_tol = 1e-4;

  std::function<soltype(soltype)> matvec;
  vec3d rhs;
  std::vector<vec3d> delta_vecs(num_solutions);
  for(int sol = 0; sol < num_solutions; ++sol) {
    history->array[sol][step][0] = history->array[sol][step - 1][0] * 0.99;
    // Make sure setting up delta_history accounts for step-1 index
    delta_history->array[sol][step][0] =
        history->array[sol][step][0] - history->array[sol][step - 1][0];
  }
  for(int iter = 0; iter < max_iter; ++iter) {
    auto start = std::chrono::high_resolution_clock::now();
    auto pulse_interactions = interactions[0]->evaluate(step);
    auto history_interactions = interactions[1]->evaluate(step);
    auto self_interactions = interactions[2]->evaluate(step);
    auto delta_history_interactions = delta_interactions[0]->evaluate(step);
    auto delta_self_interactions = delta_interactions[1]->evaluate(step);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = finish - start;
    interaction_time += elapsed_time.count();
    for(int sol = 0; sol < num_solutions; ++sol) {
      start = std::chrono::high_resolution_clock::now();
      vec3d mag_fields = pulse_interactions[sol] + history_interactions[sol] +
                         self_interactions[sol];
      vec3d delta_fields =
          delta_history_interactions[sol] + delta_self_interactions[sol];
      matvec = std::bind(matvec_funcs[sol], dt, history->array[sol][step][0],
                         std::placeholders::_1, mag_fields, delta_fields);
      rhs = residual_rhs(history->array[sol][step][0],
                         history->array[sol][step - 1][0], mag_fields, sol, dt,
                         rhs_functions);
      finish = std::chrono::high_resolution_clock::now();
      elapsed_time = finish - start;
      gmres_setup_time += elapsed_time.count();

      start = std::chrono::high_resolution_clock::now();
      int gmres_out = GMRES(matvec, delta_history->array[sol][step][0],
                            rhs, H, m, gmres_iters, gmres_tol);
      finish = std::chrono::high_resolution_clock::now();
      elapsed_time = finish - start;
      gmres_time += elapsed_time.count();

      //if(gmres_out == 1) std::cout << "Iteration = " << iter << " GMRES COULD NOT CONVERGE\n";
      history->array[sol][step][0] =
          history->array[sol][step][0] + delta_history->array[sol][step][0];
    }
  }
}
