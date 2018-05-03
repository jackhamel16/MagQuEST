#include "newton.h"

NewtonSolver::NewtonSolver(
    const double dt,
    double max_iter,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector &rhs_functions)
    : Solver(dt, history, interactions, rhs_functions), max_iter(max_iter)
{
}

Eigen::Matrix3d NewtonSolver::approx_jacob(std::function<vec3d(vec3d)> func,
                                           vec3d f,
                                           vec3d x,
                                           double eps)
{
  int n = x.size();
  Eigen::Matrix<double, 3, 3> jacob;

  for(int j = 0; j < n; ++j) {
    vec3d xh = x;
    double h = eps * std::abs(xh[j]);
    if(h == 0.0) h = eps;
    xh[j] = xh[j] + h;

    vec3d fh = func(xh);
    for(int i = 0; i < n; ++i) jacob(i, j) = (fh[i] - f[i]) / h;
  }
  return jacob;
}

vec3d newton_rhs(vec3d mag,
                 vec3d mag_past,
                 vec3d field,
                 int sol,
                 int step,
                 rhs_func_vector rhs_functions)
{
  // may have issues with not recomputing fields when mag is changed
  return (mag - mag_past) / step + rhs_functions[sol](mag, field);
}

vec3d test_func(vec3d vec1)
{
  return vec3d(std::pow(vec1[0], 2) - 4, std::pow(vec1[1], 2) - 4,
               std::pow(vec1[2], 2) - 4);
}

void NewtonSolver::solve_step(int step)
{
  //double tol = 1e-8;

  //vec3d solution(2, 2, 2);
  //vec3d iter_solution = solution + solution * 1e-2;

  //for(int iter = 0; iter < max_iter; ++iter) {
    //auto newton_func = std::bind(test_func, std::placeholders::_1);

    //vec3d f_iter = newton_func(iter_solution);
    //auto jacobian = approx_jacob(newton_func, f_iter, iter_solution, tol);

    //vec3d delta_m = -1 * jacobian.partialPivLu().solve(f_iter);

    //if(step == 1)
      //std::cout << std::endl << "f_iter: " << f_iter.transpose() << std::endl;
    //if(step == 1) std::cout << "delta: " << delta_m.transpose() << std::endl;

    //iter_solution = iter_solution + delta_m;

    //if(step == 1)
      //std::cout << "sol: " << iter_solution.transpose() << std::endl;
  //}
  double tol = 1e-8;

  for(int sol = 0; sol < num_solutions; ++sol) {
    history->array[sol][step][0] =
        history->array[sol][step - 1][0] + Eigen::Vector3d(1, 1, 1) * 1e-6;
  }

  for(int iter = 0; iter < max_iter; ++iter) {
    auto pulse_interactions = interactions[0]->evaluate(step);
    auto history_interactions = interactions[1]->evaluate(step);
    auto self_interactions = interactions[2]->evaluate(step);

    for(int sol = 0; sol < num_solutions; ++sol) {
      auto newton_func = std::bind(
          newton_rhs, std::placeholders::_1, history->array[sol][step - 1][0],
          pulse_interactions[sol] + history_interactions[sol] +
              self_interactions[sol],
          sol, step, rhs_functions);

      vec3d f_iter = newton_func(history->array[sol][step][0]);
      auto jacobian =
          approx_jacob(newton_func, f_iter, history->array[sol][step][0], tol);

      vec3d delta_m = -1 * jacobian.partialPivLu().solve(f_iter);

      if(step==40) std::cout << f_iter.transpose() << std::endl;

      history->array[sol][step][0] = history->array[sol][step][0] + delta_m;
    }
  }
}
