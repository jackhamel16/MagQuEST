#include "newton.h"

NewtonSolver::NewtonSolver(
    const double dt,
    const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &history,
    std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector &rhs_functions)
    : Solver(dt, history, interactions, rhs_functions)
{
}

Eigen::Matrix3d NewtonSolver::approx_jacob(
    std::function<vec3d(vec3d, vec3d)> func,
    vec3d f,
    vec3d x,
    vec3d x_past,
    double eps)
{
  int n = x.size();
  Eigen::Matrix<double, 3, 3> jacob;

  for(int j = 0; j < n; ++j) {
    vec3d xh = x;
    vec3d xh_past = x_past;
    double h = eps * std::abs(xh[j]);
    double h_past = eps * std::abs(xh_past[j]);
    if(h == 0.0) h = eps;
    if(h_past == 0.0) h_past = eps;
    xh[j] = xh[j] + h;
    xh_past[j] = xh_past[j] + h_past;

    vec3d fh = func(xh, xh_past);
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
  // approx_jacob
  return (mag - mag_past) / step + rhs_functions[sol](mag, field);
}

vec3d test_func(vec3d vec1, vec3d vec2, int num)
{
  return vec3d(std::pow(vec1[0], 2), std::pow(vec1[1], 2),
               std::pow(vec1[2], 2)) *
         num + vec2;
}

void NewtonSolver::solve_step(int step)
{
  double tol = 1e-8;

  auto pulse_interactions = interactions[0]->evaluate(step);
  auto history_interactions = interactions[1]->evaluate(step);
  auto self_interactions = interactions[2]->evaluate(step);
  int sol = 0;

  vec3d vec1(1,2,3);
  vec3d vec2(5,2,0);

  auto f = std::bind(test_func, std::placeholders::_1, std::placeholders::_2, 2);
  auto jacob = approx_jacob(f, f(vec1, vec2), vec1, vec2, tol);
  //auto newton_func =
      //std::bind(newton_rhs, std::placeholders::_1, std::placeholders::_2,
                //pulse_interactions[sol] + history_interactions[sol] +
                    //self_interactions[sol],
                //sol, step, rhs_functions);

  //auto jacob = approx_jacob(
      //newton_func, newton_func(history->array[sol][step][0],
                               //history->array[sol][step - 1][0]),
      //history->array[sol][step][0], history->array[sol][step][0], tol);

  if (step==1) std::cout << jacob << std::endl;

  // std::function<vec3d(vec3d,vec3d,vec3d,int,int)> f_test = newton_rhs;
  // auto f =
  // std::bind(newton_rhs, std::placeholders::_1, std::placeholders::_2,
  // pulse_interactions[sol] + history_interactions[sol] +
  // self_interactions[sol],
  // sol, step);
  //// Guess M
  // history->array[sol][step][0] =
  // history->array[sol][step - 1][0] + Eigen::Vector3d(1, 1, 1) * 1e-6;
  //// The addition is to prevent singularities
  //// if(step==1)std::cout << history->array[sol][step-1][0].transpose() <<
  //// std::endl;
  // int max_iter = 10;
  // for(int iter = 0; iter < max_iter; ++iter) {
  //// need to be recomputed for particles within 1dt of each other each loop
  // auto pulse_interactions = interactions[0]->evaluate(step);
  // auto history_interactions = interactions[1]->evaluate(step);
  // auto self_interactions = interactions[2]->evaluate(step);

  // Eigen::Vector3d f =
  //(history->array[sol][step][0] - history->array[sol][step - 1][0]) /
  // dt -
  // rhs_functions[sol](history->array[sol][step][0],
  // pulse_interactions[sol] +
  // history_interactions[sol] +
  // self_interactions[sol]);

  // Eigen::Vector3d df = (f - f_past) / dt;

  // for(int i = 0; i < 3; ++i) {
  // history->array[sol][step][0][i] =
  // history->array[sol][step][0][i] - (f[i] / df[i]);
  // if(step == 1) std::cout << f[i] / df[i] << " " << i << std::endl;
  //}
  //}
  //}
}
