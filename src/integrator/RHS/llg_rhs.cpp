#include "llg_rhs.h"
#include <fstream>
#include <ostream>
#include <string>
#include "gmres.h"

Integrator::LLG_RHS::LLG_RHS(
    const double dt,
    const std::shared_ptr<Integrator::History<soltype>> &history,
    std::vector<std::shared_ptr<Interaction>> interactions,
    rhs_func_vector rhs_functions)
    : Integrator::RHS<soltype>(dt, history),
      num_solutions(history->array.shape()[0]),
      interactions(std::move(interactions)),
      rhs_functions(std::move(rhs_functions))
{
}

void Integrator::LLG_RHS::evaluate(const int step) const
{
  auto pulse_interactions = interactions[0]->evaluate(step);
  auto history_interactions_past = interactions[1]->evaluate(step);
  //auto self_interactions = interactions[2]->evaluate(step);

  Eigen::Matrix<double, Eigen::Dynamic, 1> H_vec(3 * num_solutions);
  for(int sol = 0; sol < num_solutions; ++sol) H_vec[sol] = 0;

  int m = 20;
  int max_iter = 200;
  double tol = 1e-7;
  Eigen::Matrix<double, 21, 21> H;  // m+1 x m+1

  std::vector<Eigen::Vector3d> interactions_past(num_solutions);
  for(int i = 0; i < num_solutions; ++i) {
    interactions_past[i] = pulse_interactions[i] + history_interactions_past[i];
  }

  Eigen::Matrix<double, Eigen::Dynamic, 1> interactions_past_vec =
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
          interactions_past[0].data(), 3 * num_solutions);

  if(step == 10) std::cout << H_vec[2] << std::endl;

  auto history_interactions_now = GMRES::GMRES(
      interactions[1], H_vec, interactions_past_vec, H, m, max_iter, tol);

  if(step == 10) std::cout << H_vec[2] << std::endl; 
  // for(int sol = 0; sol < num_solutions; ++sol) {
  // history->array[sol][step][0] = (pulse_interactions[sol] +
  // history_interactions_past[sol]);
  //}
  // if(step == 10) {
  // std::cout << "step: " << step << "  "
  //<< "H_now: " << history_interactions_now[0].transpose() << std::endl;
  //}
}
