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

  Eigen::Matrix<double, Eigen::Dynamic, 1> H_vec(3 * num_solutions);
  for(int sol = 0; sol < 3 * num_solutions; ++sol) H_vec[sol] = 0;

  int m = 50;
  int max_iter = 500;
  double tol = 1e-9;
  Eigen::Matrix<double, 51, 51> H;  // m+1 x m+1

  std::vector<Eigen::Vector3d> interactions_past(num_solutions);
  for(int i = 0; i < num_solutions; ++i) {
    interactions_past[i] = pulse_interactions[i] + history_interactions_past[i];
  }

  // Mapping std:vector<Eigen::Vector3d> to an Eigen::VectorXd for GMRES
  Eigen::Matrix<double, Eigen::Dynamic, 1> interactions_past_vec =
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
          interactions_past[0].data(), 3 * num_solutions);

  auto GMRES_output = GMRES::GMRES(interactions[1], H_vec,
                                   interactions_past_vec, H, m, max_iter, tol);
  //if(step==200)for(int i=0; i <3*num_solutions; ++i) std::cout << H_vec[i] <<std::endl;

  for(int sol = 0; sol < num_solutions; ++sol) {
    Eigen::Vector3d sol_vector =
        Eigen::Vector3d(H_vec[3 * sol], H_vec[3 * sol + 1], H_vec[3 * sol + 2]);
    if(step==200) std::cout << sol_vector.transpose() << std::endl;
    history->array[sol][step][0] = sol_vector;
    history->array[sol][step][1] = pulse_interactions[sol];
  }
}
