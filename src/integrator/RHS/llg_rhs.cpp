#include "llg_rhs.h"
//#include "gmres.h"
#include <fstream>
#include <ostream>
#include <string>

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
  auto self_interactions = interactions[2]->evaluate(step);

  Eigen::Matrix<Eigen::Vector3d, num_solutions, 1> H_vec;
  for(int sol = 0; sol < num_solutions; ++sol) {
    H_vec[sol] = history->array[sol][step][0];
  }

  for(int sol = 0; sol < num_solutions; ++sol) {
    history->array[sol][step][1] = rhs_functions[sol](
        history->array[sol][step][0], pulse_interactions[sol] +
                                          history_interactions_past[sol] +
                                          self_interactions[sol]);
  }
  // if(step == 10) {
  // std::cout << "step: " << step << "  "
  //<< "H_now: " << history_interactions_now[0].transpose() << std::endl;
  //}
}
