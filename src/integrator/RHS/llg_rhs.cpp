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
      num_particles(history->array.shape()[0]),
      interactions(std::move(interactions)),
      rhs_functions(std::move(rhs_functions))
{
}

void Integrator::LLG_RHS::evaluate(const int step) const
{
  auto pulse_interactions = interactions[0]->evaluate(step);
  auto history_interactions = interactions[1]->evaluate(step);
  auto self_interactions = interactions[2]->evaluate(step);

  std::cout << history_interactions[0].transpose() << std::endl;

  for(int p_idx = 0; p_idx < num_particles; ++p_idx) {
    history->array[p_idx][step][1] =
        rhs_functions[p_idx](history->array[p_idx][step][0],
                           pulse_interactions[p_idx] + history_interactions[p_idx] +
                           //pulse_interactions[p_idx] +
                               self_interactions[p_idx]);
  }
}
