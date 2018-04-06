#include "self_interaction.h"

SelfInteraction::SelfInteraction(
    const std::shared_ptr<const DotVector> &dots,
    const std::shared_ptr<const Integrator::History<soltype>> &history)
    : Interaction(dots), history(history)
{
  chi = 1;
}

const Interaction::ResultArray &SelfInteraction::evaluate(const int time_idx)
{
  for(int i = 0; i < static_cast<int>(dots->size()); ++i) {
    results[i] = -chi * history->array[i][time_idx][0] / 3;
  }
  return results;
}
