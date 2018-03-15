#include "pulse_interaction.h"

PulseInteraction::PulseInteraction(const std::shared_ptr<const DotVector> &dots,
                                   const std::shared_ptr<PulseVector> pulses,
                                   const double hbar,
                                   const double dt)
    : Interaction(dots), pulses(std::move(pulses)), hbar(hbar), dt(dt)
{
}

const Interaction::ResultArray &PulseInteraction::evaluate(const int time_idx)
{
  const double time = time_idx * dt;

  for(size_t i = 0; i < dots->size(); ++i) {
    results[i] = Eigen::Vector3d(0, 0, 0);
    for(int n = 0; n < static_cast<int>(pulses->size()); ++n) {
      results[i] += (*pulses)[n]((*dots)[i].position(), time);
    }
  //if(time_idx==500) std::cout << "field: " << results[i].transpose() << std::endl;
  }
  return results;
}
