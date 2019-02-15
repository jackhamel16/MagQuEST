#include <Eigen/Dense>
#include <complex>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "configuration.h"
#include "integrator/RHS/llg_rhs.h"
#include "integrator/euler.h"
#include "integrator/history.h"
#include "integrator/integrator.h"
#include "interactions/green_function.h"
#include "interactions/history_interaction.h"
#include "interactions/pulse_interaction.h"
#include "interactions/self_interaction.h"
#include "solver/newton.h"
#include "solver/solver.h"

using namespace std;

int main(int argc, char *argv[])
{
  try {
    cout << setprecision(12) << scientific;
    auto vm = parse_configs(argc, argv);

    cout << "Initializing..." << endl;

    auto qds = make_shared<DotVector>(import_dots(config.qd_path));
    qds->resize(config.num_particles);

    // Set up Pulses
    auto pulses = make_shared<PulseVector>(import_pulses(config.pulse_path));
    double dc_field = 0;
    for(int p = 0; p < static_cast<int>(pulses->size()); ++p) {
      // computes sigma and td of the pulse
      (*pulses)[p].compute_parameters(config.c0);
      dc_field += (*pulses)[p].get_dc_field();
    }
    const double dt = (*pulses)[0].compute_dt();
    const double num_timesteps =
        static_cast<int>(std::ceil(config.total_time / dt));

    // Set up History
    const double chi = 1;
    auto history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 22, num_timesteps);
    history->fill(chi * dc_field * soltype(1, 1, 1));
    // history->initialize_past(qds);

    // Set up Interactions
    auto dyadic =
        make_shared<Propagation::FixedFramePropagator>(config.c0, config.e0);

    std::vector<std::shared_ptr<Interaction>> interactions{
        make_shared<PulseInteraction>(qds, pulses, config.hbar, dt),
        make_shared<HistoryInteraction>(
            qds, history, dyadic, config.interpolation_order, dt, config.c0),
        make_shared<SelfInteraction>(qds, history)};

    std::cout << "dt: " << dt << std::endl;
    std::cout << separation((*qds)[0], (*qds)[1]).norm() / dt << std::endl;
    for(int step = 0; step < num_timesteps; ++step) {
      auto pulse_interactions = interactions[0]->evaluate(step);
      auto history_interactions = interactions[1]->evaluate(step);
      auto self_interactions = interactions[2]->evaluate(step);

      history->array[0][step][0] = history_interactions[0];
      history->array[1][step][0] = history_interactions[1];
      //history->array[1][step][0] =
          //chi * (pulse_interactions[1] +
                 //history_interactions[1] +
                 //self_interactions[1]);
      //history->array[0][step][0] = history_interactions[0];
     
      //for(int particle_idx = 0; particle_idx < config.num_particles;
          //++particle_idx) {
        //history->array[particle_idx][step][0] =
            //chi * (pulse_interactions[particle_idx] +
                   //history_interactions[particle_idx] +
                   //self_interactions[particle_idx]);
      //}
    }

    cout << "Writing output..." << endl;
    ofstream outfile("output.dat");
    ofstream pulsefile("pulseout.dat");
    outfile << scientific << setprecision(15);
    pulsefile << scientific << setprecision(15);
    for(int t = 0; t < num_timesteps; ++t) {
      for(int n = 0; n < config.num_particles; ++n) {
        outfile << history->array[n][t][0].transpose() << " ";
        pulsefile << history->array[n][t][1].transpose() << " ";
        // pulsefile << (*pulses)[0](Eigen::Vector3d(0, 0, 0), t *
        // dt).transpose()
        //<< " ";
      }
      outfile << "\n";
      pulsefile << "\n";
    }
  } catch(CommandLineException &e) {
    // User most likely queried for help or version info, so we can silently
    // move on
  }

  return 0;
}
