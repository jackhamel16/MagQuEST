#include <Eigen/Dense>
#include <complex>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "configuration.h"
#include "integrator/RHS/llg_rhs.h"
#include "integrator/history.h"
#include "integrator/integrator.h"
#include "interactions/green_function.h"
#include "interactions/history_interaction.h"
#include "interactions/pulse_interaction.h"
#include "interactions/self_interaction.h"

using namespace std;

int main(int argc, char *argv[])
{
  try {
    cout << setprecision(12) << scientific;
    auto vm = parse_configs(argc, argv);

    cout << "Initializing..." << endl;

    auto qds = make_shared<DotVector>(import_dots(config.qd_path));
    qds->resize(config.num_particles);

    // Set up History
    auto history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 22, config.num_timesteps);
    history->fill(soltype(0, 0, 0));

    // Set up Interactions
    auto pulses = make_shared<PulseVector>(import_pulses(config.pulse_path));
    for(int p = 0; p < static_cast<int>(pulses->size()); ++p) {
      // computes sigma and td of the pulse
      (*pulses)[p].compute_parameters(config.c0);
    }
    auto dyadic =
        make_shared<Propagation::FixedFramePropagator>(config.c0, config.e0);
    const double dt = (*pulses)[0].compute_dt();

    std::vector<std::shared_ptr<Interaction>> interactions{
        make_shared<PulseInteraction>(qds, pulses, config.hbar, dt),
        make_shared<HistoryInteraction>(
            qds, history, dyadic, config.interpolation_order, dt, config.c0)};

    // Set up RHS functions
    auto rhs_funcs = rhs_functions(*qds);

    // Set up Bloch RHS
    std::unique_ptr<Integrator::RHS<soltype>> llg_rhs =
        std::make_unique<Integrator::LLG_RHS>(
            config.dt, history, std::move(interactions), std::move(rhs_funcs));

    Integrator::PredictorCorrector<soltype> solver(dt, 18, 22, 3.15, history,
                                                   llg_rhs);

    cout << "Solving..." << endl;
    solver.solve();

    cout << "Writing output..." << endl;
    ofstream outfile("output.dat");
    ofstream pulsefile("pulseout.dat");
    outfile << scientific << setprecision(15);
    for(int t = 0; t < config.num_timesteps; ++t) {
      for(int n = 0; n < config.num_particles; ++n) {
        outfile << history->array[n][t][0].transpose() << " ";
        pulsefile << history->array[n][t][1].transpose() << " ";
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
