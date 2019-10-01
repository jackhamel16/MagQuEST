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
#include "solver/jfnk.h"
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
    // line below sets dt based on the incident pulse. Commented out for testing.
    // const double dt = (*pulses)[0].compute_dt();
    const double dt = config.dt;  // useful for testing
    const double num_timesteps =
        static_cast<int>(std::ceil(config.total_time / dt));
    std::cout << dt << std::endl;
    std::cout << num_timesteps << std::endl;
    // Set up History
    const auto history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 22, num_timesteps);
    const auto delta_history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 0, num_timesteps);
    // Setting M for all particles. only use this for testing.
    history->fill(soltype(1.7e6, 0, 0));
    delta_history->fill(soltype(1, 0, 0));

    // Set up Interactions
    auto dyadic =
        make_shared<Propagation::FixedFramePropagator>(config.c0, config.e0);
    auto dyadic2 =
        make_shared<Propagation::FixedFramePropagator>(config.c0, config.e0);

    std::vector<std::shared_ptr<Interaction>> interactions{
        make_shared<PulseInteraction>(qds, pulses, config.hbar, dt),
        make_shared<HistoryInteraction>(
            qds, history, dyadic, config.interpolation_order, dt, config.c0),
        make_shared<SelfInteraction>(qds, history)};
    std::vector<std::shared_ptr<Interaction>> delta_interactions{
        make_shared<HistoryInteraction>(qds, delta_history, dyadic2,
                                        config.interpolation_order, dt,
                                        config.c0),
        make_shared<SelfInteraction>(qds, delta_history)};

    // Compute llg ode solutions prior to moving interactions
    std::vector<soltype> solutions(num_timesteps);
    for(int step=0; step<num_timesteps; ++step) {
      Eigen::Vector3d field = dc_field*Eigen::Vector3d(0,1,0);
      solutions[step] = (*qds)[0].ode_solution(step*dt, history->array[0][0][0], field);
    }

    rhs_func_vector rhs_funcs = rhs_functions(*qds);
    jacobian_matvec_func_vector jacobian_matvec_funcs =
        make_jacobian_matvec_funcs(*qds);

    int jfnk_iterations = 4;  // Figure out best way to set these
    auto solver = JFNKSolver(dt, jfnk_iterations, history, delta_history,
                        std::move(interactions), std::move(delta_interactions), rhs_funcs,
                        jacobian_matvec_funcs);

    solver.solve();

    cout << "Writing output..." << endl;
    ofstream outfile("output.dat");
    ofstream errorfile("errorout.dat");
    outfile << scientific << setprecision(15);
    errorfile << scientific << setprecision(15);
    for(int t = 0; t < num_timesteps; ++t) {
    errorfile << (solutions[t] - history->array[0][t][0]).norm() / solutions[t].norm() << "\n";
      for(int n = 0; n < config.num_particles; ++n) {
        outfile << history->array[n][t][0].transpose() << " ";
      }
      outfile << "\n";
    }
  } catch(CommandLineException &e) {
    // User most likely queried for help or version info, so we can silently
    // move on
  }

  return 0;
}
