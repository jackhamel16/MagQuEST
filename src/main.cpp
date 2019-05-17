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
    // const double dt = (*pulses)[0].compute_dt();
    //const double dt = 1e-13;
    const double dt = config.dt;
    const double num_timesteps =
        static_cast<int>(std::ceil(config.total_time / dt));
    std::cout << dt << std::endl;
    std::cout << num_timesteps << std::endl;
    // Set up History
    const auto history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 22, num_timesteps);
    const auto delta_history = std::make_shared<Integrator::History<soltype>>(
        config.num_particles, 0, num_timesteps);
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

    rhs_func_vector rhs_funcs = rhs_functions(*qds);
    // try passing particles to solver
    jacobian_matvec_func_vector jacobian_matvec_funcs =
        make_jacobian_matvec_funcs(*qds);

    int newton_iterations = 4;  // Figure out best way to set these
    auto solver = NewtonSolver(dt, newton_iterations, history, delta_history,
                        std::move(interactions), std::move(delta_interactions), rhs_funcs,
                        jacobian_matvec_funcs);

    solver.solve();

    //TEST AREA FOR JACOBIAN
    //vec3d test_v(0.002,-0.005,0.001);
    //vec3d test_h(1.0, 3.0, 0);
    //vec3d test_m(4.2, 0, 0);
    //double test_dt = 0.01;
    //double gamma = 2.21e5;
    //vec3d implicit_result = (*qds)[0].llg_jacobian_matvec(dt,test_m,test_v,test_h,vec3d(0,0,0));
    //vec3d explicit_result = test_v - test_dt * compute_llg_jacobian(gamma, test_m, test_h) * test_v;
    //std::cout << "Implicit = " << implicit_result.transpose() <<
      //"\nExplicit = " << explicit_result.transpose() << std::endl;


    cout << "Writing output..." << endl;
    ofstream outfile("output.dat");
    ofstream pulsefile("pulseout.dat");
    outfile << scientific << setprecision(15);
    pulsefile << scientific << setprecision(15);
    for(int t = 0; t < num_timesteps; ++t) {
    //for(int t = 0; t < num_timesteps; t=t+50) {
      for(int n = 0; n < config.num_particles; ++n) {
        //std::cout << history->array[n][t][0].norm() << " ";
        outfile << history->array[n][t][0].transpose() << " ";
        pulsefile << (*pulses)[0](soltype(0, 0, 0), dt * t).transpose() << " ";
        // pulsefile << (*pulses)[0](Eigen::Vector3d(0, 0, 0), t *
        // dt).transpose()
        //<< " ";
      }
      //std::cout <<"\n";
      outfile << "\n";
      pulsefile << "\n";
    }
  } catch(CommandLineException &e) {
    // User most likely queried for help or version info, so we can silently
    // move on
  }

  return 0;
}
