#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/integrator/history.h"
#include "../src/interactions/history_interaction.h"
#include "../src/interactions/pulse_interaction.h"
#include "../src/interactions/self_interaction.h"
#include "../src/pulse.h"
#include "../src/solver/jfnk.h"

// PREFACE: This is a terrible example of a unit test.  The JFNK solver relies
// on so many other modules and they are required to test it, unfortunately.

BOOST_AUTO_TEST_SUITE(JFNK_solver)

typedef Eigen::Vector3d vec3d;

struct Universe {
  double dt, num_solutions;

  Universe() : num_solutions(10){};

  vec3d ode(vec3d u, vec3d alpha)
  {
    double gamma0 = 221000.0;
    double damping = 0.1;
    const double gamma = gamma0 / (1 + std::pow(damping, 2));
    return -gamma * u.cross(alpha);
  }

  int factorial(int n) { return (n > 1) ? n * factorial(n - 1) : 1; }

  vec3d recursive_cross(int n, vec3d a, vec3d b)
  {
    return (n > 0) ? a.cross(recursive_cross(n - 1, a, b)) : b;
  }

  vec3d ode_solution(double t, vec3d u0, vec3d alpha)
  {
    double gamma0 = 221000.0;
    double damping = 0.1;
    const double gamma = gamma0 / (1 + std::pow(damping, 2));
    vec3d sol = u0;
    for(int n = 1; n < 20; ++n) {
      sol +=
          std::pow(t * gamma, n) / factorial(n) * recursive_cross(n, alpha, u0);
    }
    return sol;
  }

  std::vector<vec3d> ode_jacobian_matvec(vec3d v, vec3d alpha)
  {
    std::vector<vec3d> vectors(1);
    vectors[0] = v.cross(alpha);
    return vectors;
  }
};

BOOST_FIXTURE_TEST_CASE(JFNK_solver, Universe)
{
  const double tolerance = 1e-3;

  const double step_size = 1e-18;
  const int num_of_steps = 10;
  const int max_iter = 4;
  const auto history =
      std::make_shared<Integrator::History<vec3d>>(1, 22, num_of_steps);
  const auto delta_history =
      std::make_shared<Integrator::History<vec3d>>(1, 22, num_of_steps);
  std::vector<vec3d> analytical_history(num_of_steps);

  // Setup a particle
  const double Ms = 1e6;
  const vec3d M(Ms, 0, 0);
  history->fill(M);
  delta_history->fill(vec3d(1, 0, 0));
  MagneticParticle mp(vec3d(0, 0, 0), 0.1, 221000, Ms, M);
  DotVector mp_vec(1);
  mp_vec[0] = mp;
  auto mp_ptr = std::make_shared<DotVector>(mp_vec);
  // Create Pulse
  const double dc_mag = 1e3;
  const vec3d dc_orientation(0, 1, 0);
  const vec3d alpha = dc_mag * dc_orientation;
  Pulse dc_pulse(0, 1, 1, dc_mag, vec3d(1, 0, 0), dc_orientation);
  PulseVector pulse_vec(1);
  pulse_vec[0] = dc_pulse;

  auto dyadic =
      std::make_shared<Propagation::FixedFramePropagator>(1, 1);
  auto dyadic2 =
      std::make_shared<Propagation::FixedFramePropagator>(1, 1);
  // Set up interactions
  // Unfortunately, JFNK_solver class requires all interactions to be present
  std::vector<std::shared_ptr<Interaction>> interactions{
      std::make_shared<PulseInteraction>(
          mp_ptr, std::make_shared<PulseVector>(pulse_vec), 1, step_size),
      std::make_shared<HistoryInteraction>(mp_ptr, history, dyadic, 3, dt,
                                           1),
      std::make_shared<SelfInteraction>(mp_ptr, history)};
  // delta history
  std::vector<std::shared_ptr<Interaction>> delta_interactions{
      std::make_shared<HistoryInteraction>(mp_ptr, delta_history, dyadic2, 3,
                                           dt, 1),
      std::make_shared<SelfInteraction>(mp_ptr, delta_history)};
  // Set up RHS func vector and matvec_funcs
  rhs_func_vector rhs_vec = rhs_functions(*mp_ptr);
  jacobian_matvec_func_vector matvec_funcs =
      make_jacobian_matvec_funcs(*mp_ptr);
  // I think having self interactions will make the test wrong. try it before
  // changing but a possible solution is to double up history interactions
  // (since they do nothing)
  JFNKSolver solver(dt, max_iter, history, delta_history, interactions,
                    delta_interactions, rhs_vec, matvec_funcs);

  analytical_history[0] = M;

   std::cout << std::scientific << std::setprecision(6);
  for(int step = 1; step < num_of_steps; ++step) {
    analytical_history[step] =
        ode_solution(step * step_size, analytical_history[0], alpha);
  }

  for(int step = 0; step < num_of_steps; ++step) {
    BOOST_CHECK_MESSAGE(
        analytical_history[step].norm() < tolerance, "placeholder\n");
  }
}
BOOST_AUTO_TEST_SUITE_END()
