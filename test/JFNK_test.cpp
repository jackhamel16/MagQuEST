#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/pulse.h"
#include "../src/solver/solver.h"
#include "../src/interactions/pulse_interaction.h"

// PREFACE: This is a terrible example of a unit test.  The JFNK solver relies on so many other
// modules and they are required to test it, unfortunately.

BOOST_AUTO_TEST_SUITE(JFNK_solver)

typedef Eigen::Vector3d vec3d;

struct Universe {
  double dt, num_solutions;

  Universe() : num_solutions(10){};

  vec3d ode(vec3d u, vec3d alpha) { 
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
      sol += std::pow(t * gamma, n) / factorial(n) * recursive_cross(n, alpha, u0);
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
  std::vector<vec3d> history(num_of_steps);
  std::vector<vec3d> delta_vec(num_of_steps);
  std::vector<vec3d> analytical_history(num_of_steps);

  // Setup a particle
  const double Ms = 1e6;
  const vec3d M(Ms, 0, 0);
  MagneticParticle mp(vec3d(0,0,0), 0.1, 221000, Ms, M);
  DotVector mp_vec(1);
  mp_vec[0] = mp;
  // Create Pulse
  const double dc_mag = 1e3;
  const vec3d dc_orientation(0, 1, 0);
  const vec3d alpha = dc_mag * dc_orientation;
  Pulse dc_pulse(0, 1, 1, dc_mag, vec3d(1, 0, 0), dc_orientation);
  PulseVector pulse_vec(1);
  pulse_vec[0] = dc_pulse;
  //Set up interactions
  PulseInteraction pulse_interaction(std::make_shared<DotVector>(mp_vec),
                                     std::make_shared<PulseVector>(pulse_vec),
                                     1, step_size);

  history[0] = vec3d(Ms, 0, 0);
  analytical_history[0] = history[0];
   

  //std::cout << std::scientific << std::setprecision(6);
  for(int step = 1; step < num_of_steps; ++step) {
    analytical_history[step] =
        ode_solution(step * step_size, analytical_history[0], alpha);
  }


  for(int step = 0; step < num_of_steps; ++step) {
    BOOST_CHECK_MESSAGE(
        (analytical_history[step] - history[step]).norm() / analytical_history[step].norm() < tolerance,
        "Analytic Solution = "
            << analytical_history[step].transpose()
            << " and numerical solution = " << history[step].transpose()
            << " solution do match at step = " << step << "\n");
  }
  //std::string analytic_file = "~/Desktop/Research/MagQuEST/build/analytic.dat";
  //std::string numeric_file = "~/Desktop/Research/MagQuEST/build/numeric.dat";
  std::string analytic_file = "analytic.dat";
  std::string numeric_file = "numeric.dat";
  std::ofstream aout(analytic_file), nout(numeric_file);
  //aout.open(analytic_file);
  //nout.open(numeric_file);
  for(int step=0; step<num_of_steps; ++step) {
    aout << analytical_history[step].transpose() << "\n";
    nout << history[step].transpose() << "\n";
  }

}
BOOST_AUTO_TEST_SUITE_END()
