#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/solver/newton.h"
#include "../src/solver/solver.h"

BOOST_AUTO_TEST_SUITE(JFNK_solver)

typedef Eigen::Vector3d vec3d;

struct Universe {
  double dt, num_solutions;

  Universe() : dt(1e-1), num_solutions(10){};

  vec3d ode(vec3d u, vec3d alpha) { return u.cross(alpha); }

  int factorial(int n) { return (n > 1) ? n * factorial(n - 1) : 1; }

  vec3d recursive_cross(int n, vec3d a, vec3d b)
  {
    return (n > 0) ? a.cross(recursive_cross(n - 1, a, b)) : b;
  }

  vec3d ode_solution(double t, vec3d u0, vec3d alpha)
  {
    vec3d sol(0, 0, 0);
    for(int n = 0; n < 25; ++n)
      sol += std::pow(t, n) / factorial(n) * recursive_cross(n, u0, alpha);
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

  const double step_size = 0.1;
  const int num_of_steps = 10;
  const int max_iter = 8;
  const vec3d alpha(0.5, 1, 1.5);
  std::vector<vec3d> history(num_of_steps);
  std::vector<vec3d> delta_vec(num_of_steps);
  std::vector<vec3d> analytical_history(num_of_steps);
  history[0] = vec3d(1, 1, 1);

  for(int step = 1; step < num_of_steps; ++step) {
    analytical_history[step] =
        ode_solution(step * step_size, history[step], alpha);
    history[step] = history[step - 1] * 1.05;
    delta_vec[step] = history[step] - history[step - 1];
    // guess here
    for(int iter = 0; iter < max_iter; ++iter) {
      std::vector<vec3d> residual_vec(1);
      residual_vec[0] = (history[step] - history[step - 1]) / step_size -
                        ode(history[step], alpha);
      delta_vec[step] =
          JFNK_solve(1, step_size, ode_jacobian_matvec(delta_vec[step], alpha),
                     residual_vec)[0];
      history[step] = delta_vec[step] + history[step];
    }
  }
  for(int step = 0; step < num_of_steps; ++step) {
    BOOST_CHECK_MESSAGE(
        (analytical_history[step] - history[step]).norm() < tolerance,
        "Analytic Solution = "
            << analytical_history[step].transpose()
            << " and numerical solution = " << history[step].transpose()
            << " solution do match at step = " << step << "\n");
  }
}
BOOST_AUTO_TEST_SUITE_END()
