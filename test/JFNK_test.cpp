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

  Universe() : num_solutions(10){};

  vec3d ode(vec3d u, vec3d alpha) { 
    double gamma0 = 175920000000.0;
    double damping = 0.1;
    const double gamma = gamma0 / (1 + std::pow(damping, 2));
    return -gamma * u.cross(alpha);
    //return u.cross(alpha);
  }
    
  int factorial(int n) { return (n > 1) ? n * factorial(n - 1) : 1; }

  vec3d recursive_cross(int n, vec3d a, vec3d b)
  {
    return (n > 0) ? a.cross(recursive_cross(n - 1, a, b)) : b;
  }

  vec3d ode_solution(double t, vec3d u0, vec3d alpha)
  {
    double gamma0 = 175920000000.0;
    double damping = 0.1;
    const double gamma = gamma0 / (1 + std::pow(damping, 2));
    vec3d sol = u0;
    for(int n = 1; n < 20; ++n) {
    //vec3d sol = vec3d(0,0,0);
    //for(int n = 0; n < 25; ++n) {
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

  //const double step_size = 0.005;
  //const int num_of_steps = 30;
  //const int max_iter = 10;
  //const vec3d alpha(0.5, 1, 1.5);
  const double step_size = 1e-18;
  const int num_of_steps = 10;
  const int max_iter = 4;
  const vec3d alpha(0, 1e3, 0);
  std::vector<vec3d> history(num_of_steps);
  std::vector<vec3d> delta_vec(num_of_steps);
  std::vector<vec3d> analytical_history(num_of_steps);
  //history[0] = vec3d(1, 1, 1);
  history[0] = vec3d(1e6, 0, 0);
  analytical_history[0] = history[0];

  //std::cout << std::scientific << std::setprecision(6);
  for(int step = 1; step < num_of_steps; ++step) {
    analytical_history[step] =
        ode_solution(step * step_size, analytical_history[0], alpha);
    history[step] = history[step - 1] * 0.99;
    //history[step] = vec3d(0.90, .02, 0.99);
    delta_vec[step] = history[step] - history[step - 1];
    // guess here
    for(int iter = 0; iter < max_iter; ++iter) {
      std::cout << "Iteration =  " << iter <<"\n";
      std::cout << "ode= " << ode(history[step], alpha).transpose() << std::endl;
      std::vector<vec3d> residual_vec(1);
      residual_vec[0] = history[step-1] - history[step] + step_size * ode(history[step], alpha);
      std::cout << "Delta before = " << delta_vec[step].transpose() <<"\n";
      std::cout << "jmv = " << ode_jacobian_matvec(delta_vec[step], alpha)[0].transpose() <<"\n";
      delta_vec[step] =
          JFNK_solve(1, step_size, ode_jacobian_matvec(delta_vec[step], alpha),
                     residual_vec)[0];
      history[step] = delta_vec[step] + history[step];
      std::cout << "Residual = " << residual_vec[0].transpose() <<"\n";
      std::cout << "Delta = " << delta_vec[step].transpose() <<"\n";
      std::cout << "History = " << history[step].transpose() <<"\n\n";
    }
    std::cout << "History norm = " << history[step].norm()<<"\n\n";
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
