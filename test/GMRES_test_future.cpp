#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/solver/gmres.h"

BOOST_AUTO_TEST_SUITE(GMRES_solver)

typedef Eigen::Vector3d vec3d;

struct Universe {
  double dt, num_solutions;

  Universe() : num_solutions(10){};

  vec3d ode(vec3d u, vec3d alpha) { 
    //double gamma0 = 175920000000.0;
    //double damping = 0.1;
    //const double gamma = gamma0 / (1 + std::pow(damping, 2));
    //return -gamma * u.cross(alpha);
    return u.cross(alpha);
  }
    
  int factorial(int n) { return (n > 1) ? n * factorial(n - 1) : 1; }

  vec3d recursive_cross(int n, vec3d a, vec3d b)
  {
    return (n > 0) ? a.cross(recursive_cross(n - 1, a, b)) : b;
  }

  vec3d ode_solution(double t, vec3d u0, vec3d alpha)
  {
    vec3d sol = u0;
    for(int n = 1; n < 20; ++n) {
    //vec3d sol = vec3d(0,0,0);
    //for(int n = 0; n < 25; ++n) {
      sol += std::pow(t, n) / factorial(n) * recursive_cross(n, alpha, u0);
    }
    return sol;
  }

  //std::vector<vec3d> ode_jacobian_matvec(vec3d v, vec3d alpha)
  //{
    //std::vector<vec3d> vectors(1);
    //vectors[0] = v.cross(alpha);
    //return vectors;
  //}
};
vec3d ode_jacobian_matvec(vec3d v, vec3d alpha)
{
  return v.cross(alpha);
}

BOOST_FIXTURE_TEST_CASE(GMRES_solver, Universe)
{
  
  Eigen::Matrix<double, 101, 101> H;
  int m = 30;
  int max_iter = 20;
  double dt = 0.05;
  double tol = 1e-8;
  
  vec3d u0(1, 1, 1);
  vec3d v = u0 * 0.05;
  vec3d u = u0 + v;
  vec3d alpha(1.5, 1, 0.5);
  vec3d rhs = ode(u, alpha);
  vec3d b = u0 - u + dt * rhs;
  std::function<vec3d(vec3d)> matvec_func = std::bind(ode_jacobian_matvec, std::placeholders::_1, alpha);
  //auto matvec_func = std::bind(ode_jacobian_matvec, std::placeholders::_1, alpha);

  int output = GMRES(matvec_func, v, b, H, m, max_iter, tol);
  vec3d sol = ode_solution(dt, u0, alpha);
  u = u + v;

  double tolerance = 1e-6;
  BOOST_CHECK_MESSAGE((u-sol).norm()/sol.norm() < tolerance,
      "Analytic Solution = "
          << sol.transpose()
          << " and numerical solution = " << u.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
