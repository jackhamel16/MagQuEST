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

BOOST_FIXTURE_TEST_CASE(GMRES_solver, Universe)
{
  
  Eigen::Matrix<double, 101, 101> H;
  int m = 100;
  int max_iter = 20;
  double tol = 1e-8;
  Eigen::Matrix3d A;
  vec3d x;
  A << 3.1, 4, 4, 6.6, 0.1, 0, 9, 9.2, 1.8;
  vec3d b(4.4, 10.1, 44);
  vec3d sol(1.46711176,  4.17062365, -4.20763527);
  
  int output = GMRES(A, x, b, H, m, max_iter, tol);
  
  double tolerance = 1e-12;
  BOOST_CHECK_MESSAGE((x-sol).norm()/sol.norm() < tolerance,
      "Analytic Solution = "
          << sol.transpose()
          << " and numerical solution = " << x.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
