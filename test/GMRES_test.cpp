#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/solver/gmres.h"

BOOST_AUTO_TEST_SUITE(GMRES_solver)

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> dynamicVec;
typedef Eigen::Vector3d vec3d;

int factorial(int n) 
{ 
  return (n > 1) ? n * factorial(n - 1) : 1; 
}

vec3d recursive_cross(int n, vec3d a, vec3d b)
{
  return (n > 0) ? a.cross(recursive_cross(n - 1, a, b)) : b;
}

vec3d ode(vec3d u, vec3d alpha) {
  return u.cross(alpha);
}

vec3d ode_solution(double t, vec3d u0, vec3d alpha)
{
  vec3d sol = u0;
  for(int n = 1; n < 20; ++n) {
    sol += std::pow(t, n) / factorial(n) * recursive_cross(n, alpha, u0);
  }
  return sol;
}

vec3d ode_jacobian_matvec(vec3d v, vec3d alpha)
{
  return v.cross(alpha);
}

BOOST_AUTO_TEST_CASE(GMRES_solver)
{
  
  Eigen::Matrix<double, 101, 101> H;
  int m = 2;
  int max_iter = 100;
  double tol = 1e-2;
  double dt = 0.01;
  vec3d u0(1.0, 1.0, 1.0);
  vec3d alpha(1.5, 1.0, 0.5);
  vec3d v = 0.01 * u0;;
  vec3d sol = ode_solution(dt, u0, alpha);

  vec3d u = u0 + v;
  vec3d rhs = u0 - u + dt * ode(u, alpha);

  //const std::function<vec3d(vec3d)> matvec = std::bind(ode_jacobian_matvec, std::placeholders::_1, alpha);
  auto matvec = std::bind(ode_jacobian_matvec, std::placeholders::_1, alpha);
  std::cout << "test " << matvec(vec3d(1,2,3)).transpose() << std::endl; 
  int output = GMRES(matvec, v, rhs, H, m, max_iter, tol);
  if(output==1) std::cout << "GMRES Could not converge to a solution\n";
  u = u + v;

  std::cout << (v - dt*matvec(v)).transpose() << std::endl;
  double tolerance = 1e-6;
  BOOST_CHECK_MESSAGE((u-sol).norm()/sol.norm() < tolerance,
      "Analytic Solution = "
          << sol.transpose()
          << " and numerical solution = " << u.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
