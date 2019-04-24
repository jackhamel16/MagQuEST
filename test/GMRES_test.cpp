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
  vec3d sol(0, 0, 0);
  for(int n = 0; n < 20; ++n) {
    sol += std::pow(t, n) / factorial(n) * recursive_cross(n, alpha, u0);
  }
  return sol;
}

vec3d solver_matvec(vec3d v, vec3d alpha, double dt)
{
  return v - dt*v.cross(alpha);
}

BOOST_AUTO_TEST_CASE(GMRES_solver)
{
  
  Eigen::Matrix<double, 3, 3> H;
  int m = 1;
  int max_iters_gmres = 10000;
  int max_iters = 10;
  double tol = 1e-8;
  double dt = 0.0001;
  double t = dt;
  vec3d u0(1.0, 1.0, 1.0);
  vec3d alpha(1.5, 1.0, 0.5);
  vec3d v = 0.01 * u0;;
  vec3d sol = ode_solution(t, u0, alpha);
  vec3d b, b_apprx;
  vec3d u = u0 + v;

  auto matvec = std::bind(solver_matvec, std::placeholders::_1, alpha, dt);
 
  for(int iter=0; iter<max_iters; ++iter) {
    b =  u0 - u + dt * ode(u, alpha);
    std::cout << "exact b = " << b.transpose() << std::endl;
    int output = GMRES(matvec, v, b, H, m, max_iters_gmres, tol);
    b_apprx =  matvec(v);
    std::cout << "apprx b = " << b_apprx.transpose() << std::endl;
    u += v;
    std::cout << "iter = " << iter << " norm(v) = " << v.norm() << std::endl;
    if(output==1) {
      std::cout << "GMRES Could not converge to a solution\n";
      break;
    }
  }

  std::cout << "sol = " << sol.transpose() << std::endl;
  vec3d b_sol = u0 - sol + dt*ode(sol, alpha);
  std::cout << "  u = " << u.transpose() << std::endl;
  std::cout << "error = " << (u-sol).norm()/sol.norm() << std::endl;
  //double tol2 = 1e-5;
  //for(int iter=0; iter<max_iters; ++iter) {
    //rhs = u0 - u + dt * ode(u, alpha);
    //int output = GMRES(matvec, v, rhs, H, m, max_iters_gmres, tol);
    //if(output==1) std::cout << "GMRES Could not converge to a solution\n";
    //if((v.norm() / u0.norm()) > tol2) {
      //u = u + v;
      //std::cout << "new u = " << u.transpose() << std::endl;
      //rhs = u0 - u + dt * ode(u, alpha);
    //}
    //else {
      //u = u + v;
      //std::cout << "new u = " << u.transpose() << std::endl;
      //break;
    //}
  //}

  //double tolerance = 1e-6;
  //BOOST_CHECK_MESSAGE((u-sol).norm()/sol.norm() < tolerance,
      //"Analytic Solution = "
          //<< sol.transpose()
          //<< " and numerical solution = " << u.transpose() << "\n");
  //double tolerance = 1e-6;
  //BOOST_CHECK_MESSAGE((u-sol).norm()/sol.norm() < tolerance,
      //"Analytic Solution = "
          //<< sol.transpose()
          //<< " and numerical solution = " << u.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
