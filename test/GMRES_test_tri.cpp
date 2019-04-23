#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <random>
#include "../src/solver/gmres.h"

BOOST_AUTO_TEST_SUITE(GMRES_solver_simple)

typedef Eigen::Vector3d vec3d;

Eigen::Matrix<double, 20, 1> Ax(Eigen::Matrix<double, 20, 1> x) {
  Eigen::Matrix<double, 20, 20>  A;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for(int i=0; i<20; ++i) {
    for(int j=i-1; j<i+2; ++j) {
      if(j<0 || j>=20) continue;
      A(i,j) = distribution(generator);
    }
  }
  std::cout << A << std::endl;
  return A * x;
}

BOOST_AUTO_TEST_CASE(GMRES_solver_simple)
{
  
  Eigen::Matrix<double, 101, 101> H;
  int m = 18;
  int max_iter = 100;
  double tol = 1e-12;
  Eigen::Matrix<double, 20, 1> x, b, sol;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for(int i=0; i<20; ++i) 
    sol(i) = distribution(generator);
  b = Ax(sol);
  int output = GMRES(Ax, x, b, H, m, max_iter, tol);
  //int output = GMRES(Ax, std::make_shared<Matrix>(x), b, H, m, max_iter, tol);

  double tolerance = 1e-6;
  BOOST_CHECK_MESSAGE((x-sol).norm()/sol.norm() < tolerance,
      "Analytic Solution = "
          << sol.transpose()
          << " and numerical solution = " << x.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
