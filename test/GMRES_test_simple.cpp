#include <Eigen/Dense>
#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include "../src/solver/gmres.h"

BOOST_AUTO_TEST_SUITE(GMRES_solver_simple)

typedef Eigen::Vector3d vec3d;

vec3d Ax(vec3d x) {
  Eigen::Matrix3d A;
  A << 3.3, 5, 0.2, 5.5, 9, 9, 1.2, 7.8, 4.9;
  return A * x;
}

BOOST_AUTO_TEST_CASE(GMRES_solver_simple)
{
  
  Eigen::Matrix<double, 101, 101> H;
  int m = 2;
  int max_iter = 100;
  double tol = 1e-12;
  vec3d x;
  vec3d b(9.1, 23, 10.1);
  vec3d sol(2.04917113, 0.43272467, 0.87055964);

  int output = GMRES(Ax, x, b, H, m, max_iter, tol);
  //int output = GMRES(Ax, std::make_shared<Matrix>(x), b, H, m, max_iter, tol);

  double tolerance = 1e-6;
  BOOST_CHECK_MESSAGE((x-sol).norm()/sol.norm() < tolerance,
      "Analytic Solution = "
          << sol.transpose()
          << " and numerical solution = " << x.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
