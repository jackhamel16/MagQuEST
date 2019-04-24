#define EIGEN_STACK_ALLOCATION_LIMIT 0

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

Eigen::VectorXd triDiagMatvec(Eigen::VectorXd x, int N) {
  Eigen::MatrixXd A;
  A.resize(N, N);
  A.setZero();
  for(int r=0; r<N; ++r) {
    if(r>0) A(r,r-1) = -2;
    if(r<N-1) A(r,r+1) = -2;
    A(r,r) = 1;
  }
  return A * x;
}

BOOST_AUTO_TEST_CASE(GMRES_solver_simple)
{
  int N = 10;
  Eigen::MatrixXd H;
  int m = 4;
  int max_iter = 10000;
  double tol = 1e-12;
  Eigen::VectorXd x, b, sol;
  sol.resize(N);
  x.resize(N);
  H.resize(N, N);
  for(int i=0; i<N; ++i) {
    sol(i) = 1;
  }
  b = triDiagMatvec(sol, N);
  
  std::function<Eigen::VectorXd(Eigen::VectorXd)> Ax = std::bind(triDiagMatvec,
      std::placeholders::_1, N);
  
  int output = GMRES(Ax, x, b, H, m, max_iter, tol);
  if(output==1) std::cout << "GMRES UNABLE TO CONVERGE\n"; 

  std::cout << "sol = " << sol.transpose() << std::endl;
  std::cout << "  x = " << x.transpose() << std::endl;
  std::cout << "error = " << (sol-x).norm() / sol.norm() << std::endl;
  double tolerance = 1e-4;
  BOOST_CHECK_MESSAGE((x-sol).norm()/sol.norm() < tolerance,
      "\nSol = "
          << sol.transpose()
          << "\nNum = " << x.transpose() << "\n");
}
BOOST_AUTO_TEST_SUITE_END()
