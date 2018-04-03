#include <math.h>
#include <Eigen/Dense>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "gmres.h"

class foo {
 public:
  Eigen::MatrixXd mat;

  Eigen::Matrix<double, Eigen::Dynamic, 1> evaluate_now(
      Eigen::Matrix<double, Eigen::Dynamic, 1> &vec)
  {
    return mat * vec;
  };

  void build_mat(int N)
  {
    std::complex<double> j(0, 1);
    mat.resize(N, N);
    double Nd = static_cast<double>(N);
    for(double r1 = 0; r1 < N; ++r1) {
      for(double r2 = 0; r2 < N; ++r2) {
        mat(r1, r2) = 1 / std::abs(std::exp(j * 2.0 * M_PI * r1 / Nd) -
                                   std::exp(j * 2.0 * M_PI * r2 / Nd));
        if(r1 == r2) mat(r1, r2) = 1;
      }
    }
  }
};

int main()
{
  std::cout << "GMRES TESTING A.x=b" << std::endl;

  for(int size = 10; size <= 100; size += 10) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> vec =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(size, 1);

    foo bar;
    auto ptr = &bar;
    ptr->build_mat(size);

    auto vec2 = ptr->evaluate_now(vec);

    int m = 100;
    int max_iter = 6000;
    double tol = 1e-13;
    Eigen::Matrix<double, 101, 101> H;
    Eigen::Matrix<double, Eigen::Dynamic, 1> x(size);
    for(int i = 0; i < size; ++i) x[i] = 0;

    auto output = GMRES::GMRES(ptr, x, vec2, H, m, max_iter, tol);

    double error = 0;
    for(int i = 0; i < size; ++i) {
      error += (x[i] - vec[i]) / (size * vec[i]);
    }
    std::cout << "System size: " << size << std::endl;
    std::cout << "Error: " << error << std::endl << std::endl;;
  }
  return 0;
}
