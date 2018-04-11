#ifndef EULER_INTEGRATOR
#define EULER_INTEGRATOR

#include <Eigen/Dense>
#include "RHS/rhs.h"
#include "history.h"

namespace Integrator {
  template <class soltype>
  class Euler;
}

class EulerIntegrator {
 public:
  EulerIntegrator(const double,
                  const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
                  std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> &);
  void solve() const;
  void solve_step(const int) const;

 private:
  const int time_idx_ubound;
  const double dt;
  const std::shared_ptr<Integrator::History<Eigen::Vector3d>> history;
  std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> rhs_functions;
};

#endif
