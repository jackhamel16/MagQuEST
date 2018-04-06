#ifndef EULER_INTEGRATOR
#define EULER_INTEGRATOR

#include <Eigen/Dense>
#include "RHS/rhs.h"
#include "history.h"

class EulerIntegrator;

class EulerIntegrator {
 public:
  EulerIntegrator(const double,
                  const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
                  std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> &);
  void solve(const int) const;

 private:
  const double dt;
  const std::shared_ptr<Integrator::History<Eigen::Vector3d>> history;
  std::unique_ptr<Integrator::RHS<Eigen::Vector3d>> rhs_functions;
};

#endif
