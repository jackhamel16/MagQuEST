#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER

#include "solver.h"

class NewtonSolver : public Solver {
 public:
  NewtonSolver(const double,
               const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
               const std::vector<std::shared_ptr<Interaction>>,
               rhs_func_vector &);
  virtual void solve(int);
};

#endif
