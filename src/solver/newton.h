#ifndef NEWTON_SOLVER
#define NEWTON_SOLVER

#include "solver.h"

typedef Eigen::Vector3d vec3d;

class NewtonSolver : public Solver {
 public:
  NewtonSolver(const double,
               const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
               const std::vector<std::shared_ptr<Interaction>>,
               rhs_func_vector &);
  virtual void solve_step(int);
  Eigen::Matrix3d approx_jacob(
      std::function<vec3d(vec3d, vec3d)>, vec3d, vec3d, vec3d, double);
 //private:
  //vec3d newton_rhs(vec3d, vec3d, vec3d, int, int);
};

#endif
