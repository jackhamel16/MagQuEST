#ifndef JFNK_SOLVER
#define JFNK_SOLVER

#include "gmres.h"
#include "solver.h"

// Delta refers to difference between M at iteration l and l+1

typedef Eigen::Vector3d vec3d;

class JFNKSolver : public Solver {
 public:
  JFNKSolver(const double,
             int,
             const std::shared_ptr<Integrator::History<vec3d>> &,
             const std::shared_ptr<Integrator::History<vec3d>> &,
             std::vector<std::shared_ptr<Interaction>>,
             std::vector<std::shared_ptr<Interaction>>,
             rhs_func_vector &,
             jacobian_matvec_func_vector &);
  virtual void solve_step(int);

 private:
  int max_iter;
  const std::shared_ptr<Integrator::History<vec3d>> delta_history;
  const std::vector<std::shared_ptr<Interaction>> delta_interactions;
  jacobian_matvec_func_vector matvec_funcs;
  vec3d residual_rhs(vec3d, vec3d, vec3d, int, double, rhs_func_vector &);
};

#endif
