#ifndef SOLVER
#define SOLVER

#include <Eigen/Dense>
#include "../integrator/history.h"
#include "../interactions/interaction.h"

class Solver {
  public:
    Solver(const double, const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
        const std::vector<std::shared_ptr<Interaction>>, rhs_func_vector &);
    virtual void solve() = 0;

  private:
    const double dt;
    std::shared_ptr<Integrator::History<Eigen::Vector3d>> history;
    std::vector<std::shared_ptr<Interaction>> interactions;
    rhs_func_vector rhs_functions;
    const int time_idx_ubound;
    const int num_solutions;
};

#endif
