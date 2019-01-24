#ifndef SOLVER
#define SOLVER

#include <functional>
#include <Eigen/Dense>
#include "../integrator/history.h"
#include "../interactions/interaction.h"

class Solver {
  public:
    Solver(const double, const std::shared_ptr<Integrator::History<Eigen::Vector3d>> &,
        const std::vector<std::shared_ptr<Interaction>>, rhs_func_vector &);
    void solve();
    virtual void solve_step(int) = 0;

  protected:
    const double dt;
    std::shared_ptr<Integrator::History<Eigen::Vector3d>> history;
    std::vector<std::shared_ptr<Interaction>> interactions;
    rhs_func_vector rhs_functions;
    const int time_idx_ubound;
    const int num_solutions;
};

#endif
