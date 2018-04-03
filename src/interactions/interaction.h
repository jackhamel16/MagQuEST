#ifndef INTERACTION_H
#define INTERACTION_H

#include <Eigen/Dense>
#include <memory>

#include "../magnetic_particle.h"

class Interaction {
 public:
  typedef std::vector<Eigen::Vector3d> ResultArray;

  Interaction(const std::shared_ptr<const DotVector> &dots)
      : dots(dots), results(dots->size()), results_now(3 * (dots->size())){};
  const Eigen::Vector3d &operator[](const int i) const { return results[i]; }
  virtual const ResultArray &evaluate(const int) = 0;
  virtual const Eigen::Matrix<double, Eigen::Dynamic, 1> &evaluate_now(
      Eigen::Matrix<double, Eigen::Dynamic, 1> &)
  {
    return results_now;
  }

 protected:
  std::shared_ptr<const DotVector> dots;
  ResultArray results;
  Eigen::Matrix<double, Eigen::Dynamic, 1> results_now;
};

#endif
