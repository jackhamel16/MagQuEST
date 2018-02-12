#ifndef INTERACTION_H
#define INTERACTION_H

#include <Eigen/Dense>
#include <memory>

#include "../magnetic_particle.h"

class Interaction {
 public:
  typedef std::vector<Eigen::Vector3d> ResultArray;

  Interaction(const std::shared_ptr<const DotVector> &dots)
      : dots(dots), results(dots->size()), results_now(dots->size()){};
  const Eigen::Vector3d &operator[](const int i) const { return results[i]; }
  virtual const ResultArray &evaluate(const int) = 0;
  virtual const ResultArray &evaluate_now(
      const int, Eigen::Matrix<soltype, Eigen::Dynamic, 1>)
  {
    return results_now;
  }

 protected:
  std::shared_ptr<const DotVector> dots;
  ResultArray results;
  ResultArray results_now;
};

#endif
