#ifndef INTERACTION_TABLE_H
#define INTERACTION_TABLE_H

#include <Eigen/Dense>
#include <algorithm>
#include <boost/multi_array.hpp>
#include <cmath>
#include <vector>

#include "configuration.h"
#include "lagrange_set.h"
#include "quantum_dot.h"

class InteractionTable {
 public:
  InteractionTable(const std::vector<QuantumDot> &);

  boost::multi_array<double, 2> coefficients;

  // private:
  size_t num_dots;

  size_t coord2idx(size_t, size_t);
  std::pair<size_t, size_t> idx2coord(const size_t);
};

#endif