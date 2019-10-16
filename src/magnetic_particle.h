#ifndef MAGNETIC_PARTICLE_H
#define MAGNETIC_PARTICLE_H

#include <Eigen/Dense>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class MagneticParticle;

typedef Eigen::Vector3d soltype;
typedef std::vector<MagneticParticle> DotVector;
typedef std::vector<
    std::function<soltype(const soltype &, const Eigen::Vector3d &)>>
    rhs_func_vector;
typedef std::vector<std::function<soltype(const double,
                                          const soltype &,
                                          const soltype &,
                                          const Eigen::Vector3d &,
                                          const Eigen::Vector3d)>>
    jacobian_matvec_func_vector;
typedef std::vector<std::function<Eigen::Matrix3d(const double,
    const soltype &,
    const soltype &,
    const Eigen::Vector3d &,
    const Eigen::Vector3d)>> jacobian_func_vector;

class MagneticParticle {
 public:
  MagneticParticle() = default;
  MagneticParticle(const Eigen::Vector3d &,
                   const double,
                   const double,
                   const double,
                   const soltype &);

  soltype llg_rhs(const soltype &, const Eigen::Vector3d &);
  soltype llg_jacobian_matvec(const double,
                              const soltype &,
                              const soltype &,
                              const Eigen::Vector3d &,
                              const Eigen::Vector3d &);
  Eigen::Matrix3d llg_jacobian(const double,
                              const soltype &,
                              const soltype &,
                              const Eigen::Vector3d &,
                              const Eigen::Vector3d &);

  const Eigen::Vector3d &position() const { return pos; }
  const soltype &magnetization() const { return mag; }
  friend const Eigen::Vector3d separation(const MagneticParticle &,
                                          const MagneticParticle &);

  friend std::ostream &operator<<(std::ostream &, const MagneticParticle &);
  friend std::istream &operator>>(std::istream &, MagneticParticle &);

 private:
  Eigen::Vector3d pos;
  double alpha;
  double gamma0;
  soltype mag;
  double sat_mag;
};

rhs_func_vector rhs_functions(const DotVector &);
jacobian_matvec_func_vector make_jacobian_matvec_funcs(const DotVector &);
jacobian_func_vector make_jacobian_vector(const DotVector &);
DotVector import_dots(const std::string &);

#endif
