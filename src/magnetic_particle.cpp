#include "magnetic_particle.h"

MagneticParticle::MagneticParticle(const Eigen::Vector3d &pos,
                                   const double alpha,
                                   const double gamma0,
                                   const double sat_mag,
                                   const soltype &mag)
    : pos(pos), alpha(alpha), gamma0(gamma0), mag(mag), sat_mag(sat_mag)
{
}

soltype MagneticParticle::llg_rhs(const soltype &mag,
                                  const Eigen::Vector3d &hfield)
{
  const Eigen::Vector3d mxh = mag.cross(hfield);
  const double gamma = gamma0 / (1 + std::pow(alpha, 2));

  return -gamma * mxh - gamma * alpha / sat_mag * mag.cross(mxh);
}

soltype MagneticParticle::llg_jacobian_matvec(
    // bind parameters and use as Ax in GMRES to solve Ax=b
    const double dt,
    const soltype &mag,
    const soltype &delta_mag,
    const Eigen::Vector3d &hfield,
    const Eigen::Vector3d &delta_field)
{
  // Jacobian matvec approximation used in the Jacobian Free Newton Krylov solver
  Eigen::Vector3d precession = mag.cross(delta_field) + delta_mag.cross(hfield);
  Eigen::Vector3d damping = mag.cross(mag.cross(delta_field)) +
                            mag.cross(delta_mag.cross(hfield)) +
                            delta_mag.cross(mag.cross(hfield));
  const double gamma = gamma0 / (1 + std::pow(alpha, 2));

  Eigen::Vector3d jacobian_matvec =  -gamma * precession - gamma * alpha / sat_mag * damping;
  return delta_mag - dt * jacobian_matvec; //note this is a little more than just the matvec
}

soltype MagneticParticle::llg_jacobian_matvec_explicit(
    // bind parameters and use as Ax in GMRES to solve Ax=b using explicitly formed jacobian
    const double dt,
    const soltype &mag,
    const soltype &delta_mag,
    const Eigen::Vector3d &H,
    const Eigen::Vector3d &delta_field)
{
  const double gamma = gamma0 / (1 + std::pow(alpha, 2));
  const double gamma2 = gamma * alpha / sat_mag;
  Eigen::Matrix3d jacobian; 
  jacobian(0, 0) = -gamma2 * (mag[2] * H[2] + mag[1] * H[1]);
  jacobian(0, 1) =
      -gamma * H[2] + gamma2 * 2 * mag[1] * H[0] - gamma2 * mag[0] * H[1];
  jacobian(0, 2) = gamma * H[1] + gamma2 * 2 * mag[2] * H[0] - gamma2 * mag[0] * H[2];
  jacobian(1, 0) = gamma * H[2] + gamma2 * 2 * mag[0] * H[1] - gamma2 * mag[1] * H[0];
  jacobian(1, 1) = -gamma2 * (mag[2] * H[2] + mag[0] * H[0]);
  jacobian(1, 2) =
      -gamma * H[0] + gamma2 * 2 * mag[2] * H[1] - gamma2 * mag[1] * H[2];
  jacobian(2, 0) =
      -gamma * H[1] + gamma2 * 2 * mag[0] * H[2] - gamma2 * mag[2] * H[0];
  jacobian(2, 1) = gamma * H[0] + gamma2 * 2 * mag[1] * H[2] - gamma2 * mag[2] * H[1];
  jacobian(2, 2) = -gamma2 * (mag[1] * H[1] + mag[0] * H[0]);

  return delta_mag - dt * jacobian * delta_mag; //note this is a little more than just the matvec
}

const Eigen::Vector3d separation(const MagneticParticle &mp1,
                                 const MagneticParticle &mp2)
{
  return mp2.pos - mp1.pos;
}

rhs_func_vector rhs_functions(const DotVector &dots)
{
  rhs_func_vector funcs(dots.size());

  using std::placeholders::_1;
  using std::placeholders::_2;
  std::transform(dots.begin(), dots.end(), funcs.begin(),
                 [](const MagneticParticle &mp) {
                   return std::bind(&MagneticParticle::llg_rhs, mp, _1, _2);
                 });
  return funcs;
}

jacobian_matvec_func_vector make_jacobian_matvec_funcs(const DotVector &dots)
{
  // Builds data structure of matvecs for gmres for each particle
  jacobian_matvec_func_vector funcs(dots.size());

  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  using std::placeholders::_4;
  using std::placeholders::_5;
  std::transform(dots.begin(), dots.end(), funcs.begin(),
                 [](const MagneticParticle &mp) {
                   return std::bind(&MagneticParticle::llg_jacobian_matvec, mp, _1, _2, _3, _4, _5);
                 });
  return funcs;
}

std::ostream &operator<<(std::ostream &os, const MagneticParticle &mp)
{
  os << mp.pos.transpose() << " " << mp.alpha << " " << mp.gamma0 << " "
     << mp.sat_mag << " " << mp.mag.transpose() << std::endl;
  return os;
}

std::istream &operator>>(std::istream &is, MagneticParticle &mp)
{
  is >> mp.pos[0] >> mp.pos[1] >> mp.pos[2] >> mp.alpha >> mp.gamma0 >>
      mp.sat_mag >> mp.mag[0] >> mp.mag[1] >> mp.mag[2];
  return is;
}

DotVector import_dots(const std::string &fname)
{
  std::ifstream ifs(fname);
  if(!ifs) throw std::runtime_error("Could not open " + fname);

  std::istream_iterator<MagneticParticle> in_iter(ifs), eof;
  return DotVector(in_iter, eof);
}
