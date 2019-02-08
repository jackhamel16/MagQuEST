#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

typedef Eigen::Vector3d vec3d;

Eigen::Vector3d mag_d0_source(double t, double delay, double dt)
{
  return Eigen::Vector3d(0, exp(-std::pow((t - 500 * dt - delay) / 4e-10, 2) / 2),
                         0);
}

Eigen::Vector3d mag_d1_source(double t, double delay, double dt)
{
  return Eigen::Vector3d(0,
                         -6.25e18 * (t - 500 * dt - delay) *
                             exp(-std::pow((t - 500 * dt - delay) / 4e-10, 2) / 2),
                         0);
}

Eigen::Vector3d mag_d2_source(double t, double delay, double dt)
{
  return Eigen::Vector3d(
      0,
      exp(-std::pow((t - 500 * dt - delay) / 4e-10, 2) / 2) *
          (3.9062e37 * std::pow((-500 * dt - delay + t), 2) - 6.25e18),
      0);
}

Eigen::Vector3d analytic_interaction(Eigen::Vector3d &mag,
                                     Eigen::Vector3d &magd1,
                                     Eigen::Vector3d &magd2,
                                     Eigen::Vector3d &dr,
                                     double c,
                                     double dist)
{
  Eigen::Matrix3d rr = dr * dr.transpose() / dr.squaredNorm();
  Eigen::Matrix3d irr = Eigen::Matrix3d::Identity() - rr;
  Eigen::Matrix3d i3rr = Eigen::Matrix3d::Identity() - 3 * rr;

  return -1 / (4 * M_PI) *
         (i3rr * mag / std::pow(dist, 3) +
          i3rr * magd1 / (c * std::pow(dist, 2)) +
          irr * magd2 / (std::pow(c, 2) * dist));
}

Eigen::Matrix<double, 3, 3> G_Matrix(Eigen::Vector3d &dr, double c, double dist)
{
  Eigen::Matrix3d rr = dr * dr.transpose() / dr.squaredNorm();
  Eigen::Matrix3d irr = Eigen::Matrix3d::Identity() - rr;
  Eigen::Matrix3d i3rr = Eigen::Matrix3d::Identity() - 3 * rr;

  return -1 / (4 * M_PI) *
         (i3rr / std::pow(dist, 3) + i3rr / (c * std::pow(dist, 2)) +
          irr / (std::pow(c, 2) * dist));
}

int main()
{
  const double c = 2.99792458e8;
  // const double eps = 8.85418782e-12;
  const double dt = 4.545454545455e-12;
  const vec3d pos1(0, 0, 0);
  const vec3d pos2(0.001362693, 0, 0);
  //const vec3d pos2(0.001362692990909, 0, 0);
  const double total_t = 1e-8;
  int steps = total_t / dt;
  double dist = (pos2 - pos1).norm();
  double delay = dist / c;
  Eigen::Vector3d dr(pos1 - pos2);  // corresponds to separation calculation

  std::cout << dist << std::endl;

  std::vector<Eigen::Vector3d> fields_at_1(steps);  // Set to be gaussian
  std::vector<Eigen::Vector3d> fields_at_2(steps);

  for(int i = 1; i < steps; i = i + 50) {
    std::cout << mag_d0_source(i * dt, 0, dt).transpose() << std::endl;
  }
  for(int i = 1; i < steps; ++i) {
    Eigen::Vector3d magd0 = mag_d0_source(i * dt, delay, dt);
    // Eigen::Vector3d magd0 = Eigen::Vector3d(0, 0, 0);
    // Eigen::Vector3d magd1 = Eigen::Vector3d(0, 0, 0);
    // Eigen::Vector3d magd2 = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d magd1 = mag_d1_source(i * dt, delay, dt);
    Eigen::Vector3d magd2 = mag_d2_source(i * dt, delay, dt);

    fields_at_1[i] = mag_d0_source(i * dt, 0, dt);
    fields_at_2[i] = analytic_interaction(magd0, magd1, magd2, dr, c, dist);
  }

  // Testing to see what G matrix looks like
  std::cout << "G Matrix: \n" << G_Matrix(dr, c, dist) << std::endl << std::endl;

  std::ofstream outfile;
  outfile.open("analytic_results.dat");
  outfile << std::scientific << std::setprecision(15);
  for(int i = 0; i < steps; ++i) {
    outfile << fields_at_1[i].transpose() << " " << fields_at_2[i].transpose()
            << std::endl;
  }
  outfile.close();
}
