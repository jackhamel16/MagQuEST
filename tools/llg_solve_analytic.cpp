#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>

typedef Eigen::Vector3d vec3d;

vec3d rec_cross(vec3d a, vec3d b, int n)
{
  // A recursive cross product
  // Crosses a n-times with b
  return n > 0 ? a.cross(rec_cross(a, b, n-1)) : b;
}

double factorial(double x)
{
  double answer = 1;
  for(int term = 1; term <= x; ++term)
  {
    answer = answer * term;
  }
  return answer;
}

vec3d field(double t, double delay, double dt)
{
  double sigma = 4e-10;
  double mu = 500 * dt;
  return vec3d(0, exp(-pow((t - mu - delay) / sigma, 2) / 2),
                         0); 
}

vec3d field_integral(double amplitude, double t, double delay, double dt)
{
  double sigma = 4e-10;
  double mu = 500 * dt;
  double arg1 = (t - mu) / (sqrt(2) * sigma);
  double arg2 = -mu / (sqrt(2) * sigma);
  return vec3d(0, sqrt(M_PI/2) * sigma * amplitude * (erf(arg1) - erf(arg2)), 0);
}

vec3d simple_field(double t)
{
  return vec3d(0, t * 1e5, 0);
}

vec3d simple_field_integral(double t)
{
  return vec3d(0, t * t * 5e4, 0);
}

vec3d llg_solver(vec3d M0, vec3d H, double t, int max_iters, double gamma)
{
  vec3d M(0,0,0);
  for(int n = 0; n < max_iters; ++n)
  {
    M = M + std::pow(t * gamma, n) * rec_cross(H, M0, n) / factorial(n); 
  }
  return M;
}

int main()
{
  vec3d H_integral;
  const int max_iters = 24;
  const vec3d M0(1,0,0);
  const vec3d H_dc(0,1e-3,0);
  const double c = 2.99792458e8;
  const double dt = 4.545454545455e-12;
  const double total_t = 1e-8;
  const double alpha = 0.1;
  const double gamma0 = 175920000000;

  int num_steps = total_t / dt + 1;
  double gamma = gamma0 / ( 1 + pow(alpha,2));

  std::vector<vec3d> M_array(num_steps);
  std::ofstream outfile;
  outfile.open("llg_results.dat");
  outfile << std::scientific << std::setprecision(15);
  
  for(int step = 0; step < num_steps; ++step)
  {
    H_integral = field_integral(1e-2, step * dt, 0, dt);
    M_array[step] = llg_solver(M0, simple_field_integral(step * dt), step * dt, max_iters, gamma);
    outfile << M_array[step].transpose() << std::endl;
  }
  //for(int iters = 0; iters < max_iters; ++iters)
  //{ 
    //for(int step = 0; step < num_steps; ++step)
    //{
      //M_array[step] = llg_solver(M0, H, step * dt, iters, gamma);
    //}
  //}

  return 0;
}
