#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>

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

vec3d simple_field(double t)
{
  return vec3d(0, t * 1e5, 0);
}

vec3d llg_solver(vec3d M0, vec3d H, double t, int num_terms, double gamma)
{
  vec3d M(0,0,0);
  for(int n = 0; n < num_terms; ++n)
  {
    M = M + std::pow(t * gamma, n) * rec_cross(H, M0, n) / factorial(n); 
  }
  return M;
}

int main(int argc, char* argv[])
{
  double dt;
  std::string file_name;
  vec3d M;
  const int num_terms = 24;
  const vec3d M0(1,0,0);
  const vec3d H_dc(0,1e-3,0);
  const double c = 2.99792458e8;
  const double total_t = 1e-8;
  const double alpha = 0.1;
  const double gamma0 = 175920000000;
  double gamma = gamma0 / ( 1 + pow(alpha,2));

  if(argc > 1)
    dt = atof(argv[1]);
  else
    dt = total_t / 1000;

  int num_steps = (int) std::ceil(total_t / dt);
  
  if(argc > 2)
    file_name = argv[2];
  else
    file_name = "llg_output.dat";
  
  std::ofstream outfile;
  outfile.open(file_name);
  outfile << std::scientific << std::setprecision(15);
  
  for(int step = 0; step < num_steps; ++step)
  {
    M = llg_solver(M0, H_dc, step * dt, num_terms, gamma);
    outfile << M.transpose() << std::endl;
  }

  std::cout << "Step Size = " << dt << std::endl;
  std::cout << "Number of Steps = " << num_steps << std::endl;
  
  return 0;
}
