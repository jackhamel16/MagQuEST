#ifndef GMRES_H
#define GMRES_H

#include <math.h>
#include <Eigen/Dense>
#include <vector>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

void GeneratePlaneRotation(double &, double &, double &, double &);

void ApplyPlaneRotation(double &, double &, double &, double &);

void Update(Eigen::Ref<Matrix>,
            int,
            Eigen::Ref<Matrix>,
            Eigen::Ref<Matrix>,
            std::vector<Matrix> &);

double dot(Eigen::Matrix<double, Eigen::Dynamic, 1>,
           Eigen::Matrix<double, Eigen::Dynamic, 1>);

int GMRES(std::function<Matrix(Matrix)>,
          Eigen::Ref<Matrix>,
          const Eigen::Ref<const Matrix>,
          Eigen::Ref<Matrix>,
          int &,
          int &,
          double &);

#endif
