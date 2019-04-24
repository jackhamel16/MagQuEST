#ifndef GMRES_H
#define GMRES_H

//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the 
// Generalized Minimum Residual method
//
// GMRES follows the algorithm described on p. 20 of the 
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//  
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//  
//*****************************************************************
#include <math.h> 

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

void GeneratePlaneRotation(double &dx, double &dy, double &cs, double &sn)
{
  if (dy == 0.0) {
    cs = 1.0;
    sn = 0.0;
  } else if (std::abs(dy) > std::abs(dx)) {
    double temp = dx / dy;
    sn = 1.0 / sqrt( 1.0 + temp*temp );
    cs = temp * sn;
  } else {
    double temp = dy / dx;
    cs = 1.0 / sqrt( 1.0 + temp*temp );
    sn = temp * cs;
  }
}


void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn)
{
  double temp  =  cs * dx + sn * dy;
  dy = -sn * dx + cs * dy;
  dx = temp;
}

void Update(Eigen::Ref<Matrix> x, int k, Eigen::Ref<Matrix> h, Matrix &s, std::vector<Matrix> v)
{
  Matrix y(s);

  // Backsolve:  
  for (int i = k; i >= 0; i--) {
    y(i) /= h(i,i);
    for (int j = i - 1; j >= 0; j--)
      y(j) -= h(j,i) * y(i);
  }
  for (int j = 0; j <= k; j++) {
    x += v[j] * y(j);
  }
}

double dot(Eigen::Matrix<double, Eigen::Dynamic, 1> v, Eigen::Matrix<double, Eigen::Dynamic, 1> u) {
  double sum = 0;
  for(int i=0; i<v.size(); ++i) {
    sum += v(i) * u(i);
  }
  return sum;
}


int GMRES(std::function<Matrix(Matrix)> Ax, Eigen::Ref<Matrix> x, const Eigen::Ref<const Matrix> b,
      Eigen::Ref<Matrix> H, int &m, int &max_iter,
      double &tol)
{
  double resid;
  int i, j = 1, k;
  Matrix s(m+1, 1), cs(m+1, 1), sn(m+1, 1), w;
  double normb = b.norm();
  Matrix r = b - Ax(x);
  double beta = r.norm();
    
  if (normb == 0.0)
    normb = 1;// exit gmres?
  
  if ((resid = r.norm() / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  std::vector<Matrix> v(m+1);
  for(i=0; i<m+1; ++i) 
    v[i].resize(x.size(), 1);

  while (j <= max_iter) {
    v[0] = r * (1.0 / beta);    // ??? r / beta
    s.setZero();
    s(0) = beta;
    for (i = 0; i < m && j <= max_iter; i++, j++) {
      w = Ax(v[i]);
      for (k = 0; k <= i; k++) {
        H(k, i) = dot(w, v[k]);
        //H(k, i) = w.dot(v[k]);
        w -= H(k, i) * v[k];
      }
      H(i+1, i) = w.norm();
      v[i+1] = w * (1.0 / H(i+1, i)); // ??? w / H(i+1, i)

      for (k = 0; k < i; k++)
        ApplyPlaneRotation(H(k,i), H(k+1,i), cs(k), sn(k));
      
      GeneratePlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
      ApplyPlaneRotation(H(i,i), H(i+1,i), cs(i), sn(i));
      ApplyPlaneRotation(s(i), s(i+1), cs(i), sn(i));
      
      if ((resid = std::abs(s(i+1)) / normb) < tol) {
        Update(x, i, H, s, v);
        tol = resid;
        max_iter = j;
        //delete v;
        return 0;
      }
    }
    Update(x, m - 1, H, s, v);
    r = b - Ax(x);
    beta = r.norm();
    if ((resid = beta / normb) < tol) {
      tol = resid;
      max_iter = j;
      //delete v;
      return 0;
    }
  }
  
  tol = resid;
  //delete [] v;
  return 1;
}

#endif
