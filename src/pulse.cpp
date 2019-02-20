#include "pulse.h"

Pulse::Pulse(const double amplitude,
             const double fc,
             const double bw,
             const double dc,
             const Eigen::Vector3d &wavevector,
             const Eigen::Vector3d &field_orientation)
    : amplitude(amplitude),
      fc(fc),
      bw(bw),
      dc(dc),
      wavevector(wavevector.normalized()),
      field_orientation(field_orientation.normalized())
{
}

void Pulse::compute_parameters(const double c0)
{
  c = c0;
  sigma = 3 / (2 * M_PI * bw);
  td = 6 * sigma;
  return;
}

double Pulse::compute_dt()
{
  double dt = 1 / (20 * (fc + bw));
  return dt;
}

double Pulse::get_dc_field() {return dc;}

Eigen::Vector3d Pulse::operator()(const Eigen::Vector3d &r,
                                  const double t) const
{
  return Eigen::Vector3d(0, 1e5 * t, 0);
  //const double arg = wavevector.normalized().dot(r) / c - (t - td);
  //return (amplitude * field_orientation.normalized() * gaussian(arg / sigma));// *
          //cos(2 * M_PI * fc * arg));
}

std::ostream &operator<<(std::ostream &os, const Pulse &p)
{
  os << p.amplitude << " " << p.fc << " " << p.bw << " " << p.dc << " "
     << p.wavevector.transpose() << " " << p.field_orientation.transpose();
  return os;
}

std::istream &operator>>(std::istream &is, Pulse &p)
{
  is >> p.amplitude >> p.fc >> p.bw >> p.dc >> p.wavevector[0] >>
      p.wavevector[1] >> p.wavevector[2] >> p.field_orientation[0] >>
      p.field_orientation[1] >> p.field_orientation[2];
  return is;
}

// First pulse should have largest bw + fc to set step size properly
PulseVector import_pulses(const std::string &fname)
{
  std::ifstream ifs(fname);
  if(!ifs) throw std::runtime_error("Could not open " + fname);

  std::istream_iterator<Pulse> in_iter(ifs), eof;
  return PulseVector(in_iter, eof);
}
