#ifndef UNIVERSE_H
#define UNIVERSE_H

#include <boost/program_options.hpp>

enum class Domain {FREQUENCY, TIME};

template <typename Enumeration>
auto as_integer(Enumeration const value)
  -> typename std::underlying_type<Enumeration>::type 
{
  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

struct Universal{
  size_t dimensions, num_particles;
  double c0, hbar;
};

extern Universal Universe;

#endif
