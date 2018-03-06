#include "history_interaction.h"

using Vec3d = Eigen::Vector3d;

HistoryInteraction::HistoryInteraction(
    const std::shared_ptr<const DotVector> &dots,
    const std::shared_ptr<const Integrator::History<soltype>> &history,
    const std::shared_ptr<Propagation::FixedFramePropagator> &dyadic,
    const int interp_order,
    const double dt,
    const double c0)
    : Interaction(dots),
      history(history),
      dyadic(dyadic),
      interp_order(interp_order),
      num_interactions(dots->size() * (dots->size() - 1) / 2),
      floor_delays(num_interactions),
      coefficients(boost::extents[num_interactions][interp_order + 1]),
      dt(dt),
      c0(c0)
{
  build_coefficient_table();
  chi = 1;
}

void HistoryInteraction::build_coefficient_table()
{
  Interpolation::UniformLagrangeSet lagrange(interp_order);

  for(int pair_idx = 0; pair_idx < num_interactions; ++pair_idx) {
    int src, obs;
    std::tie(src, obs) = idx2coord(pair_idx);

    Vec3d dr(separation((*dots)[src], (*dots)[obs]));

    std::pair<int, double> delay(split_double(dr.norm() / (c0 * dt)));

    floor_delays[pair_idx] = delay.first;
    lagrange.calculate_weights(delay.second, dt);

    if(delay.first == 0) {
      now_pairs.push_back(pair_idx);
      std::cout << pair_idx << std::endl;
    }

    std::vector<Eigen::Matrix3d> interp_dyads(
        dyadic->coefficients(dr, lagrange));

    for(int i = 0; i <= interp_order; ++i) {
      coefficients[pair_idx][i] = interp_dyads[i];
    }
  }
}

const Interaction::ResultArray &HistoryInteraction::evaluate(const int time_idx)
{
  for(int i = 0; i < static_cast<int>(results.size()); ++i)
    results[i] = Eigen::Vector3d(0, 0, 0);

  for(int pair_idx = 0; pair_idx < num_interactions; ++pair_idx) {
    int src, obs;
    std::tie(src, obs) = idx2coord(pair_idx);
    const int s = time_idx - floor_delays[pair_idx];

    Vec3d dr(separation((*dots)[src], (*dots)[obs]));

    for(int i = 0; i <= interp_order; ++i) {
      if(s - i < 0) continue;

      if(now_pairs.size() > 0) {
        auto pair_found =
            std::find(now_pairs.begin(), now_pairs.end(), pair_idx);
        if(*pair_found == pair_idx && i == 0) continue;
      }
      results[src] += coefficients[pair_idx][i] * history->array[obs][s - i][0];
      results[obs] += coefficients[pair_idx][i] * history->array[src][s - i][0];
    }
  }
  return results;
}

const Eigen::Matrix<double, Eigen::Dynamic, 1>
    &HistoryInteraction::evaluate_now(
        Eigen::Matrix<double, Eigen::Dynamic, 1> &H_vec)
{
  for(int i = 0; i < static_cast<int>(results_now.size()); ++i)
    results_now[i] = -chi / 3 * H_vec[i];  // self-interaction

  for(int i = 0; i < static_cast<int>(now_pairs.size()); ++i) {
    int src, obs;
    std::tie(src, obs) = idx2coord(now_pairs[i]);

    Eigen::Vector3d obs_vec = Eigen::Map<Eigen::Vector3d>(&H_vec[3 * obs]);
    Eigen::Vector3d src_vec = Eigen::Map<Eigen::Vector3d>(&H_vec[3 * src]);

    Eigen::Vector3d src_field = coefficients[now_pairs[i]][0] * obs_vec;
    Eigen::Vector3d obs_field = coefficients[now_pairs[i]][0] * src_vec;

    results_now[3 * src] += src_field[0];
    results_now[3 * src + 1] += src_field[1];
    results_now[3 * src + 2] += src_field[2];
    results_now[3 * obs] += obs_field[0];
    results_now[3 * obs + 1] += obs_field[1];
    results_now[3 * obs + 2] += obs_field[2];
  }
  return results_now;
}

int HistoryInteraction::coord2idx(int row, int col)
{
  assert(row != col);
  if(col > row) std::swap(row, col);

  return row * (row - 1) / 2 + col;
}

std::pair<int, int> HistoryInteraction::idx2coord(const int idx)
{
  const int row = std::floor((std::sqrt(1 + 8 * idx) + 1) / 2.0);
  const int col = idx - row * (row - 1) / 2;

  return std::pair<int, int>(row, col);
}

std::pair<int, double> HistoryInteraction::split_double(const double delay)
{
  std::pair<int, double> result;

  double idelay;
  result.second = std::modf(delay, &idelay);
  result.first = static_cast<int>(idelay);

  return result;
}
