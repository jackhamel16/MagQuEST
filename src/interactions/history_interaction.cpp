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

    // if(time_idx == 1) {
    // auto test = std::find(rhs_pairs.begin(), rhs_pairs.end(), pair_idx);
    // if(*test == pair_idx) std::cout << "dt < 1 : " << pair_idx << std::endl;
    //}

    for(int i = 0; i <= interp_order; ++i) {
      if(s - i < 0) continue;

      auto pair_found = std::find(now_pairs.begin(), now_pairs.end(), pair_idx);
      if(*pair_found == pair_idx && i == 0) continue;

      results[src] += coefficients[pair_idx][i] * history->array[obs][s - i][0];
      results[obs] += coefficients[pair_idx][i] * history->array[src][s - i][0];
    }
  }
  return results;
}

const Interaction::ResultArray &HistoryInteraction::evaluate_now(
    const int time_idx, Eigen::Matrix<soltype, Eigen::Dynamic, 1> &H_vec)
{
  for(int i = 0; i < static_cast<int>(results_now.size()); ++i)
    results_now[i] = Eigen::Vector3d(0, 0, 0);

  for(int i = 0; i < static_cast<int>(now_pairs.size()); ++i) {
    int src, obs;
    std::tie(src, obs) = idx2coord(now_pairs[i]);

    // if(time_idx == 10) std::cout << coefficients[now_pairs[i]][0] <<
    // std::endl;
    results_now[src] += coefficients[now_pairs[i]][0] * H_vec[obs];
    results_now[obs] += coefficients[now_pairs[i]][0] * H_vec[src];
    // if(time_idx == 10) std::cout << results_now[src].transpose() <<
    // std::endl;
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
