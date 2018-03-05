#include <assert.h>
#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  assert(estimations.size() == ground_truth.size());

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  for (int i = 0; i < estimations.size(); i++) {
	VectorXd residual = estimations[i] - ground_truth[i];
	residual = residual.array() * residual.array();
	rmse += residual;
  }
  rmse = rmse / estimations.size();  // Calculate the mean.
  return rmse.array().sqrt();
}
