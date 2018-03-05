#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <assert.h>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace {
const float kSmallFloat = 0.0001;

float EnforceNotZero(float x) {
  return fabs(x) < kSmallFloat ? kSmallFloat : x;
}

double NormalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2. * M_PI;
  while (angle < -M_PI) angle += 2. * M_PI;
  return angle;
}
} // end anonymous namespace

UKF::UKF() {
  is_initialized_ = false;
  use_laser_ = true;
  use_radar_ = true;
  time_us_ = 0;

  // Process noise (had to be tweaked).
  std_a_ = 0.2;      // Process noise standard deviation longitudinal acceleration in m/s^2.
  std_yawdd_ = 0.2;  // Process noise standard deviation yaw acceleration in rad/s^2.

  // Provided by the manufacturer (do not modify!).
  const double std_laser_px = 0.15;  // Laser measurement noise standard deviation position1 in m.
  const double std_laser_py = 0.15;  // Laser measurement noise standard deviation position2 in m.

  // Laser measurement noise covariance matrix.
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << pow(std_laser_px, 2), 0,
              0, pow(std_laser_py, 2);

  // Provided by the manufacturer (do not modify!).
  const double std_radar_rho = 0.3;      // Radar measurement noise standard deviation radius in m.
  const double std_radar_phi = 0.03;     // Radar measurement noise standard deviation angle in rad.
  const double std_radar_rho_dot = 0.3;  // Radar measurement noise standard deviation radius change in m/s.

  // Radar measurement noise covariance matrix.
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << pow(std_radar_rho, 2), 0, 0,
              0, pow(std_radar_phi, 2), 0,
              0, 0, pow(std_radar_rho_dot, 2);

  // Variables to hold state.
  n_x_ = 5;
  x_ = VectorXd(n_x_);
  P_ = MatrixXd(n_x_, n_x_);

  // Variables for sigma points.
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {
}

void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  if (!is_initialized_) {
    Initialize(meas_package);
    is_initialized_ = true;
  } else {
    const float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;  // microseconds to seconds
    time_us_ = meas_package.timestamp_;
    Predict(delta_t);

    switch (meas_package.sensor_type_) {
      case MeasurementPackage::RADAR: {
        if (use_radar_) {
          UpdateRadar(meas_package);
        }
        break;
      }
      case MeasurementPackage::LASER: {
        if (use_laser_) {
          UpdateLidar(meas_package);
        }
        break;
      }
      default: {
        assert(false && "Unrecognized sensor type.");
      }
    }
  }
}

void UKF::Initialize(const MeasurementPackage& meas_package) {
  float px, py;
  switch (meas_package.sensor_type_) {
    case MeasurementPackage::RADAR: {
      const float rho = meas_package.raw_measurements_[0];
      const float phi = meas_package.raw_measurements_[1];
      const float rho_dot = meas_package.raw_measurements_[2];
      // Convert polar coordinates to cartesian coordinates.
      px = rho * cos(phi);
      py = rho * sin(phi);
      break;
    }
    case MeasurementPackage::LASER: {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      break;
    }
    default: {
      assert(false && "Unrecognized sensor type.");
    }
  }

  // TODO: Experiment with different ways of initializing v, phi, phi_dot and P.
  x_ << px, py, 0, 0, 0;
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Set the weights of the sigma points.
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  time_us_ = meas_package.timestamp_;
}

void UKF::Predict(double delta_t) {
  GenerateAugmentedSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();
}

void UKF::GenerateAugmentedSigmaPoints() {
  // Augmented mean state.
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // Augmented covariance matrix.
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;
  MatrixXd Sqr = P_aug.llt().matrixL();

  // Augmented sigma points matrix.
  Xsig_aug_.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    MatrixXd mean_modifier = sqrt(lambda_ + n_aug_) * Sqr.col(i);
    Xsig_aug_.col(i + 1) = x_aug + mean_modifier;
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - mean_modifier;
  }
}

void UKF::PredictSigmaPoints(double delta_t) {
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Extract values for readability.
    const double px = Xsig_aug_(0, i);
    const double py = Xsig_aug_(1, i);
    const double v = Xsig_aug_(2, i);
    const double yaw = Xsig_aug_(3, i);
    const double yawd = Xsig_aug_(4, i);
    const double nu_a = Xsig_aug_(5, i);
    const double nu_yawdd = Xsig_aug_(6, i);

    // Predict state values.
    const double px_p = fabs(yawd) > 0.001
        ? px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw))
        : px + v * delta_t * cos(yaw);
    const double py_p = fabs(yawd) > 0.001
        ? py + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t))
        : py + v * delta_t * sin(yaw);
    const double v_p = v;
    const double yaw_p = yaw + yawd * delta_t;
    const double yawd_p = yawd;

    // Add noise.
    Xsig_pred_(0, i) = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    Xsig_pred_(1, i) = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    Xsig_pred_(2, i) = v_p + nu_a * delta_t;
    Xsig_pred_(3, i) = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    Xsig_pred_(4, i) = yawd_p + nu_yawdd * delta_t;
  }
}

void UKF::PredictMeanAndCovariance() {
  // Predict the state mean.
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict the covariance matrix.
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = NormalizeAngle(x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  // Project sigma points onto measurement space.
  MatrixXd Zsig = MatrixXd(2, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Zsig(0,i) = Xsig_pred_(0,i);  // px
    Zsig(1,i) = Xsig_pred_(1,i);  // py
  }

  Update(meas_package, Zsig, R_laser_);
}

void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  // Project sigma points onto measurement space.
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // Extract values for better readability.
    const double px = Xsig_pred_(0, i);
    const double py = Xsig_pred_(1, i);
    const double v = Xsig_pred_(2, i);
    const double yaw = Xsig_pred_(3, i);
    const double vx = cos(yaw) * v;
    const double vy = sin(yaw) * v;

    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * vx + py * vy) / EnforceNotZero(Zsig(0, i));
  }

  Update(meas_package, Zsig, R_radar_);
}

void UKF::Update(const MeasurementPackage& meas_package,
                 const MatrixXd& Zsig,
                 const MatrixXd& R /* noise covariance matrix */) {
  const int nz = Zsig.rows();

  // Predicted mean measurement.
  VectorXd z_pred = VectorXd(nz);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Predicted measurement covariance matrix.
  MatrixXd S = MatrixXd(nz, nz);
  S.fill(0.0);

  // Cross-correlation matrix.
  MatrixXd Tc = MatrixXd(n_x_, nz);
  Tc.fill(0.0);

  bool should_normalize_angle = (meas_package.sensor_type_ == MeasurementPackage::RADAR);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (should_normalize_angle) {
      z_diff(1) = NormalizeAngle(z_diff(1));
    }
    S = S + weights_(i) * z_diff * z_diff.transpose();

    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    if (should_normalize_angle) {
      x_diff(3) = NormalizeAngle(x_diff(3));
    }
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  S = S + R;
  MatrixXd K = Tc * S.inverse();  // Kalman gain.
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  z_diff(1) = NormalizeAngle(z_diff(1));

  // Update state mean and covariance matrix.
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  std::cout << "x: " << x_ << std::endl;
  std::cout << "P: " << P_ << std::endl;

  // TODO: Calculate NIS.
}
