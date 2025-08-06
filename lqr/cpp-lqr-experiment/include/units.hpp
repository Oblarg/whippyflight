#pragma once

#include "mp-units/systems/si.h"
#include "mp-units/systems/isq/electromagnetism.h"
#include "mp-units/systems/angular.h"
#include <Eigen/Core>

namespace lqr {

using namespace mp_units;

// Simple type aliases for rotations per second
using RotationsPerSecond = quantity<angular::revolution / si::second>;
using RotationsPerSecondSquared = quantity<angular::revolution / si::second / si::second>;

// Custom unit types for LQR controller (units visible in IDE hover)
using VoltPerRotation = quantity<isq::electric_potential[si::volt] / angular::revolution>;
using VoltSecondPerRotation = quantity<isq::electric_potential[si::volt] * si::second / angular::revolution>;
using VoltSecondSquaredPerRotation = quantity<isq::electric_potential[si::volt] * si::second * si::second / angular::revolution>;

// Matrix types with unit-aware elements
using KiMatrix = Eigen::Matrix<VoltPerRotation, 4, 3>;
using KdMatrix = Eigen::Matrix<VoltSecondSquaredPerRotation, 4, 3>;
using KpMatrix = Eigen::Matrix<VoltSecondPerRotation, 4, 3>;
using BInvMatrix = Eigen::Matrix<VoltPerRotation, 4, 3>;

// Matrix shape aliases for augmented system construction
using Matrix3x3 = Eigen::Matrix<double, 3, 3>;
using Matrix3x7 = Eigen::Matrix<double, 3, 7>;
using Matrix7x3 = Eigen::Matrix<double, 7, 3>;
using Matrix3x4 = Eigen::Matrix<double, 3, 4>;

// Interpolated model structure with unit-aware matrices
struct InterpolatedModel {
    KiMatrix ki;
    KdMatrix kd;
    KpMatrix kp;
    BInvMatrix binv;
};

// Custom operator overloads for scalar multiplication with mp-units quantities
// These enable Eigen to work with our unit-aware matrices

// Single template for scalar multiplication with any unit-aware matrix
template<typename ScalarType>
inline auto operator*(float scalar, const Eigen::Matrix<ScalarType, 4, 3>& matrix) {
    Eigen::Matrix<ScalarType, 4, 3> result = matrix;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            // Extract the numerical value, multiply by scalar, then reconstruct
            auto value = result(i, j).numerical_value_in(result(i, j).unit);
            result(i, j) = ScalarType(scalar * value * result(i, j).unit);
        }
    }
    return result;
}

} // namespace lqr 