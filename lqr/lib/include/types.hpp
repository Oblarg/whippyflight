#pragma once

#include "mp-units/systems/si.h"
#include "mp-units/systems/isq/electromagnetism.h"
#include "mp-units/systems/angular.h"
#include <array>
#include <Eigen/Core>

namespace lqr {

using namespace mp_units;

// Simple type aliases for rotations per second
using RotationsPerSecond = quantity<angular::revolution / si::second>;

// C-compatible types (prefixed with C as per memory rule)
struct CMotorSpeeds {
    float rotations_per_second[4];
};

struct CGyroRates {
    float rotations_per_second[3]; // 0: yaw, 1: pitch, 2: roll
};

struct CMotorOutputs {
    float volts[4];
};

struct CLqrCoefficients {
    // Motor coefficients
    float motorVel;
    float motorVelSq;
    float motorAccel;
    
    // Yaw coefficients
    float yawRate;
    float yawPropVelSumSq;
    float yawPropAccelSum;
    
    // Pitch coefficients
    float pitchRate;
    float pitchPropVelSumSq;
    float pitchDFront;
    float pitchDBack;
    
    // Roll coefficients
    float rollRate;
    float rollPropVelSumSq;
    float rollDLeft;
    float rollDRight;
};

// C++-native types with unit awareness using mp-units
struct MotorSpeeds {
    std::array<RotationsPerSecond, 4> rotations_per_second;
};

struct GyroRates {
    std::array<RotationsPerSecond, 3> rotations_per_second; // 0: yaw, 1: pitch, 2: roll
};

struct MotorOutputs {
    std::array<quantity<isq::electric_potential[si::volt]>, 4> volts;
};

struct LqrCoefficients {
    // Motor coefficients
    float motorVel;
    float motorVelSq;
    float motorAccel;
    
    // Yaw coefficients
    float yawRate;
    float yawPropVelSumSq;
    float yawPropAccelSum;
    
    // Pitch coefficients
    float pitchRate;
    float pitchPropVelSumSq;
    float pitchDFront;
    float pitchDBack;
    
    // Roll coefficients
    float rollRate;
    float rollPropVelSumSq;
    float rollDLeft;
    float rollDRight;
};

// Matrix types are defined in units.hpp to avoid duplication

// Conversion functions
MotorSpeeds from_c_motor_speeds(const CMotorSpeeds& c_speeds);
LqrCoefficients from_c_lqr_coefficients(const CLqrCoefficients& c_coeffs);
CMotorOutputs to_c_motor_outputs(const MotorOutputs& outputs);

} // namespace lqr 