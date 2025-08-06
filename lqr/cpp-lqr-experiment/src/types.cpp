#include "types.hpp"

namespace lqr {

MotorSpeeds from_c_motor_speeds(const CMotorSpeeds& c_speeds) {
    MotorSpeeds speeds;
    for (int i = 0; i < 4; ++i) {
        speeds.rotations_per_second[i] = c_speeds.rotations_per_second[i] * mp_units::angular::revolution / mp_units::si::second;
    }
    return speeds;
}

GyroRates from_c_gyro_rates(const CGyroRates& c_rates) {
    GyroRates rates;
    for (int i = 0; i < 3; ++i) {
        rates.rotations_per_second[i] = c_rates.rotations_per_second[i] * mp_units::angular::revolution / mp_units::si::second;
    }
    return rates;
}

MotorOutputs from_c_motor_outputs(const CMotorOutputs& c_outputs) {
    MotorOutputs outputs;
    for (int i = 0; i < 4; ++i) {
        outputs.volts[i] = c_outputs.volts[i] * mp_units::si::volt;
    }
    return outputs;
}

LqrCoefficients from_c_lqr_coefficients(const CLqrCoefficients& c_coeffs) {
    LqrCoefficients coeffs;
    
    // Motor coefficients
    coeffs.motorVel = c_coeffs.motorVel;
    coeffs.motorVelSq = c_coeffs.motorVelSq;
    coeffs.motorAccel = c_coeffs.motorAccel;
    
    // Yaw coefficients
    coeffs.yawRate = c_coeffs.yawRate;
    coeffs.yawPropVelSumSq = c_coeffs.yawPropVelSumSq;
    coeffs.yawPropAccelSum = c_coeffs.yawPropAccelSum;
    
    // Pitch coefficients
    coeffs.pitchRate = c_coeffs.pitchRate;
    coeffs.pitchPropVelSumSq = c_coeffs.pitchPropVelSumSq;
    coeffs.pitchDFront = c_coeffs.pitchDFront;
    coeffs.pitchDBack = c_coeffs.pitchDBack;
    
    // Roll coefficients
    coeffs.rollRate = c_coeffs.rollRate;
    coeffs.rollPropVelSumSq = c_coeffs.rollPropVelSumSq;
    coeffs.rollDLeft = c_coeffs.rollDLeft;
    coeffs.rollDRight = c_coeffs.rollDRight;
    
    return coeffs;
}

CMotorOutputs to_c_motor_outputs(const MotorOutputs& outputs) {
    CMotorOutputs c_outputs;
    for (int i = 0; i < 4; ++i) {
        c_outputs.volts[i] = outputs.volts[i].numerical_value_in(mp_units::si::volt);
    }
    return c_outputs;
}

} // namespace lqr 