#pragma once

// C-compatible data structures matching existing LQR types
typedef struct {
    float rotationsPerSecond[4];
} MotorSpeeds;

typedef struct {
    // 0: yaw, 1: pitch, 2: roll
    float rotationsPerSecond[3];
} GyroRates;

typedef struct {
    float volts[4];
} MotorOutputs;

// LQR coefficients passed from CLI interface
typedef struct {
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
} LqrCoefficients;

// C++ LQR API - same interface as Rust implementation
#ifdef __cplusplus
extern "C" {
#endif

// Initialize LQR at boot (generate LUT with coefficients)
int cpp_lqr_init(const LqrCoefficients* coefficients);

// Main control function (drop-in replacement for calculateAttitudeControl)
MotorOutputs cpp_lqr_calculate_attitude_control(void);

// Reset integral states
void cpp_lqr_reset_integral_states(void);

// Optional: For testing/debugging
int cpp_lqr_test_interpolation(const MotorSpeeds* speeds, MotorOutputs* result);

#ifdef __cplusplus
}
#endif 