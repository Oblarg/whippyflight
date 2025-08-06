#pragma once

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

typedef struct {
    // Integral gains: volts per rotation (since we've integrated with time constant)
    float KiVoltPerRotation[4][3];
    // Derivative gains: volts per rotation per second squared
    float KdVoltPerRotationPerSecondSquared[4][3];
    // Rate gains: volts per rotation per second
    float KpVoltPerRotationPerSecond[4][3];
    float BDiscreteInvVoltSecondPerRotation[4][3];
} InterpolatedModel;

MotorOutputs calculateAttitudeControl(void);
MotorOutputs calculate_feedback(const InterpolatedModel *interpolatedModel, GyroRates proportionalErrors, GyroRates integralErrors, GyroRates derivativeErrors);
MotorOutputs calculate_feedforward(const InterpolatedModel *interpolatedModel, GyroRates setpointDiff);
void reset_integral_states(void);

// Global variables for blackbox logging (populated during LQR calculation)
extern GyroRates setpoints;
extern GyroRates prevSetpoints;
extern GyroRates setpointDiff;
extern GyroRates measurements;
extern GyroRates prevMeasurements;
extern GyroRates proportionalErrors;
extern GyroRates integralErrors;
extern GyroRates derivativeErrors;
extern MotorOutputs feedback;
extern MotorOutputs feedforward;
extern MotorOutputs affine;
extern MotorOutputs attitudeControl;
extern MotorSpeeds motorSpeeds;
extern InterpolatedModel interpolatedModel;

// P, I, and D control voltages for individual motors (for blackbox logging)
extern MotorOutputs pControl;
extern MotorOutputs iControl;
extern MotorOutputs dControl;
