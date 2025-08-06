#include "blackbox.h"
#include "lqr_controller.h"
#include "../../flight/pid.h"
#include "../../fc/rc.h"
#include "../../build/debug.h"

// Function to update PID setpoints for blackbox logging
void updateBlackboxSetpoints(const GyroRates *setpoints) {
    // Update PID runtime setpoints for blackbox logging
    // Convert from rotations per second to degrees per second (multiply by 360)
    // The PID system expects setpoints in degrees per second
    
    // Convert rotations per second to degrees per second and update
    // LQR setpoints are in rotations per second, PID expects degrees per second
    pidRuntime.previousPidSetpoint[0] = setpoints->rotationsPerSecond[0] * 360.0f; // Roll
    pidRuntime.previousPidSetpoint[1] = setpoints->rotationsPerSecond[1] * 360.0f; // Pitch  
    pidRuntime.previousPidSetpoint[2] = setpoints->rotationsPerSecond[2] * 360.0f; // Yaw
}

// Function to shim LQR internal state to PID handles for blackbox logging
void updateBlackboxLqrState(void) {
    
    for (int motor = 0; motor < 3; motor++) {
        // P term: proportional error (rate error) - convert from rotations/sec to degrees/sec
        pidData[motor].P = pControl.volts[motor] * 100.0f;
        
        // I term: integral error - convert from rotations to degrees, multiply by 1000 for integer representation
        pidData[motor].I = iControl.volts[motor] * 100.0f;
        
        // D term: derivative error - convert from rotations/sec to degrees/sec
        pidData[motor].D = dControl.volts[motor] * 100.0f;
        
        // F term: feedforward - convert from volts to a scaled value for display
        // Scale feedforward voltage to a reasonable range for blackbox
        pidData[motor].F = feedforward.volts[motor] * 100.0f;
        
        // S term: affine term - convert from volts to a scaled value for display
        // Repurpose the unused S term for affine control voltage
        pidData[motor].S = affine.volts[motor] * 100.0f;
        
        // Sum: total control output for motor (feedback + feedforward + affine)
        pidData[motor].Sum = attitudeControl.volts[motor] * 100.0f;
    }
    
    // Debug field layout:
    //   debug[0-2]: PID gains for motor 0 yaw (Kp, Ki, Kd)
    //   debug[3-5]: PID gains for motor 0 pitch (Kp, Ki, Kd)
    //   debug[6-8]: PID gains for motor 0 roll (Kp, Ki, Kd)
    
    // Log PID gains for motor 0 yaw
    debug[0] = (int16_t)(interpolatedModel.KpVoltPerRotationPerSecond[0][0] * 1000.0f); // Yaw Kp gain
    debug[1] = (int16_t)(interpolatedModel.KiVoltPerRotation[0][0] * 1000.0f); // Yaw Ki gain
    debug[2] = (int16_t)(interpolatedModel.KdVoltPerRotationPerSecondSquared[0][0] * 1000.0f); // Yaw Kd gain
    
    // Log PID gains for motor 0 pitch
    debug[3] = (int16_t)(interpolatedModel.KpVoltPerRotationPerSecond[0][1] * 1000.0f); // Pitch Kp gain
    debug[4] = (int16_t)(interpolatedModel.KiVoltPerRotation[0][1] * 1000.0f); // Pitch Ki gain
    debug[5] = (int16_t)(interpolatedModel.KdVoltPerRotationPerSecondSquared[0][1] * 1000.0f); // Pitch Kd gain
    
    // Log PID gains for motor 0 roll
    debug[6] = (int16_t)(interpolatedModel.KpVoltPerRotationPerSecond[0][2] * 1000.0f); // Roll Kp gain
    debug[7] = (int16_t)(interpolatedModel.KiVoltPerRotation[0][2] * 1000.0f); // Roll Ki gain
    debug[8] = (int16_t)(interpolatedModel.KdVoltPerRotationPerSecondSquared[0][2] * 1000.0f); // Roll Kd gain
}