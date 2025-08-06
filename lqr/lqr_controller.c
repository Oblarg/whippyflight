#include "lqr_constants.h"
#include "lqr_controller.h"
#include "blackbox.h"
#include "../../sensors/gyro.h"
#include "../../fc/rc.h"
#include "../../fc/runtime_config.h"
#include "../../drivers/dshot.h"
#include "../../flight/pid.h"
#include <math.h>
#include <stdbool.h>

// Global variables for blackbox logging
GyroRates setpoints = {.rotationsPerSecond = {0, 0, 0}};
GyroRates prevSetpoints = {.rotationsPerSecond = {0, 0, 0}};
GyroRates setpointDiff = {.rotationsPerSecond = {0, 0, 0}};
GyroRates measurements = {.rotationsPerSecond = {0, 0, 0}};
GyroRates prevMeasurements = {.rotationsPerSecond = {0, 0, 0}};
GyroRates proportionalErrors = {.rotationsPerSecond = {0, 0, 0}};
GyroRates integralErrors = {.rotationsPerSecond = {0, 0, 0}};
GyroRates derivativeErrors = {.rotationsPerSecond = {0, 0, 0}};
MotorOutputs feedback = {.volts = {0, 0, 0, 0}};
MotorOutputs feedforward = {.volts = {0, 0, 0, 0}};
MotorOutputs affine = {.volts = {0, 0, 0, 0}};
MotorOutputs attitudeControl = {.volts = {0, 0, 0, 0}};
MotorSpeeds motorSpeeds = {.rotationsPerSecond = {0, 0, 0, 0}};
MotorOutputs pControl = {.volts = {0, 0, 0, 0}};
MotorOutputs iControl = {.volts = {0, 0, 0, 0}};
MotorOutputs dControl = {.volts = {0, 0, 0, 0}};

// loop time
const float dt = 0.00025f;

InterpolatedModel interpolatedModel = {
    .KiVoltPerRotation = {
        {0, 0, 0},
        {0, 0, 0}, 
        {0, 0, 0},
        {0, 0, 0}
    },
    .KdVoltPerRotationPerSecondSquared = {
        {0, 0, 0},
        {0, 0, 0}, 
        {0, 0, 0},
        {0, 0, 0}
    },
    .KpVoltPerRotationPerSecond = {
        {0, 0, 0},
        {0, 0, 0},  
        {0, 0, 0},
        {0, 0, 0}
    },
    .BDiscreteInvVoltSecondPerRotation = {
        {0, 0, 0},
        {0, 0, 0},  
        {0, 0, 0},
        {0, 0, 0}
    }
};

float signed_square(float arg) {
    return copysignf(arg, arg * arg);
}



InterpolatedModel lerp_lqr(MotorSpeeds motorSpeeds) {
    // [floor(0) or ceiling(1)][motor index]
    int lookupIndex[2][4];
    float rpsAtLookupIndex[2][4];

    for (int motor = 0; motor < 4; motor++) {
        lookupIndex[0][motor] = (int) floorf((motorSpeeds.rotationsPerSecond[motor] - lqrLookupMinRps) / lqrLookupStepSize);
        rpsAtLookupIndex[0][motor] = (float) lookupIndex[0][motor] * lqrLookupStepSize + lqrLookupMinRps;
        lookupIndex[1][motor] = lookupIndex[0][motor] + 1;
        rpsAtLookupIndex[1][motor] = (float) lookupIndex[1][motor] * lqrLookupStepSize + lqrLookupMinRps;
    }

    InterpolatedModel model = {
        .KiVoltPerRotation = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
        },
        .KdVoltPerRotationPerSecondSquared = {
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}
        },
        .KpVoltPerRotationPerSecond = {
            {0, 0, 0},
            {0, 0, 0},  
            {0, 0, 0},
            {0, 0, 0}
        },
        .BDiscreteInvVoltSecondPerRotation = {
            {0, 0, 0},
            {0, 0, 0},  
            {0, 0, 0},
            {0, 0, 0}
        }
    };

    float weights[16];
    float sumWeights = 0;

    // Calculate weights for each vertex of the 4-cube
    int vertexIndex = 0;
    for (int flFloorCeil = 0; flFloorCeil < 2; flFloorCeil++) {
        for (int frFloorCeil = 0; frFloorCeil < 2; frFloorCeil++) {
            for (int blFloorCeil = 0; blFloorCeil < 2; blFloorCeil++) {
                for (int brFloorCeil = 0; brFloorCeil < 2; brFloorCeil++) {
                    float weight = 1.0f;
                    int floorCeil[4] = {flFloorCeil, frFloorCeil, blFloorCeil, brFloorCeil};
                    
                    for (int motor = 0; motor < 4; motor++) {
                        float diff = fabsf(rpsAtLookupIndex[floorCeil[motor]][motor] - motorSpeeds.rotationsPerSecond[motor]) / lqrLookupStepSize;
                        weight *= (1 - diff);
                    }
                    weights[vertexIndex] = weight;
                    sumWeights += weight;
                    vertexIndex++;
                }
            }
        }
    }

    // Normalize weights
    for (int i = 0; i < 16; i++) {
        weights[i] /= sumWeights;
    }

    // Interpolate gains using all 16 vertices
    vertexIndex = 0;
    for (int flFloorCeil = 0; flFloorCeil < 2; flFloorCeil++) {
        for (int frFloorCeil = 0; frFloorCeil < 2; frFloorCeil++) {
            for (int blFloorCeil = 0; blFloorCeil < 2; blFloorCeil++) {
                for (int brFloorCeil = 0; brFloorCeil < 2; brFloorCeil++) {
                    int floorCeil[4] = {flFloorCeil, frFloorCeil, blFloorCeil, brFloorCeil};
                    
                                            for (int motor = 0; motor < 4; motor++) {
                            for (int axis = 0; axis < 3; axis++) {
                                // Interpolate integral gains
                                model.KiVoltPerRotation[motor][axis] += weights[vertexIndex] * 
                                    modelLookup.KiVoltPerRotation[lookupIndex[floorCeil[0]][0]]
                                                          [lookupIndex[floorCeil[1]][1]]
                                                          [lookupIndex[floorCeil[2]][2]]
                                                          [lookupIndex[floorCeil[3]][3]]
                                                          [motor][axis];
                                
                                // Interpolate derivative gains
                                model.KdVoltPerRotationPerSecondSquared[motor][axis] += weights[vertexIndex] * 
                                    modelLookup.KdVoltPerRotationPerSecondSquared[lookupIndex[floorCeil[0]][0]]
                                                          [lookupIndex[floorCeil[1]][1]]
                                                          [lookupIndex[floorCeil[2]][2]]
                                                          [lookupIndex[floorCeil[3]][3]]
                                                          [motor][axis];
                                
                                // Interpolate rate gains
                                model.KpVoltPerRotationPerSecond[motor][axis] += weights[vertexIndex] * 
                                    modelLookup.KpVoltPerRotationPerSecond[lookupIndex[floorCeil[0]][0]]
                                                          [lookupIndex[floorCeil[1]][1]]
                                                          [lookupIndex[floorCeil[2]][2]]
                                                          [lookupIndex[floorCeil[3]][3]]
                                                          [motor][axis];
                                
                                // Interpolate B matrix
                                model.BDiscreteInvVoltSecondPerRotation[motor][axis] += weights[vertexIndex] * 
                                    modelLookup.BDiscreteInvVoltSecondPerRotation[lookupIndex[floorCeil[0]][0]]
                                                          [lookupIndex[floorCeil[1]][1]]
                                                          [lookupIndex[floorCeil[2]][2]]
                                                          [lookupIndex[floorCeil[3]][3]]
                                                          [motor][axis];
                            }
                        }
                    vertexIndex++;
                }
            }
        }
    }

    return model;
}

MotorOutputs calculate_feedback(const InterpolatedModel *model, GyroRates proportionalErrors, GyroRates integralErrors, GyroRates derivativeErrors) {
    MotorOutputs motorOutputs = {.volts = {0, 0, 0, 0}};
    
    // Reset P, I, and D control voltages for this iteration
    pControl = (MotorOutputs){.volts = {0, 0, 0, 0}};
    iControl = (MotorOutputs){.volts = {0, 0, 0, 0}};
    dControl = (MotorOutputs){.volts = {0, 0, 0, 0}};

    for (int motor = 0; motor < 4; motor++) {
        // Integral feedback: volts per rotation
        for (int axis = 0; axis < 3; axis++) {
            float iVoltage = model->KiVoltPerRotation[motor][axis] * integralErrors.rotationsPerSecond[axis];
            motorOutputs.volts[motor] += iVoltage;
            iControl.volts[motor] += iVoltage;
        }
        
        // Derivative feedback: volts per rotation per second squared
        for (int axis = 0; axis < 3; axis++) {
            float dVoltage = model->KdVoltPerRotationPerSecondSquared[motor][axis] * derivativeErrors.rotationsPerSecond[axis];
            motorOutputs.volts[motor] += dVoltage;
            dControl.volts[motor] += dVoltage;
        }
        
        // Rate feedback: volts per rotation per second
        for (int axis = 0; axis < 3; axis++) {
            float pVoltage = model->KpVoltPerRotationPerSecond[motor][axis] * proportionalErrors.rotationsPerSecond[axis];
            motorOutputs.volts[motor] += pVoltage;
            pControl.volts[motor] += pVoltage;
        }
    }

    return motorOutputs;
}

void reset_integral_states(void) {
    // Reset integral and derivative states to zero
    for (int axis = 0; axis < 3; axis++) {
        integralErrors.rotationsPerSecond[axis] = 0.0f;
        derivativeErrors.rotationsPerSecond[axis] = 0.0f;
    }
}

MotorOutputs calculate_feedforward(const InterpolatedModel *model, GyroRates setpointDiff) {
    MotorOutputs motorOutputs = {.volts = {0, 0, 0, 0}};

    for (int motor = 0; motor < 4; motor++) {
        for (int axis = 0; axis < 3; axis++) {
            motorOutputs.volts[motor] += model->BDiscreteInvVoltSecondPerRotation[motor][axis]
                                         * setpointDiff.rotationsPerSecond[axis];
        }
    }

    return motorOutputs;
}

MotorOutputs calculate_affine_linearizer(MotorSpeeds motorSpeeds) {
    MotorOutputs motorOutputs = {.volts = {0, 0, 0, 0}};

    float affineIntercept[7];

    affineIntercept[0] = attitudeModel.affineInterceptPartial[0]
                         * (-signed_square(motorSpeeds.rotationsPerSecond[0])
                            + signed_square(motorSpeeds.rotationsPerSecond[1])
                            + signed_square(motorSpeeds.rotationsPerSecond[2])
                            - signed_square(motorSpeeds.rotationsPerSecond[3]));

    affineIntercept[1] = attitudeModel.affineInterceptPartial[1]
                         * (-attitudeModel.dFront * (signed_square(motorSpeeds.rotationsPerSecond[0])
                                                     + signed_square(motorSpeeds.rotationsPerSecond[1]))
                            + attitudeModel.dBack * (signed_square(motorSpeeds.rotationsPerSecond[2])
                                                     + signed_square(motorSpeeds.rotationsPerSecond[3])));

    affineIntercept[2] = attitudeModel.affineInterceptPartial[2]
                         * (-attitudeModel.dLeft * (signed_square(motorSpeeds.rotationsPerSecond[0])
                                                    + signed_square(motorSpeeds.rotationsPerSecond[2]))
                            + attitudeModel.dRight * (signed_square(motorSpeeds.rotationsPerSecond[1])
                                                      + signed_square(motorSpeeds.rotationsPerSecond[3])));

    for (int motor = 0; motor < 4; motor++) {
        affineIntercept[motor + 3] =
                attitudeModel.affineInterceptPartial[motor + 3] * signed_square(motorSpeeds.rotationsPerSecond[motor]);
    }

    for (int motor = 0; motor < 4; motor++) {
        for (int i = 0; i < 7; i++) {
            motorOutputs.volts[motor] +=
                    affineIntercept[i] * attitudeModel.BContinuousInvVoltSecondSquaredPerRotation[motor][i];
        }
    }

    return motorOutputs;
}

MotorOutputs calculateAttitudeControl(void) {
    // Reset LQR integral states when disarmed
    if (!ARMING_FLAG(ARMED)) {
        reset_integral_states();
    }
    
    measurements = (GyroRates) {
        .rotationsPerSecond = {
            - (double) gyro.gyroADCf[2] / 360.,
            (double) gyro.gyroADCf[1] / 360.,
            - (double) gyro.gyroADCf[0] / 360.
        }
    };

    setpoints = (GyroRates) {
        .rotationsPerSecond = {
            - (double) getSetpointRate(2) / 360.,
            (double) getSetpointRate(1) / 360.,
            - (double) getSetpointRate(0) / 360.
        }
    };

    // Update PID setpoints for blackbox logging
    updateBlackboxSetpoints(&setpoints);

    const float maxLookupRPS = lqrLookupMinRps + lqrLookupStepSize * lqrLookupMaxIndex;

    for (int motor = 0; motor < 4; motor++) {
        motorSpeeds.rotationsPerSecond[motor] = constrainf(getMotorFrequencyHz(motor), lqrLookupMinRps, maxLookupRPS);
    }

    interpolatedModel = lerp_lqr(motorSpeeds);

    // Leaky integrator constant
    const float alpha = dt / integralTimeConstant;
    // Derivative filter constant
    const float alpha_deriv = dt / derivativeTimeConstant;

    for (int axis = 0; axis < 3; axis++) {
        proportionalErrors.rotationsPerSecond[axis] = setpoints.rotationsPerSecond[axis] - measurements.rotationsPerSecond[axis];
        setpointDiff.rotationsPerSecond[axis] = setpoints.rotationsPerSecond[axis] - prevSetpoints.rotationsPerSecond[axis];
        integralErrors.rotationsPerSecond[axis] = constrainf(
            (1.0f - alpha) * integralErrors.rotationsPerSecond[axis] + alpha * proportionalErrors.rotationsPerSecond[axis],
            -integralWindupLimitRotations,
            integralWindupLimitRotations
        );
        
        // Calculate derivative error using leaky derivative filter
        // This matches the derivative filter in the Python script
        derivativeErrors.rotationsPerSecond[axis] = (1.0f - alpha_deriv) * derivativeErrors.rotationsPerSecond[axis] 
                                                   + alpha_deriv * (proportionalErrors.rotationsPerSecond[axis] - (measurements.rotationsPerSecond[axis] - prevMeasurements.rotationsPerSecond[axis]) / dt);
    }

    feedback = calculate_feedback(&interpolatedModel, proportionalErrors, integralErrors, derivativeErrors);

    feedforward = calculate_feedforward(&interpolatedModel, setpointDiff);

    affine = calculate_affine_linearizer(motorSpeeds);

    for (int motor = 0; motor < 4; motor++) {
        attitudeControl.volts[motor] = feedback.volts[motor] + feedforward.volts[motor] + affine.volts[motor];
    }

    // Update LQR internal state for blackbox logging
    updateBlackboxLqrState();

    prevSetpoints = setpoints;
    prevMeasurements = measurements;

    return attitudeControl;
}
