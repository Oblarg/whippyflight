#pragma once

#include "lqr_controller.h"

// Function to update PID setpoints for blackbox logging
void updateBlackboxSetpoints(const GyroRates *setpoints);

// Function to wire up LQR internal state to PID handles for blackbox logging
void updateBlackboxLqrState(void); 