#pragma once

#include "types.hpp"
#include "gain_generation.hpp"
#include "gain_interpolation.hpp"
#include <memory>

namespace lqr {

// Global state for LQR controller
extern std::unique_ptr<GainLookup> g_lqr_lut;
extern std::unique_ptr<LqrCoefficients> g_lqr_coefficients;

// C++ interface functions
namespace internal {
    // Initialize the LQR controller with coefficients
    bool init_lqr_controller(const CLqrCoefficients& coefficients);
    
    // Calculate attitude control outputs
    MotorOutputs calculate_attitude_control();
    
    // Reset integral states
    void reset_integral_states();
    
    // Test interpolation with given motor speeds
    bool test_interpolation(const CMotorSpeeds& speeds, CMotorOutputs& result);
}

} // namespace lqr 