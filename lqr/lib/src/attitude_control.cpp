#include "../include/attitude_control.hpp"
#include "../include/cpp_lqr.h"  // for C types
#include <iostream>
#include <cstring>  // for std::memcpy

namespace lqr {

// Global state for LQR controller
std::unique_ptr<GainLookup> g_lqr_lut;
std::unique_ptr<LqrCoefficients> g_lqr_coefficients;

namespace internal {

bool init_lqr_controller(const CLqrCoefficients& coefficients) {
    try {
        // Convert C coefficients to C++ coefficients
        g_lqr_coefficients = std::make_unique<LqrCoefficients>(from_c_lqr_coefficients(coefficients));
        
        // Allocate LUT on heap with unit-aware matrix types
        g_lqr_lut = std::make_unique<GainLookup>();
        
        // Generate LUT with unit-aware coefficients
        generate_lut(*g_lqr_lut, coefficients);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing LQR controller: " << e.what() << std::endl;
        return false;
    }
}

MotorOutputs calculate_attitude_control() {
    if (!g_lqr_lut) {
        // Not initialized, return zero outputs
        return MotorOutputs{};
    }
    
    // TODO: Read gyro measurements and setpoints from existing C code
    // For now, return dummy values
    MotorOutputs outputs;
    for (int i = 0; i < 4; ++i) {
        outputs.volts[i] = 0.0 * si::volt;
    }
    
    return outputs;
}

void reset_integral_states() {
    // TODO: Reset integral states for the LQR controller
    // This would typically involve resetting any accumulated error states
}

bool test_interpolation(const CMotorSpeeds& speeds, CMotorOutputs& result) {
    if (!g_lqr_lut) {
        return false; // Not initialized
    }
    
    try {
        MotorSpeeds motor_speeds = from_c_motor_speeds(speeds);
        InterpolatedModel interpolated = lerp_lqr(motor_speeds, *g_lqr_lut);
        
        // Convert interpolated model to motor outputs (simplified)
        MotorOutputs output;
        for (int i = 0; i < 4; ++i) {
            output.volts[i] = 0.0 * si::volt; // TODO: Calculate from interpolated gains
        }
        
        result = to_c_motor_outputs(output);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in test_interpolation: " << e.what() << std::endl;
        return false;
    }
}

} // namespace internal

} // namespace lqr

// C ABI function implementations
int cpp_lqr_init(const LqrCoefficients* coefficients) {
    if (!coefficients) {
        return -1; // Invalid parameters
    }
    
    // Convert C types to lqr namespace types
    lqr::CLqrCoefficients c_coeffs;
    std::memcpy(&c_coeffs, coefficients, sizeof(LqrCoefficients));
    
    return lqr::internal::init_lqr_controller(c_coeffs) ? 0 : -1;
}

MotorOutputs cpp_lqr_calculate_attitude_control(void) {
    lqr::MotorOutputs outputs = lqr::internal::calculate_attitude_control();
    lqr::CMotorOutputs c_outputs = lqr::to_c_motor_outputs(outputs);
    
    // Convert back to C types
    MotorOutputs result;
    std::memcpy(&result, &c_outputs, sizeof(MotorOutputs));
    return result;
}

void cpp_lqr_reset_integral_states(void) {
    lqr::internal::reset_integral_states();
}

int cpp_lqr_test_interpolation(const MotorSpeeds* speeds, MotorOutputs* result) {
    if (!lqr::g_lqr_lut || !speeds || !result) {
        return -1; // Not initialized or invalid parameters
    }
    
    // Convert C types to lqr namespace types
    lqr::CMotorSpeeds c_speeds;
    std::memcpy(&c_speeds, speeds, sizeof(MotorSpeeds));
    
    lqr::CMotorOutputs c_result;
    bool success = lqr::internal::test_interpolation(c_speeds, c_result);
    
    if (success) {
        // Convert back to C types
        std::memcpy(result, &c_result, sizeof(MotorOutputs));
    }
    
    return success ? 0 : -1;
} 