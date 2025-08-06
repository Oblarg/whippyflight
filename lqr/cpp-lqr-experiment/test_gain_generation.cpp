#include "gain_generation.hpp"
#include "gain_lookup.hpp"
#include "types.hpp"
#include <iostream>

int main() {
    // Create test coefficients
    lqr::CLqrCoefficients coeffs;
    
    // Motor coefficients
    coeffs.motorVel = 0.1f;
    coeffs.motorVelSq = 0.001f;
    coeffs.motorAccel = 10.0f;
    
    // Yaw coefficients
    coeffs.yawRate = -0.5f;
    coeffs.yawPropVelSumSq = 0.001f;
    coeffs.yawPropAccelSum = 0.1f;
    
    // Pitch coefficients
    coeffs.pitchRate = -0.5f;
    coeffs.pitchPropVelSumSq = 0.001f;
    coeffs.pitchDFront = 1.0f;
    coeffs.pitchDBack = 1.0f;
    
    // Roll coefficients
    coeffs.rollRate = -0.5f;
    coeffs.rollPropVelSumSq = 0.001f;
    coeffs.rollDLeft = 1.0f;
    coeffs.rollDRight = 1.0f;
    
    // Create lookup table
    lqr::GainLookup lut;
    
    // Generate the lookup table
    std::cout << "Generating LUT..." << std::endl;
    lqr::generate_lut(lut, coeffs);
    std::cout << "LUT generation complete!" << std::endl;
    
    // Test accessing some gains
    auto ki = lut.get_ki_matrix(0, 0, 0, 0);
    auto kd = lut.get_kd_matrix(0, 0, 0, 0);
    auto kp = lut.get_kp_matrix(0, 0, 0, 0);
    auto binv = lut.get_binv_matrix(0, 0, 0, 0);
    
    std::cout << "Successfully accessed gain matrices from LUT" << std::endl;
    std::cout << "KI matrix shape: " << ki.rows() << "x" << ki.cols() << std::endl;
    std::cout << "KD matrix shape: " << kd.rows() << "x" << kd.cols() << std::endl;
    std::cout << "KP matrix shape: " << kp.rows() << "x" << kp.cols() << std::endl;
    std::cout << "BINV matrix shape: " << binv.rows() << "x" << binv.cols() << std::endl;
    
    return 0;
} 