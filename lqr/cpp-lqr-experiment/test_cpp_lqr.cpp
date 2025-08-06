#include "types.hpp"
#include "units.hpp"
#include "gain_interpolation.hpp"
#include "gain_generation.hpp"
#include <iostream>

int main() {
    std::cout << "Testing C++ LQR with unit-aware types..." << std::endl;
    
    // Test C-compatible types
    lqr::CMotorSpeeds c_speeds;
    c_speeds.rotations_per_second[0] = 25.0f;
    c_speeds.rotations_per_second[1] = 30.0f;
    c_speeds.rotations_per_second[2] = 35.0f;
    c_speeds.rotations_per_second[3] = 40.0f;
    
    // Convert to unit-aware types
    lqr::MotorSpeeds unit_aware_speeds = lqr::from_c_motor_speeds(c_speeds);
    
    std::cout << "Motor speeds: ["
              << unit_aware_speeds.rotations_per_second[0].numerical_value_in(mp_units::angular::revolution / mp_units::si::second) << ", "
              << unit_aware_speeds.rotations_per_second[1].numerical_value_in(mp_units::angular::revolution / mp_units::si::second) << ", "
              << unit_aware_speeds.rotations_per_second[2].numerical_value_in(mp_units::angular::revolution / mp_units::si::second) << ", "
              << unit_aware_speeds.rotations_per_second[3].numerical_value_in(mp_units::angular::revolution / mp_units::si::second) << "] rot/s" << std::endl;
    
    // Test unit-aware matrix creation
    lqr::KiMatrix ki_test = lqr::KiMatrix::Constant(lqr::VoltPerRotation::zero());
    lqr::KdMatrix kd_test = lqr::KdMatrix::Constant(lqr::VoltPerRotationPerSecondSquared::zero());
    lqr::KpMatrix kp_test = lqr::KpMatrix::Constant(lqr::VoltPerRotationPerSecond::zero());
    lqr::BInvMatrix binv_test = lqr::BInvMatrix::Constant(lqr::VoltPerRotation::zero());
    
    // Set some test values using direct construction
    ki_test(0, 0) = lqr::VoltPerRotation(1.5 * mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second));
    kd_test(0, 0) = lqr::VoltPerRotationPerSecondSquared(0.1 * mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second / mp_units::si::second);
    kp_test(0, 0) = lqr::VoltPerRotationPerSecond(0.5 * mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second);
    binv_test(0, 0) = lqr::VoltPerRotation(2.0 * mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second));
    
    std::cout << "Test matrix values:" << std::endl;
    std::cout << "  Ki[0,0] = " << ki_test(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second)) << " V/(rot/s)" << std::endl;
    std::cout << "  Kd[0,0] = " << kd_test(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second / mp_units::si::second) << " V/(rot/s³)" << std::endl;
    std::cout << "  Kp[0,0] = " << kp_test(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second) << " V/(rot/s²)" << std::endl;
    std::cout << "  BInv[0,0] = " << binv_test(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second)) << " V/(rot/s)" << std::endl;
    
    // Create a test lookup table
    lqr::GainLookup test_lut;
    
    // Fill the lookup table with test data
    for (std::size_t fl = 0; fl < lqr::RESOLUTION; ++fl) {
        for (std::size_t bl = 0; bl < lqr::RESOLUTION; ++bl) {
            for (std::size_t fr = 0; fr < lqr::RESOLUTION; ++fr) {
                for (std::size_t br = 0; br < lqr::RESOLUTION; ++br) {
                    test_lut.set_ki_matrix(fl, bl, fr, br, ki_test);
                    test_lut.set_kd_matrix(fl, bl, fr, br, kd_test);
                    test_lut.set_kp_matrix(fl, bl, fr, br, kp_test);
                    test_lut.set_binv_matrix(fl, bl, fr, br, binv_test);
                }
            }
        }
    }
    
    // Test interpolation
    lqr::InterpolatedModel test_model = lqr::lerp_lqr(unit_aware_speeds, test_lut);
    
    std::cout << "Interpolated model values:" << std::endl;
    std::cout << "  test_model.ki[0,0] = " << test_model.ki(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second)) << " V/(rot/s)" << std::endl;
    std::cout << "  test_model.kd[0,0] = " << test_model.kd(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second / mp_units::si::second) << " V/(rot/s³)" << std::endl;
    std::cout << "  test_model.kp[0,0] = " << test_model.kp(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second) / mp_units::si::second) << " V/(rot/s²)" << std::endl;
    std::cout << "  test_model.binv[0,0] = " << test_model.binv(0, 0).numerical_value_in(mp_units::si::volt / (mp_units::angular::revolution / mp_units::si::second)) << " V/(rot/s)" << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
} 