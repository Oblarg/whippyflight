#include "gain_generation.hpp"
#include <iostream>
#include <Eigen/Dense>

int main() {
    using namespace lqr;
    
    // Simple test case: 2x2 system
    StateMatrix a = StateMatrix::Identity();
    a(0, 1) = 0.1; // Simple coupling
    
    InputMatrix b = InputMatrix::Zero();
    b(0, 0) = 1.0;
    b(1, 1) = 1.0;
    b(2, 2) = 1.0;
    b(3, 3) = 1.0;
    
    StateMatrix q = StateMatrix::Identity();
    Eigen::Matrix<double, INPUTS, INPUTS> r = Eigen::Matrix<double, INPUTS, INPUTS>::Identity();
    
    std::cout << "Testing LQR solver..." << std::endl;
    
    // Test the standalone reference implementation
    try {
        GainMatrix k_ref = solve_lqr_reference(a, b, q, r);
        std::cout << "Reference LQR solver succeeded!" << std::endl;
        std::cout << "Gain matrix K (first 4x4 block):" << std::endl;
        std::cout << k_ref.block<4, 4>(0, 0) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Reference LQR solver failed: " << e.what() << std::endl;
    }
    
    // Test the class-based implementation
    try {
        auto k_opt = LqrSolver::solve_lqr(a, b, q, r);
        if (k_opt) {
            std::cout << "Class-based LQR solver succeeded!" << std::endl;
            std::cout << "Gain matrix K (first 4x4 block):" << std::endl;
            std::cout << k_opt->block<4, 4>(0, 0) << std::endl;
        } else {
            std::cout << "Class-based LQR solver failed to converge" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Class-based LQR solver failed: " << e.what() << std::endl;
    }
    
    return 0;
} 