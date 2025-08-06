#pragma once

#include <optional>
#include <Eigen/Core>

namespace lqr {

// Matrix dimensions for LQR solver
constexpr std::size_t STATES = 16;
constexpr std::size_t INPUTS = 4;

// Matrix types for LQR solver (using float for internal calculations)
using StateMatrix = Eigen::Matrix<float, STATES, STATES>;
using InputMatrix = Eigen::Matrix<float, STATES, INPUTS>;
using GainMatrix = Eigen::Matrix<float, INPUTS, STATES>;

// LQR solver class
class LqrSolver {
public:
    // Solve Discrete Algebraic Riccati Equation (DARE)
    static auto solve_discrete_are(
        const StateMatrix& a,
        const InputMatrix& b,
        const StateMatrix& q,
        const Eigen::Matrix<float, INPUTS, INPUTS>& r
    ) -> std::optional<StateMatrix>;
    
    // Compute optimal LQR gain matrix from DARE solution
    static auto compute_lqr_gains(
        const StateMatrix& a,
        const InputMatrix& b,
        const StateMatrix& s,
        const Eigen::Matrix<float, INPUTS, INPUTS>& r
    ) -> std::optional<GainMatrix>;
    
    // Complete LQR solution: solve DARE and compute gains
    static auto solve_lqr(
        const StateMatrix& a,
        const InputMatrix& b,
        const StateMatrix& q,
        const Eigen::Matrix<float, INPUTS, INPUTS>& r
    ) -> std::optional<GainMatrix>;
};



} // namespace lqr 