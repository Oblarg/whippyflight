#pragma once

#include "types.hpp"
#include "units.hpp"
#include "gain_lookup.hpp"
#include "lqr_solver.hpp"
#include <vector>
#include <memory>
#include <cstddef>  // for size_t
#include <optional> // for std::optional
#include <tuple>    // for std::tuple
#include <array>    // for std::array>
#include <Eigen/Core>

namespace lqr {

// Matrix dimensions for gain generation
constexpr std::size_t GAIN_ROWS = 4;
constexpr std::size_t GAIN_COLS = 3;

// Matrix dimensions for linearization (7-state attitude model)
constexpr std::size_t LINEARIZATION_STATES = 7;
constexpr std::size_t LINEARIZATION_INPUTS = 4;

// Matrix types for linearization (using float for internal calculations)
using LinearizationStateMatrix = Eigen::Matrix<float, LINEARIZATION_STATES, LINEARIZATION_STATES>;
using LinearizationInputMatrix = Eigen::Matrix<float, LINEARIZATION_STATES, LINEARIZATION_INPUTS>;

// Linearized model structure
struct LinearizedModel {
    LinearizationStateMatrix a_continuous;  // 7x7 continuous A matrix
    LinearizationInputMatrix b_continuous;  // 7x4 continuous B matrix
    StateMatrix a_discrete;    // 16x16 discrete A matrix (augmented)
    InputMatrix b_discrete;    // 16x4 discrete B matrix (augmented)
    KiMatrix k_integral;       // 4x3 integral gains
    KdMatrix k_derivative;     // 4x3 derivative gains
    KpMatrix k_proportional;   // 4x3 proportional gains
};

// Main functions
void generate_lut(GainLookup& lut, const CLqrCoefficients& coeffs);
auto create_linearized_model(const std::array<float, 4>& motor_speeds, const CLqrCoefficients& coeffs) -> LinearizedModel;

// Matrix creation helpers (matching Python make_cost_matrix pattern)
auto create_state_cost_matrix(const std::array<float, STATES>& elements) -> StateMatrix;
auto create_input_cost_matrix(const std::array<float, INPUTS>& elements) -> Eigen::Matrix<float, INPUTS, INPUTS>;

} // namespace lqr 