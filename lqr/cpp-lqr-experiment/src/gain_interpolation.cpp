#include "../include/gain_interpolation.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <Eigen/Core>

namespace lqr {

struct GridIndicesAndFractions {
    std::array<int, 4> indices;
    std::array<float, 4> fractions;
};

auto compute_grid_indices_and_fractions(
    const std::array<RotationsPerSecond, 4>& query_points) -> GridIndicesAndFractions {
    
    GridIndicesAndFractions result{};
    
    // Convert from rotations per second and compute grid indices
    for (int i = 0; i < 4; ++i) {
        float rps = query_points[i].numerical_value_in(RotationsPerSecond::unit);
        
        // Compute grid index based on lookup table parameters
        float grid_idx = (rps - MIN_RPS) / RPS_STEP;
        result.indices[i] = static_cast<int>(std::floor(grid_idx));
        result.indices[i] = std::max(0, std::min(result.indices[i], static_cast<int>(RESOLUTION) - 2));  // Clamp to valid range
        
        // Compute fractional position within the grid cell
        result.fractions[i] = grid_idx - result.indices[i];
    }
    
    return result;
}

/**
 * Iterates through all 16 vertices of a 4D hypercube for multilinear interpolation.
 * 
 * In a 4D interpolation grid, each query point falls within a hypercube defined by
 * 16 vertices (2^4 = 16). Each vertex represents a combination of floor/ceil values
 * for the 4 motor speeds:
 * 
 * Vertex mapping (vertex_idx -> [FL, BL, FR, BR] where 0=floor, 1=ceil):
 *   0:  [0,0,0,0] - All motors at floor values
 *   1:  [0,0,0,1] - FL,BL,FR at floor, BR at ceil
 *   2:  [0,0,1,0] - FL,BL,BR at floor, FR at ceil
 *   3:  [0,0,1,1] - FL,BL at floor, FR,BR at ceil
 *   4:  [0,1,0,0] - FL,FR,BR at floor, BL at ceil
 *   5:  [0,1,0,1] - FL,FR at floor, BL,BR at ceil
 *   6:  [0,1,1,0] - FL,BR at floor, BL,FR at ceil
 *   7:  [0,1,1,1] - FL at floor, BL,FR,BR at ceil
 *   8:  [1,0,0,0] - BL,FR,BR at floor, FL at ceil
 *   9:  [1,0,0,1] - BL,FR at floor, FL,BR at ceil
 *   10: [1,0,1,0] - BL,BR at floor, FL,FR at ceil
 *   11: [1,0,1,1] - BL at floor, FL,FR,BR at ceil
 *   12: [1,1,0,0] - FR,BR at floor, FL,BL at ceil
 *   13: [1,1,0,1] - FR at floor, FL,BL,BR at ceil
 *   14: [1,1,1,0] - BR at floor, FL,BL,FR at ceil
 *   15: [1,1,1,1] - All motors at ceil values
 * 
 * The callback receives the vertex index and an array of 4 flags indicating
 * whether each motor uses floor (0) or ceil (1) values for interpolation.
 */
template<typename F>
void iterate_4d_vertices(F&& callback) {
    for (std::size_t vertex_idx = 0; vertex_idx < 16; ++vertex_idx) {
        std::size_t fl = (vertex_idx >> 3) & 1;  // Extract bit 3: front-left (0 or 1)
        std::size_t bl = (vertex_idx >> 2) & 1;  // Extract bit 2: back-left (0 or 1)
        std::size_t fr = (vertex_idx >> 1) & 1;  // Extract bit 1: front-right (0 or 1)
        std::size_t br = vertex_idx & 1;         // Extract bit 0: back-right (0 or 1)
        
        std::array<std::size_t, 4> floor_ceil_flags = {fl, bl, fr, br};
        callback(vertex_idx, floor_ceil_flags);
    }
}

auto compute_interpolation_weights(
    const std::array<float, 4>& fractions) -> std::array<float, 16> {
    
    // Use Eigen vector for atomic weight computation and normalization
    Eigen::VectorXf weights(16);
    
    iterate_4d_vertices([&](std::size_t vertex_idx, const std::array<std::size_t, 4>& floor_ceil_flags) {
        float weight = 1.0f;
        for (int motor = 0; motor < 4; ++motor) {
            float frac = fractions[motor];
            float distance = (floor_ceil_flags[motor] == 0) ? (1.0f - frac) : frac;
            weight *= distance;
        }
        
        weights(vertex_idx) = weight;
    });
    
    // Atomic normalization using Eigen's sum() and array() operations
    float sum_weights = weights.sum();
    if (sum_weights > 0.0f) {
        weights = weights / sum_weights;
    }
    
    // Convert to std::array inline using std::copy (most idiomatic C++ approach)
    std::array<float, 16> result{};
    std::copy_n(weights.data(), 16, result.begin());
    return result;
}

auto lerp_lqr(const MotorSpeeds& motor_speeds, const GainLookup& lut) -> InterpolatedModel {
    auto [grid_indices, interpolation_fractions] = compute_grid_indices_and_fractions(motor_speeds.rotations_per_second);
    
    std::array<float, 16> weights = compute_interpolation_weights(interpolation_fractions);
    
    KiMatrix ki_result = KiMatrix::Constant(VoltPerRotation::zero());
    KdMatrix kd_result = KdMatrix::Constant(VoltSecondSquaredPerRotation::zero());
    KpMatrix kp_result = KpMatrix::Constant(VoltSecondPerRotation::zero());
    BInvMatrix binv_result = BInvMatrix::Constant(VoltPerRotation::zero());
    
    iterate_4d_vertices([&](std::size_t vertex_idx, const std::array<std::size_t, 4>& floor_ceil_flags) {
        std::array<std::size_t, 4> lookup_indices = {
            std::clamp(grid_indices[0] + floor_ceil_flags[0], std::size_t{0}, RESOLUTION - 1),
            std::clamp(grid_indices[1] + floor_ceil_flags[1], std::size_t{0}, RESOLUTION - 1),
            std::clamp(grid_indices[2] + floor_ceil_flags[2], std::size_t{0}, RESOLUTION - 1),
            std::clamp(grid_indices[3] + floor_ceil_flags[3], std::size_t{0}, RESOLUTION - 1)
        };
        
        float weight = weights[vertex_idx];
        
        ki_result += weight * lut.get_ki_matrix(lookup_indices);
        kd_result += weight * lut.get_kd_matrix(lookup_indices);
        kp_result += weight * lut.get_kp_matrix(lookup_indices);
        binv_result += weight * lut.get_binv_matrix(lookup_indices);
    });
    
    return InterpolatedModel{.ki=ki_result, .kd=kd_result, .kp=kp_result, .binv=binv_result};
}

} // namespace lqr 