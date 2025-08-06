#include "gain_lookup.hpp"
#include "units.hpp"
#include <algorithm>
#include <cmath>

namespace lqr {

// Constants for LUT generation
constexpr float MIN_RPS = 0.0f;
constexpr float MAX_RPS = 100.0f;
constexpr float RPS_STEP = 25.0f;
constexpr std::size_t RESOLUTION = 5; // 5^4 = 625 LUT entries

// GainLookup implementation
GainLookup::GainLookup() {
    // Initialize all vectors with the correct size (5^4 = 625)
    flat.ki_gains.resize(RESOLUTION * RESOLUTION * RESOLUTION * RESOLUTION);
    flat.kd_gains.resize(RESOLUTION * RESOLUTION * RESOLUTION * RESOLUTION);
    flat.kp_gains.resize(RESOLUTION * RESOLUTION * RESOLUTION * RESOLUTION);
    flat.binv_gains.resize(RESOLUTION * RESOLUTION * RESOLUTION * RESOLUTION);
    
    // Initialize all matrices to zero using Eigen's Constant with proper unit-aware zeros
    for (auto& matrix : flat.ki_gains) {
        matrix = KiMatrix::Constant(VoltPerRotation::zero());
    }
    for (auto& matrix : flat.kd_gains) {
        matrix = KdMatrix::Constant(VoltSecondSquaredPerRotation::zero());
    }
    for (auto& matrix : flat.kp_gains) {
        matrix = KpMatrix::Constant(VoltSecondPerRotation::zero());
    }
    for (auto& matrix : flat.binv_gains) {
        matrix = BInvMatrix::Constant(VoltPerRotation::zero());
    }
}

std::size_t GainLookup::index(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const {
    return fl * RESOLUTION * RESOLUTION * RESOLUTION + 
           bl * RESOLUTION * RESOLUTION + 
           fr * RESOLUTION + 
           br;
}

std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> GainLookup::rps_to_indices(float fl_rps, float bl_rps, float fr_rps, float br_rps) const {
    std::size_t fl_idx = std::min(static_cast<std::size_t>((fl_rps - MIN_RPS) / RPS_STEP), RESOLUTION - 1);
    std::size_t bl_idx = std::min(static_cast<std::size_t>((bl_rps - MIN_RPS) / RPS_STEP), RESOLUTION - 1);
    std::size_t fr_idx = std::min(static_cast<std::size_t>((fr_rps - MIN_RPS) / RPS_STEP), RESOLUTION - 1);
    std::size_t br_idx = std::min(static_cast<std::size_t>((br_rps - MIN_RPS) / RPS_STEP), RESOLUTION - 1);
    return {fl_idx, bl_idx, fr_idx, br_idx};
}

const KiMatrix& GainLookup::get_ki_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const {
    return flat.ki_gains[index(fl, bl, fr, br)];
}

const KiMatrix& GainLookup::get_ki_matrix(const std::array<std::size_t, 4>& indices) const {
    return flat.ki_gains[index(indices[0], indices[1], indices[2], indices[3])];
}

void GainLookup::set_ki_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KiMatrix& matrix) {
    flat.ki_gains[index(fl, bl, fr, br)] = matrix;
}

void GainLookup::set_ki_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KiMatrix& matrix) {
    auto [fl, bl, fr, br] = rps_to_indices(fl_rps, bl_rps, fr_rps, br_rps);
    set_ki_matrix(fl, bl, fr, br, matrix);
}

const KdMatrix& GainLookup::get_kd_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const {
    return flat.kd_gains[index(fl, bl, fr, br)];
}

const KdMatrix& GainLookup::get_kd_matrix(const std::array<std::size_t, 4>& indices) const {
    return flat.kd_gains[index(indices[0], indices[1], indices[2], indices[3])];
}

void GainLookup::set_kd_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KdMatrix& matrix) {
    flat.kd_gains[index(fl, bl, fr, br)] = matrix;
}

void GainLookup::set_kd_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KdMatrix& matrix) {
    auto [fl, bl, fr, br] = rps_to_indices(fl_rps, bl_rps, fr_rps, br_rps);
    set_kd_matrix(fl, bl, fr, br, matrix);
}

const KpMatrix& GainLookup::get_kp_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const {
    return flat.kp_gains[index(fl, bl, fr, br)];
}

const KpMatrix& GainLookup::get_kp_matrix(const std::array<std::size_t, 4>& indices) const {
    return flat.kp_gains[index(indices[0], indices[1], indices[2], indices[3])];
}

void GainLookup::set_kp_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KpMatrix& matrix) {
    flat.kp_gains[index(fl, bl, fr, br)] = matrix;
}

void GainLookup::set_kp_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KpMatrix& matrix) {
    auto [fl, bl, fr, br] = rps_to_indices(fl_rps, bl_rps, fr_rps, br_rps);
    set_kp_matrix(fl, bl, fr, br, matrix);
}

const BInvMatrix& GainLookup::get_binv_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const {
    return flat.binv_gains[index(fl, bl, fr, br)];
}

const BInvMatrix& GainLookup::get_binv_matrix(const std::array<std::size_t, 4>& indices) const {
    return flat.binv_gains[index(indices[0], indices[1], indices[2], indices[3])];
}

void GainLookup::set_binv_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const BInvMatrix& matrix) {
    flat.binv_gains[index(fl, bl, fr, br)] = matrix;
}

void GainLookup::set_binv_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const BInvMatrix& matrix) {
    auto [fl, bl, fr, br] = rps_to_indices(fl_rps, bl_rps, fr_rps, br_rps);
    set_binv_matrix(fl, bl, fr, br, matrix);
}

} // namespace lqr 