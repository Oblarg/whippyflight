#pragma once

#include "types.hpp"
#include "units.hpp"
#include <vector>
#include <cstddef>  // for size_t
#include <tuple>    // for std::tuple
#include <array>    // for std::array>

namespace lqr {

// Gain lookup table that provides 4D indexing while maintaining flat storage
struct GainLookupFlat {
    std::vector<KiMatrix> ki_gains;           // 625 entries
    std::vector<KdMatrix> kd_gains;           // 625 entries
    std::vector<KpMatrix> kp_gains;           // 625 entries
    std::vector<BInvMatrix> binv_gains;       // 625 entries
};

class GainLookup {
public:
    GainLookup();
    
    // Indexing functions
    std::size_t index(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const;
    std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> rps_to_indices(float fl_rps, float bl_rps, float fr_rps, float br_rps) const;
    
    // Matrix access functions
    const KiMatrix& get_ki_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const;
    const KiMatrix& get_ki_matrix(const std::array<std::size_t, 4>& indices) const;
    void set_ki_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KiMatrix& matrix);
    void set_ki_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KiMatrix& matrix);
    
    const KdMatrix& get_kd_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const;
    const KdMatrix& get_kd_matrix(const std::array<std::size_t, 4>& indices) const;
    void set_kd_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KdMatrix& matrix);
    void set_kd_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KdMatrix& matrix);
    
    const KpMatrix& get_kp_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const;
    const KpMatrix& get_kp_matrix(const std::array<std::size_t, 4>& indices) const;
    void set_kp_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const KpMatrix& matrix);
    void set_kp_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const KpMatrix& matrix);
    
    const BInvMatrix& get_binv_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br) const;
    const BInvMatrix& get_binv_matrix(const std::array<std::size_t, 4>& indices) const;
    void set_binv_matrix(std::size_t fl, std::size_t bl, std::size_t fr, std::size_t br, const BInvMatrix& matrix);
    void set_binv_matrix_rps(float fl_rps, float bl_rps, float fr_rps, float br_rps, const BInvMatrix& matrix);
    
    // Access to flat storage
    const GainLookupFlat& as_flat() const { return flat; }
    GainLookupFlat& as_flat_mut() { return flat; }

private:
    GainLookupFlat flat;
};

} // namespace lqr 