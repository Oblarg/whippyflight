#include "lqr_solver.hpp"
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

namespace lqr {

// LqrSolver implementation
std::optional<StateMatrix> LqrSolver::solve_discrete_are(
    const StateMatrix& a,
    const InputMatrix& b,
    const StateMatrix& q,
    const Eigen::Matrix<float, INPUTS, INPUTS>& r
) {
    // Implementation based on the reference DARE solver
    StateMatrix a_k = a;
    StateMatrix g_k = b * r.llt().solve(b.transpose());
    StateMatrix h_k;
    StateMatrix h_k1 = q;
    
    const float tolerance = 1e-10f;
    const int max_iterations = 100;
    int iteration = 0;
    
    do {
        h_k = h_k1;
        
        // W = I + G_k * H_k
        StateMatrix w = StateMatrix::Identity() + g_k * h_k;
        
        // Solve W * V_1 = A_k and W * V_2 = G_k^T
        auto w_solver = w.lu();
        StateMatrix v_1 = w_solver.solve(a_k);
        StateMatrix v_2 = w_solver.solve(g_k.transpose()).transpose();
        
        // Update G_k and H_k
        g_k += a_k * v_2 * a_k.transpose();
        h_k1 = h_k + v_1.transpose() * h_k * a_k;
        a_k *= v_1;
        
        iteration++;
    } while ((h_k1 - h_k).norm() > tolerance * h_k1.norm() && iteration < max_iterations);
    
    if (iteration >= max_iterations) {
        return std::nullopt; // Failed to converge
    }
    
    return h_k1;
}

std::optional<GainMatrix> LqrSolver::compute_lqr_gains(
    const StateMatrix& a,
    const InputMatrix& b,
    const StateMatrix& s,
    const Eigen::Matrix<float, INPUTS, INPUTS>& r
) {
    // K = (BᵀSB + R)⁻¹BᵀSA
    // This matches the reference implementation exactly
    GainMatrix k = (b.transpose() * s * b + r).llt().solve(b.transpose() * s * a);
    return k;
}

std::optional<GainMatrix> LqrSolver::solve_lqr(
    const StateMatrix& a,
    const InputMatrix& b,
    const StateMatrix& q,
    const Eigen::Matrix<float, INPUTS, INPUTS>& r
) {
    auto s = solve_discrete_are(a, b, q, r);
    if (!s) return std::nullopt;
    return compute_lqr_gains(a, b, *s, r);
}



} // namespace lqr 