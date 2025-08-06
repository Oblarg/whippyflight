#include "gain_generation.hpp"
#include "gain_lookup.hpp"
#include "lqr_solver.hpp"
#include "units.hpp"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>
#include <limits>

namespace lqr {

// Constants for motor speed range
constexpr float MAX_MOTOR_SPEED = 1000.0f; // rotations per second

// Constants for state augmentation dimensions
constexpr std::size_t ATTITUDE_STATES = 3;      // roll, pitch, yaw attitude rates
constexpr std::size_t MOTOR_STATES = 4;         // fl, bl, fr, br motor speeds
constexpr std::size_t INTEGRAL_STATES = ATTITUDE_STATES;      // roll, pitch, yaw integral states
constexpr std::size_t DERIVATIVE_STATES = ATTITUDE_STATES;    // roll, pitch, yaw derivative states
constexpr std::size_t BUFFER_STATES = ATTITUDE_STATES;        // roll, pitch, yaw buffer states
constexpr std::size_t AUGMENTED_STATES = INTEGRAL_STATES + DERIVATIVE_STATES + BUFFER_STATES;     // integral + derivative + buffer states
constexpr std::size_t ORIGINAL_STATES = ATTITUDE_STATES + MOTOR_STATES;      // attitude + motor states

// State indices for the augmented system
constexpr std::size_t INTEGRAL_START = 0;       // integral states start index
constexpr std::size_t DERIVATIVE_START = 3;     // derivative states start index
constexpr std::size_t BUFFER_START = 6;         // buffer states start index
constexpr std::size_t ATTITUDE_START = 9;       // attitude states start index
constexpr std::size_t MOTOR_START = 12;         // motor states start index

// Type aliases for block matrices to improve readability
// Note: "to" indicates the direction of the *linear transformation*, not the order of the state parameters
// i.e. AttitudeToMotor is the linear transformation from 3 attitude states to 4 motor states,
// meaning 4 columns (inputs) and 3 rows (outputs).  The dimension ordering is (rows, columns) - opposite of the
// transformation direction.
using AttitudeMatrix = Eigen::Matrix<float, ATTITUDE_STATES, ATTITUDE_STATES>;
using MotorToAttitude = Eigen::Matrix<float, ATTITUDE_STATES, MOTOR_STATES>;
using AttitudeToMotor = Eigen::Matrix<float, MOTOR_STATES, ATTITUDE_STATES>;
using OriginalToAttitude = Eigen::Matrix<float, ATTITUDE_STATES, ORIGINAL_STATES>;
using AttitudeToOriginal = Eigen::Matrix<float, ORIGINAL_STATES, ATTITUDE_STATES>;
using OriginalMatrix = Eigen::Matrix<float, ORIGINAL_STATES, ORIGINAL_STATES>;
using MotorMatrix = Eigen::Matrix<float, MOTOR_STATES, MOTOR_STATES>;

// Main LUT generation function
auto generate_lut(GainLookup& lut, const CLqrCoefficients& coeffs) -> void {
    // Grid parameters (matching Python resolution)
    constexpr std::size_t resolution = 5;  // 5x5x5x5 = 625 total points
    constexpr float min_rps = 0.0f;
    constexpr float max_rps = 1000.0f;
    constexpr float rps_step = (max_rps - min_rps) / (resolution - 1);
    
    // Decay time parameters (matching Python)
    constexpr float integral_decay_time = 0.3f;
    constexpr float derivative_decay_time = 0.05f;
    
    // Cost weights (matching Python values)
    constexpr float qelm_deriv = 0.1f;    // RPS^2
    constexpr float qelm_int = 0.005f;    // Rotations  
    constexpr float qelm = 0.01f;         // RPS
    constexpr float relm = 0.8f;          // volt
    
    // Convert C coefficients to unit-aware types
    auto lqr_coeffs = from_c_lqr_coefficients(coeffs);
    
    // Iterate through 4D grid of motor speeds
    for (std::size_t fl_index = 0; fl_index < resolution; ++fl_index) {
        for (std::size_t bl_index = 0; bl_index < resolution; ++bl_index) {
            for (std::size_t fr_index = 0; fr_index < resolution; ++fr_index) {
                for (std::size_t br_index = 0; br_index < resolution; ++br_index) {
                    
                    // Calculate motor speeds for this grid point
                    float fl_rps = fl_index * rps_step + min_rps;
                    float bl_rps = bl_index * rps_step + min_rps;
                    float fr_rps = fr_index * rps_step + min_rps;
                    float br_rps = br_index * rps_step + min_rps;
                    
                    std::array<float, 4> motor_speeds = {fl_rps, bl_rps, fr_rps, br_rps};
                    
                    // Create linearized model at this operating point
                    auto linearized_model = create_linearized_model(motor_speeds, coeffs);
                    
                    // Create cost matrices for the augmented system (16 states total)
                    // State order: [integral_roll, integral_pitch, integral_yaw, 
                    //              derivative_roll, derivative_pitch, derivative_yaw,
                    //              buffer_roll, buffer_pitch, buffer_yaw,
                    //              attitude_rates, motor_speeds]
                    std::array<float, STATES> q_elements = {
                        qelm_int,           // integral_roll
                        qelm_int,           // integral_pitch
                        qelm_int,           // integral_yaw
                        qelm_deriv,         // derivative_roll
                        qelm_deriv,         // derivative_pitch
                        qelm_deriv,         // derivative_yaw
                        std::numeric_limits<float>::infinity(), // buffer_roll (no cost)
                        std::numeric_limits<float>::infinity(), // buffer_pitch (no cost)
                        std::numeric_limits<float>::infinity(), // buffer_yaw (no cost)
                        qelm,               // attitude_rate_roll
                        qelm,               // attitude_rate_pitch
                        qelm,               // attitude_rate_yaw
                        std::numeric_limits<float>::infinity(), // motor_speed_fl (no cost)
                        std::numeric_limits<float>::infinity(), // motor_speed_bl (no cost)
                        std::numeric_limits<float>::infinity(), // motor_speed_fr (no cost)
                        std::numeric_limits<float>::infinity()  // motor_speed_br (no cost)
                    };
                    
                    std::array<float, INPUTS> r_elements = {
                        relm, // fl_motor
                        relm, // bl_motor
                        relm, // fr_motor
                        relm  // br_motor
                    };
                    
                    // Create cost matrices
                    StateMatrix q_cost = create_state_cost_matrix(q_elements);
                    auto r_cost = create_input_cost_matrix(r_elements);
                    
                    // Solve LQR for the augmented system
                    auto lqr_result = LqrSolver::solve_lqr(
                        linearized_model.a_discrete, 
                        linearized_model.b_discrete, 
                        q_cost, 
                        r_cost
                    );
                    
                    if (lqr_result) {
                        // Extract gains for the 9 controllable states:
                        // States 0-2: integral gains
                        // States 3-5: derivative gains  
                        // States 9-11: proportional gains (attitude rates)
                        
                        const auto& k_full = *lqr_result;
                        
                        // Initialize gain matrices with zeros
                        KiMatrix ki = KiMatrix::Constant(VoltPerRotation::zero());
                        KdMatrix kd = KdMatrix::Constant(VoltSecondSquaredPerRotation::zero());
                        KpMatrix kp = KpMatrix::Constant(VoltSecondPerRotation::zero());
                        
                        // Extract integral gains (states 0-2)
                        for (int motor = 0; motor < 4; ++motor) {
                            for (int axis = 0; axis < 3; ++axis) {
                                ki(motor, axis) = VoltPerRotation(k_full(motor, axis) * si::volt / angular::revolution);
                            }
                        }
                        
                        // Extract derivative gains (states 3-5)
                        for (int motor = 0; motor < 4; ++motor) {
                            for (int axis = 0; axis < 3; ++axis) {
                                kd(motor, axis) = VoltSecondSquaredPerRotation(k_full(motor, axis + 3) * si::volt * si::second * si::second / angular::revolution);
                            }
                        }
                        
                        // Extract proportional gains (states 9-11)
                        for (int motor = 0; motor < 4; ++motor) {
                            for (int axis = 0; axis < 3; ++axis) {
                                kp(motor, axis) = VoltSecondPerRotation(k_full(motor, axis + 9) * si::volt * si::second / angular::revolution);
                            }
                        }
                        
                        // Store gains in lookup table
                        lut.set_ki_matrix(fl_index, bl_index, fr_index, br_index, ki);
                        lut.set_kd_matrix(fl_index, bl_index, fr_index, br_index, kd);
                        lut.set_kp_matrix(fl_index, bl_index, fr_index, br_index, kp);
                        
                        // Store B inverse matrix (pseudo-inverse of discrete B matrix)
                        // This is used for feedforward control
                        BInvMatrix binv = BInvMatrix::Constant(VoltPerRotation::zero());
                        auto b_pinv = linearized_model.b_discrete.completeOrthogonalDecomposition().pseudoInverse();
                        
                        // Extract the first 3 columns (for attitude control)
                        for (int motor = 0; motor < 4; ++motor) {
                            for (int axis = 0; axis < 3; ++axis) {
                                binv(motor, axis) = VoltPerRotation(b_pinv(motor, axis) * si::volt / angular::revolution);
                            }
                        }
                        
                        lut.set_binv_matrix(fl_index, bl_index, fr_index, br_index, binv);
                    }
                }
            }
        }
    }
}

// Helper function to discretize continuous system using zero-order hold
auto discretize_system(const LinearizationStateMatrix& a_continuous, 
                       const LinearizationInputMatrix& b_continuous, 
                       float dt) -> std::pair<LinearizationStateMatrix, LinearizationInputMatrix> {
    // For zero-order hold discretization:
    // A_d = exp(A_c * dt)
    // B_d = A_c^(-1) * (A_d - I) * B_c
    
    // Compute A_d = exp(A_c * dt) using Eigen's built-in matrix exponential
    LinearizationStateMatrix a_discrete = (a_continuous * dt).exp();
    
    // Compute B_d = A_c^(-1) * (A_d - I) * B_c
    LinearizationStateMatrix a_inv = a_continuous.inverse();
    LinearizationStateMatrix a_diff = a_discrete - LinearizationStateMatrix::Identity();
    LinearizationInputMatrix b_discrete = a_inv * a_diff * b_continuous;
    
    return {a_discrete, b_discrete};
}

// Helper function to create augmented system matrices
auto create_augmented_system(const LinearizationStateMatrix& a_discrete,
                             const LinearizationInputMatrix& b_discrete,
                             float dt,
                             float integral_decay_time,
                             float derivative_decay_time) -> std::pair<StateMatrix, InputMatrix> {
    
    // Create the augmented system with 16 states:
    // States 0-2:   Integral states (roll, pitch, yaw)
    // States 3-5:   Derivative states (roll, pitch, yaw) 
    // States 6-8:   Buffer states (previous attitude rates)
    // States 9-11:  Attitude rates (roll, pitch, yaw)
    // States 12-15: Motor speeds (fl, bl, fr, br)
    
    // Decay parameters
    float alpha = dt / integral_decay_time;
    float alpha_deriv = dt / derivative_decay_time;
    
    // Create block matrices for the augmented system
    // Integral dynamics: leaky integrator with decay
    auto integral_dynamics = AttitudeMatrix{
        {1.0f - alpha, 0.0f,         0.0f},        // integral_roll row
        {0.0f,         1.0f - alpha, 0.0f},        // integral_pitch row
        {0.0f,         0.0f,         1.0f - alpha} // integral_yaw row
    };
    
    // Integral coupling: integrates attitude rates (spans first 3 columns of a_discrete)
    auto integral_coupling = OriginalToAttitude{
        {alpha * dt, 0.0f,         0.0f,        0.0f, 0.0f, 0.0f, 0.0f},  // integral_roll coupling row
        {0.0f,       alpha * dt,   0.0f,        0.0f, 0.0f, 0.0f, 0.0f},  // integral_pitch coupling row
        {0.0f,       0.0f,         alpha * dt,  0.0f, 0.0f, 0.0f, 0.0f}   // integral_yaw coupling row
    };
    
    // Derivative dynamics: first-order difference with decay
    auto derivative_dynamics = AttitudeMatrix{
        {1.0f - alpha_deriv, 0.0f,               0.0f},        // derivative_roll row
        {0.0f,               1.0f - alpha_deriv, 0.0f},        // derivative_pitch row
        {0.0f,               0.0f,               1.0f - alpha_deriv} // derivative_yaw row
    };
    
    // Derivative coupling: current attitude rates (spans first 3 columns of a_discrete)
    auto derivative_coupling_current = OriginalToAttitude{
        {alpha_deriv / dt, 0.0f,              0.0f,              0.0f, 0.0f, 0.0f, 0.0f},  // derivative_roll current coupling row
        {0.0f,             alpha_deriv / dt,  0.0f,              0.0f, 0.0f, 0.0f, 0.0f},  // derivative_pitch current coupling row
        {0.0f,             0.0f,              alpha_deriv / dt,  0.0f, 0.0f, 0.0f, 0.0f}   // derivative_yaw current coupling row
    };
    
    auto derivative_coupling_buffer = AttitudeMatrix{
        {-alpha_deriv / dt, 0.0f,               0.0f},              // derivative_roll previous coupling row
        {0.0f,              -alpha_deriv / dt,  0.0f},              // derivative_pitch previous coupling row
        {0.0f,              0.0f,               -alpha_deriv / dt}  // derivative_yaw previous coupling row
    };
    
    // Buffer dynamics: copy from attitude rates
    auto buffer_dynamics = AttitudeMatrix::Identity(); // Copy attitude rates to buffer
    
    // Create the augmented A matrix using block notation
    // Structure: [integral_dynamics    0_3x3                0_3x3              integral_coupling]
    //            [0_3x3                derivative_dynamics  derivative_buffer  derivative_current]
    //            [0_3x3                0_3x3                buffer_dynamics    0_3x7]
    //            [0_7x3                0_7x3                0_7x3              a_discrete]
    
    // Create the augmented A matrix using comma initializer (Eigen doesn't support nested initializer lists)
    // Note: This only checks total element count, not individual block dimensions
    StateMatrix a_augmented = (StateMatrix() <<
        integral_dynamics,          AttitudeMatrix::Zero(),     AttitudeMatrix::Zero(),     integral_coupling,
        AttitudeMatrix::Zero(),     derivative_dynamics,        derivative_coupling_buffer, derivative_coupling_current,
        AttitudeMatrix::Zero(),     AttitudeMatrix::Zero(),     buffer_dynamics,            OriginalToAttitude::Zero(),
        AttitudeToOriginal::Zero(), AttitudeToOriginal::Zero(), AttitudeToOriginal::Zero(), a_discrete
    ).finished();
    
    // Create the augmented B matrix using initializer-list for full shape-checking
    auto b_augmented = (InputMatrix() <<
        MotorToAttitude::Zero(),    // integral states (3x4)
        MotorToAttitude::Zero(),    // derivative states (3x4)
        MotorToAttitude::Zero(),    // buffer states (3x4)
        b_discrete                  // original states (7x4)
    ).finished();
    
    return {a_augmented, b_augmented};
}

auto make_cost_matrices(auto sigma_deriv, auto sigma_int, auto sigma, auto sigma_input) -> std::pair<StateMatrix, Eigen::Matrix<float, INPUTS, INPUTS>> {
    // Extract numeric values from mp-units quantities and apply Bryson's rule: weight = 1/σ²
    float sigma_deriv_val = sigma_deriv.numerical_value_in(sigma_deriv.unit);
    float sigma_int_val = sigma_int.numerical_value_in(sigma_int.unit);
    float sigma_val = sigma.numerical_value_in(sigma.unit);
    float sigma_input_val = sigma_input.numerical_value_in(sigma_input.unit);
    
    // Apply Bryson's rule: Q_ii = 1/σ²
    float q_deriv = 1.0f / (sigma_deriv_val * sigma_deriv_val);
    float q_int = 1.0f / (sigma_int_val * sigma_int_val);
    float q_attitude = 1.0f / (sigma_val * sigma_val);
    float r_input = 1.0f / (sigma_input_val * sigma_input_val);
    
    // Create state cost elements (matching Python Q = [...] pattern)
    std::array<float, STATES> q_elements = {
        q_int,           // integral_roll
        q_int,           // integral_pitch
        q_int,           // integral_yaw
        q_deriv,         // derivative_roll
        q_deriv,         // derivative_pitch
        q_deriv,         // derivative_yaw
        std::numeric_limits<float>::infinity(), // buffer_roll (no cost)
        std::numeric_limits<float>::infinity(), // buffer_pitch (no cost)
        std::numeric_limits<float>::infinity(), // buffer_yaw (no cost)
        q_attitude,      // attitude_rate_roll
        q_attitude,      // attitude_rate_pitch
        q_attitude,      // attitude_rate_yaw
        std::numeric_limits<float>::infinity(), // motor_speed_fl (no cost)
        std::numeric_limits<float>::infinity(), // motor_speed_bl (no cost)
        std::numeric_limits<float>::infinity(), // motor_speed_fr (no cost)
        std::numeric_limits<float>::infinity()  // motor_speed_br (no cost)
    };
    
    // Create input cost elements (matching Python R = [...] pattern)
    std::array<float, INPUTS> r_elements = {
        r_input, // fl_motor
        r_input, // bl_motor
        r_input, // fr_motor
        r_input  // br_motor
    };
    
    StateMatrix q_cost = create_state_cost_matrix(q_elements);
    Eigen::Matrix<float, INPUTS, INPUTS> r_cost = create_input_cost_matrix(r_elements);
    
    return {q_cost, r_cost};
}

auto create_linearized_model(const std::array<float, 4>& motor_speeds, const CLqrCoefficients& coeffs) -> LinearizedModel {
    // Implementation of linearized quadcopter dynamics
    // See LaTeX source: https://www.overleaf.com/read/spqqbbpcvhwh#7fe96e
    
    // Extract motor speeds as raw floats
    float fl_rps = motor_speeds[0];
    float bl_rps = motor_speeds[1];
    float fr_rps = motor_speeds[2];
    float br_rps = motor_speeds[3];
    
    // Extract coefficients as raw floats
    float k_alpha = coeffs.motorAccel;
    float k_omega = coeffs.motorVel;
    float k_omega_sq = coeffs.motorVelSq;
    
    float k_I = coeffs.yawPropAccelSum;
    float k_tau = coeffs.yawPropVelSumSq;
    float k_rho_yaw = coeffs.yawRate;
    
    float k_T_pitch = coeffs.pitchPropVelSumSq;
    float k_rho_pitch = coeffs.pitchRate;
    float d_front = coeffs.pitchDFront;
    float d_back = coeffs.pitchDBack;
    
    float k_T_roll = coeffs.rollPropVelSumSq;
    float k_rho_roll = coeffs.rollRate;
    float d_left = coeffs.rollDLeft;
    float d_right = coeffs.rollDRight;
    
    // Helper functions for linearized quadcopter dynamics
    
    // Signed square: x * |x| (used in motor dynamics for quadratic terms)
    auto signed_square = [](float x) {
        return x * std::abs(x);
    };
    
    // Motor dynamics coefficient: kᵦ,ₘ = -(kω + 2kω²ω̂ₘ)/kₐ
    // From linearization of voltage equation: V = kω(ω) + kω²(ω²) + kₐ(ω̇)
    // Represents the linearized motor speed dynamics at operating point ω̂ₘ
    auto k_beta = [&](float omega_0) {
        return -(k_omega + 2.0f * k_omega_sq * omega_0) / k_alpha;
    };
    
    // Yaw coupling coefficient: kᵧ = 2kᵣω̂ₘ + kᵢkᵦ,ₘ
    // Combines rotor torque (kᵣ) and inertial coupling (kᵢ) effects
    // Represents how motor speed affects yaw rate through rotor torque and inertia
    auto k_gamma = [&](float omega_0) {
        return 2.0f * k_tau * omega_0 + k_I * k_beta(omega_0);
    };
    
    // Pitch coupling coefficient: kφ = 2kₜDₑω̂ₘ
    // Represents thrust-induced pitch moment from rotor displacement Dₑ
    // Dₑ = ||Dₑₗ × ŷ|| = ||Dₑᵣ × ŷ|| (front rotor displacement)
    auto k_phi = [&](float omega_0, float d = 1.0f) {
        return 2.0f * k_T_pitch * d * omega_0;
    };
    
    // Roll coupling coefficient: kθ = 2kₜDₗω̂ₘ  
    // Represents thrust-induced roll moment from rotor displacement Dₗ
    // Dₗ = ||Dₑₗ × x̂|| = ||Dᵦₗ × x̂|| (left rotor displacement)
    auto k_theta = [&](float omega_0, float d = 1.0f) {
        return 2.0f * k_T_roll * d * omega_0;
    };
    
    // Create continuous B matrix (7x4) - Input coupling matrix
    // States: [yaw_rate, pitch_rate, roll_rate, fl_speed, bl_speed, fr_speed, br_speed]
    // Inputs: [fl_motor, bl_motor, fr_motor, br_motor]
    // From derivation: B = [kᵢ/kₐ, -kᵢ/kₐ, -kᵢ/kₐ, kᵢ/kₐ; 0,0,0,0; 0,0,0,0; 1/kₐ,0,0,0; ...]
    // Yaw row: kᵢ/kₐ * [1,-1,-1,1] (inertial coupling with rotor direction)
    // Attitude rows: [0,0,0,0] (no direct voltage-to-attitude coupling)
    // Motor rows: diag(1/kₐ) (direct voltage-to-motor coupling)
    auto b_continuous = LinearizationInputMatrix{
        {k_I / k_alpha,  -k_I / k_alpha, -k_I / k_alpha,  k_I / k_alpha},  // yaw_rate row
        {0.0f,            0.0f,           0.0f,           0.0f},            // pitch_rate row
        {0.0f,            0.0f,           0.0f,           0.0f},            // roll_rate row
        {1.0f / k_alpha,  0.0f,           0.0f,           0.0f},            // fl_speed row
        {0.0f,            1.0f / k_alpha, 0.0f,           0.0f},            // bl_speed row
        {0.0f,            0.0f,           1.0f / k_alpha, 0.0f},            // fr_speed row
        {0.0f,            0.0f,           0.0f,           1.0f / k_alpha}   // br_speed row
    };
    
    // Create continuous A matrix (7x7) - State dynamics matrix
    // States: [yaw_rate, pitch_rate, roll_rate, fl_speed, bl_speed, fr_speed, br_speed]
    // Structure: [attitude_rate_dynamics    motor_to_attitude_coupling]
    //            [0                         motor_speed_dynamics]
    
    // Attitude rate dynamics: diagonal matrix with damping coefficients
    // From derivation: ρᵧ, ρφ, ρθ (linear damping terms)
    // Represents aerodynamic damping of attitude rates
    auto attitude_rate_dynamics = AttitudeMatrix{
        {k_rho_yaw,   0.0f,        0.0f},        // yaw rate row
        {0.0f,        k_rho_pitch, 0.0f},        // pitch rate row
        {0.0f,        0.0f,        k_rho_roll}   // roll rate row
    };
    
    // Motor-to-attitude coupling: how motor speeds affect attitude rates
    // From derivation: coupling coefficients kᵧ, kφ, kθ for yaw, pitch, roll
    // Yaw row: kᵧ = 2kᵣω̂ₘ + kᵢkᵦ,ₘ (torque + inertial effects)
    // Pitch row: kφ = 2kₜDₑω̂ₘ (thrust × front displacement)
    // Roll row: kθ = 2kₜDₗω̂ₘ (thrust × left displacement)
    auto motor_to_attitude_coupling = MotorToAttitude{
        {k_gamma(fl_rps),         -k_gamma(bl_rps),         -k_gamma(fr_rps),           k_gamma(br_rps)},      // yaw coupling row
        {k_phi(fl_rps, d_front),  -k_phi(bl_rps, d_back),    k_phi(fr_rps, d_front),   -k_phi(br_rps, d_back)}, // pitch coupling row
        {k_theta(fl_rps, d_left),  k_theta(bl_rps, d_left), -k_theta(fr_rps, d_right), -k_theta(br_rps, d_right)} // roll coupling row
    };
    
    // Motor speed dynamics: diagonal matrix with motor dynamics coefficients
    // From derivation: kᵦ,ₘ = -(kᵦ + 2kᵦ²ω̂ₘ)/kₐ (linearized motor dynamics)
    // Represents how each motor's speed evolves based on its operating point
    // Diagonal structure assumes no cross-coupling between motors
    auto motor_speed_dynamics = MotorMatrix{
        {k_beta(fl_rps), 0.0f,           0.0f,           0.0f},           // front left row
        {0.0f,           k_beta(bl_rps), 0.0f,           0.0f},           // back left row
        {0.0f,           0.0f,           k_beta(fr_rps), 0.0f},           // front right row
        {0.0f,           0.0f,           0.0f,           k_beta(br_rps)}  // back right row
    };
    
    // Assemble the continuous-time state matrix A from block components (7x7)
    // Structure: [attitude_rate_dynamics    motor_to_attitude_coupling]
    //            [0                         motor_speed_dynamics]
    // This represents the linearized quadcopter dynamics in state-space form
    // where motor speeds couple to attitude rates through thrust and torque effects
    // States: [yaw_rate, pitch_rate, roll_rate, fl_speed, bl_speed, fr_speed, br_speed]
    LinearizationStateMatrix a_continuous = (LinearizationStateMatrix() <<
        attitude_rate_dynamics,  motor_to_attitude_coupling,
        AttitudeToMotor::Zero(), motor_speed_dynamics
    ).finished();
    
    // Discretize the continuous system
    float dt = 0.00025f; // 4kHz = 0.25ms
    auto [a_discrete_7x7, b_discrete_7x7] = discretize_system(a_continuous, b_continuous, dt);
    
    // Create augmented system with integral and derivative states
    float integral_decay_time = 0.3f;
    float derivative_decay_time = 0.05f;
    auto [a_discrete_augmented, b_discrete_augmented] = create_augmented_system(
        a_discrete_7x7, b_discrete_7x7, dt, integral_decay_time, derivative_decay_time);
    
    // Cost weights following Bryson's rule: Q_ii = 1 / (characteristic_deviation_i)²
    // This ensures states with smaller characteristic deviations get higher penalties

    // The sigma values are the σ (sigma) values themselves, and weights = 1/σ²:
    // - Smaller σ = larger weight = higher penalty for deviations
    // - Larger σ = smaller weight = lower penalty for deviations
    constexpr auto sigma_deriv = 0.1f * angular::revolution / si::second / si::second;
    constexpr auto sigma = 0.05f * angular::revolution / si::second;
    constexpr auto sigma_int = 0.001f * angular::revolution;
    
    // Input cost weight: R_ii = 1 / (characteristic_input_deviation_i)²
    // Note: Input weights work in opposite direction - larger σ = higher penalty for input effort
    constexpr auto sigma_input = 0.8f * si::volt;
    
    // Create cost matrices using Bryson's rule weights
    auto [q_cost, r_cost] = make_cost_matrices(sigma_deriv, sigma_int, sigma, sigma_input);
    
    // Solve LQR for the augmented system
    auto lqr_result = LqrSolver::solve_lqr(a_discrete_augmented, b_discrete_augmented, q_cost, r_cost);
    
    // Initialize gain matrices with zeros
    KiMatrix ki = KiMatrix::Constant(VoltPerRotation::zero());
    KdMatrix kd = KdMatrix::Constant(VoltSecondSquaredPerRotation::zero());
    KpMatrix kp = KpMatrix::Constant(VoltSecondPerRotation::zero());
    
    if (lqr_result) {
        // Extract gains for the 9 controllable states:
        // States 0-2: integral gains
        // States 3-5: derivative gains  
        // States 9-11: proportional gains (attitude rates)
        
        const auto& k_full = *lqr_result;
        
        // Extract integral gains (states 0-2)
        for (int motor = 0; motor < 4; ++motor) {
            for (int axis = 0; axis < 3; ++axis) {
                ki(motor, axis) = VoltPerRotation(k_full(motor, axis) * si::volt / angular::revolution);
            }
        }
        
        // Extract derivative gains (states 3-5)
        for (int motor = 0; motor < 4; ++motor) {
            for (int axis = 0; axis < 3; ++axis) {
                kd(motor, axis) = VoltSecondSquaredPerRotation(k_full(motor, axis + 3) * si::volt * si::second * si::second / angular::revolution);
            }
        }
        
        // Extract proportional gains (states 9-11)
        for (int motor = 0; motor < 4; ++motor) {
            for (int axis = 0; axis < 3; ++axis) {
                kp(motor, axis) = VoltSecondPerRotation(k_full(motor, axis + 9) * si::volt * si::second / angular::revolution);
            }
        }
    }
    
    // Create the linearized model
    LinearizedModel model;
    model.a_continuous = a_continuous;
    model.b_continuous = b_continuous;
    model.a_discrete = a_discrete_augmented;
    model.b_discrete = b_discrete_augmented;
    model.k_integral = ki;
    model.k_derivative = kd;
    model.k_proportional = kp;
    
    return model;
}

auto create_state_cost_matrix(const std::array<float, STATES>& elements) -> StateMatrix {
    // Create diagonal matrix from elements (matching Python make_cost_matrix)
    StateMatrix matrix = StateMatrix::Zero();
    for (size_t i = 0; i < STATES; ++i) {
        matrix(i, i) = static_cast<float>(elements[i]);
    }
    return matrix;
}

auto create_input_cost_matrix(const std::array<float, INPUTS>& elements) -> Eigen::Matrix<float, INPUTS, INPUTS> {
    // Create diagonal matrix from elements (matching Python make_cost_matrix)
    Eigen::Matrix<float, INPUTS, INPUTS> matrix = Eigen::Matrix<float, INPUTS, INPUTS>::Zero();
    for (size_t i = 0; i < INPUTS; ++i) {
        matrix(i, i) = static_cast<float>(elements[i]);
    }
    return matrix;
}

} // namespace lqr 