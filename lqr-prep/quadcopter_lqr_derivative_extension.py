import math
from typing import Literal, Generic, TypeVar
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matrepr
from mako.template import Template
from dataclasses import dataclass
import control as ct
import frccontrol as frcct
import scipy as sc
from scipy import linalg

N = TypeVar('N')


def solve_discrete_lqr(A, B, Qelems, Relems):
    if not frcct.is_stabilizable(A, B):
        raise RuntimeError(
            f"The system is unstabilizable!\n\nA = {A}\nB = {B}\n"
        )

    Q = frcct.make_cost_matrix(Qelems)
    R = frcct.make_cost_matrix(Relems)

    S = sc.linalg.solve_discrete_are(a=A, b=B, q=Q, r=R)

    # K = (BᵀSB + R)⁻¹BᵀSA
    K = np.linalg.solve(B.T @ S @ B + R, B.T @ S @ A)
    return K


@dataclass
class Model(Generic[N]):
    lookup_table_resolution: N
    lookup_table_rps_step: float
    lookup_table_min_rps: float
    lookup_table_max_rps: float
    A_continuous: np.matrix[tuple[N, N, N, N, Literal[7], Literal[7]], float]
    A_continuous_inv: np.matrix[tuple[N, N, N, N, Literal[7], Literal[7]], float]
    A_discrete: np.matrix[tuple[N, N, N, N, Literal[10], Literal[10]], float]
    A_discrete_inv: np.matrix[tuple[N, N, N, N, Literal[10], Literal[10]], float]
    B_continuous: np.matrix[tuple[Literal[7], Literal[4]], float]
    B_continuous_inv: np.matrix[tuple[Literal[4], Literal[7]], float]
    B_discrete: np.matrix[tuple[N, N, N, N, Literal[7], Literal[4]], float]
    B_discrete_inv: np.matrix[tuple[N, N, N, N, Literal[4], Literal[3]], float]
    K: np.matrix[tuple[N, N, N, N, Literal[4], Literal[9]], float]
    affine_partial: np.matrix[tuple[Literal[7]], float]
    motor_mix: np.matrix[tuple[4,3]]
    d_front: float
    d_back: float
    d_left: float
    d_right: float
    integral_decay_time: float


def calculate_model(coeffs_file: str, resolution: int, min_rps: float, max_rps: float, integral_decay_time: float = 1.0, derivative_decay_time: float = 1.0):
    A_continuous = np.empty([resolution, resolution, resolution, resolution, 7, 7])
    A_continuous_inv = np.empty([resolution, resolution, resolution, resolution, 7, 7])
    A_discrete = np.empty([resolution, resolution, resolution, resolution, 10, 10])
    A_discrete_inv = np.empty([resolution, resolution, resolution, resolution, 10, 10])
    B_discrete = np.empty([resolution, resolution, resolution, resolution, 7, 4])
    B_discrete_inv_partial = np.empty([resolution, resolution, resolution, resolution, 4, 3])
    K_partial = np.empty([resolution, resolution, resolution, resolution, 4, 9])

    rps_step = (max_rps - min_rps)/(resolution - 1)

    with open(coeffs_file, 'r') as file:
        data = json.load(file)

    motor_fit = data['motors']
    yaw_fit = data['yaw']
    pitch_fit = data['pitch']
    roll_fit = data['roll']

    k_alpha = motor_fit['motorAccel']
    k_omega = motor_fit['motorVel']
    k_omega_sq = motor_fit['motorVelSq']

    k_I = yaw_fit['propAccelSum']
    k_tau = yaw_fit['propVelSumSq']
    k_rho_yaw = yaw_fit['yawRate']

    k_T_pitch = pitch_fit['propVelSumSq']
    k_rho_pitch = pitch_fit['pitchRate']
    d_front = pitch_fit.get('dFront', 1.0)
    d_back = pitch_fit.get('dBack', 1.0)

    k_T_roll = roll_fit['propVelSumSq']
    k_rho_roll = roll_fit['rollRate']
    d_left = roll_fit.get('dLeft', 1.0)
    d_right = roll_fit.get('dRight', 1.0)

    def signed_square(x):
        return x * abs(x)

    def k_beta(omega_0):
        return (k_omega + 2 * k_omega_sq * omega_0) / k_alpha

    def k_gamma(omega_0):
        return 2 * k_tau * omega_0 + k_I * k_beta(omega_0)

    def k_phi(omega_0, d=1):
        return 2 * k_T_pitch * d * omega_0

    def k_theta(omega_0, d=1):
        return 2 * k_T_roll * d * omega_0

    B = np.array([
        [k_I / k_alpha, -k_I / k_alpha, -k_I / k_alpha, k_I / k_alpha],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1 / k_alpha, 0, 0, 0],
        [0, 1 / k_alpha, 0, 0],
        [0, 0, 1 / k_alpha, 0],
        [0, 0, 0, 1 / k_alpha]
    ])

    B_continuous = B
    B_continuous_inv = np.linalg.pinv(B)

    def linearized_attitude_model(omega_fl, omega_bl, omega_fr, omega_br):
        A = np.array([
            [k_rho_yaw, 0, 0, k_gamma(omega_fl), -k_gamma(omega_bl), -k_gamma(omega_fr), k_gamma(omega_br)],
            [0, k_rho_pitch, 0, k_phi(omega_fl, d_front), -k_phi(omega_bl, d_back), k_phi(omega_fr, d_front),
             -k_phi(omega_br, d_back)],
            [0, 0, k_rho_roll, k_theta(omega_fl, d_left), k_theta(omega_bl, d_left), -k_theta(omega_fr, d_right),
             -k_theta(omega_br, d_right)],
            [0, 0, 0, -k_beta(omega_fl), 0, 0, 0],
            [0, 0, 0, 0, -k_beta(omega_bl), 0, 0],
            [0, 0, 0, 0, 0, -k_beta(omega_fr), 0],
            [0, 0, 0, 0, 0, 0, -k_beta(omega_br)]
        ])

        matrepr.mprint(A)

        affine_intercept_partial = np.array([
            (k_tau + k_I * k_omega_sq / k_alpha),
            k_T_pitch,
            k_T_roll,
            k_omega_sq / k_alpha,
            k_omega_sq / k_alpha,
            k_omega_sq / k_alpha,
            k_omega_sq / k_alpha
        ])
        return A, affine_intercept_partial

    for front_left_index in range(resolution):
        for back_left_index in range(resolution):
            for front_right_index in range(resolution):
                for back_right_index in range(resolution):
                    [A, affine_intercept_partial] = linearized_attitude_model(
                        front_left_index * rps_step + min_rps,
                        back_left_index * rps_step + min_rps,
                        front_right_index * rps_step + min_rps,
                        back_right_index * rps_step + min_rps)

                    dt = .00025  # 4kHz = 0.25ms
                    alpha = dt / integral_decay_time
                    alpha_deriv = dt / derivative_decay_time
                

                    C = np.array([[1, 0, 0, 0, 0, 0, 0]])
                    D = np.array([[0, 0, 0, 0]])

                    sys = ct.ss(A, B, C, D)
                    dsys = sys.sample(dt)

                    # Add integral and derivative to the state (9 new states - 3 integral, 3 derivatives, 3 buffers) (9 + 7 = 16 states total)
                    A_int = np.array([
                        # leaky integrator
                        [(1-alpha), 0, 0, 0, 0, 0, 0, 0, 0, alpha * dt, 0, 0, 0, 0, 0, 0],
                        [0, (1-alpha), 0, 0, 0, 0, 0, 0, 0, 0, alpha * dt, 0, 0, 0, 0, 0],
                        [0, 0, (1-alpha), 0, 0, 0, 0, 0, 0, 0, 0, alpha * dt, 0, 0, 0, 0],
                        # derivative
                        [0, 0, 0, (1 - alpha_deriv), 0, 0, - alpha_deriv / dt, 0, 0, alpha_deriv / dt, 0, 0, 0, 0, 0, 0],    
                        [0, 0, 0, 0, (1 - alpha_deriv), 0, 0, - alpha_deriv / dt, 0, 0, alpha_deriv / dt, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, (1 - alpha_deriv), 0, 0, - alpha_deriv / dt, 0, 0, alpha_deriv / dt, 0, 0, 0, 0],
                        # previous state (delay buffer)
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        
                    ])

                    # Combine the integrator and the attitude model
                    # The integrator is in the first 3 rows, the attitude model is in the last 7 rows
                    A_discrete_aug = np.block([
                        [A_int],
                        [np.zeros((7, 9)), dsys.A]
                    ])

                    B_discrete_aug = np.block([
                        [np.zeros((9, 4))],
                        [dsys.B]
                    ])

                    qelm_deriv= 0.1 # RPS^2
                    qelm_int = 0.005 # Rotations
                    qelm = 0.01 # RPS
                    relm = .8 # volt
                    
                    # Create cost matrices for the augmented system (16 states total)
                    # State order: [integral_roll, integral_pitch, integral_yaw, 
                    #              derivative_roll, derivative_pitch, derivative_yaw,
                    #              buffer_roll, buffer_pitch, buffer_yaw,
                    #              attitude_rates, motor_speeds]
                    Q = [
                        qelm_int,    # integral_roll
                        qelm_int,    # integral_pitch  
                        qelm_int,    # integral_yaw
                        qelm_deriv,  # derivative_roll
                        qelm_deriv,  # derivative_pitch
                        qelm_deriv,  # derivative_yaw
                        math.inf,    # buffer_roll (no cost)
                        math.inf,    # buffer_pitch (no cost)
                        math.inf,    # buffer_yaw (no cost)
                        qelm,        # attitude_rate_roll
                        qelm,        # attitude_rate_pitch
                        qelm,        # attitude_rate_yaw
                        math.inf,    # motor_speed_fl (no cost)
                        math.inf,    # motor_speed_bl (no cost)
                        math.inf,    # motor_speed_fr (no cost)
                        math.inf     # motor_speed_br (no cost)
                    ]
                    R = [relm, relm, relm, relm]
                    
                    # Solve discrete-time LQR with our augmented system
                    K_full = solve_discrete_lqr(A_discrete_aug, B_discrete_aug, Q, R)
                    
                    # Extract the gain matrix for the 4 motors and controllable states
                    # States: [integral_roll, integral_pitch, integral_yaw, derivative_roll, derivative_pitch, derivative_yaw, buffer_roll, buffer_pitch, buffer_yaw, attitude_rates, ...]
                    # We want: [integral_roll, integral_pitch, integral_yaw, derivative_roll, derivative_pitch, derivative_yaw, attitude_rate_roll, attitude_rate_pitch, attitude_rate_yaw]
                    K_partial[front_left_index, back_left_index, front_right_index, back_right_index, :, :] = np.hstack([
                        K_full[:, 0:3],   # integral gains (states 0-2)
                        K_full[:, 3:6],   # derivative gains (states 3-5) 
                        K_full[:, 9:12]   # proportional gains (attitude rates, states 9-11)
                    ])

                    A_continuous[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = A

                    A_continuous_inv[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = np.linalg.pinv(A)

                    A_discrete[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = A_discrete_aug[:10, :10]

                    A_discrete_inv[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = np.linalg.pinv(A_discrete_aug[:10, :10])

                    B_discrete[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = dsys.B

                    B_discrete_inv_partial[
                    front_left_index,
                    back_left_index,
                    front_right_index,
                    back_right_index,
                    :,
                    :
                    ] = np.linalg.pinv(dsys.B)[:,:3]

    model = Model(
        lookup_table_resolution=resolution,
        lookup_table_min_rps=min_rps,
        lookup_table_max_rps=max_rps,
        lookup_table_rps_step=rps_step,
        A_continuous=A_continuous,
        A_continuous_inv=A_continuous_inv,
        A_discrete=A_discrete,
        A_discrete_inv=A_discrete_inv,
        B_continuous=B_continuous,
        B_continuous_inv=B_continuous_inv,
        B_discrete=B_discrete,
        B_discrete_inv=B_discrete_inv_partial,
        K=K_partial,
        affine_partial=affine_intercept_partial,
        motor_mix=np.array([
                [1, 1, 1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, -1]
             ]),
        d_front=d_front,
        d_back=d_back,
        d_left=d_left,
        d_right=d_right,
        integral_decay_time=integral_decay_time
    )
    return model


motor_mix = np.array([1, -1, -1, 1])  # Example value, adjust as needed

with open('regression_coefficients.json', 'r') as file:
    data = json.load(file)

pitch_fit = data['pitch']
roll_fit = data['roll']

d_front = pitch_fit.get('dFront', 1.0)  # Default value of 1.0
d_back = pitch_fit.get('dBack', 1.0)  # Default value of 1.0
d_left = roll_fit.get('dLeft', 1.0)  # Default value of 1.0
d_right = roll_fit.get('dRight', 1.0)  # Default value of 1.0


def gen_constants(model: Model, rps_step, derivative_decay_time):
    with open('lqr_constants.c.mako') as template_file:
        template = Template(template_file.read())

    # Split the 4x9 gain matrix into three 4x3 matrices
    # First 3 columns are integral gains, middle 3 are derivative gains, last 3 are proportional gains
    ki_lookup = model.K[:, :, :, :, :, :3].tolist()  # Integral gains: volts per rotation
    kd_lookup = model.K[:, :, :, :, :, 3:6].tolist()  # Derivative gains: volts per rotation per second squared
    kp_lookup = model.K[:, :, :, :, :, 6:].tolist()  # Proportional gains: volts per rotation per second
    B_discrete_inv_lookup = model.B_discrete_inv[:,:,:,:,:,:].tolist()

    return template.render(
        B_continuous=model.B_continuous.tolist(),
        B_continuous_inverse=model.B_continuous_inv.tolist(),
        affine_intercept_partial=model.affine_partial.tolist(),
        d_left=model.d_left,
        d_right=model.d_right,
        d_front=model.d_front,
        d_back=model.d_back,
        motor_mix=model.motor_mix.tolist(),
        lookup_step_size=rps_step,
        lookup_min_rps=model.lookup_table_min_rps,
        ki_lookup=ki_lookup,
        kd_lookup=kd_lookup,
        kp_lookup=kp_lookup,
        B_discrete_inv_lookup=B_discrete_inv_lookup,
        resolution=model.lookup_table_resolution,
        integral_decay_time=model.integral_decay_time,
        derivative_decay_time=derivative_decay_time
    )

def lerp_lqr(motor_speeds, model):
    """
    Interpolates LQR gains based on motor speeds using linear interpolation.

    Args:
        motor_speeds: A list or numpy array of 4 motor speeds (rotations per second).
        model: The LQR model containing the lookup table.

    Returns:
        A tuple containing the interpolated Ki (integral), Kp (rate), and BDiscreteInv gains.
    """

    lookup_index = np.zeros((2, 4), dtype=int)
    rps_at_lookup_index = np.zeros((2, 4))

    for motor in range(4):
        lookup_index[0, motor] = int(np.floor((motor_speeds[motor] - model.lookup_table_min_rps) / model.lookup_table_rps_step))
        rps_at_lookup_index[0, motor] = lookup_index[0, motor] * model.lookup_table_rps_step + model.lookup_table_min_rps
        lookup_index[1, motor] = int(np.ceil((motor_speeds[motor] - model.lookup_table_min_rps) / model.lookup_table_rps_step))
        rps_at_lookup_index[1, motor] = lookup_index[1, motor] * model.lookup_table_rps_step + model.lookup_table_min_rps

    ki_volt_per_rotation = np.zeros((4, 3))  # Integral gains: volts per rotation
    kd_volt_per_rotation_per_second_squared = np.zeros((4, 3))  # Derivative gains: volts per rotation per second squared
    kp_volt_per_rotation_per_second = np.zeros((4, 3))  # Proportional gains: volts per rotation per second
    b_discrete_inv_volt_second_per_rotation = np.zeros((4, 3))

    weights = np.zeros(16)

    # Calculate weights for each vertex
    vertex_index = 0
    for fl_floor_ceil in range(2):
        for bl_floor_ceil in range(2):
            for fr_floor_ceil in range(2):
                for br_floor_ceil in range(2):
                    vertex_speeds = [fl_floor_ceil, bl_floor_ceil, fr_floor_ceil, br_floor_ceil]
                    weight = 1.0
                    for motor in range(4):
                        diff = np.fabs(rps_at_lookup_index[vertex_speeds[motor], motor] - motor_speeds[motor]) / model.lookup_table_rps_step
                        weight *= (1 - diff)
                    weights[vertex_index] = weight
                    vertex_index += 1

    # Normalize weights
    weights /= sum(weights)

    # Interpolate gains
    vertex_index = 0
    for fl_floor_ceil in range(2):
        for bl_floor_ceil in range(2):
            for fr_floor_ceil in range(2):
                for br_floor_ceil in range(2):
                    vertex_speeds = [fl_floor_ceil, bl_floor_ceil, fr_floor_ceil, br_floor_ceil]
                    for motor in range(4):
                        # Interpolate integral gains (first 3 columns: 0-2)
                        for axis in range(3):
                            ki_volt_per_rotation[motor, axis] += weights[vertex_index] * model.K[
                                lookup_index[vertex_speeds[0], 0],
                                lookup_index[vertex_speeds[1], 1],
                                lookup_index[vertex_speeds[2], 2],
                                lookup_index[vertex_speeds[3], 3],
                                motor,
                                axis
                            ]
                        
                        # Interpolate derivative gains (middle 3 columns: 3-5)
                        for axis in range(3):
                            kd_volt_per_rotation_per_second_squared[motor, axis] += weights[vertex_index] * model.K[
                                lookup_index[vertex_speeds[0], 0],
                                lookup_index[vertex_speeds[1], 1],
                                lookup_index[vertex_speeds[2], 2],
                                lookup_index[vertex_speeds[3], 3],
                                motor,
                                axis + 3
                            ]
                        
                        # Interpolate proportional gains (last 3 columns: 6-8)
                        for axis in range(3):
                            kp_volt_per_rotation_per_second[motor, axis] += weights[vertex_index] * model.K[
                                lookup_index[vertex_speeds[0], 0],
                                lookup_index[vertex_speeds[1], 1],
                                lookup_index[vertex_speeds[2], 2],
                                lookup_index[vertex_speeds[3], 3],
                                motor,
                                axis + 6
                            ]
                            
                            # Interpolate B matrix
                            b_discrete_inv_volt_second_per_rotation[motor, axis] += weights[
                                vertex_index] * model.B_discrete_inv[
                                lookup_index[vertex_speeds[0], 0],
                                lookup_index[vertex_speeds[1], 1],
                                lookup_index[vertex_speeds[2], 2],
                                lookup_index[vertex_speeds[3], 3],
                                motor,
                                axis
                            ]
                    vertex_index += 1

    return ki_volt_per_rotation, kd_volt_per_rotation_per_second_squared, kp_volt_per_rotation_per_second, b_discrete_inv_volt_second_per_rotation


def plot_all_gains_together(model, prop_index=0, axis_index=1, num_points=20):
    """
    Plots all three LQR gains (integral, derivative, proportional) side by side with synchronized rotation.
    
    Args:
        model: The LQR model containing the lookup table.
        prop_index: The index of the motor (0-3).
        axis_index: The index of the axis (0-2).
        num_points: The number of points to use in the meshgrid.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np

    # Create a finer meshgrid
    motor_speeds = np.linspace(model.lookup_table_min_rps, model.lookup_table_max_rps, num_points)
    X, Y = np.meshgrid(motor_speeds, motor_speeds)

    # Interpolate all gains for each point in the meshgrid
    Z_integral = np.zeros_like(X)
    Z_derivative = np.zeros_like(X)
    Z_proportional = np.zeros_like(X)
    
    for i in range(num_points):
        for j in range(num_points):
            speeds = np.array([X[i, j], Y[i, j], Y[i, j], Y[i, j]])
            ki, kd, kp, _ = lerp_lqr(speeds, model)
            Z_integral[i, j] = ki[prop_index, axis_index]
            Z_derivative[i, j] = kd[prop_index, axis_index]
            Z_proportional[i, j] = kp[prop_index, axis_index]

    # Create subplots in a 2x2 grid layout
    fig = plt.figure(figsize=(12, 10))
    
    # Integral gain plot (top left)
    ax1 = fig.add_subplot(221, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_integral, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(X.min(), X.max())
    ax1.set_ylim(Y.min(), Y.max())
    ax1.set_zlim(Z_integral.min(), Z_integral.max())
    ax1.set_title('Integral Gain', fontsize=10)
    ax1.set_xlabel('Front left motor (RPS)', fontsize=8)
    ax1.set_ylabel('Other motors (RPS)', fontsize=8)
    ax1.set_zlabel('Gain (V/rotation)', fontsize=8)
    ax1.zaxis.set_major_locator(LinearLocator(5))
    ax1.zaxis.set_major_formatter('{x:.03f}')
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=8, pad=0.05)
    cbar1.set_label('Gain (V/rotation)', fontsize=8)
    ax1.view_init(azim=60, elev=30)
    
    # Derivative gain plot (top right)
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_derivative, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(X.min(), X.max())
    ax2.set_ylim(Y.min(), Y.max())
    ax2.set_zlim(Z_derivative.min(), Z_derivative.max())
    ax2.set_title('Derivative Gain', fontsize=10)
    ax2.set_xlabel('Front left motor (RPS)', fontsize=8)
    ax2.set_ylabel('Other motors (RPS)', fontsize=8)
    ax2.set_zlabel('Gain (V/(rotation/s²))', fontsize=8)
    ax2.zaxis.set_major_locator(LinearLocator(5))
    ax2.zaxis.set_major_formatter('{x:.03f}')
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=8, pad=0.05)
    cbar2.set_label('Gain (V/(rotation/s²))', fontsize=8)
    ax2.view_init(azim=60, elev=30)
    
    # Proportional gain plot (bottom left)
    ax3 = fig.add_subplot(223, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_proportional, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_xlim(X.min(), X.max())
    ax3.set_ylim(Y.min(), Y.max())
    ax3.set_zlim(Z_proportional.min(), Z_proportional.max())
    ax3.set_title('Proportional Gain', fontsize=10)
    ax3.set_xlabel('Front left motor (RPS)', fontsize=8)
    ax3.set_ylabel('Other motors (RPS)', fontsize=8)
    ax3.set_zlabel('Gain (V/(rotation/s))', fontsize=8)
    ax3.zaxis.set_major_locator(LinearLocator(5))
    ax3.zaxis.set_major_formatter('{x:.03f}')
    cbar3 = fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=8, pad=0.05)
    cbar3.set_label('Gain (V/(rotation/s))', fontsize=8)
    ax3.view_init(azim=60, elev=30)
    
    # Synchronize rotation using a more robust approach
    def sync_views(source_ax, target_axes):
        """Synchronize view angles from source to target axes"""
        azim, elev = source_ax.azim, source_ax.elev
        for ax in target_axes:
            ax.view_init(azim=azim, elev=elev)
        fig.canvas.draw_idle()
    
    # Connect events to synchronize views
    def on_rotate_ax1(event):
        if event.inaxes == ax1:
            sync_views(ax1, [ax2, ax3])
    
    def on_rotate_ax2(event):
        if event.inaxes == ax2:
            sync_views(ax2, [ax1, ax3])
    
    def on_rotate_ax3(event):
        if event.inaxes == ax3:
            sync_views(ax3, [ax1, ax2])
    
    # Connect the events with a small delay to prevent conflicts
    fig.canvas.mpl_connect('button_release_event', on_rotate_ax1)
    fig.canvas.mpl_connect('button_release_event', on_rotate_ax2)
    fig.canvas.mpl_connect('button_release_event', on_rotate_ax3)
    
    plt.tight_layout(pad=5.0, h_pad=4.0, w_pad=4.0)
    plt.show()

resolution = 5 # Changed back to 5 for higher resolution
min_rps = 50
max_rps = 500

model = calculate_model("regression_coefficients.json", resolution, min_rps, max_rps, integral_decay_time=0.3, derivative_decay_time=0.05)

# Test plotting all three gain types together
plot_all_gains_together(model, 0, 1, 30)

# Generate the constants file
constants = gen_constants(model, model.lookup_table_rps_step, 0.05)  # derivative_decay_time = 0.05
# Write the constants to lqr_constants.c
with open('lqr_constants.c', 'w') as f:
    f.write(constants)

# Generate the header file from template
with open('lqr_constants.h.mako') as template_file:
    header_template = Template(template_file.read())

header_content = header_template.render(resolution=resolution)

# Write the header to lqr_constants.h in lqr-prep folder for inspection
with open('lqr_constants.h', 'w') as f:
    f.write(header_content)
