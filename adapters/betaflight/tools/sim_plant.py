"""
Nonlinear rate/motor plant that gen_wfg.py linearizes, for closing the loop
around SITL.  Same model as whippyflight-sim's plant:

  ω̇_i   = (V_i − kω·ω_i − kω²·ω_i|ω_i|) / kα
  roll'  = ρ_r·roll  + kT_r·Σ s_r·ω|ω|                  s_r = [w_L, w_L, −w_R, −w_R]
  pitch' = ρ_p·pitch + kT_p·Σ s_p·ω|ω|                  s_p = [w_F, −w_B, w_F, −w_B]
  yaw'   = ρ_y·yaw   + kτ·Σ s_y·ω|ω| + k_I·Σ s_y·ω̇     s_y = [+, −, −, +]

with the patterns over the slots [front_left, rear_left, front_right,
rear_right], each BF motor taking its slot's weight (the fit's `frame`).
Rates are FRD [roll, pitch, yaw] in rot/s; motor speeds are by BF motor
index in shaft rev/s; V is duty × pack voltage.
"""

import math

from whippyflight import coefficients, gen_wfg

TWO_PI = 2.0 * math.pi


def erpm_from_rps(rps, poles):
    """Electrical RPM of a motor turning at `rps` shaft rev/s."""
    return rps * 60.0 * poles / 2.0


class Plant:
    def __init__(self, coeffs_path, omega0=0.0):
        c = coefficients.load(coeffs_path)
        if c.get("motor_speed") != "shaft rev/s":
            raise SystemExit(f"{coeffs_path}: fitted before motor speeds were in shaft rev/s; fit it again")
        self.motor_poles = c["airframe"]["motor_poles"]
        w = gen_wfg.arm_weights(c)
        w_f, w_b, w_l, w_r = w["front"], w["back"], w["left"], w["right"]
        self.k_alpha = c["motors"]["motorAccel"]
        self.k_w = c["motors"]["motorVel"]
        self.k_w2 = c["motors"]["motorVelSq"]
        self.rho = (c["roll"]["rollRate"], c["pitch"]["pitchRate"], c["yaw"]["yawRate"])
        self.k_t = (c["roll"]["propVelSumSq"], c["pitch"]["propVelSumSq"], c["yaw"]["propVelSumSq"])
        self.k_i = c["yaw"]["propAccelSum"]
        slots, _ = coefficients.layout(c)
        by_slot = ((w_l, w_l, -w_r, -w_r), (w_f, -w_b, w_f, -w_b), (1.0, -1.0, -1.0, 1.0))
        self.patterns = [[p[s] for s in slots] for p in by_slot]
        self.rate = [0.0, 0.0, 0.0]
        self.omega = [omega0] * 4

    def _deriv(self, rate, omega, volts):
        sq = [w * abs(w) for w in omega]
        wdot = [(volts[i] - self.k_w * omega[i] - self.k_w2 * sq[i]) / self.k_alpha
                for i in range(4)]
        out = [self.rho[a] * rate[a] + self.k_t[a] * sum(p * q for p, q in zip(self.patterns[a], sq))
               for a in range(3)]
        out[2] += self.k_i * sum(p * d for p, d in zip(self.patterns[2], wdot))
        return out, wdot

    def step(self, volts, h):
        """Advance h seconds with the motor voltages held (RK4)."""
        r0, w0 = self.rate, self.omega
        k1r, k1w = self._deriv(r0, w0, volts)
        k2r, k2w = self._deriv([r0[i] + h / 2 * k1r[i] for i in range(3)],
                               [w0[i] + h / 2 * k1w[i] for i in range(4)], volts)
        k3r, k3w = self._deriv([r0[i] + h / 2 * k2r[i] for i in range(3)],
                               [w0[i] + h / 2 * k2w[i] for i in range(4)], volts)
        k4r, k4w = self._deriv([r0[i] + h * k3r[i] for i in range(3)],
                               [w0[i] + h * k3w[i] for i in range(4)], volts)
        self.rate = [r0[i] + h / 6 * (k1r[i] + 2 * k2r[i] + 2 * k3r[i] + k4r[i]) for i in range(3)]
        self.omega = [w0[i] + h / 6 * (k1w[i] + 2 * k2w[i] + 2 * k3w[i] + k4w[i]) for i in range(4)]

    def gyro_bf(self):
        """Betaflight gyro [roll, pitch, yaw] in rot/s: FLU, the FRD rates
        with pitch and yaw negated."""
        roll, pitch, yaw = self.rate
        return (roll, -pitch, -yaw)

    def imu_rpy(self):
        """fdm_packet imu_angular_velocity_rpy (rad/s) that SITL maps onto
        gyro_bf(): BF roll = rpy[0], pitch = −rpy[1], yaw = rpy[2]."""
        roll, pitch, yaw = self.gyro_bf()
        return (TWO_PI * roll, -TWO_PI * pitch, TWO_PI * yaw)

    def erpm(self):
        return [erpm_from_rps(w, self.motor_poles) for w in self.omega]
