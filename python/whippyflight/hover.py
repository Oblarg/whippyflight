"""
The closed loop at hover as the sim flies it (whippyflight-sim margin.rs),
linearized, and the front's measures on it: knock recovery, delay margin,
CG-shift trim and the motor roughness the fit's recorded sensor noise
causes.

The loop is the fit's plant at its hover speed with the firmware's control
law around it: the feedback gains interpolated at hover from the gain grid
(interpolation.rs), or with --hover-only solved at hover, the integral and
derivative filters as controller.rs runs them, the affine linearizer on the
motor-speed telemetry, the fit's
motor-command and telemetry delays, its gyro low-pass stages (as PT1s) and
motor-speed low-pass.  The feedforward's reference model runs on its own
state at hover and drops out.  Saturation, the integral clamp and the
sim's clean-flight test are nonlinear and left to the sim.
"""

import contextlib
import io
import json
import os
import struct

import numpy as np
import scipy as sc

from whippyflight import coefficients
from whippyflight import gen_wfg as g

# The sim's gyro path (margin.rs pt1_cutoffs, plant.rs DEFAULT_GYRO_LPF).
PT1_STAGES = {"PT1": (1.0, 1), "PT2": (1.553773974, 2), "PT3": (1.961459177, 3)}
DEFAULT_GYRO_LPF = [{"type": "PT1", "hz": 275.0}, {"type": "PT1", "hz": 550.0}]
# The sim's flights (margin.rs, flick.rs ROUGH_HP).
SETTLE = 1.0
HOVER_FLIGHT = 2 * 0.030 + 0.5 + 0.6
TRIM = 3.0
ROUGH_HP_HZ = 50.0
# Noise files (noise.rs).
GYRO_LSB, ERPM_LSB = 1e-5, 1e-3
NOISE, NOISE_INFO = "noise.bin", "noise.json"
# Model axis order [yaw, pitch, roll] from FRD [roll, pitch, yaw].
MODEL_FROM_FRD = [2, 1, 0]
ROLL, PITCH = 2, 1
# The loop's inputs (HoverLoop.system).
INPUTS = 8


class Recording:
    """A fit's recorded sensor noise (noise.rs), next to its coefficients."""

    def __init__(self, fit_dir, perm, hover):
        with open(os.path.join(fit_dir, NOISE), "rb") as f:
            data = f.read()
        fs, n = struct.unpack_from("<fI", data)
        frames = np.frombuffer(data, "<i2", 7 * n, 8).reshape(n, 7).astype(float)
        self.fs = fs
        self.gyro = frames[:, 0:3] * GYRO_LSB
        motor = frames[:, 3:7] * ERPM_LSB
        law = {}
        with contextlib.suppress(OSError, ValueError):
            with open(os.path.join(fit_dir, NOISE_INFO)) as f:
                law = json.load(f).get("motor_noise_speed") or {}
        if law:
            motor = motor * (abs(hover) / law["reference_rps"]) ** law["exponent"]
        self.motor = np.zeros_like(motor)
        self.motor[:, perm] = motor

    def at(self, t):
        """[gyro (model axes), motor speeds (model slots)] at times `t`, interpolated, looped."""
        n = len(self.gyro)
        x = np.mod(t * self.fs, n)
        i = x.astype(int) % n
        f = (x - np.floor(x))[:, None]
        mix = lambda a: a[i] + f * (a[(i + 1) % n] - a[i])
        return np.hstack([mix(self.gyro)[:, MODEL_FROM_FRD], mix(self.motor)])


class HoverLoop:
    """The linearized loop at hover for a fit and gen_wfg arguments."""

    def __init__(self, coefficients_path, gen_argv):
        self.args = g.parse_args([coefficients_path, os.devnull, *gen_argv])
        with contextlib.redirect_stderr(io.StringIO()):
            self.coeffs = g.load_coefficients(coefficients_path)
            g.fill_defaults(self.args, self.coeffs)
        a, c = self.args, self.coeffs
        self.dt = a.dt
        self.w = g.arm_weights(c, a.d_front, a.d_back, a.d_left, a.d_right)
        _, hover, _ = g.operating_point(c)
        self.hover = hover
        n = a.resolution
        step = (a.max_rps - a.min_rps) / (n - 1)
        position = (min(max(hover, a.min_rps), a.max_rps) - a.min_rps) / step
        lower = min(int(position), n - 2)
        frac = position - lower
        speeds = [a.min_rps + (lower + i) * step for i in (0, 1)]
        self.corners = []
        if a.hover_only:
            self.corners.append((1.0, [hover] * 4))
        else:
            for corner in range(16):
                upper = [(corner >> m) & 1 for m in range(4)]
                weight = np.prod([frac if u else 1.0 - frac for u in upper])
                if weight > 0.0:
                    self.corners.append((weight, [speeds[u] for u in upper]))
        k_I, k_alpha = c['yaw']['propAccelSum'], c['motors']['motorAccel']
        A = g.build_A_matrix(*[hover] * 4, c, self.w)
        B = np.hstack([g.build_B_matrix(k_I, k_alpha), np.eye(7)[:, [PITCH]]])
        M = sc.linalg.expm(np.block([[A, B], [np.zeros((5, 12))]]) * self.dt)
        self.Ad, self.Bd, self.Bt = M[:7, :7], M[:7, 7:11], M[:7, 11]
        self.linearizer = g.build_affine_linearizer(c, self.w) @ np.diag(np.full(4, 2.0 * abs(hover)))
        airframe = c.get('airframe') or {}
        self.gyro_gains = []
        for stage in airframe.get('gyro_lpf') or DEFAULT_GYRO_LPF:
            if stage['type'] not in PT1_STAGES:
                raise SystemExit(f"gyro LPF type {stage['type']} is not modeled")
            scale, count = PT1_STAGES[stage['type']]
            x = 2 * np.pi * stage['hz'] * scale * self.dt
            self.gyro_gains += [x / (x + 1.0)] * count
        lpf = airframe.get('rpm_lpf_hz') or 0.0
        x = 2 * np.pi * lpf * self.dt
        self.motor_lpf = x / (1.0 + x) if lpf > 0 else None
        latency = c.get('latency') or {}
        # The fit's measured loop latency, s.
        self.latency = 1e-3 * (latency.get('actuator_ms', 0.0) + latency.get('telemetry_ms', 0.0))
        ticks = lambda key: max(int(round(1e-3 * latency.get(key, 0.0) / self.dt)), 0)
        self.actuator_ticks, self.telemetry_ticks = ticks('actuator_ms'), ticks('telemetry_ms')
        perm, _ = coefficients.layout(c)
        fit_dir = os.path.dirname(os.path.abspath(coefficients_path))
        self.noise = Recording(fit_dir, perm, hover) if os.path.isfile(os.path.join(fit_dir, NOISE)) else None

    def _augmented(self, tau_deriv, tau_int):
        """[(weight, (A_aug, B_aug))] at the corners, for the last time constants asked."""
        if getattr(self, "_augmented_at", None) != (tau_deriv, tau_int):
            self._augmented_systems = [
                (weight, g.augmented_system(omegas, self.coeffs, self.w, self.dt, tau_int,
                                            tau_deriv)[:2])
                for weight, omegas in self.corners]
            self._augmented_at = (tau_deriv, tau_int)
        return self._augmented_systems

    def gains(self, design):
        """The feedback gains the firmware interpolates at hover, over the augmented state;
        None where the DARE can't be solved."""
        sigma_deriv, sigma_int, tau_deriv, tau_int = design
        a = self.args
        q, r = g.lqr_weights(sigma_int, sigma_deriv, a.sigma_rate, a.sigma_volt)
        K = 0.0
        try:
            for weight, (A_aug, B_aug) in self._augmented(tau_deriv, tau_int):
                K = K + weight * g.applied_gains(A_aug, B_aug, q, r, a.structured)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return K if np.all(np.isfinite(K)) else None

    def system(self, design, K=None, delay_ticks=0.0):
        """One tick of the loop, x' = M·x + W·w and u = C·x + D·w, with an extra gyro delay
        (fractional ticks) on top of the gyro path.  The state starts with the plant (model
        order [yaw, pitch, roll, motors]); w is [pitch torque (rot/s²), gyro noise (model
        axes), motor-speed noise (model slots)]; u the commanded motor voltages."""
        _, _, tau_deriv, tau_int = design
        K = self.gains(design) if K is None else K
        k_int, k_deriv, k_rate = K[:, 0:3], K[:, 3:6], K[:, 9:12]
        alpha, alpha_d = self.dt / tau_int, self.dt / tau_deriv
        ns, nt, na = len(self.gyro_gains), self.telemetry_ticks, self.actuator_ticks
        whole, f = int(np.floor(delay_ticks)), delay_ticks - np.floor(delay_ticks)
        nh = whole + 1
        sizes = [7, 3 * ns, 3 * nh, 3, 3, 3, 4 * nt, 4 if self.motor_lpf else 0, 4 * na]
        cuts = np.cumsum([0] + sizes)

        def tick(x, w):
            """One tick on each column of states `x` and inputs `w`."""
            plant, filt, hist, integ, deriv, prev, tel, lp, act = (x[cuts[i]:cuts[i + 1]] for i in range(9))
            y, stages = plant[0:3], []
            for i, k in enumerate(self.gyro_gains):
                s = filt[3 * i:3 * i + 3]
                y = s + k * (y - s)
                stages.append(y)
            filtered = np.concatenate([y, hist])
            y = (1 - f) * filtered[3 * whole:3 * whole + 3] + f * filtered[3 * whole + 3:3 * whole + 6]
            e = -(y + w[1:4])
            integ = (1 - alpha) * integ + self.dt * e
            deriv = (1 - alpha_d) * deriv + alpha_d / self.dt * (e - prev)
            speeds = np.concatenate([plant[3:7], tel])
            measured = speeds[4 * nt:4 * nt + 4] + w[4:8]
            if self.motor_lpf:
                lp = lp + self.motor_lpf * (measured - lp)
                measured = lp
            u = k_rate @ e + k_int @ integ + k_deriv @ deriv + self.linearizer @ measured
            line = np.concatenate([u, act])
            plant = self.Ad @ plant + self.Bd @ line[4 * na:4 * na + 4] + np.outer(self.Bt, w[0])
            x = np.concatenate([plant, *stages, filtered[:3 * nh], integ, deriv, e, speeds[:4 * nt], lp,
                                line[:4 * na]])
            return x, u

        n = cuts[-1]
        M, C = tick(np.eye(n), np.zeros((INPUTS, n)))
        W, D = tick(np.zeros((n, INPUTS)), np.eye(INPUTS))
        return M, W, C, D

    def stable(self, design, K=None, delay_ticks=0.0):
        K = self.gains(design) if K is None else K
        if K is None:
            return False
        M = self.system(design, K, delay_ticks)[0]
        return np.abs(np.linalg.eigvals(M)).max() < 1.0

    def kick_iae(self, design, K=None, system=None):
        """∫|roll rate| dt over a roll-rate knock, per unit knock, over the sim's flight
        after it (s); inf if the loop is unstable."""
        if system is None:
            K = self.gains(design) if K is None else K
            if K is None:
                return np.inf
            system = self.system(design, K)
        M = system[0]
        if np.abs(np.linalg.eigvals(M)).max() >= 1.0:
            return np.inf
        x0 = np.zeros(len(M))
        x0[ROLL] = 1.0
        return np.abs(free_response(M, x0, x0, int(HOVER_FLIGHT / self.dt))).sum() * self.dt

    def delay_margin(self, design, K=None, tolerance=1e-3):
        """Largest extra gyro delay (s) the loop stays stable with, to `tolerance` relative;
        −inf if unstable without."""
        K = self.gains(design) if K is None else K
        if K is None or not self.stable(design, K):
            return -np.inf
        lo, hi = 0.0, 1.0
        while self.stable(design, K, hi):
            lo, hi = hi, 2.0 * hi
        while hi - lo > tolerance * hi:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.stable(design, K, mid) else (lo, mid)
        return lo * self.dt

    def trim(self, design, K=None, system=None):
        """|∫ pitch rate| over TRIM after a unit step pitch torque (rot per rot/s²)."""
        M, W, _, _ = self.system(design, K) if system is None else system
        n = len(M)
        # The step held in a last state, so the response is free.
        stepped = np.block([[M, W[:, :1]], [np.zeros((1, n)), np.ones((1, 1))]])
        x0, row = np.zeros(n + 1), np.zeros(n + 1)
        x0[n], row[PITCH] = 1.0, 1.0
        return abs(free_response(stepped, x0, row, int(TRIM / self.dt) + 1).sum()) * self.dt

    def roughness(self, design, K=None, system=None):
        """The fit's recorded noise's motor roughness at hover (V), as flick.rs roughness; None
        without a recording."""
        if self.noise is None:
            return None
        if getattr(self, "_noise_input", None) is None:
            steps = int((SETTLE + HOVER_FLIGHT) / self.dt)
            self._noise_input = np.zeros((steps, INPUTS))
            self._noise_input[:, 1:] = self.noise.at(np.arange(steps) * self.dt)
        volts = forced_response(*(self.system(design, K) if system is None else system),
                                self._noise_input)[int(SETTLE / self.dt):]
        x = 2 * np.pi * ROUGH_HP_HZ * self.dt
        k = x / (x + 1.0)
        lp = sc.signal.lfilter([k], [1.0, k - 1.0], volts, axis=0, zi=(1.0 - k) * volts[:1])[0]
        return float(np.sqrt(((volts - lp) ** 2).sum() / 4.0 / len(volts)))

    def measures(self, design, tolerance=1e-3):
        """(kick IAE s, delay margin s, trim, roughness V or None); None if unstable."""
        K = self.gains(design)
        margin = self.delay_margin(design, K, tolerance)
        if K is None or margin < 0:
            return None
        system = self.system(design, K)
        return (self.kick_iae(design, K, system), margin, self.trim(design, K, system),
                self.roughness(design, K, system))


# Ticks per block of the responses' block powers.
BLOCK = 64


def free_response(M, x0, row, n):
    """row·Mᵏ·x0 for k < n."""
    rows = np.empty((BLOCK, len(M)))
    r = row
    for j in range(BLOCK):
        rows[j] = r
        r = r @ M
    jump = np.linalg.matrix_power(M, BLOCK)
    out, x = [], x0
    for start in range(0, n, BLOCK):
        out.append(rows[:min(BLOCK, n - start)] @ x)
        x = jump @ x
    return np.concatenate(out)


def forced_response(M, W, C, D, w):
    """Outputs C·x + D·w over inputs `w` (ticks × inputs) of x' = M·x + W·w, from rest."""
    n, (steps, m), p = len(M), w.shape, len(C)
    blocks = -(-steps // BLOCK)
    w = np.vstack([w, np.zeros((blocks * BLOCK - steps, m))]).reshape(blocks, BLOCK * m)
    # Over a block: outputs from its start state (seen) and its inputs (toeplitz), and the
    # next block's start state from them (jump, reach).
    seen, toeplitz = np.empty((BLOCK * p, n)), np.zeros((BLOCK * p, BLOCK * m))
    reach = np.empty((n, BLOCK * m))
    power, impulse = np.eye(n), [D]
    for j in range(BLOCK):
        seen[j * p:(j + 1) * p] = C @ power
        reach[:, (BLOCK - 1 - j) * m:(BLOCK - j) * m] = power @ W
        impulse.append(C @ power @ W)
        power = M @ power
    for i in range(BLOCK):
        for j in range(i + 1):
            toeplitz[i * p:(i + 1) * p, j * m:(j + 1) * m] = impulse[i - j]
    starts, x = np.empty((blocks, n)), np.zeros(n)
    for b in range(blocks):
        starts[b] = x
        x = power @ x + reach @ w[b]
    return (starts @ seen.T + w @ toeplitz.T).reshape(blocks * BLOCK, p)[:steps]
