//! The core LQR attitude-rate controller.
//!
//! Every matrix-vector operation flows through whippyalgebra's
//! [`MixedUnitMatrix`], ensuring that the dimensional algebra is correct at
//! compile time:
//!
//! - `Kp (V / rot·s⁻¹) × e_prop (rot/s)  →  voltage (V)` ✓
//! - `Ki (V / rot)      × e_int  (rot)    →  voltage (V)` ✓
//! - `Kd (V / rot·s⁻²) × e_deriv(rot/s²) →  voltage (V)` ✓
//! - `G_h⁺(V / rot·s⁻¹) × e_ff  (rot/s)  →  voltage (V)` ✓
//! - `G  (V / rot²·s⁻²) × ω|ω|  (rot²/s²) →  voltage (V)` ✓
//!
//! Passing the wrong error vector to the wrong gain matrix is a compile error.
//!
//! # Feedforward
//!
//! The feedforward runs an internal reference model in
//! deviation coordinates, δx = [r_ref; δω_ref], relinearized every tick at
//! the measured (floored) motor speeds.  It holds u constant over an h-tick
//! horizon (ZOH) and inverts the attitude response over that horizon:
//!
//! ```text
//!   r_target = r + h·Δr                       (extrapolate the smoothed setpoint)
//!   u_ff     = G_h⁺ · (r_target − F_h·δx)     (F_h·δx: h-tick free response)
//!   δx      ← Ad·δx + Bd·ū_ff                  (one-tick ZOH propagation)
//! ```
//!
//! ū_ff is the part of u_ff the allocator actually delivered, so under
//! saturation the model follows what the motors flew rather than what was
//! asked.  The feedback regulates the measured rate onto r_ref, the model's
//! own rate, not onto the setpoint.  What ū_ff reaches over the horizon is
//! reported back ([`ControllerState::reachable_target`]) so the setpoint
//! estimator can hold its target there instead of running ahead.
//!
//! The reference model sees no measurements, so gyro/eRPM noise never
//! passes through G_h⁺; measured errors are left to the LQR feedback.
//!
//! On the ground (armed before takeoff, or after landing) there is no
//! feedforward, and the reference model stays seeded at the setpoint.
//!
//! # Airframe drag
//!
//! With the `airframe-drag` feature and a blob that carries a drag model,
//! the rotor-drag torque of a tall frame, α_d = C·ā (ā: low-passed body x/y
//! accelerometer), enters the reference model as a known disturbance held
//! over the horizon.  The rate block of Ad is diagonal and rates don't drive
//! motor speeds, so its h-tick response is Σ_{j<h} ad_rr^j ⊙ α_d·dt:
//!
//! ```text
//!   u_ff  = G_h⁺ · (r_target − F_h·δx − Σ_{j<h} ad_rr^j ⊙ α_d·dt)
//!   r_ref ← (one-tick propagation) + α_d·dt
//! ```
//!
//! so the feedforward cancels the torque and r_ref stays on the setpoint.
//!
//! # Thrust curvature
//!
//! Thrust goes as ω|ω|, but the reference model is linear in δω_ref,
//! relinearized at the measured speed ω̂: it credits motor m with
//! k_m·2ω̂·δω where its torque changed by k_m·(ω̂² − (ω̂ − δω)²), short by
//! k_m·δω².  Those terms cancel across a symmetric command, but not when
//! the motors start at different speeds (a CG offset) or move unequally: a
//! hard yaw then tilts the airframe about a diagonal the model never sees.
//! With the blob's thrust torque k_m (see [`crate::gains::ThrustTorque`]),
//! the shortfall is a known disturbance, held over the horizon and
//! propagated like the drag:
//!
//! ```text
//!   α_c = −Σ_m k_m·δω_m²          (roll, pitch)
//! ```
//!
//! It reads only δω_ref, which sees no measurements, so it adds no noise
//! path of its own.  Yaw needs none: its curvature is in range of the
//! affine linearizer.

use crate::allocation::{allocate, balanced_direction};
use crate::gains::{GainStorage, ReferenceModel, ThrustTorque};
use crate::interpolation::{interpolate_feedback, interpolate_gains, InterpolatedGains, ScheduleStart};
use crate::types::*;

use whippyalgebra::nalgebra::zeros;
use whippyunits::quantity;

/// Internal controller state.
pub struct ControllerState {
    /// Leaky integral error: [roll, pitch, yaw].
    integral_error: IntegralErrorVec,
    /// Filtered derivative error: [roll, pitch, yaw].
    derivative_error: DerivErrorVec,
    /// Previous proportional error (for derivative).
    prev_prop_error: PropErrorVec,
    /// Previous setpoint (for feedforward diff).
    prev_setpoint: PropErrorVec,
    /// Minimum linearization speed (shaft rev/s); see [`Self::set_min_rps`].
    min_rps: RotPerSec,
    /// See [`Self::set_min_battery`].
    min_battery: Volt,
    /// Cutoff of the low-pass on reported motor speeds; see [`Self::set_motor_lpf`].
    motor_lpf: Hertz,
    /// Low-passed motor speeds; `None` until the first sample.
    motor_rps_lp: Option<[RotPerSec; 4]>,
    /// Cutoff of the low-pass on the battery reading; see [`Self::set_battery_lpf`].
    battery_lpf: Hertz,
    /// Low-passed battery reading; `None` until the first sample.
    battery_lp: Option<Volt>,
    /// See [`Self::set_effort_trace`].
    effort_trace: bool,
    /// Scheduled speeds through [`SCHEDULE_TRACE_LPF`]; `None` until the
    /// first traced sample.
    schedule_rps_lp: Option<[RotPerSec; 4]>,
    /// Feedforward reference-model rate state r_ref.
    ref_rate: PropErrorVec,
    /// Feedforward reference-model motor-speed deviation δω_ref.
    ref_speed: MotorSpeedVec,
    /// Whether the reference model has been seeded since the last reset.
    ref_seeded: bool,
    /// Whether the previous command was scaled or clipped by allocation.
    saturated: bool,
    /// Flying, as opposed to on the ground (armed, before takeoff or after
    /// landing); see [`Self::update_flight_state`].
    airborne: bool,
    /// How long the landing condition has held.
    landing_s: Seconds,
    /// Low-passed body x/y specific force for the airframe drag model;
    /// `None` until the first sample.
    #[cfg(feature = "airframe-drag")]
    accel_lp: Option<[MetersPerSecSq; 2]>,
    /// Latest telemetry.
    telemetry: LqrTelemetry,
    /// See [`Self::reachable_target`].
    reachable: Option<[RotPerSec; 3]>,
    /// The gains in use, `None` until the first tick.  Interpolation is the
    /// costliest part of a tick, so after the first tick it is spread over
    /// two unless [`Self::set_schedule_every_tick`]: one starts it
    /// ([`ScheduleStart`]) at that tick's speeds, the next finishes it there
    /// and takes the result.  The motor speeds it is scheduled on are
    /// low-passed far below the loop rate.  Takeoff interpolates in full.
    schedule: Option<InterpolatedGains>,
    /// An interpolation started last tick.
    schedule_start: Option<ScheduleStart>,
    /// See [`Self::set_schedule_every_tick`].
    schedule_every_tick: bool,
}

impl ControllerState {
    pub fn new() -> Self {
        Self {
            integral_error: zeros![IntegralAxes, Scalar as f32],
            derivative_error: zeros![DerivativeAxes, Scalar as f32],
            prev_prop_error: zeros![AttitudeAxes, Scalar as f32],
            prev_setpoint: zeros![AttitudeAxes, Scalar as f32],
            min_rps: RotPerSec::new(0.0),
            min_battery: Volt::new(0.0),
            motor_lpf: Hertz::new(0.0),
            motor_rps_lp: None,
            battery_lpf: Hertz::new(0.0),
            battery_lp: None,
            effort_trace: false,
            schedule_rps_lp: None,
            ref_rate: zeros![AttitudeAxes, Scalar as f32],
            ref_speed: zeros![MotorSpeedAxes, Scalar as f32],
            ref_seeded: false,
            saturated: false,
            airborne: false,
            landing_s: Seconds::new(0.0),
            #[cfg(feature = "airframe-drag")]
            accel_lp: None,
            telemetry: LqrTelemetry {
                feedback_v: Default::default(),
                feedback_p_v: Default::default(),
                feedback_i_v: Default::default(),
                feedback_d_v: Default::default(),
                schedule_jitter_v: Default::default(),
                feedforward_v: Default::default(),
                affine_v: Default::default(),
                error_prop: Default::default(),
                error_integral: Default::default(),
                error_integral_rate: Default::default(),
                error_derivative: Default::default(),
                kp_motor0: Default::default(),
                applied_v: Default::default(),
                filtered_rps: Default::default(),
                collective_v: Volt::new(0.0),
                command_scale: 1.0,
                setpoint_rps: Default::default(),
                ff_target_rps: Default::default(),
                status: 0,
            },
            reachable: None,
            schedule: None,
            schedule_start: None,
            schedule_every_tick: false,
        }
    }

    pub fn reset(&mut self) {
        self.integral_error = zeros![IntegralAxes, Scalar as f32];
        self.derivative_error = zeros![DerivativeAxes, Scalar as f32];
        self.prev_prop_error = zeros![AttitudeAxes, Scalar as f32];
        self.prev_setpoint = zeros![AttitudeAxes, Scalar as f32];
        self.ref_rate = zeros![AttitudeAxes, Scalar as f32];
        self.ref_speed = zeros![MotorSpeedAxes, Scalar as f32];
        self.ref_seeded = false;
        self.saturated = false;
        self.airborne = false;
        self.landing_s = Seconds::new(0.0);
        self.motor_rps_lp = None;
        self.schedule_rps_lp = None;
        self.reachable = None;
        self.schedule = None;
        self.schedule_start = None;
        #[cfg(feature = "airframe-drag")]
        {
            self.accel_lp = None;
        }
    }

    /// Set the minimum linearization speed, in shaft rev/s.
    ///
    /// Gains grow without bound as any pitch/roll motor's speed → 0 (thrust
    /// authority ∝ ω), so reported speeds are clamped up to
    /// `max(min_rps, grid min_rps)`.  The same floor sets the idle voltage
    /// the allocator keeps every motor above, so the airframe idles where
    /// the gains were solved.
    pub fn set_min_rps(&mut self, min_rps: RotPerSec) {
        self.min_rps = min_rps;
    }

    /// Set the lowest battery voltage the controller compensates for.
    ///
    /// Voltages are commanded in volts and converted to duty at the battery
    /// reading, so a sagging pack is driven harder to hold them: its current
    /// climbs as it empties, and the pilot feels no sag.  Below this the
    /// controller takes the battery to be at it, so the motors get their
    /// command scaled by reading / floor and the airframe weakens like an
    /// uncompensated one.  0 compensates down to any reading.
    pub fn set_min_battery(&mut self, floor: Volt) {
        self.min_battery = floor.max(Volt::new(0.0));
    }

    /// Set the cutoff of a one-pole low-pass on the reported motor speeds,
    /// which everything downstream reads (gain schedule, feedforward,
    /// affine linearizer, takeoff/landing); 0 turns it off.  Telemetry
    /// jitter in the speeds otherwise reaches the motors through the
    /// schedule's feedforward, most during big rate commands.  The same
    /// filter as BF's `pt1FilterGain` (e.g. on rpm_filter_lpf_hz).
    pub fn set_motor_lpf(&mut self, cutoff: Hertz) {
        self.motor_lpf = cutoff.max(Hertz::new(0.0));
    }

    /// Set the cutoff of a one-pole low-pass on the battery reading, which
    /// converts commanded volts to duty; 0 turns it off.  The pack sags
    /// within a flip, so a reading that lags it overstates the volts the
    /// motors get while they are saturated, when nothing corrects it.
    pub fn set_battery_lpf(&mut self, cutoff: Hertz) {
        self.battery_lpf = cutoff.max(Hertz::new(0.0));
    }

    /// Interpolate the gains in full every tick, for a loop with the time to,
    /// rather than spreading each interpolation over two.
    pub fn set_schedule_every_tick(&mut self, on: bool) {
        self.schedule_every_tick = on;
    }

    /// Trace the feedback's schedule jitter into
    /// [`LqrTelemetry::schedule_jitter_v`], at the cost of a second
    /// interpolation of the feedback gains every tick.
    pub fn set_effort_trace(&mut self, on: bool) {
        self.effort_trace = on;
        if !on {
            self.schedule_rps_lp = None;
        }
    }

    /// This tick's battery reading, through the [`Self::set_battery_lpf`] filter.
    fn filter_battery(&mut self, raw: Volt, dt: Seconds) -> Volt {
        if self.battery_lpf <= Hertz::new(0.0) {
            self.battery_lp = None;
            return raw;
        }
        let x: f32 = (2.0 * core::f32::consts::PI * self.battery_lpf * dt).into();
        let k = x / (1.0 + x);
        let lp = self.battery_lp.map_or(raw, |p| p + k * (raw - p));
        self.battery_lp = Some(lp);
        lp
    }

    /// This tick's motor speeds, through the [`Self::set_motor_lpf`] filter.
    fn filter_motor_speeds(&mut self, raw: [RotPerSec; 4], dt: Seconds) -> [RotPerSec; 4] {
        if self.motor_lpf <= Hertz::new(0.0) {
            self.motor_rps_lp = None;
            return raw;
        }
        low_pass(&mut self.motor_rps_lp, raw, self.motor_lpf, dt)
    }

    /// The feedback voltage the gains' motion faster than
    /// [`SCHEDULE_TRACE_LPF`] adds, or zero with the trace off.
    fn schedule_jitter(&mut self, feedback: MotorVoltageVec, scheduled: [RotPerSec; 4],
                       prop_error: PropErrorVec, gains: &GainStorage, dt: Seconds) -> MotorVoltageVec {
        if !self.effort_trace {
            return zeros![MotorAxes, Scalar as f32];
        }
        let slow = low_pass(&mut self.schedule_rps_lp, scheduled, SCHEDULE_TRACE_LPF, dt);
        let k = interpolate_feedback(&slow, gains);
        feedback - (k.kp * prop_error + k.ki * self.integral_error + k.kd * self.derivative_error)
    }

    pub fn telemetry(&self) -> &LqrTelemetry {
        &self.telemetry
    }

    /// When the last command saturated, the feedforward target
    /// [roll, pitch, yaw] the voltage the motors got reaches over the
    /// horizon; `None` otherwise.  Feed it to
    /// [`crate::setpoint::SetpointEstimator::condition`].
    pub fn reachable_target(&self) -> Option<[RotPerSec; 3]> {
        self.reachable
    }

    /// Run one control iteration.
    ///
    /// The matrix multiplications here are the core of the control law.  Each
    /// one is unit-checked at compile time:
    ///
    /// ```text
    ///   Kp [V / (rot/s)]  ×  e_prop [rot/s]   =  [V]
    ///   Ki [V / rot]       ×  e_int  [rot]      =  [V]
    ///   Kd [V / (rot/s²)] ×  e_deriv [rot/s²]  =  [V]
    ///   G_h⁺[V / (rot/s)] ×  e_ff    [rot/s]   =  [V]
    /// ```
    ///
    /// If any of these pairings were wrong (e.g. passing `e_prop` to `Ki`),
    /// the code would fail to compile because `dims![rot/s]` ≠ `dims![rot]`.
    pub fn update(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        gains: &GainStorage,
    ) -> ControlOutput {
        self.step(sensors, setpoint, None, gains)
    }

    /// [`Self::update`] with the feedforward's h-tick-ahead target supplied
    /// by the caller ([roll, pitch, yaw]) instead of extrapolated as
    /// r + h·Δr.
    pub fn update_with_target(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target: [RotPerSec; 3],
        gains: &GainStorage,
    ) -> ControlOutput {
        self.step(sensors, setpoint, Some(ff_target), gains)
    }

    fn step(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target: Option<[RotPerSec; 3]>,
        gains: &GainStorage,
    ) -> ControlOutput {
        if !gains.loaded {
            self.schedule = None;
            self.schedule_start = None;
            self.telemetry.status = 0;
            return ControlOutput { motor_duty: [0.0; 4] };
        }

        let dt = gains.header.dt;
        let alpha_int: f32 = (dt / gains.header.integral_tau).into();
        let alpha_deriv: f32 = (dt / gains.header.derivative_tau).into();

        let measurement: PropErrorVec = from_array(sensors.gyro_rps);
        let setpoint_vec: SetpointDiffVec = from_array(setpoint.rate_rps);

        let measured_rps = self.filter_motor_speeds(sensors.motor_rps, dt);
        let was_airborne = self.airborne;
        self.update_flight_state(sensors, measured_rps, gains, dt);
        let took_off = self.airborne && !was_airborne;

        // Seed the reference model at the current setpoint so the first tick
        // after a reset produces no feedforward transient.  On the ground it
        // stays seeded there (see the feedforward below), so takeoff doesn't
        // produce one either.
        if !self.ref_seeded || !self.airborne {
            self.ref_rate = setpoint_vec;
            self.ref_speed = zeros![MotorSpeedAxes, Scalar as f32];
            self.prev_setpoint = setpoint_vec;
            self.ref_seeded = true;
        }

        // Proportional error: e = r_ref - y  (both in rot/s → result in rot/s).
        // r_ref is the rate the reference model predicts the feedforward has
        // produced by now, so the feedback corrects only departures from
        // that trajectory instead of pushing toward the same goal (which,
        // while the command saturates, adds up to overshoot).
        let prop_error: PropErrorVec = self.ref_rate - measurement;

        // Setpoint diff for feedforward: Δr = r[k] - r[k-1]  (rot/s)
        let setpoint_diff: SetpointDiffVec = setpoint_vec - self.prev_setpoint;

        // Leaky integrator: e_int[k] = (1-α)·e_int[k-1] + dt·e[k]  (rot), the
        // error's integral over spans short against τ_int; a held error
        // settles at τ_int·e.
        //
        // Must match gen_wfg.py's augmented model (A_aug rows 0–2), which Ki
        // was solved against.  The clamp bounds the low-passed error
        // e_int/τ_int at INTEGRAL_ERROR_RATE_LIMIT.  While the last command
        // was saturated, or on the ground, the error isn't correctable, so
        // the state only leaks (anti-windup).
        let tau_int = gains.header.integral_tau;
        let int_gain = if self.saturated || !self.airborne { Seconds::new(0.0) } else { dt };
        let int_raw: IntegralErrorVec =
            self.integral_error.scale(1.0 - alpha_int) + prop_error * int_gain;
        let int_limit = INTEGRAL_ERROR_RATE_LIMIT * tau_int;
        self.integral_error =
            from_array(to_array(&int_raw).map(|x| x.clamp(-int_limit, int_limit)));

        // Derivative filter: ė[k] = (1-α_d)·ė[k-1] + α_d·(e[k]-e[k-1])/dt
        self.derivative_error = self.derivative_error.scale(1.0 - alpha_deriv)
            + (prop_error - self.prev_prop_error) * (alpha_deriv / dt);

        // --- Gain interpolation (returns typed matrices) ---
        let header = &gains.header;
        let floor = self.min_rps.max(header.min_rps);
        let motor_rps = measured_rps.map(|w| w.clamp(floor, header.max_rps));
        // The feedforward switches on at takeoff, so it starts from gains at
        // this tick's speeds rather than the ground's.
        let interp = match (self.schedule.take(), self.schedule_start.take()) {
            (Some(_), _) if took_off || self.schedule_every_tick => interpolate_gains(&motor_rps, gains),
            (Some(_), Some(start)) => start.finish(gains),
            (Some(current), None) => {
                self.schedule_start = Some(ScheduleStart::new(&motor_rps, gains));
                current
            }
            (None, _) => interpolate_gains(&motor_rps, gains),
        };

        // --- Feedback law: u_fb = Kp·e + Ki·e_int + Kd·ė ---
        //
        // Each of these multiplications is compile-time unit-checked:
        //   KpMatrix [MotorAxes × AttitudeAxes]   ×  PropErrorVec [AttitudeAxes × Scalar]    → MotorVoltageVec
        //   KiMatrix [MotorAxes × IntegralAxes]   ×  IntegralErrorVec [IntegralAxes × Scalar] → MotorVoltageVec
        //   KdMatrix [MotorAxes × DerivativeAxes] ×  DerivErrorVec [DerivativeAxes × Scalar]  → MotorVoltageVec
        let battery_v = self.filter_battery(sensors.battery_voltage, dt).max(self.min_battery).max(MIN_BATTERY_V);

        // Bound the integral's voltage on every motor by scaling its state:
        // enough to trim a generous imbalance (see TRIM_CG_ARM_FRACTION),
        // little enough that windup from a collision stays recoverable.
        let mut u_i: MotorVoltageVec = interp.ki * self.integral_error;
        let u_i_max = u_i.as_uniform().amax();
        let u_i_limit = integral_voltage_limit(gains, battery_v);
        if u_i_max > u_i_limit {
            let s: f32 = (u_i_limit / u_i_max).into();
            self.integral_error = self.integral_error.scale(s);
            u_i = u_i.scale(s);
        }

        let u_p: MotorVoltageVec = interp.kp * prop_error;
        let u_d: MotorVoltageVec = interp.kd * self.derivative_error;
        let feedback: MotorVoltageVec = u_p + u_i + u_d;
        let jitter = self.schedule_jitter(feedback, motor_rps, prop_error, gains, dt);

        // --- Feedforward ---
        // r_target = r + h·Δr unless the caller predicted it.
        let target: PropErrorVec = match ff_target {
            Some(t) => from_array(t),
            None => setpoint_vec + setpoint_diff.scale(gains.header.ff_horizon as f32),
        };
        let disturbance = match (self.airframe_drag(sensors, gains, dt), self.thrust_curvature(gains, dt)) {
            (Some(drag), Some(curvature)) => Some(drag + curvature),
            (drag, curvature) => drag.or(curvature),
        };
        // None on the ground: the speeds there are idle, far below hover,
        // where thrust per volt is small and G_h⁺ large, so a gentle stick
        // ramp asks for ±Vbat, and the airframe resting on the ground can't
        // follow it anyway.
        let feedforward: MotorVoltageVec = if !self.airborne {
            zeros![MotorAxes, Scalar as f32]
        } else {
            self.reference_feedforward(&interp.ref_model, &interp.binv, target, disturbance,
                                       gains.header.ff_horizon, battery_v)
        };

        // --- Affine linearizer: u_aff = −B⁺·C(ω̂) = G · ω̂|ω̂| ---
        //   AffineMatrix [MotorAxes × MotorSpeedSqAxes] × MotorSpeedSqVec → MotorVoltageVec
        // ω̂ is the same clamped operating point the gains were interpolated at.
        let speed_sq: MotorSpeedSqVec = from_array(motor_rps.map(|w| w * w.abs()));
        let affine: MotorVoltageVec = gains.affine_matrix() * speed_sq;

        // --- Total control voltage ---
        let total: MotorVoltageVec = feedback + feedforward + affine;

        // --- Allocation: fit the command into [v_idle, Vbat] along the
        // thrust-balanced direction (see `allocation`), then voltage → duty.
        let drag_fit = gains.motor_drag;
        let idle_v = (drag_fit.k_w * floor + drag_fit.k_w2 * floor * floor)
            .clamp(Volt::new(0.0), 0.5 * battery_v);
        let command = to_array(&total);
        let pilot_collective = setpoint.throttle.clamp(0.0, 1.0) * battery_v;
        let direction = balanced_direction(gains.thrust_balance, gains.motor_drag, &gains.affine,
                                           pilot_collective.max(idle_v));
        let alloc = allocate(command, direction, pilot_collective, idle_v, battery_v);
        let output = ControlOutput {
            motor_duty: alloc.voltage.map(|v| f32::from(v / battery_v).clamp(0.0, 1.0)),
        };

        self.saturated = alloc.saturated;
        self.reachable = None;

        // Propagate the reference model with the feedforward the motors
        // actually got: each motor's share of its command that survived
        // scaling/clipping (the collective c·n is not part of the command).
        if self.airborne {
            let realized: MotorVoltageVec = if alloc.saturated {
                let ff = to_array(&feedforward);
                from_array(core::array::from_fn(|m| {
                    let want = command[m];
                    let got = alloc.voltage[m] - alloc.collective_v * direction[m];
                    let kept = if want > KEPT_MIN_COMMAND || want < -KEPT_MIN_COMMAND {
                        f32::from(got / want).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    kept * ff[m]
                }))
            } else {
                feedforward
            };
            if alloc.saturated {
                self.reachable = Some(to_array(&self.horizon_rate(
                    &interp.ref_model, realized, disturbance, gains.header.ff_horizon)));
            }
            self.propagate_reference(&interp.ref_model, realized, disturbance);
        }

        // --- Populate telemetry ---
        let kp = interp.kp.as_uniform();
        self.telemetry = LqrTelemetry {
            feedback_v: to_array(&feedback),
            feedback_p_v: to_array(&u_p),
            feedback_i_v: to_array(&u_i),
            feedback_d_v: to_array(&u_d),
            schedule_jitter_v: to_array(&jitter),
            feedforward_v: to_array(&feedforward),
            affine_v: to_array(&affine),
            error_prop: to_array(&prop_error),
            error_integral: to_array(&self.integral_error),
            error_integral_rate: to_array(&self.integral_error).map(|x| x / tau_int),
            error_derivative: to_array(&self.derivative_error),
            kp_motor0: [kp.get(0, 0), kp.get(0, 1), kp.get(0, 2)],
            applied_v: alloc.voltage,
            filtered_rps: motor_rps,
            collective_v: alloc.collective_v,
            command_scale: alloc.command_scale,
            setpoint_rps: to_array(&setpoint_vec),
            ff_target_rps: to_array(&target),
            status: 1,
        };

        // --- Save state ---
        self.prev_prop_error = prop_error;
        self.prev_setpoint = setpoint_vec;
        self.schedule = Some(interp);

        output
    }
}

impl Default for ControllerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ControllerState {
    /// Track takeoff and landing, which gate integral accumulation and the
    /// feedforward.  Latches
    /// airborne at TAKEOFF_SPEED_FRACTION of hover; lands after the slow,
    /// quiet (and, with an accelerometer, 1 g) condition holds.  Without a
    /// hover speed in the blob the controller counts as always airborne.
    fn update_flight_state(&mut self, sensors: &SensorData, motor_rps: [RotPerSec; 4],
                           gains: &GainStorage, dt: Seconds) {
        let hover = gains.hover_rps;
        if hover <= RotPerSec::new(0.0) {
            self.airborne = true;
            return;
        }
        let w = motor_rps;
        let mean_rps = 0.25 * (w[0] + w[1] + w[2] + w[3]);
        if !self.airborne {
            if mean_rps >= TAKEOFF_SPEED_FRACTION * hover {
                self.airborne = true;
                self.landing_s = Seconds::new(0.0);
            }
            return;
        }
        let quiet = sensors.gyro_rps.iter().all(|r| r.abs() < LANDED_RATE);
        let a = sensors.accel;
        let accel_mag: MetersPerSecSq = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let has_accel = accel_mag > ACCEL_PRESENT;
        let supported =
            !has_accel || (accel_mag - STANDARD_GRAVITY).abs() < LANDED_ACCEL_TOLERANCE;
        if mean_rps < LANDED_SPEED_FRACTION * hover && quiet && supported {
            self.landing_s += dt;
            let hold = if has_accel { LANDED_HOLD } else { LANDED_HOLD_NO_ACCEL };
            if self.landing_s >= hold {
                self.airborne = false;
                self.landing_s = Seconds::new(0.0);
            }
        } else {
            self.landing_s = Seconds::new(0.0);
        }
    }

    /// This tick's airframe drag disturbance α_d·dt, or `None`
    /// when the blob has no drag model or the feature is off.
    #[cfg(feature = "airframe-drag")]
    fn airframe_drag(&mut self, sensors: &SensorData, gains: &GainStorage, dt: Seconds) -> Option<PropErrorVec> {
        let drag = gains.airframe_drag;
        if drag.lp <= Hertz::new(0.0) {
            return None;
        }
        let limit = DRAG_ACCEL_LIMIT;
        let a = [0, 1].map(|i| sensors.accel[i].clamp(-limit, limit));
        // One-pole low-pass; 2πf·dt/(1 + 2πf·dt) ≈ 1 − e^{−2πf·dt} at f ≪ 1/dt.
        let x: f32 = (2.0 * core::f32::consts::PI * drag.lp * dt).into();
        let k = x / (1.0 + x);
        let lp = match self.accel_lp {
            Some(p) => [p[0] + k * (a[0] - p[0]), p[1] + k * (a[1] - p[1])],
            None => a,
        };
        self.accel_lp = Some(lp);
        Some(from_array([
            (drag.roll[0] * lp[0] + drag.roll[1] * lp[1]) * dt,
            (drag.pitch[0] * lp[0] + drag.pitch[1] * lp[1]) * dt,
            RotPerSec::new(0.0),
        ]))
    }

    #[cfg(not(feature = "airframe-drag"))]
    fn airframe_drag(&mut self, _: &SensorData, _: &GainStorage, _: Seconds) -> Option<PropErrorVec> {
        None
    }

    /// This tick's thrust-curvature correction to the reference model,
    /// −Σ_m k_m·δω_m²·dt on roll and pitch (see module docs), or `None` when
    /// the blob has no thrust torque.
    fn thrust_curvature(&self, gains: &GainStorage, dt: Seconds) -> Option<PropErrorVec> {
        let torque = gains.thrust_torque;
        if torque == ThrustTorque::ZERO {
            return None;
        }
        let speed = to_array(&self.ref_speed);
        let lost = |k: &[PerRot; 4]| {
            (0..4).fold(RotPerSecSq::new(0.0), |acc, m| acc - k[m] * speed[m] * speed[m]) * dt
        };
        Some(from_array([lost(&torque.roll), lost(&torque.pitch), RotPerSec::new(0.0)]))
    }

    /// The reference-model feedforward command for this tick (see module docs).
    ///
    /// `ref_model` is the interpolated reference model at the current
    /// operating point; `binv` is the interpolated horizon inverse G_h⁺.
    fn reference_feedforward(
        &self,
        ref_model: &ReferenceModel,
        binv: &BInvMatrix,
        target: PropErrorVec,
        disturbance: Option<PropErrorVec>,
        horizon: u8,
        battery_v: Volt,
    ) -> MotorVoltageVec {
        let mut request: SetpointDiffVec = target - self.free_response(ref_model);
        if let Some(d) = disturbance {
            request -= held_response(ref_model, d, horizon);
        }
        let u_ff: MotorVoltageVec = *binv * request;
        from_array(to_array(&u_ff).map(|v| v.clamp(-battery_v, battery_v)))
    }

    /// The reference model's h-tick free response F_h·δx.  The rate→rate
    /// blocks are diagonal (dimensionless decay), applied elementwise.
    fn free_response(&self, m: &ReferenceModel) -> PropErrorVec {
        let rate = to_array(&self.ref_rate);
        let from_speed = to_array(&(m.fh_rw * self.ref_speed));
        from_array(core::array::from_fn(|i| m.fh_rr[i] * rate[i] + from_speed[i]))
    }

    /// The reference model's rate after `horizon` ticks of `u` and the
    /// disturbance held, F_h·δx + G_h·u + Σ_{j<h} ad_rr^j ⊙ d: what that
    /// feedforward reaches.
    fn horizon_rate(&self, m: &ReferenceModel, u: MotorVoltageVec, disturbance: Option<PropErrorVec>,
                    horizon: u8) -> PropErrorVec {
        let rate = self.free_response(m) + m.gh * u;
        match disturbance {
            Some(d) => rate + held_response(m, d, horizon),
            None => rate,
        }
    }

    /// One-tick ZOH propagation of the reference model with the voltage
    /// it actually got, plus the disturbance (airframe drag, thrust
    /// curvature): δx ← Ad·δx + Bd·u + [d; 0].
    fn propagate_reference(
        &mut self,
        ref_model: &ReferenceModel,
        u_ff: MotorVoltageVec,
        disturbance: Option<PropErrorVec>,
    ) {
        let rate = to_array(&self.ref_rate);
        let speed = to_array(&self.ref_speed);
        let u = to_array(&u_ff);
        let from_speed = to_array(&(ref_model.ad_rw * self.ref_speed));
        let from_volt = to_array(&(ref_model.bd_r * u_ff));
        let disturbance = disturbance.map_or([RotPerSec::new(0.0); 3], |d| to_array(&d));
        self.ref_rate = from_array(core::array::from_fn(|i| {
            ref_model.ad_rr[i] * rate[i] + from_speed[i] + from_volt[i] + disturbance[i]
        }));
        self.ref_speed = from_array(core::array::from_fn(|m| {
            ref_model.ad_ww[m] * speed[m] + ref_model.bd_w[m] * u[m]
        }));
    }
}

/// The reference model's rate response to the disturbance `d` held over
/// `horizon` ticks: Σ_{j<h} ad_rr^j ⊙ d.
fn held_response(m: &ReferenceModel, d: PropErrorVec, horizon: u8) -> PropErrorVec {
    let mut decay = [1.0f32; 3];
    let mut sum = [0.0f32; 3];
    for _ in 0..horizon {
        for i in 0..3 {
            sum[i] += decay[i];
            decay[i] *= m.ad_rr[i];
        }
    }
    let d = to_array(&d);
    from_array(core::array::from_fn(|i| sum[i] * d[i]))
}

/// Bound on each accelerometer axis fed to the drag model, so impacts
/// don't charge its low-pass.
#[cfg(feature = "airframe-drag")]
const DRAG_ACCEL_LIMIT: MetersPerSecSq = mps2(quantity!(3.0, g_n, f32));

/// Cutoff of the reference schedule for
/// [`LqrTelemetry::schedule_jitter_v`]: a decade under the band where
/// roughness is measured, so the jitter holds all of the schedule's motion
/// in that band and little of a maneuver's.
pub const SCHEDULE_TRACE_LPF: Hertz = Hertz::new(5.0);

/// One step of a one-pole low-pass (BF's `pt1FilterGain`) on `state`,
/// seeded at the first sample.
fn low_pass(state: &mut Option<[RotPerSec; 4]>, raw: [RotPerSec; 4], cutoff: Hertz,
            dt: Seconds) -> [RotPerSec; 4] {
    let x: f32 = (2.0 * core::f32::consts::PI * cutoff * dt).into();
    let k = x / (1.0 + x);
    let lp = match *state {
        Some(p) => core::array::from_fn(|m| p[m] + k * (raw[m] - p[m])),
        None => raw,
    };
    *state = Some(lp);
    lp
}

/// Bound on the integral state's low-passed error e_int/τ_int.
const INTEGRAL_ERROR_RATE_LIMIT: RotPerSec = RotPerSec::new(0.25);

/// Battery reading below which the controller assumes this voltage (a
/// missing or implausible reading), so voltage → duty stays finite.
const MIN_BATTERY_V: Volt = Volt::new(1.0);

/// Commands smaller than this count as fully delivered when sharing out a
/// saturated command, so the ratio stays finite.
const KEPT_MIN_COMMAND: Volt = Volt::new(1e-6);
/// Largest integral voltage on any motor, as a fraction of battery voltage.
const INTEGRAL_VOLTAGE_FRACTION: f32 = 0.2;

/// Per-motor integral voltage bound: the blob's (what trimming a generous
/// CG offset at hover takes — 25% of the arm, the steady part of wind
/// included — so trim never runs out, while a collision's windup leaves a
/// recoverable torque), capped at INTEGRAL_VOLTAGE_FRACTION of the battery;
/// that cap alone for blobs without one.
fn integral_voltage_limit(gains: &GainStorage, battery_v: Volt) -> Volt {
    let cap = INTEGRAL_VOLTAGE_FRACTION * battery_v;
    let trim = gains.integral_limit_v;
    if trim > Volt::new(0.0) && trim < cap { trim } else { cap }
}

/// Takeoff: mean motor speed reaches this fraction of hover (≈70% of the
/// hover thrust).
const TAKEOFF_SPEED_FRACTION: f32 = 0.85;
/// Landing candidates: mean motor speed below this fraction of hover
/// (thrust under ≈⅓ of weight) with body rates under LANDED_RATE.
const LANDED_SPEED_FRACTION: f32 = 0.6;
const LANDED_RATE: RotPerSec = RotPerSec::new(0.05);
/// With an accelerometer, ground contact also reads a steady 1 g₀ (±this)
/// although the motors make well under it; in the air at that thrust the
/// specific force is the low thrust itself (falling reads ≪ 1 g₀).
const LANDED_ACCEL_TOLERANCE: MetersPerSecSq = mps2(quantity!(0.15, g_n, f32));
/// How long the landing condition must hold: with / without accelerometer.
const LANDED_HOLD: Seconds = Seconds::new(0.5);
const LANDED_HOLD_NO_ACCEL: Seconds = Seconds::new(1.0);
/// Below this magnitude the accelerometer is taken as absent (not fitted,
/// or USE_ACC off: the glue leaves it zero).
const ACCEL_PRESENT: MetersPerSecSq = mps2(quantity!(0.2, g_n, f32));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gains::REF_FLOATS;

    /// A reference model whose F_h and G_h tabulate `horizon` steps of its
    /// own one-tick blocks, as the blob's do.
    fn consistent_model(horizon: u8) -> ReferenceModel {
        let mut r = [0.0f32; REF_FLOATS];
        r[0..3].copy_from_slice(&[0.999, 0.998, 0.9995]);
        for k in 0..12 {
            r[3 + k] = 1e-3 * (k as f32 - 5.5);
            r[19 + k] = 0.02 * (if k % 2 == 0 { 1.0 } else { -0.7 }) * (1.0 + 0.1 * k as f32);
        }
        r[15..19].copy_from_slice(&[0.97, 0.96, 0.975, 0.965]);
        r[31..35].copy_from_slice(&[1.5, 1.4, 1.6, 1.45]);
        let one_tick = r;
        let step = |rate: [f32; 3], speed: [f32; 4], u: [f32; 4]| {
            let r = &one_tick;
            let rate = core::array::from_fn(|i| {
                r[i] * rate[i] + (0..4).map(|m| r[3 + 4 * i + m] * speed[m] + r[19 + 4 * i + m] * u[m]).sum::<f32>()
            });
            (rate, core::array::from_fn(|m| r[15 + m] * speed[m] + r[31 + m] * u[m]))
        };
        let run = |mut rate: [f32; 3], mut speed: [f32; 4], u: [f32; 4]| {
            for _ in 0..horizon {
                (rate, speed) = step(rate, speed, u);
            }
            rate
        };
        for i in 0..3 {
            let mut e = [0.0; 3];
            e[i] = 1.0;
            r[35 + i] = run(e, [0.0; 4], [0.0; 4])[i];
        }
        for m in 0..4 {
            let mut e = [0.0; 4];
            e[m] = 1.0;
            let (free, forced) = (run([0.0; 3], e, [0.0; 4]), run([0.0; 3], [0.0; 4], e));
            for i in 0..3 {
                r[38 + 4 * i + m] = free[i];
                r[50 + 4 * i + m] = forced[i];
            }
        }
        ReferenceModel::from_slots(&r)
    }

    #[test]
    fn horizon_rate_is_the_stepped_reference_model() {
        let horizon = 16;
        let m = consistent_model(horizon);
        let mut ctrl = ControllerState::new();
        ctrl.ref_rate = from_array([0.3, -0.2, 0.5].map(RotPerSec::new));
        ctrl.ref_speed = from_array([4.0, -3.0, 2.0, -1.0].map(RotPerSec::new));
        let u: MotorVoltageVec = from_array([1.0, -0.5, 0.25, 2.0].map(Volt::new));
        let d: PropErrorVec = from_array([0.01, -0.02, 0.005].map(RotPerSec::new));
        let reached = to_array(&ctrl.horizon_rate(&m, u, Some(d), horizon));
        for _ in 0..horizon {
            ctrl.propagate_reference(&m, u, Some(d));
        }
        let stepped = to_array(&ctrl.ref_rate);
        for i in 0..3 {
            assert!((reached[i] - stepped[i]).abs() < RotPerSec::new(1e-4), "{reached:?} vs {stepped:?}");
        }
    }
}
