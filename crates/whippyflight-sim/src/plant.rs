//! The nonlinear plant that gen_wfg.py linearizes, from a coefficient JSON
//! (see the Python `whippyflight.coefficients` for its frame).

use crate::units::*;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{MotorLayout, MotorPosition, Spin};
use whippyflight_control::setpoint::{DEFAULT_LAMBDA_500HZ, SetpointEstimator};
use whippyflight_control::types::{RotPerSec, Volt};

/// fit_model.py's `motor_speed`: the unit its motor speeds are in.  Older fits
/// (without it) are in eRPM/600, which this can't convert without the pole
/// count they were logged with.
pub const MOTOR_SPEED_UNIT: &str = "shaft rev/s";

/// Battery voltage and hover motor speed when the JSON has no fitted
/// `airframe` block: the Meteor65 Pro (1S) test airframe.
const DEFAULT_BATTERY_V: Voltage = Voltage::new(4.2);
const DEFAULT_HOVER_RPS: Rate = Rate::new(450.0);
/// The most latency a fit may record, per stage.
const MAX_LATENCY_MS: f64 = 50.0;
/// Its gyro low-pass stages (type, cutoff), for JSONs without `gyro_lpf`.
const DEFAULT_GYRO_LPF: [(&str, Freq); 2] = [("PT1", Freq::new(275.0)), ("PT1", Freq::new(550.0))];

/// Betaflight's QUADX mixer by motor index and its default (props-in)
/// spins, for legacy fits (without a `frame` block); `yaw_motors_reversed`
/// flips the spins.
const BETAFLIGHT_QUADX: [MotorPosition; 4] = [
    MotorPosition::RearRight,
    MotorPosition::FrontRight,
    MotorPosition::RearLeft,
    MotorPosition::FrontLeft,
];
const BETAFLIGHT_PROPS_IN: [Spin; 4] = [Spin::Clockwise, Spin::CounterClockwise, Spin::CounterClockwise, Spin::Clockwise];

/// Plant coefficients in the canonical frame: FRD axes, the torque patterns
/// by slot ([`MotorPosition`] code), motors by firmware index.
pub struct Model {
    pub yaw_rate: PerTime,
    pub yaw_prop_vel_sq: PerRot,
    /// rot/s² of yaw per rot/s² of motor acceleration.
    pub yaw_prop_accel: f64,
    pub pitch_rate: PerTime,
    pub pitch_prop_vel_sq: PerRot,
    pub roll_rate: PerTime,
    pub roll_prop_vel_sq: PerRot,
    pub motor_vel: MotorVel,
    pub motor_vel_sq: MotorVelSq,
    pub motor_accel: MotorAccel,
    /// gen_wfg arm_weights: 2·d/(d_F + d_B) and 2·d/(d_L + d_R).
    pub w_front: f64,
    pub w_back: f64,
    pub w_left: f64,
    pub w_right: f64,
    /// Where each firmware motor sits and how it spins.
    pub layout: MotorLayout,
    /// Flight battery voltage and median hover motor speed (shaft rev/s).
    pub battery_v: Voltage,
    pub hover_rps: Rate,
    /// The firmware's gyro low-pass stages on the logged airframe (type,
    /// cutoff); a dynamic stage is at its minimum (lowest, most lag) cutoff.
    pub gyro_lpf: Vec<(String, Freq)>,
    /// Cutoff of the controller's low-pass on reported motor speeds
    /// (`ControllerState::set_motor_lpf`); 0, off, without `rpm_lpf_hz`.
    pub motor_lpf: Freq,
    /// The firmware's minimum linearization speed, `idle_rps`
    /// (`ControllerState::set_min_rps`); 0, the gain grid's own floor,
    /// without it.
    pub min_rps: RotPerSec,
    /// Stick setpoint tracking index λ at 500 Hz
    /// (`SetpointEstimator::set_lambda`), from `stick_tracking`.
    pub stick_tracking: f32,
    /// Dead time between a motor command and the motor answering it (the
    /// ESC), and between the motor's speed and its report (the telemetry),
    /// from `latency`; 0 without it.
    pub actuator_delay: Time,
    pub telemetry_delay: Time,
    /// Where the coefficients came from.
    pub source: String,
}

/// A fixed delay of whole controller ticks: [`Model::actuator_delay`] on the
/// motor commands, [`Model::telemetry_delay`] on the speed reports.
pub struct DelayLine<T: Copy> {
    queue: std::collections::VecDeque<T>,
    ticks: usize,
}

impl<T: Copy> DelayLine<T> {
    /// `delay` at ticks of `dt`, rounded; the first value pushed fills it.
    pub fn new(delay: Time, dt: Time) -> Self {
        let ticks = Self::ticks(delay, dt);
        Self { queue: std::collections::VecDeque::with_capacity(ticks + 1), ticks }
    }

    /// The delay a line of `delay` at ticks of `dt` holds: whole ticks.
    pub fn flown(delay: Time, dt: Time) -> Time {
        Self::ticks(delay, dt) as f64 * dt
    }

    fn ticks(delay: Time, dt: Time) -> usize {
        f64::from(delay / dt).round().max(0.0) as usize
    }

    /// Push this tick's value; returns the one from `delay` ago.
    pub fn push(&mut self, x: T) -> T {
        if self.queue.is_empty() {
            self.queue.extend(core::iter::repeat_n(x, self.ticks));
        }
        self.queue.push_back(x);
        self.queue.pop_front().unwrap()
    }
}

/// The motor layout of a `frame` block.
fn parse_frame(frame: &serde_json::Value, source: &str) -> Result<MotorLayout, String> {
    if frame["axes"].as_str() != Some("FRD") {
        return Err(format!("{source}: frame.axes must be \"FRD\""));
    }
    let motors = frame["motors"].as_array().filter(|m| m.len() == 4)
        .ok_or_else(|| format!("{source}: frame.motors must list 4 motors"))?;
    let mut codes = [0u8; 8];
    for (m, motor) in motors.iter().enumerate() {
        codes[m] = match motor["position"].as_str() {
            Some("front_left") => MotorPosition::FrontLeft,
            Some("rear_left") => MotorPosition::RearLeft,
            Some("front_right") => MotorPosition::FrontRight,
            Some("rear_right") => MotorPosition::RearRight,
            p => return Err(format!("{source}: motor {m} position {p:?}")),
        } as u8;
        codes[4 + m] = match motor["spin"].as_str() {
            Some("cw") => Spin::Clockwise,
            Some("ccw") => Spin::CounterClockwise,
            s => return Err(format!("{source}: motor {m} spin {s:?}")),
        } as u8;
    }
    MotorLayout::from_codes(&codes).ok_or_else(|| format!("{source}: frame.motors is not a quad-X layout"))
}

/// The layout of a legacy (Betaflight QUADX) fit, from its `airframe` block.
fn betaflight_layout(airframe: &serde_json::Value, source: &str) -> Result<MotorLayout, String> {
    if airframe["mixer"].as_str().is_some_and(|m| m != "QUADX") {
        return Err(format!("{source}: mixer {}: only QUADX is supported", airframe["mixer"]));
    }
    let reversed = airframe["yaw_motors_reversed"].as_bool().unwrap_or(false);
    let flip = |s: Spin| match s {
        Spin::Clockwise => Spin::CounterClockwise,
        Spin::CounterClockwise => Spin::Clockwise,
    };
    Ok(MotorLayout { position: BETAFLIGHT_QUADX, spin: BETAFLIGHT_PROPS_IN.map(|s| if reversed { flip(s) } else { s }) })
}

impl Model {
    /// Parse a coefficient JSON.  A legacy Betaflight fit (no `frame`
    /// block, the frame turned 180° about z) is converted as
    /// `whippyflight.coefficients.normalize` does.
    pub fn from_json(text: &str, source: &str) -> Result<Model, String> {
        let c: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{source}: {e}"))?;
        if c["motor_speed"].as_str() != Some(MOTOR_SPEED_UNIT) {
            return Err(format!(
                "{source}: fitted before motor speeds were in {MOTOR_SPEED_UNIT}; fit it again"
            ));
        }
        let get = |axis: &str, key: &str| c[axis][key].as_f64().ok_or_else(|| format!("{source}: no {axis}.{key}"));
        let legacy = c.get("frame").is_none();
        // Arms without fitted ones: symmetric, or what gen_wfg assumed for
        // legacy fits (in their frame, whose front and left are the
        // physical rear and right).
        let arm = |axis: &str, key: &str, legacy_default: f64| {
            c[axis][key].as_f64().unwrap_or(if legacy { legacy_default } else { 1.0 })
        };
        let (mut d_front, mut d_back) = (arm("pitch", "dFront", 1.0), arm("pitch", "dBack", 1.7));
        let (mut d_left, mut d_right) = (arm("roll", "dLeft", 1.0), arm("roll", "dRight", 1.0));
        let layout = if legacy {
            core::mem::swap(&mut d_front, &mut d_back);
            core::mem::swap(&mut d_left, &mut d_right);
            betaflight_layout(&c["airframe"], source)?
        } else {
            parse_frame(&c["frame"], source)?
        };
        let gyro_lpf = match c["airframe"]["gyro_lpf"].as_array() {
            Some(stages) => stages
                .iter()
                .map(|s| {
                    let kind = s["type"].as_str().ok_or_else(|| format!("{source}: gyro_lpf type"))?;
                    let hz = s["hz"].as_f64().ok_or_else(|| format!("{source}: gyro_lpf hz"))?;
                    Ok((kind.to_string(), Freq::new(hz)))
                })
                .collect::<Result<_, String>>()?,
            None => DEFAULT_GYRO_LPF.iter().map(|&(k, f)| (k.to_string(), f)).collect(),
        };
        let latency = |key: &str| -> Result<Time, String> {
            match c["latency"][key].as_f64() {
                None => Ok(Time::new(0.0)),
                Some(ms) if ms.is_finite() && (0.0..=MAX_LATENCY_MS).contains(&ms) => Ok(Time::new(ms * 1e-3)),
                Some(ms) => Err(format!("{source}: latency.{key} {ms} ms")),
            }
        };
        Ok(Model {
            yaw_rate: PerTime::new(get("yaw", "yawRate")?),
            yaw_prop_vel_sq: PerRot::new(get("yaw", "propVelSumSq")?),
            yaw_prop_accel: get("yaw", "propAccelSum")?,
            pitch_rate: PerTime::new(get("pitch", "pitchRate")?),
            pitch_prop_vel_sq: PerRot::new(get("pitch", "propVelSumSq")?),
            roll_rate: PerTime::new(get("roll", "rollRate")?),
            roll_prop_vel_sq: PerRot::new(get("roll", "propVelSumSq")?),
            motor_vel: MotorVel::new(get("motors", "motorVel")?),
            motor_vel_sq: MotorVelSq::new(get("motors", "motorVelSq")?),
            motor_accel: MotorAccel::new(get("motors", "motorAccel")?),
            w_front: 2.0 * d_front / (d_front + d_back),
            w_back: 2.0 * d_back / (d_front + d_back),
            w_left: 2.0 * d_left / (d_left + d_right),
            w_right: 2.0 * d_right / (d_left + d_right),
            layout,
            battery_v: c["airframe"]["vbat_v"].as_f64().map_or(DEFAULT_BATTERY_V, Voltage::new),
            hover_rps: c["airframe"]["hover_rps"].as_f64().map_or(DEFAULT_HOVER_RPS, Rate::new),
            gyro_lpf,
            motor_lpf: Freq::new(c["airframe"]["rpm_lpf_hz"].as_f64().unwrap_or(0.0)),
            min_rps: RotPerSec::new(c["airframe"]["idle_rps"].as_f64().unwrap_or(0.0) as f32),
            stick_tracking: c["airframe"]["stick_tracking"]
                .as_f64()
                .map_or(DEFAULT_LAMBDA_500HZ, |l| l as f32),
            actuator_delay: latency("actuator_ms")?,
            telemetry_delay: latency("telemetry_ms")?,
            source: source.to_string(),
        })
    }

    /// Set up a controller and setpoint estimator as the firmware would on
    /// the logged airframe's settings.
    pub fn configure(&self, ctrl: &mut ControllerState, sp: &mut SetpointEstimator) {
        ctrl.set_motor_lpf(self.motor_lpf.lossy_into());
        ctrl.set_min_rps(self.min_rps);
        sp.set_lambda(self.stick_tracking);
    }

    /// Steady-state voltage holding a motor at `omega`.
    pub fn motor_volts(&self, omega: Rate) -> Voltage {
        self.motor_vel * omega + self.motor_vel_sq * omega * omega.abs()
    }

    /// Throttle (fraction of battery) that holds every motor at `scale` × hover speed.
    pub fn throttle_at(&self, scale: f64) -> f32 {
        f64::from(self.motor_volts(scale * self.hover_rps) / self.battery_v) as f32
    }

    pub fn battery_v(&self) -> Volt {
        self.battery_v.lossy_into()
    }

    /// Motor voltages for the plant from the controller's duty output.
    pub fn duty_volts(&self, duty: [f32; 4]) -> [Voltage; 4] {
        duty.map(|d| (d * self.battery_v()).lossy_into())
    }

    /// Per firmware motor, its weight in the [roll, pitch, yaw] torques of
    /// the speeds' signed squares: by its slot, front and left motors push
    /// pitch and roll up, front-left and rear-right yaw.
    pub fn torque_patterns(&self) -> [[f64; 4]; 3] {
        let by_slot = [
            [self.w_left, self.w_left, -self.w_right, -self.w_right],
            [self.w_front, -self.w_back, self.w_front, -self.w_back],
            [1.0, -1.0, -1.0, 1.0],
        ];
        by_slot.map(|p| self.layout.position.map(|pos| p[pos as usize]))
    }
}

/// Plant state: FRD rates [roll, pitch, yaw], motor shaft speeds by
/// firmware index.
#[derive(Clone, Copy)]
pub struct Plant<'m> {
    pub rate: [Rate; 3],
    pub omega: [Rate; 4],
    pub model: &'m Model,
    patterns: [[f64; 4]; 3],
}

/// Time derivative of a [`Plant`].
#[derive(Clone, Copy)]
struct PlantDeriv {
    rate: [AngAccel; 3],
    omega: [AngAccel; 4],
}

impl<'m> Plant<'m> {
    /// At rest with every motor at `scale` × the model's hover speed.
    pub fn hovering(model: &'m Model, scale: f64) -> Self {
        Plant {
            rate: [Rate::new(0.0); 3],
            omega: [scale * model.hover_rps; 4],
            model,
            patterns: model.torque_patterns(),
        }
    }

    /// `torque`: external angular acceleration, FRD.
    fn deriv(&self, volts: &[Voltage; 4], torque: &[AngAccel; 3]) -> PlantDeriv {
        let m = self.model;
        let sq = self.omega.map(|w| w * w.abs());
        let mut wdot = [AngAccel::new(0.0); 4];
        for i in 0..4 {
            wdot[i] = (volts[i] - m.motor_vel * self.omega[i] - m.motor_vel_sq * sq[i]) / m.motor_accel;
        }
        let [s_roll, s_pitch, s_yaw] = &self.patterns;
        let dot_sq = |s: &[f64; 4]| (0..4).fold(SpeedSq::new(0.0), |acc, i| acc + s[i] * sq[i]);
        let dot_wdot = |s: &[f64; 4]| (0..4).fold(AngAccel::new(0.0), |acc, i| acc + s[i] * wdot[i]);
        PlantDeriv {
            rate: [
                m.roll_rate * self.rate[0] + m.roll_prop_vel_sq * dot_sq(s_roll) + torque[0],
                m.pitch_rate * self.rate[1] + m.pitch_prop_vel_sq * dot_sq(s_pitch) + torque[1],
                m.yaw_rate * self.rate[2]
                    + m.yaw_prop_vel_sq * dot_sq(s_yaw)
                    + m.yaw_prop_accel * dot_wdot(s_yaw)
                    + torque[2],
            ],
            omega: wdot,
        }
    }

    fn axpy(&self, k: &PlantDeriv, h: Time) -> Self {
        Plant {
            rate: core::array::from_fn(|i| self.rate[i] + h * k.rate[i]),
            omega: core::array::from_fn(|i| self.omega[i] + h * k.omega[i]),
            ..*self
        }
    }

    pub fn rk4(&mut self, volts: &[Voltage; 4], h: Time) {
        self.rk4_with_torque(volts, h, [AngAccel::new(0.0); 3]);
    }

    /// [`Self::rk4`] with an external angular acceleration (FRD), held over
    /// the step.
    pub fn rk4_with_torque(&mut self, volts: &[Voltage; 4], h: Time, torque: [AngAccel; 3]) {
        let k1 = self.deriv(volts, &torque);
        let k2 = self.axpy(&k1, h / 2.0).deriv(volts, &torque);
        let k3 = self.axpy(&k2, h / 2.0).deriv(volts, &torque);
        let k4 = self.axpy(&k3, h).deriv(volts, &torque);
        self.rate = core::array::from_fn(|i| {
            self.rate[i] + h / 6.0 * (k1.rate[i] + 2.0 * k2.rate[i] + 2.0 * k3.rate[i] + k4.rate[i])
        });
        self.omega = core::array::from_fn(|i| {
            self.omega[i] + h / 6.0 * (k1.omega[i] + 2.0 * k2.omega[i] + 2.0 * k3.omega[i] + k4.omega[i])
        });
    }

    /// The gyro: body rates, FRD.
    pub fn gyro(&self) -> [RotPerSec; 3] {
        self.rate.map(|r| r.lossy_into())
    }

    /// Motor speed telemetry as the controller reads it.
    pub fn motor_rps(&self) -> [RotPerSec; 4] {
        self.omega.map(|w| w.lossy_into())
    }
}
