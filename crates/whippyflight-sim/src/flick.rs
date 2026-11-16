//! Stick flicks on the nonlinear plant: the controller and setpoint
//! estimator fly a raised-cosine flick from hover, held and released, with
//! RC packets at a receiver rate like the real link.
//!
//! Each flick is flown twice, with and without the stick, and the response
//! is the difference, so the hover's own settling doesn't count against it.
//! The flight without the stick is calm: the [`Conditions`]' disturbances and
//! (if [`Conditions::noisy_flights`]) sensor noise stay in the response.

use whippyunits::value;

use crate::margin::GyroPath;
use crate::noise::Noise;
use crate::plant::{DelayLine, Model, Plant};
use crate::units::*;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::GainStorage;
use whippyflight_control::setpoint::{SetpointEstimator, target_limits};
use whippyflight_control::types::*;

pub const F_RX: Freq = Freq::new(500.0);
/// Hover before the stick moves, recorded from its end.
pub const SETTLE: Time = Time::new(0.2);
pub const HOLD: Time = Time::new(0.3);
pub const REST: Time = Time::new(0.3);
/// The flicks the evals and the Configurator fly, gentle to saturating.
pub const AMPS: [Rate; 6] = [
    Rate::new(0.05),
    Rate::new(0.2),
    Rate::new(0.5),
    Rate::new(1.0),
    Rate::new(2.0),
    Rate::new(4.0),
];
pub const RISES: [Time; 2] = [Time::new(0.030), Time::new(0.010)];
/// Axes flown (FRD): roll, pitch.
pub const AXES: [usize; 2] = [0, 1];
/// Motor roughness is the RMS of the commanded voltages high-passed here
/// (whippyflight_betaflight.extract_noise ROUGH_HP_HZ, BF's own figure for a log).
pub const ROUGH_HP: Freq = Freq::new(50.0);

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// The controller's microsecond clock at plant time `t` (truncated).
pub fn micros(t: Time) -> Micros {
    Micros::new(value!(t, us) as u32)
}

/// One flight, per controller tick from the end of [`SETTLE`], on the flicked axis.
pub struct Trace {
    pub gyro: Vec<Rate>,
    /// The setpoint estimator's reconstructed stick.
    pub setpoint: Vec<Rate>,
    /// The feedforward target it hands the controller.
    pub target: Vec<Rate>,
    pub stick: Vec<Rate>,
    /// Whether the allocator scaled the command down to fit the motors.
    pub saturated: Vec<bool>,
    /// Commanded motor voltages.
    pub volts: Vec<[Voltage; 4]>,
    /// The motors' shaft speeds, by firmware index (the plant's, without
    /// telemetry noise).
    pub motor_speeds: Vec<[Rate; 4]>,
}

/// The setpoint estimator's target slew (rot/s²) and jerk (rot/s³) limits;
/// [`target_limits`] of the blob when not overridden.
#[derive(Clone, Copy, Default)]
pub struct Limits {
    pub slew: Option<f32>,
    pub jerk: Option<f32>,
}

/// Extra hover before the stick when the CG is offset, for the integral to
/// take up its torque first, as it has in flight.
pub const CG_SETTLE: Time = Time::new(3.0);

/// Turbulence: on each axis, white noise low-passed (one pole) at
/// `bandwidth` and scaled to that axis's `rms` angular acceleration (FRD
/// [roll, pitch, yaw]), from `seed` so flights repeat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turbulence {
    pub rms: [AngAccel; 3],
    pub bandwidth: Freq,
    pub seed: u64,
}

/// What a flick flies in besides the stick.  `Default` is an ideal gyro on a
/// calm, balanced airframe; [`Conditions::of`] has the airframe's gyro filters.
#[derive(Clone, Copy, Default)]
pub struct Conditions<'a> {
    /// The gyro low-pass stages between the body rates and the controller,
    /// as PT1 cutoffs ([`crate::margin::pt1_cutoffs`]).
    pub gyro_filter: &'a [Freq],
    /// Sensor noise on the gyro (after the filters) and motor speeds.
    pub noise: Option<&'a Noise>,
    /// Fly the measured flight with `noise`; otherwise it only sets
    /// [`Metrics::rough_v`].
    pub noisy_flights: bool,
    /// CG offset [right, forward] as fractions of the arm: the hover thrust's
    /// torque on that lever, held throughout.
    pub cg_shift: [f64; 2],
    pub turbulence: Option<Turbulence>,
}

impl<'a> Conditions<'a> {
    /// The gyro through `gyro_filter`, nothing else.
    pub fn of(gyro_filter: &'a [Freq]) -> Self {
        Self { gyro_filter, ..Self::default() }
    }

    fn settle(&self) -> Time {
        if self.cg_shift == [0.0; 2] { SETTLE } else { SETTLE + CG_SETTLE }
    }

    /// The CG offset's angular acceleration (FRD): a forward CG pitches the
    /// nose down, a right one rolls right.
    fn cg_torque(&self, model: &Model) -> [AngAccel; 3] {
        let w = model.hover_rps;
        let thrust = 4.0 * (w * w);
        [
            self.cg_shift[0] * model.roll_prop_vel_sq.abs() * thrust,
            -self.cg_shift[1] * model.pitch_prop_vel_sq.abs() * thrust,
            AngAccel::new(0.0),
        ]
    }

    /// Only the gyro filters: the stick-free reference flight.
    fn calm(&self) -> Self {
        Self::of(self.gyro_filter)
    }
}

/// [`Turbulence`]'s angular accelerations, one per controller tick.
struct Gusts {
    k: f64,
    drive: f64,
    rms: [AngAccel; 3],
    /// Each axis's low-passed noise, unit variance.
    state: [f64; 3],
    rng: u64,
}

impl Gusts {
    fn new(t: &Turbulence, dt: Time) -> Self {
        let w = f64::from(2.0 * core::f64::consts::PI * t.bandwidth * dt);
        let k = w / (w + 1.0);
        let mut rng = t.seed;
        // A one-pole low-pass of white noise σ has variance σ²·k/(2 − k).
        let drive = ((2.0 - k) / k).sqrt();
        let state = core::array::from_fn(|_| gauss(&mut rng));
        Self { k, drive, rms: t.rms, state, rng }
    }

    fn next(&mut self) -> [AngAccel; 3] {
        for s in &mut self.state {
            let x = self.drive * gauss(&mut self.rng);
            *s += self.k * (x - *s);
        }
        core::array::from_fn(|i| self.state[i] * self.rms[i])
    }
}

/// splitmix64.
fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal sample (Box–Muller).
fn gauss(s: &mut u64) -> f64 {
    let unit = |s: &mut u64| (next_u64(s) >> 11) as f64 / (1u64 << 53) as f64;
    let (u1, u2) = (1.0 - unit(s), unit(s));
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Fly a flick to `amp` on FRD axis `axis`, rising over `rise`, in
/// `conditions` (with its noise, if any).
pub fn fly(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
    conditions: &Conditions,
) -> Trace {
    fly_settled(gains, model, axis, amp, rise, limits, conditions, conditions.settle())
}

/// [`fly`], hovering `settle` before the stick.
#[allow(clippy::too_many_arguments)]
fn fly_settled(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
    conditions: &Conditions,
    settle: Time,
) -> Trace {
    let dt: Time = gains.header.dt.lossy_into();
    let stick = |t: Time| {
        let t = t - settle;
        amp * (s_curve(f64::from(t / rise)) - s_curve(f64::from((t - rise - HOLD) / rise)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    let (slew, jerk) = target_limits(gains);
    sp.configure(
        model.stick_tracking,
        limits.slew.map_or(slew, RotPerSecSq::new),
        limits.jerk.map_or(jerk, RotPerSecCu::new),
    );
    model.configure(&mut ctrl, &mut sp);
    let mut plant = Plant::hovering(model, 1.0);
    let mut path = GyroPath::new(conditions.gyro_filter, 0, dt);
    let (mut actuator, mut telemetry) = (DelayLine::new(model.actuator_delay, dt), DelayLine::new(model.telemetry_delay, dt));
    let cg = conditions.cg_torque(model);
    let mut gusts = conditions.turbulence.as_ref().map(|t| Gusts::new(t, dt));
    let throttle = model.throttle_at(1.0);
    let t_end = settle + 2.0 * rise + HOLD + REST;
    let mut next_pkt = Time::new(0.0);
    let mut tr = Trace {
        gyro: vec![],
        setpoint: vec![],
        target: vec![],
        stick: vec![],
        saturated: vec![],
        volts: vec![],
        motor_speeds: vec![],
    };
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = micros(t);
        if t >= next_pkt {
            let mut rc = [RotPerSec::new(0.0); 3];
            rc[axis] = stick(t).lossy_into();
            sp.packet(rc, now_us, (1.0 / F_RX).lossy_into());
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let (mut gyro_rps, mut motor_rps) = (path.filter(plant.gyro()), telemetry.push(plant.motor_rps()));
        if let Some(noise) = conditions.noise {
            let s = noise.at(t, plant.motor_rps());
            (0..3).for_each(|a| gyro_rps[a] += s.gyro[a]);
            (0..4).for_each(|m| motor_rps[m] += s.motor[m]);
        }
        let sensors = SensorData {
            gyro_rps,
            motor_rps,
            battery_voltage: model.battery_v(),
            accel: [MetersPerSecSq::new(0.0); 3],
        };
        let out = ctrl.update_with_target(
            &sensors,
            &SetpointData {
                rate_rps: rate,
                throttle,
            },
            target,
            gains,
        );
        sp.condition(ctrl.reachable_target());
        let volts = model.duty_volts(out.motor_duty);
        let gust = gusts.as_mut().map_or([AngAccel::new(0.0); 3], Gusts::next);
        let torque: [AngAccel; 3] = core::array::from_fn(|a| cg[a] + gust[a]);
        let applied = actuator.push(volts);
        for _ in 0..4 {
            plant.rk4_with_torque(&applied, dt / 4.0, torque);
        }
        if t >= settle {
            tr.gyro.push(plant.gyro()[axis].lossy_into());
            tr.setpoint.push(rate[axis].lossy_into());
            tr.target.push(target[axis].lossy_into());
            tr.stick.push(stick(t));
            tr.saturated.push(ctrl.telemetry().command_scale < 0.999);
            tr.volts.push(volts);
            tr.motor_speeds.push(plant.omega);
        }
    }
    tr
}

/// How one flick was tracked, in its own direction; ratios are of the
/// flick's amplitude's magnitude.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    /// Feedforward target overshoot.
    pub target_ovs: f64,
    /// Target and gyro lag behind the stick at half amplitude.
    pub target_lag: Time,
    pub gyro_lag: Time,
    /// Time to 90% of the amplitude; infinite if never reached before release.
    pub rise90: Time,
    pub overshoot: f64,
    /// Mean error over the last 100 ms of the hold.
    pub hold_err: f64,
    /// RMS gyro − setpoint through the release and 50 ms after.
    pub track_rms: f64,
    /// Largest gyro excursion after that.
    pub residual: f64,
    /// Fraction of ticks the motors saturated.
    pub sat_frac: f64,
    /// What sensor noise adds to the motor voltages over the flick: RMS of the
    /// [`ROUGH_HP`]-high-passed difference from the noise-free flight, mean
    /// over motors.  None when flown without noise.
    pub rough_v: Option<Voltage>,
}

/// A flick's response: the gyro difference from the stick-free flight, and
/// the stick, setpoint, target and saturation of the flight with the stick.
pub struct Flick {
    pub dt: Time,
    pub response: Vec<Rate>,
    pub run: Trace,
    pub metrics: Metrics,
}

/// Fly a flick to `amp` on FRD axis `axis`, rising over `rise`, in
/// `conditions`, against the calm flight without the stick; with noise also
/// the motor roughness it causes (a third flight).
pub fn measure(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
    conditions: &Conditions,
) -> Flick {
    let dt: Time = gains.header.dt.lossy_into();
    let settle = conditions.settle();
    let flight = |amp, c: &Conditions| fly_settled(gains, model, axis, amp, rise, limits, c, settle);
    let base = flight(Rate::new(0.0), &conditions.calm());
    let quiet = Conditions { noise: None, ..*conditions };
    let run = flight(amp, if conditions.noisy_flights { conditions } else { &quiet });
    let y: Vec<Rate> = run
        .gyro
        .iter()
        .zip(&base.gyro)
        .map(|(&a, &b)| a - b)
        .collect();
    // The metrics are of the flick mirrored into its own direction, so a
    // negative flick scores as its mirror image would.
    let sign = if amp < Rate::new(0.0) { -1.0 } else { 1.0 };
    let along = |v: &[Rate]| v.iter().map(|&x| sign * x).collect::<Vec<_>>();
    let (y_dir, stick_dir, target_dir) = (along(&y), along(&run.stick), along(&run.target));
    let mag = sign * amp;
    let idx = |t: Time| (f64::from(t / dt) as usize).min(y.len() - 1);
    let (release, end) = (idx(rise + HOLD), idx(2.0 * rise + HOLD));
    let rise90 = y_dir[..release]
        .iter()
        .position(|&v| v >= 0.9 * mag)
        .map_or(Time::new(f64::INFINITY), |i| i as f64 * dt);
    let peak = y_dir[..release]
        .iter()
        .fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold = &y_dir[idx(rise + HOLD - Time::new(0.1))..release];
    let track_end = idx(2.0 * rise + HOLD + Time::new(0.05));
    let track = (0..track_end)
        .map(|i| (y[i] - run.setpoint[i]) * (y[i] - run.setpoint[i]))
        .fold(SpeedSq::new(0.0), |acc, e| acc + e)
        / track_end as f64;
    let half = |v: &[Rate]| {
        v[..release]
            .iter()
            .position(|&x| x >= 0.5 * mag)
            .unwrap_or(release) as f64
            * dt
    };
    let t_stick = half(&stick_dir);
    let target_peak = target_dir[..release]
        .iter()
        .fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold_mean = hold.iter().fold(Rate::new(0.0), |acc, &v| acc + v) / hold.len() as f64;
    let residual = y[track_end..]
        .iter()
        .fold(Rate::new(0.0), |m, v| m.max(v.abs()));
    let metrics = Metrics {
        target_ovs: f64::from(target_peak / mag) - 1.0,
        target_lag: half(&target_dir) - t_stick,
        gyro_lag: half(&y_dir) - t_stick,
        rise90,
        overshoot: f64::from(peak / mag) - 1.0,
        hold_err: f64::from(hold_mean / mag) - 1.0,
        track_rms: f64::from(track.sqrt() / mag),
        residual: f64::from(residual / mag),
        sat_frac: run.saturated[..end].iter().filter(|&&s| s).count() as f64 / end as f64,
        rough_v: conditions.noise.map(|_| {
            let other = flight(amp, if conditions.noisy_flights { &quiet } else { conditions });
            let (noisy, calm) = if conditions.noisy_flights { (&run, &other) } else { (&other, &run) };
            roughness(&noisy.volts, &calm.volts, dt)
        }),
    };
    Flick {
        dt,
        response: y,
        run,
        metrics,
    }
}

/// RMS of the [`ROUGH_HP`] one-pole high-pass of `a − b`, mean over motors.
pub(crate) fn roughness(a: &[[Voltage; 4]], b: &[[Voltage; 4]], dt: Time) -> Voltage {
    let w = f64::from(2.0 * core::f64::consts::PI * ROUGH_HP * dt);
    let k = w / (w + 1.0);
    let diff = |i: usize, m: usize| a[i][m] - b[i][m];
    let mut lp: [Voltage; 4] = core::array::from_fn(|m| diff(0, m));
    let mut sum = VoltageSq::new(0.0);
    for i in 0..a.len() {
        for (m, lp) in lp.iter_mut().enumerate() {
            *lp += k * (diff(i, m) - *lp);
            let hp = diff(i, m) - *lp;
            sum += hp * hp / 4.0;
        }
    }
    (sum / a.len() as f64).sqrt()
}

/// One entry of [`all_flicks`]: FRD axis, rise time, amplitude, and the flick.
pub type FlickRun = (usize, Time, Rate, Flick);

/// Every (axis, rise, amplitude) flick, in table order.
pub fn all_flicks(gains: &GainStorage, model: &Model, limits: Limits, conditions: &Conditions) -> Vec<FlickRun> {
    let mut out = vec![];
    for axis in AXES {
        for rise in RISES {
            for amp in AMPS {
                out.push((
                    axis,
                    rise,
                    amp,
                    measure(gains, model, axis, amp, rise, limits, conditions),
                ));
            }
        }
    }
    out
}
