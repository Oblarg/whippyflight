//! Robustness of the closed loop, with the gyro passed through the airframe's
//! gyro low-pass stages (as PT1s): the delay margin, how much extra gyro lag
//! it tolerates (the largest pure delay a gentle and a medium roll flick
//! still fly cleanly with), and the integrator trim, how much attitude a CG
//! shift costs before the integral takes up the torque.  Also how fast a
//! knock dies out and how much measured sensor noise roughens the motors at
//! hover: with the delay margin and trim, the tradeoffs the LQR weights set.

use whippyunits::api::rescale;

use crate::flick::roughness;
use crate::noise::Noise;
use crate::plant::{DelayLine, Model, Plant};
use crate::units::*;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::GainStorage;
use whippyflight_control::setpoint::SetpointEstimator;
use whippyflight_control::types::*;

const F_RX: Freq = Freq::new(500.0);
const SETTLE: Time = Time::new(1.0);
const RISE: Time = Time::new(0.030);
const HOLD: Time = Time::new(0.5);
const REST: Time = Time::new(0.6);
/// Motor speeds the margin flights hover at, of the model's hover speed.
pub const HOVER_SCALE: f64 = 250.0 / 270.0;
/// The flicks that must fly cleanly.
const AMPS: [Rate; 2] = [Rate::new(0.05), Rate::new(0.5)];
pub const MAX_MARGIN: Millis = Millis::new(16.0);
const MARGIN_RESOLUTION: Millis = Millis::new(0.1);
/// Fraction of the front-back arm the CG shifts by in [`trim`].
pub const TRIM_CG_SHIFT: f64 = 0.05;
/// How long [`trim`] flies after the shift.
pub const TRIM: Time = Time::new(3.0);
/// The rate knock [`kick_iae`] recovers from.
pub const KICK: Rate = Rate::new(0.2);
/// [`group_delay_margin`]'s cap, in the model's loop latencies: a loop delayed
/// on some axes only can stay stable at any delay.
pub const MARGIN_CAP_LATENCIES: f64 = 16.0;

/// The axes a set of LQR weights acts on, FRD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    PitchRoll,
    Yaw,
}

impl Group {
    pub fn axes(self) -> [bool; 3] {
        match self {
            Group::PitchRoll => [true, true, false],
            Group::Yaw => [false, false, true],
        }
    }

    /// The axis knocked and flicked: roll for pitch/roll.
    pub fn knock_axis(self) -> usize {
        match self {
            Group::PitchRoll => 0,
            Group::Yaw => 2,
        }
    }

    /// The axis a step torque trims: pitch (a CG shift) for pitch/roll.
    pub fn trim_axis(self) -> usize {
        match self {
            Group::PitchRoll => 1,
            Group::Yaw => 2,
        }
    }
}

/// Gyro path: PT1 stages (BF's pt1FilterGain) then a delay in PID ticks on
/// some axes, fractional delays linearly interpolated between ticks.
pub struct GyroPath {
    gains: Vec<f64>,
    state: Vec<[Rate; 3]>,
    /// Filtered samples, newest last.
    history: std::collections::VecDeque<[Rate; 3]>,
    delay_ticks: f64,
    axes: [bool; 3],
}

impl GyroPath {
    pub fn new(cutoffs: &[Freq], delay_ticks: usize, dt: Time) -> Self {
        Self::delayed(cutoffs, delay_ticks as f64, dt)
    }

    pub fn delayed(cutoffs: &[Freq], delay_ticks: f64, dt: Time) -> Self {
        Self::delayed_on(cutoffs, delay_ticks, dt, [true; 3])
    }

    /// Delayed on `axes` only.
    pub fn delayed_on(cutoffs: &[Freq], delay_ticks: f64, dt: Time, axes: [bool; 3]) -> Self {
        let gains = cutoffs
            .iter()
            .map(|&f| {
                let w = f64::from(2.0 * core::f64::consts::PI * f * dt);
                w / (w + 1.0)
            })
            .collect::<Vec<_>>();
        let n = gains.len();
        let history = vec![[Rate::new(0.0); 3]; delay_ticks as usize + 2].into();
        Self {
            gains,
            state: vec![[Rate::new(0.0); 3]; n],
            history,
            delay_ticks,
            axes,
        }
    }

    pub fn filter(&mut self, x: [RotPerSec; 3]) -> [RotPerSec; 3] {
        let mut y: [Rate; 3] = x.map(|v| v.lossy_into());
        for (&k, s) in self.gains.iter().zip(self.state.iter_mut()) {
            for a in 0..3 {
                s[a] += k * (y[a] - s[a]);
            }
            y = *s;
        }
        self.history.pop_front();
        self.history.push_back(y);
        let (n, f) = (self.delay_ticks.floor() as usize, self.delay_ticks.fract());
        let last = self.history.len() - 1;
        let (a, b) = (self.history[last - n], self.history[last - n - 1]);
        core::array::from_fn(|i| {
            if self.axes[i] { (1.0 - f) * a[i] + f * b[i] } else { y[i] }.lossy_into()
        })
    }
}

/// `model`'s gyro low-pass stages as PT1 cutoffs (see [`stage_cutoffs`]).
pub fn pt1_cutoffs(model: &Model) -> Result<Vec<Freq>, String> {
    stage_cutoffs(&model.gyro_lpf, "gyro")
}

/// Low-pass `stages` as PT1 cutoffs: BF's PT2/PT3 are 2/3 PT1s at the
/// cutoff × CUTOFF_CORRECTION_PT2/PT3 (filter.c).
pub fn stage_cutoffs(stages: &[(String, Freq)], name: &str) -> Result<Vec<Freq>, String> {
    let mut out = vec![];
    for (kind, hz) in stages {
        match kind.as_str() {
            "PT1" => out.push(*hz),
            "PT2" => out.extend([*hz * 1.553_773_974; 2]),
            "PT3" => out.extend([*hz * 1.961_459_177; 3]),
            _ => return Err(format!("{name} LPF type {kind} is not modeled")),
        }
    }
    Ok(out)
}

/// Peak-to-peak of `(t, y)` samples about their least-squares line.
pub fn detrended_p2p(samples: &[(Time, Rate)]) -> Rate {
    let n = samples.len() as f64;
    let (mt, my) = samples
        .iter()
        .fold((Time::new(0.0), Rate::new(0.0)), |(a, b), &(t, y)| {
            (a + t / n, b + y / n)
        });
    let (stt, sty) = samples
        .iter()
        .fold((TimeSq::new(0.0), Angle::new(0.0)), |(a, b), &(t, y)| {
            (a + (t - mt) * (t - mt), b + (t - mt) * (y - my))
        });
    let slope = if stt > TimeSq::new(0.0) {
        sty / stt
    } else {
        AngAccel::new(0.0)
    };
    let (lo, hi) = samples.iter().fold(
        (Rate::new(f64::MAX), Rate::new(f64::MIN)),
        |(lo, hi), &(t, y)| {
            let r = y - my - slope * (t - mt);
            (lo.min(r), hi.max(r))
        },
    );
    hi - lo
}

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

struct Flight {
    /// Detrended peak-to-peak rate on the flick axis over the last 0.3 s.
    rest_p2p: Rate,
    /// Fraction of ticks the motors saturated.
    sat_frac: f64,
    /// True body rates (FRD) from SETTLE on.
    rates: Vec<[Rate; 3]>,
    /// Commanded motor voltages from SETTLE on.
    volts: Vec<[Voltage; 4]>,
}

/// What disturbs a [`flight`] at SETTLE: an angular acceleration (FRD) from
/// then on, a step in rate, and sensor noise throughout.
struct Disturbance<'n> {
    torque: [AngAccel; 3],
    kick: [Rate; 3],
    noise: Option<&'n Noise>,
}

const CALM: Disturbance<'static> = Disturbance {
    torque: [AngAccel::new(0.0); 3],
    kick: [Rate::new(0.0); 3],
    noise: None,
};

/// Fly a flick to `amp` on `axis` (FRD) through `path`, disturbed by
/// `disturbance`, for `after` past SETTLE.
fn flight(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    mut path: GyroPath,
    disturbance: Disturbance,
    after: Time,
) -> Flight {
    let Disturbance { torque, kick, noise } = disturbance;
    let dt: Time = gains.header.dt.lossy_into();
    let stick = |t: Time| {
        let t = t - SETTLE;
        amp * (s_curve(f64::from(t / RISE)) - s_curve(f64::from((t - RISE - HOLD) / RISE)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    sp.configure_for(gains);
    model.configure(&mut ctrl, &mut sp);
    let mut plant = Plant::hovering(model, HOVER_SCALE);
    let (mut actuator, mut telemetry) = (DelayLine::new(model.actuator_delay, dt), DelayLine::new(model.telemetry_delay, dt));
    let throttle = model.throttle_at(1.0);
    let t_end = SETTLE + after;
    let mut next_pkt = Time::new(0.0);
    let (mut rest, mut sat, mut n, mut rates) = (Vec::new(), 0usize, 0usize, Vec::new());
    let (mut kicked, mut volts_trace) = (false, Vec::new());
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = crate::flick::micros(t);
        if !kicked && t >= SETTLE {
            (0..3).for_each(|a| plant.rate[a] += kick[a]);
            kicked = true;
        }
        if t >= next_pkt {
            let mut sticks = [RotPerSec::new(0.0); 3];
            sticks[axis] = stick(t).lossy_into();
            sp.packet(
                sticks,
                now_us,
                (1.0 / F_RX).lossy_into(),
            );
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let mut gyro_rps = path.filter(plant.gyro());
        let mut motor_rps = telemetry.push(plant.motor_rps());
        if let Some(noise) = noise {
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
        let torque = if t >= SETTLE { torque } else { [AngAccel::new(0.0); 3] };
        let applied = actuator.push(volts);
        for _ in 0..4 {
            plant.rk4_with_torque(&applied, dt / 4.0, torque);
        }
        if t >= SETTLE {
            n += 1;
            rates.push(plant.rate);
            volts_trace.push(volts);
            if ctrl.telemetry().command_scale < 0.999 {
                sat += 1;
            }
            if t - SETTLE >= t_end - SETTLE - Time::new(0.3) {
                rest.push((t, plant.gyro()[axis].lossy_into()));
            }
        }
    }
    Flight {
        rest_p2p: detrended_p2p(&rest),
        sat_frac: sat as f64 / n as f64,
        rates,
        volts: volts_trace,
    }
}

/// Integrator trim: hover, shift the CG by [`TRIM_CG_SHIFT`] (a step pitch
/// torque: the hover thrust on that lever arm of the unit front-back arm) at
/// SETTLE, gyro through `cutoffs`, and return (attitude it costs, = |∫ pitch
/// rate| over [`TRIM`]; residual pitch rate over the last 0.5 s).  The rate
/// loop alone holds a constant rate error under a constant torque; the
/// (leaky) integrator is what brings it back toward zero.
pub fn trim(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> (Angle, Rate) {
    group_trim(gains, model, cutoffs, Group::PitchRoll)
}

/// [`trim`] on `group`'s trim axis; for yaw the step is [`TRIM_CG_SHIFT`] of
/// the motors' summed drag torque at hover.
pub fn group_trim(gains: &GainStorage, model: &Model, cutoffs: &[Freq], group: Group) -> (Angle, Rate) {
    let dt: Time = gains.header.dt.lossy_into();
    let w = HOVER_SCALE * model.hover_rps;
    let axis = group.trim_axis();
    let per_rot = match group {
        Group::PitchRoll => model.pitch_prop_vel_sq,
        Group::Yaw => model.yaw_prop_vel_sq,
    };
    let mut torque = [AngAccel::new(0.0); 3];
    torque[axis] = TRIM_CG_SHIFT * per_rot * 4.0 * (w * w);
    let f = flight(
        gains,
        model,
        group.knock_axis(),
        Rate::new(0.0),
        GyroPath::new(cutoffs, 0, dt),
        Disturbance { torque, ..CALM },
        TRIM,
    );
    let rate: Vec<Rate> = f.rates.iter().map(|r| r[axis]).collect();
    let angle: Angle = rate.iter().fold(Rate::new(0.0), |acc, &r| acc + r) * dt;
    let tail = f64::from(Time::new(0.5) / dt) as usize;
    let residual = rate[rate.len() - tail..]
        .iter()
        .fold(Rate::new(0.0), |acc, &r| acc + r)
        / tail as f64;
    (angle.abs(), residual)
}

/// Largest extra gyro delay on top of `cutoffs` that still flies both
/// flicks cleanly (settled without limit cycling or chattering on the
/// rails): bisected over 0–[`MAX_MARGIN`] with a fractional delay, so it
/// ranks candidates instead of saturating at a tick cap.  −1 ms if the path
/// isn't clean without extra delay; [`MAX_MARGIN`] if it never breaks.  A
/// time rather than ticks so loop rates compare.
pub fn delay_margin(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> Millis {
    margin_on(gains, model, cutoffs, Group::PitchRoll, [true; 3], MAX_MARGIN)
}

/// [`group_delay_margin`]'s cap for `model` flown at `dt`.
pub fn margin_cap(model: &Model, dt: Time) -> Millis {
    let latency = model.actuator_delay + model.telemetry_delay;
    MARGIN_CAP_LATENCIES * rescale(if latency > dt { latency } else { dt })
}

/// [`delay_margin`] with the delay on `group`'s axes only and its flicks on
/// its knock axis, bisected over 0–[`margin_cap`].
pub fn group_delay_margin(gains: &GainStorage, model: &Model, cutoffs: &[Freq], group: Group) -> Millis {
    let cap = margin_cap(model, gains.header.dt.lossy_into());
    margin_on(gains, model, cutoffs, group, group.axes(), cap)
}

fn margin_on(
    gains: &GainStorage,
    model: &Model,
    cutoffs: &[Freq],
    group: Group,
    axes: [bool; 3],
    cap: Millis,
) -> Millis {
    let dt: Time = gains.header.dt.lossy_into();
    let ok = |delay: Millis| {
        let delay: Time = rescale(delay);
        AMPS.iter().all(|&a| {
            let path = GyroPath::delayed_on(cutoffs, f64::from(delay / dt), dt, axes);
            let f = flight(gains, model, group.knock_axis(), a, path, CALM, 2.0 * RISE + HOLD + REST);
            f.rest_p2p < Rate::new(0.005) && f.sat_frac < 0.1
        })
    };
    if !ok(Millis::new(0.0)) {
        return Millis::new(-1.0);
    }
    if ok(cap) {
        return cap;
    }
    let (mut lo, mut hi) = (Millis::new(0.0), cap);
    while hi - lo > MARGIN_RESOLUTION {
        let mid = 0.5 * (lo + hi);
        if ok(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// Hover, knock the roll rate by [`KICK`] at SETTLE, gyro through `cutoffs`,
/// and return ∫|roll rate| dt / KICK: how long the knock takes to die out,
/// continuous where a time to 10% would jump with each undershoot lobe.
pub fn kick_iae(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> Millis {
    group_kick_iae(gains, model, cutoffs, Group::PitchRoll)
}

/// [`kick_iae`] on `group`'s knock axis.
pub fn group_kick_iae(gains: &GainStorage, model: &Model, cutoffs: &[Freq], group: Group) -> Millis {
    let dt: Time = gains.header.dt.lossy_into();
    let axis = group.knock_axis();
    let mut kick = [Rate::new(0.0); 3];
    kick[axis] = KICK;
    let f = flight(
        gains,
        model,
        axis,
        Rate::new(0.0),
        GyroPath::new(cutoffs, 0, dt),
        Disturbance { kick, ..CALM },
        2.0 * RISE + HOLD + REST,
    );
    let iae: Time = f
        .rates
        .iter()
        .fold(Rate::new(0.0), |acc, r| acc + r[axis].abs())
        * dt
        / KICK;
    rescale(iae)
}

/// Hover with `noise` on the gyro (after `cutoffs`) and motor speeds, and
/// return the motor roughness it causes: as [`super::flick::Metrics::rough_v`],
/// against the same hover without it.
pub fn hover_roughness(
    gains: &GainStorage,
    model: &Model,
    cutoffs: &[Freq],
    noise: &Noise,
) -> Voltage {
    let dt: Time = gains.header.dt.lossy_into();
    let hover = |noise| {
        flight(
            gains,
            model,
            0,
            Rate::new(0.0),
            GyroPath::new(cutoffs, 0, dt),
            Disturbance { noise, ..CALM },
            2.0 * RISE + HOLD + REST,
        )
    };
    roughness(&hover(Some(noise)).volts, &hover(None).volts, dt)
}
