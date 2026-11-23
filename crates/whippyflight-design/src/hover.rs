//! The closed loop at hover as the sim flies it (whippyflight-sim margin.rs),
//! linearized, and the front's measures on it: knock recovery, delay margin,
//! CG-shift trim and the motor roughness the fit's recorded sensor noise
//! causes.
//!
//! The loop is the fit's plant at its hover speed with the firmware's control
//! law around it: the structured LQR gains solved at hover, the integral and
//! derivative filters as controller.rs runs them, the affine linearizer on
//! the motor-speed telemetry, the fit's motor-command and telemetry delays,
//! its gyro low-pass stages (as PT1s) and motor-speed low-pass. The
//! feedforward's reference model runs on its own state at hover and drops
//! out. Saturation, the integral clamp and the sim's clean-flight test are
//! nonlinear and left to the sim.

use std::path::Path;

use whippyalgebra::nalgebra::DMatrix;
use whippyflight_sim::margin::{MARGIN_CAP_LATENCIES, pt1_cutoffs};
use whippyflight_sim::noise::Noise;
use whippyunits::{qty, quantity};

use crate::disturbance::Disturbances;
use crate::fit::Fit;
use crate::lqr::{self, Costs, Group, Weights};
use crate::plant::{self, Discrete, pt1_gain, ticks};
use crate::spaces::{DerivativeGain, Gain, IntegralGain, RateGain};
use crate::spectrum::{Endomorphism, spectral_radius};
use crate::units::*;

/// Settling time before the noise is measured.
const SETTLE: Time = quantity!(1.0, s);
/// The sim's flight after a disturbance.
const HOVER_FLIGHT: Time = quantity!(1.16, s);
/// How long a CG shift is trimmed for.
const TRIM: Time = quantity!(3.0, s);
/// The roughness high-pass (flick.rs ROUGH_HP).
const ROUGH_HP: Freq = quantity!(50.0, Hz);
/// Attitude a step torque costs, per unit torque: rot per rot/s².
pub type TrimCost = qty!(s ^ 2);

/// What disturbs the loop over a tick.
#[derive(Clone, Copy)]
struct Input {
    torque: [AngAccel; 3],
    gyro: [Rate; 3],
    motor: [Rate; 4],
}

const CALM: Input = Input {
    torque: [quantity!(0.0, rot / s ^ 2); 3],
    gyro: [quantity!(0.0, rot / s); 3],
    motor: [quantity!(0.0, rot / s); 4],
};

/// Extra gyro delay, fractional ticks, on a group's axes.
#[derive(Clone, Copy)]
struct Delay {
    whole: usize,
    frac: f64,
    axes: [bool; 3],
}

const UNDELAYED: Delay = Delay { whole: 0, frac: 0.0, axes: [false; 3] };

impl Delay {
    fn on(group: Group, ticks: f64) -> Self {
        Delay { whole: ticks.floor() as usize, frac: ticks - ticks.floor(), axes: group.axes() }
    }
}

/// The loop's state, model order.
#[derive(Clone)]
struct State {
    rates: [Rate; 3],
    motors: [Rate; 4],
    /// The gyro low-pass stages.
    stages: Vec<[Rate; 3]>,
    /// Filtered gyro, newest first.
    history: Vec<[Rate; 3]>,
    integ: [Angle; 3],
    deriv: [AngAccel; 3],
    prev: [Rate; 3],
    /// Motor-speed telemetry in flight, newest first.
    telemetry: Vec<[Rate; 4]>,
    lpf: [Rate; 4],
    /// Motor commands in flight, newest first.
    commands: Vec<[Voltage; 4]>,
}

/// The applied gains, slot × model axis, and the filters.
struct Feedback {
    integ: IntegralGain,
    deriv: DerivativeGain,
    rate: RateGain,
    leak: f64,
    smoothing: f64,
    tau_deriv: Time,
}

impl Feedback {
    fn new(gain: &Gain, weights: &Weights, tau_deriv: Time, dt: Time) -> Self {
        Feedback {
            integ: gain.block::<0, 0, 4, 3>().to_uniform(),
            deriv: gain.block::<0, 3, 4, 3>().to_uniform(),
            rate: gain.block::<0, 6, 4, 3>().to_uniform(),
            leak: f64::from((weights.tau_int - dt) / weights.tau_int),
            smoothing: f64::from((tau_deriv - dt) / tau_deriv),
            tau_deriv,
        }
    }
}

const RATE: Rate = quantity!(1.0, rot / s);
const ANGLE: Angle = quantity!(1.0, rot);
const ANG_ACCEL: AngAccel = quantity!(1.0, rot / s ^ 2);
const VOLT: Voltage = quantity!(1.0, V);

#[derive(Clone)]
pub struct HoverLoop {
    pub dt: Time,
    /// The fit's measured loop latency (actuator and telemetry).
    pub latency: Time,
    volt: Voltage,
    /// The LQR's plant: the effective dynamics at hover.
    lqr_plant: Discrete,
    /// The loop's plant: the dynamics at hover, the linearizer in the loop.
    ad: UnitMatrix<Dimensionless, 7, 7>,
    bd: UnitMatrix<RotPerSecondVolt, 7, 4>,
    bt: UnitMatrix<Seconds, 7, 3>,
    slope: UnitMatrix<VoltSecondPerRot, 4, 4>,
    gyro_gains: Vec<f64>,
    motor_lpf: Option<f64>,
    actuator_ticks: usize,
    telemetry_ticks: usize,
    /// What the LQR is solved against.
    pub disturbances: Disturbances,
    /// The derivative filter's time constant ([`plant::derivative_tau`]).
    pub tau_deriv: Time,
    /// The recorded sensor noise over a roughness flight, by tick.
    noise: Vec<Input>,
    /// What every solve starts from, if set.
    anchor: Option<Gain>,
    solved: Option<(Weights, Option<Gain>)>,
}

impl HoverLoop {
    /// The loop of `fit` at a loop period of `dt`, disturbed as measured
    /// beside it in `fit_dir`.
    pub fn new(fit: &Fit, fit_dir: &Path, dt: Time) -> Result<Self, String> {
        let m = &fit.model;
        let hover = [m.hover_rps; 4];
        let actuation = plant::actuation(fit);
        let lqr_plant = Discrete::sample(&plant::effective_dynamics(fit, hover), &actuation, dt);
        let sampled = Discrete::sample(&plant::dynamics(fit, hover), &actuation, dt);
        let gyro_gains = pt1_cutoffs(m)?.into_iter().map(|f| pt1_gain(f, dt)).collect();
        let motor_lpf = (m.motor_lpf > quantity!(0.0, Hz)).then(|| pt1_gain(m.motor_lpf, dt));
        let disturbances = Disturbances::load(fit_dir, dt)?;
        let tau_deriv = plant::derivative_tau(fit, dt)?;
        let noise = Noise::load_fit(fit_dir)?.0.ok_or("the fit has no recorded sensor noise")?;
        let at_hover = [m.hover_rps.lossy_into(); 4];
        let steps = f64::from((SETTLE + HOVER_FLIGHT) / dt) as usize;
        let noise = (0..steps)
            .map(|k| {
                let s = noise.at(k as f64 * dt, at_hover);
                let gyro: [Rate; 3] = s.gyro.map(|g| g.lossy_into());
                let mut motor = [quantity!(0.0, rot / s); 4];
                for (i, &v) in s.motor.iter().enumerate() {
                    motor[fit.slot_of(i)] = v.lossy_into();
                }
                Input { gyro: [gyro[2], gyro[1], gyro[0]], motor, ..CALM }
            })
            .collect();
        Ok(HoverLoop {
            dt,
            latency: m.actuator_delay + m.telemetry_delay,
            volt: fit.cells * quantity!(1.0, V),
            lqr_plant,
            ad: sampled.ad.to_uniform(),
            bd: sampled.bd.to_uniform(),
            bt: sampled.bt.to_uniform(),
            slope: plant::linearizer_slope(fit, hover),
            gyro_gains,
            motor_lpf,
            actuator_ticks: ticks(m.actuator_delay, dt),
            telemetry_ticks: ticks(m.telemetry_delay, dt),
            disturbances,
            tau_deriv,
            noise,
            anchor: None,
            solved: None,
        })
    }

    /// Start every later solve from the one at `weights`, solved cold.
    pub fn anchor(&mut self, weights: &Weights) -> Result<(), String> {
        self.anchor = None;
        self.solved = None;
        self.anchor = Some(self.gain(weights).ok_or("no stabilizing gains at hover for the current weights")?);
        Ok(())
    }

    /// The gains at hover for `weights`, solved from the anchor if set, else
    /// cold, so they don't depend on what else was solved; None where no
    /// stabilizing gains were found.
    pub fn gain(&mut self, weights: &Weights) -> Option<Gain> {
        if let Some((at, gain)) = &self.solved
            && at == weights
        {
            return *gain;
        }
        let costs = Costs::new(weights.pitch_roll, weights.yaw, self.volt);
        let sys = plant::augmented(&self.lqr_plant, self.dt, weights.tau_int, self.tau_deriv, &self.disturbances);
        let gain = lqr::solve(&sys, &costs, self.anchor.as_ref());
        self.solved = Some((*weights, gain));
        gain
    }

    fn rest(&self, delay: Delay) -> State {
        State {
            rates: [quantity!(0.0, rot / s); 3],
            motors: [quantity!(0.0, rot / s); 4],
            stages: vec![[quantity!(0.0, rot / s); 3]; self.gyro_gains.len()],
            history: vec![[quantity!(0.0, rot / s); 3]; delay.whole + 1],
            integ: [quantity!(0.0, rot); 3],
            deriv: [quantity!(0.0, rot / s ^ 2); 3],
            prev: [quantity!(0.0, rot / s); 3],
            telemetry: vec![[quantity!(0.0, rot / s); 4]; self.telemetry_ticks],
            lpf: [quantity!(0.0, rot / s); 4],
            commands: vec![[quantity!(0.0, V); 4]; self.actuator_ticks],
        }
    }

    /// One tick of the loop: the next state and the commanded voltages.
    fn tick(&self, x: &State, w: &Input, k: &Feedback, delay: Delay) -> (State, [Voltage; 4]) {
        let dt = self.dt;
        let mut y = x.rates;
        let mut stages = Vec::with_capacity(x.stages.len());
        for (&g, s) in self.gyro_gains.iter().zip(&x.stages) {
            y = core::array::from_fn(|a| s[a] + g * (y[a] - s[a]));
            stages.push(y);
        }
        let filtered = |i: usize| if i == 0 { y } else { x.history[i - 1] };
        let e: [Rate; 3] = core::array::from_fn(|a| {
            let seen = if delay.axes[a] {
                (1.0 - delay.frac) * filtered(delay.whole)[a] + delay.frac * filtered(delay.whole + 1)[a]
            } else {
                y[a]
            };
            -(seen + w.gyro[a])
        });
        let integ: [Angle; 3] = core::array::from_fn(|a| k.leak * x.integ[a] + dt * e[a]);
        let deriv: [AngAccel; 3] = core::array::from_fn(|a| k.smoothing * x.deriv[a] + (e[a] - x.prev[a]) / k.tau_deriv);
        let reported = if self.telemetry_ticks == 0 { x.motors } else { x.telemetry[self.telemetry_ticks - 1] };
        let mut measured: [Rate; 4] = core::array::from_fn(|m| reported[m] + w.motor[m]);
        let mut lpf = x.lpf;
        if let Some(g) = self.motor_lpf {
            lpf = core::array::from_fn(|m| lpf[m] + g * (measured[m] - lpf[m]));
            measured = lpf;
        }
        let u: [Voltage; 4] = core::array::from_fn(|m| {
            (0..3).fold(quantity!(0.0, V), |v, a| {
                v + k.rate.get(m, a) * e[a] + k.integ.get(m, a) * integ[a] + k.deriv.get(m, a) * deriv[a]
            }) + (0..4).fold(quantity!(0.0, V), |v, j| v + self.slope.get(m, j) * measured[j])
        });
        let applied = if self.actuator_ticks == 0 { u } else { x.commands[self.actuator_ticks - 1] };
        let plant = |i: usize| if i < 3 { x.rates[i] } else { x.motors[i - 3] };
        let next = |i: usize| {
            (0..7).fold(quantity!(0.0, rot / s), |v, j| v + self.ad.get(i, j) * plant(j))
                + (0..4).fold(quantity!(0.0, rot / s), |v, j| v + self.bd.get(i, j) * applied[j])
                + (0..3).fold(quantity!(0.0, rot / s), |v, j| v + self.bt.get(i, j) * w.torque[j])
        };
        let mut history = Vec::with_capacity(x.history.len());
        history.extend((0..x.history.len()).map(filtered));
        let mut telemetry = Vec::with_capacity(self.telemetry_ticks);
        if self.telemetry_ticks > 0 {
            telemetry.push(x.motors);
            telemetry.extend_from_slice(&x.telemetry[..self.telemetry_ticks - 1]);
        }
        let mut commands = Vec::with_capacity(self.actuator_ticks);
        if self.actuator_ticks > 0 {
            commands.push(u);
            commands.extend_from_slice(&x.commands[..self.actuator_ticks - 1]);
        }
        let state = State {
            rates: core::array::from_fn(next),
            motors: core::array::from_fn(|m| next(3 + m)),
            stages,
            history,
            integ,
            deriv,
            prev: e,
            telemetry,
            lpf,
            commands,
        };
        (state, u)
    }

    /// The state as numbers, each in its unit; of the gyro history only the
    /// delayed axes', the others' are never read.
    fn flatten(&self, x: &State, delay: Delay) -> Vec<f64> {
        let mut out = Vec::new();
        let rates = |out: &mut Vec<f64>, v: &[Rate]| out.extend(v.iter().map(|&r| f64::from(r / RATE)));
        rates(&mut out, &x.rates);
        rates(&mut out, &x.motors);
        x.stages.iter().for_each(|s| rates(&mut out, s));
        for s in &x.history {
            (0..3).filter(|&a| delay.axes[a]).for_each(|a| out.push(f64::from(s[a] / RATE)));
        }
        out.extend(x.integ.iter().map(|&v| f64::from(v / ANGLE)));
        out.extend(x.deriv.iter().map(|&v| f64::from(v / ANG_ACCEL)));
        rates(&mut out, &x.prev);
        x.telemetry.iter().for_each(|s| rates(&mut out, s));
        if self.motor_lpf.is_some() {
            rates(&mut out, &x.lpf);
        }
        x.commands.iter().for_each(|c| out.extend(c.iter().map(|&v| f64::from(v / VOLT))));
        out
    }

    /// [`Self::flatten`]'s inverse, on a state shaped as `like`.
    fn unflatten(&self, flat: &[f64], like: &State, delay: Delay) -> State {
        let mut it = flat.iter().copied();
        let mut next = || it.next().unwrap();
        let mut x = like.clone();
        x.rates.iter_mut().for_each(|v| *v = next() * RATE);
        x.motors.iter_mut().for_each(|v| *v = next() * RATE);
        x.stages.iter_mut().flatten().for_each(|v| *v = next() * RATE);
        for s in &mut x.history {
            (0..3).filter(|&a| delay.axes[a]).for_each(|a| s[a] = next() * RATE);
        }
        x.integ.iter_mut().for_each(|v| *v = next() * ANGLE);
        x.deriv.iter_mut().for_each(|v| *v = next() * ANG_ACCEL);
        x.prev.iter_mut().for_each(|v| *v = next() * RATE);
        x.telemetry.iter_mut().flatten().for_each(|v| *v = next() * RATE);
        if self.motor_lpf.is_some() {
            x.lpf.iter_mut().for_each(|v| *v = next() * RATE);
        }
        x.commands.iter_mut().flatten().for_each(|v| *v = next() * VOLT);
        x
    }

    /// The largest eigenvalue magnitude of a tick of the loop.
    fn spectral_radius(&self, k: &Feedback, delay: Delay) -> f64 {
        let rest = self.rest(delay);
        let n = self.flatten(&rest, delay).len();
        let mut tick = DMatrix::<f64>::zeros(n, n);
        let mut basis = vec![0.0; n];
        for j in 0..n {
            basis[j] = 1.0;
            let (next, _) = self.tick(&self.unflatten(&basis, &rest, delay), &CALM, k, delay);
            tick.set_column(j, &nalgebra::DVector::from_vec(self.flatten(&next, delay)));
            basis[j] = 0.0;
        }
        spectral_radius(&Endomorphism::from_nalgebra(tick))
    }

    fn feedback(&mut self, weights: &Weights) -> Option<Feedback> {
        let gain = self.gain(weights)?;
        Some(Feedback::new(&gain, weights, self.tau_deriv, self.dt))
    }

    fn stable_with(&self, k: &Feedback, delay: Delay) -> bool {
        self.spectral_radius(k, delay) < 1.0
    }

    /// Whether the loop at `weights` is stable with an extra gyro delay of
    /// `ticks` on `group`'s axes.
    pub fn stable(&mut self, weights: &Weights, group: Group, ticks: f64) -> bool {
        self.feedback(weights).is_some_and(|k| self.stable_with(&k, Delay::on(group, ticks)))
    }

    /// Ok if the loop at `weights` is stable at hover, else why not.
    pub fn flies(&mut self, weights: &Weights) -> Result<(), String> {
        if self.gain(weights).is_none() {
            return Err("no stabilizing gains at hover for these weights".into());
        }
        if !self.stable(weights, Group::PitchRoll, 0.0) {
            return Err("these weights' gains are unstable at hover with the fit's filters and delays; reset the weights, or pick them from the fronts".into());
        }
        Ok(())
    }

    /// ∫|rate| dt over a knock on `group`'s first axis (roll for pitch/roll),
    /// per unit knock, over the sim's flight after it; ∞ if the loop is
    /// unstable.
    pub fn kick_iae(&mut self, weights: &Weights, group: Group) -> Time {
        let unstable = quantity!(f64::INFINITY, s);
        let Some(k) = self.feedback(weights) else { return unstable };
        if !self.stable_with(&k, UNDELAYED) {
            return unstable;
        }
        let axis = kick_axis(group);
        let mut x = self.rest(UNDELAYED);
        x.rates[axis] = RATE;
        let mut iae = quantity!(0.0, rot);
        for _ in 0..f64::from(HOVER_FLIGHT / self.dt) as usize {
            iae += x.rates[axis].abs() * self.dt;
            x = self.tick(&x, &CALM, &k, UNDELAYED).0;
        }
        iae / RATE
    }

    /// The most delay margin measured: a loop delayed on some axes only can
    /// stay stable at any delay.
    pub fn margin_cap(&self) -> Time {
        MARGIN_CAP_LATENCIES * if self.latency > self.dt { self.latency } else { self.dt }
    }

    /// The largest extra gyro delay on `group`'s axes the loop stays stable
    /// with, to `tolerance` relative, up to [`Self::margin_cap`]; None if it
    /// is unstable without.
    pub fn delay_margin(&mut self, weights: &Weights, group: Group, tolerance: f64) -> Option<Time> {
        let k = self.feedback(weights)?;
        let cap = self.margin_cap();
        let top = f64::from(cap / self.dt);
        let stable = |ticks: f64| self.stable_with(&k, Delay::on(group, ticks));
        if !stable(0.0) {
            return None;
        }
        if stable(top) {
            return Some(cap);
        }
        let (mut lo, mut hi) = (0.0, 1.0_f64.min(top));
        while stable(hi) {
            (lo, hi) = (hi, (2.0 * hi).min(top));
        }
        while hi - lo > tolerance * hi {
            let mid = 0.5 * (lo + hi);
            if stable(mid) { lo = mid } else { hi = mid }
        }
        Some(lo * self.dt)
    }

    /// |∫ rate| over [`TRIM`] after a unit step torque on `group`'s trim axis
    /// (pitch for pitch/roll).
    pub fn trim(&mut self, weights: &Weights, group: Group) -> Option<TrimCost> {
        let k = self.feedback(weights)?;
        let axis = trim_axis(group);
        let mut step = CALM;
        step.torque[axis] = ANG_ACCEL;
        let mut x = self.rest(UNDELAYED);
        let mut angle = quantity!(0.0, rot);
        for _ in 0..f64::from(TRIM / self.dt) as usize + 1 {
            angle += x.rates[axis] * self.dt;
            x = self.tick(&x, &step, &k, UNDELAYED).0;
        }
        Some(angle.abs() / ANG_ACCEL)
    }

    /// The fit's recorded noise's motor roughness at hover, as flick.rs
    /// roughness.
    pub fn roughness(&mut self, weights: &Weights) -> Option<Voltage> {
        let k = self.feedback(weights)?;
        let noise = &self.noise;
        let mut x = self.rest(UNDELAYED);
        let settle = f64::from(SETTLE / self.dt) as usize;
        let g = pt1_gain(ROUGH_HP, self.dt);
        let (mut lpf, mut sum, mut count) = ([quantity!(0.0, V); 4], quantity!(0.0, V ^ 2), 0usize);
        for (i, w) in noise.iter().enumerate() {
            let (next, u) = self.tick(&x, w, &k, UNDELAYED);
            x = next;
            if i < settle {
                continue;
            }
            if i == settle {
                lpf = u;
            }
            for m in 0..4 {
                lpf[m] = lpf[m] + g * (u[m] - lpf[m]);
                let d = u[m] - lpf[m];
                sum += d * d;
            }
            count += 1;
        }
        Some((sum / (4.0 * count as f64)).sqrt())
    }

    /// The measures of `group` at `weights`; None if the loop is unstable.
    pub fn measures(&mut self, weights: &Weights, group: Group, tolerance: f64) -> Option<Measures> {
        let margin = self.delay_margin(weights, group, tolerance)?;
        Some(Measures {
            kick: self.kick_iae(weights, group),
            margin,
            margin_capped: margin >= self.margin_cap(),
            trim: self.trim(weights, group)?,
            roughness: self.roughness(weights)?,
        })
    }
}

/// A group's knock axis: roll for pitch/roll, yaw for yaw.
pub fn kick_axis(group: Group) -> usize {
    match group {
        Group::PitchRoll => 2,
        Group::Yaw => 0,
    }
}

/// A group's trim axis: pitch for pitch/roll (a CG shift), yaw for yaw.
pub fn trim_axis(group: Group) -> usize {
    match group {
        Group::PitchRoll => 1,
        Group::Yaw => 0,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Measures {
    pub kick: Time,
    pub margin: Time,
    /// The margin is at least [`HoverLoop::margin_cap`], not measured.
    pub margin_capped: bool,
    pub trim: TrimCost,
    pub roughness: Voltage,
}
