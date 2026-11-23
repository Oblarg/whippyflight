//! The fit's plant linearized at motor speeds (gen_wfg.py's), in model
//! order, and the LQR's augmented system on it.

use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, UniformUnitMatrix, block_matrix, gauge, zeros};
use whippyflight_sim::margin::stage_cutoffs;
use whippyunits::{quantity, unit, value};

use crate::disturbance::Disturbances;
use crate::fit::Fit;
use crate::lqr::System;
use crate::spaces::*;
use crate::units::*;

/// Continuous dynamics: d[rates, motors]/dt per [rates, motors].
pub type Dynamics = UniformUnitMatrix<unit!(1 / s), SMatrix<f64, 7, 7>>;
/// Continuous input: d[rates, motors]/dt per volt.
pub type Actuation = UniformUnitMatrix<unit!(rot / s ^ 2 / V), SMatrix<f64, 7, 4>>;
/// The affine term's map from the speeds' signed squares to d[rates, motors]/dt.
pub type Curvature = UniformUnitMatrix<unit!(1 / rot), SMatrix<f64, 7, 4>>;
/// Volts per signed squared speed, slot × slot.
pub type Linearizer = UniformUnitMatrix<unit!(V * s ^ 2 / rot ^ 2), SMatrix<f64, 4, 4>>;

/// Per slot, its weight in the [yaw, pitch, roll] torques of the speeds'
/// signed squares.
pub fn patterns(fit: &Fit) -> [[f64; 4]; 3] {
    let m = &fit.model;
    [
        YAW_PATTERN,
        [m.w_front, -m.w_back, m.w_front, -m.w_back],
        [m.w_left, m.w_left, -m.w_right, -m.w_right],
    ]
}

/// The [yaw, pitch, roll] thrust coefficients of the patterns.
pub fn prop_coefficients(fit: &Fit) -> [PerRot; 3] {
    let m = &fit.model;
    [m.yaw_prop_vel_sq, m.pitch_prop_vel_sq, m.roll_prop_vel_sq]
}

fn rate_damping(fit: &Fit) -> [PerTime; 3] {
    let m = &fit.model;
    [m.yaw_rate, m.pitch_rate, m.roll_rate]
}

/// The motors' own dynamics: dω̇/dω at ω.
fn motor_pole(fit: &Fit, omega: Rate) -> PerTime {
    let m = &fit.model;
    -(m.motor_vel + 2.0 * m.motor_vel_sq * omega) / m.motor_accel
}

/// The plant linearized at slot speeds `omegas`, holding the affine term.
pub fn dynamics(fit: &Fit, omegas: [Rate; 4]) -> Dynamics {
    let pats = patterns(fit);
    let props = prop_coefficients(fit);
    let damping = rate_damping(fit);
    let k_i = fit.model.yaw_prop_accel;
    let zero = quantity!(0.0, 1 / s);
    Dynamics::from_fn(|i, j| match (i, j) {
        (0..3, 0..3) if i == j => damping[i],
        (0..3, 0..3) => zero,
        (0..3, _) => {
            let w = omegas[j - 3];
            let thrust = 2.0 * props[i] * w;
            pats[i][j - 3] * if i == 0 { thrust + k_i * motor_pole(fit, w) } else { thrust }
        }
        (_, 0..3) => zero,
        _ if i == j => motor_pole(fit, omegas[i - 3]),
        _ => zero,
    })
}

pub fn actuation(fit: &Fit) -> Actuation {
    let pats = patterns(fit);
    let per_volt = 1.0 / fit.model.motor_accel;
    let k_i = fit.model.yaw_prop_accel;
    Actuation::from_fn(|i, j| match i {
        0 => pats[0][j] * k_i * per_volt,
        1 | 2 => 0.0 * per_volt,
        _ if i - 3 == j => per_volt,
        _ => 0.0 * per_volt,
    })
}

/// C(ω̂) = P·[ω̂ᵢ|ω̂ᵢ|]: the true dynamics less the linear model's, at ω̂.
pub fn curvature(fit: &Fit) -> Curvature {
    let pats = patterns(fit);
    let props = prop_coefficients(fit);
    let m = &fit.model;
    let motor = m.motor_vel_sq / m.motor_accel;
    let yaw = m.yaw_prop_accel * motor - props[0];
    Curvature::from_fn(|i, j| match i {
        0 => pats[0][j] * yaw,
        1 | 2 => -pats[i][j] * props[i],
        _ if i - 3 == j => motor,
        _ => 0.0 * motor,
    })
}

/// The affine linearizer G: u_aff = G·[ω̂ᵢ|ω̂ᵢ|] cancels the affine term's
/// component in range(B) (the yaw and motor rows); pitch and roll's are
/// unmatched and left to the integrator.
pub fn linearizer(fit: &Fit) -> Linearizer {
    let b = actuation(fit);
    let bt = b.transpose();
    let pinv = (bt * b).try_inverse().expect("the motors actuate independently") * bt;
    -(pinv * curvature(fit))
}

/// The linearizer's slope on the speeds at slot speeds `omegas`.
pub fn linearizer_slope(fit: &Fit, omegas: [Rate; 4]) -> UnitMatrix<VoltSecondPerRot, 4, 4> {
    let g = linearizer(fit);
    UniformUnitMatrix::from_fn(|i, j| g.get(i, j) * (2.0 * omegas[j].abs()))
}

/// The plant as the controller drives it at slot speeds `omegas`: the
/// linearizer, re-evaluated on the measured speeds every tick, adds its
/// slope as feedback through them. The motor rows keep its diagonal only.
pub fn effective_dynamics(fit: &Fit, omegas: [Rate; 4]) -> Dynamics {
    let mut a = dynamics(fit, omegas);
    let feedback = actuation(fit) * linearizer_slope(fit, omegas);
    for i in 0..7 {
        for j in 0..4 {
            if i < 3 || i - 3 == j {
                a.set(i, 3 + j, a.get(i, 3 + j) + feedback.get(i, j));
            }
        }
    }
    a
}

/// The plant's state: rates then motor speeds.
pub type PlantState = whippyalgebra::Concatenated<Rates, Motors>;
/// One tick of the plant.
pub type PlantMap = MixedUnitMatrix<PlantState, PlantState, SMatrix<f64, 7, 7>>;
/// Voltage, held over a tick, into the plant.
pub type PlantInput = MixedUnitMatrix<PlantState, Volts, SMatrix<f64, 7, 4>>;
/// External angular acceleration, held over a tick, into the plant.
pub type PlantTorque = MixedUnitMatrix<PlantState, Torques, SMatrix<f64, 7, 3>>;

/// The plant sampled at `dt` (zero-order hold on its inputs).
#[derive(Clone)]
pub struct Discrete {
    pub ad: PlantMap,
    pub bd: PlantInput,
    pub bt: PlantTorque,
}

impl Discrete {
    pub fn sample(a: &Dynamics, b: &Actuation, dt: Time) -> Discrete {
        let torque = UniformUnitMatrix::<unit!(s), SMatrix<f64, 7, 3>>::from_fn(|i, j| {
            if i == j { dt } else { 0.0 * dt }
        });
        let van_loan = block_matrix![
            [gauge!(*a * dt => rot / s, rot / s), gauge!(*b * dt => rot / s, V), gauge!(torque => rot / s, rot / s ^ 2)],
            [zeros![Volts, PlantState], zeros![Volts, Volts], zeros![Volts, Torques]],
            [zeros![Torques, PlantState], zeros![Torques, Volts], zeros![Torques, Torques]],
        ]
        .exp();
        Discrete {
            ad: van_loan.block::<0, 0, 7, 7>(),
            bd: van_loan.block::<0, 7, 7, 4>(),
            bt: van_loan.block::<0, 11, 7, 3>(),
        }
    }
}

/// A PT1's gain per tick of `dt` at `cutoff`.
pub fn pt1_gain(cutoff: Freq, dt: Time) -> f64 {
    let x = f64::from(2.0 * core::f64::consts::PI * cutoff * dt);
    x / (x + 1.0)
}

/// `delay` in whole ticks of `dt`, rounded.
pub fn ticks(delay: Time, dt: Time) -> usize {
    f64::from(delay / dt).round().max(0.0) as usize
}

/// Which filter chain a low-pass stage is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaySource {
    Gyro,
    DTerm,
}

/// One low-pass stage between the rate and the controller's derivative.
#[derive(Clone, Debug)]
pub struct Delay {
    pub source: DelaySource,
    /// The stage's type ("PT1", "PT2", "PT3").
    pub kind: String,
    pub cutoff: Freq,
    /// The top of the stage's dynamic range, where it has one; it's held at
    /// its cutoff.
    pub max_cutoff: Option<Freq>,
    /// Whether the fit recorded it; a default stands in if not.
    pub recorded: bool,
    /// Its mean delay.
    pub delay: Time,
}

impl Delay {
    /// What it is and how long, as one line.
    pub fn describe(&self) -> String {
        let name = if self.source == DelaySource::Gyro { "gyro" } else { "D-term" };
        let dynamic =
            self.max_cutoff.map_or(String::new(), |max| format!(" (dynamic to {:.0} Hz, held at its floor)", value!(max, Hz)));
        let assumed = if self.recorded { "" } else { " — not in the fit; default" };
        format!(
            "{name} {} {:.0} Hz{dynamic}: {:.2} ms{assumed}",
            self.kind,
            value!(self.cutoff, Hz),
            value!(self.delay, ms)
        )
    }
}

/// The low-pass stages between the rate and the controller's derivative on
/// `fit`'s airframe, the gyro's and the D-term's, each with its mean delay.
pub fn derivative_delays(fit: &Fit) -> Result<Vec<Delay>, String> {
    let (m, r) = (&fit.model, &fit.recorded);
    let mut delays = vec![];
    for (source, stages, recorded, maxima, name) in [
        (DelaySource::Gyro, &m.gyro_lpf, r.gyro_lpf, &r.gyro_lpf_max, "gyro"),
        (DelaySource::DTerm, &m.dterm_lpf, r.dterm_lpf, &r.dterm_lpf_max, "D-term"),
    ] {
        for (i, stage) in stages.iter().enumerate() {
            let delay = stage_cutoffs(std::slice::from_ref(stage), name)?
                .into_iter()
                .map(|f| 1.0 / (2.0 * core::f64::consts::PI * f))
                .reduce(|a, b| a + b)
                .ok_or_else(|| format!("{name} LPF stage {i} has no poles"))?;
            let max_cutoff = maxima.get(i).copied().flatten();
            delays.push(Delay { source, kind: stage.0.clone(), cutoff: stage.1, max_cutoff, recorded, delay });
        }
    }
    Ok(delays)
}

/// The derivative filter's time constant on `fit`'s airframe: the summed mean
/// delay of its [`derivative_delays`], as one first-order lag. An error if
/// it's under a loop tick of `dt`.
pub fn derivative_tau(fit: &Fit, dt: Time) -> Result<Time, String> {
    let tau = derivative_delays(fit)?
        .into_iter()
        .map(|d| d.delay)
        .reduce(|a, b| a + b)
        .ok_or("the airframe has no gyro or D-term low-pass stages")?;
    if tau < dt {
        return Err(format!(
            "the airframe's gyro and D-term filters add up to {:.3} ms, under a {:.3} ms loop tick",
            value!(tau, ms),
            value!(dt, ms)
        ));
    }
    Ok(tau)
}

/// Rows of the augmented state's map, in `D`'s units.
type Rows<D, const N: usize> = MixedUnitMatrix<D, State, SMatrix<f64, N, STATE>>;

/// The plant at slot speeds `omegas` augmented with the controller's
/// filters and what disturbs it: the noise on the measured rate, the leaky
/// integral and the derivative filter (time constant `tau_deriv`) of the
/// measured rate error, the derivative's one-tick delay and the turbulence's
/// torques. The firmware updates its integral and derivative with this
/// tick's error before applying them, so C holds those updates.
pub fn augmented(plant: &Discrete, dt: Time, tau_int: Time, tau_deriv: Time, disturbances: &Disturbances) -> System {
    let id = StateMap::identity();
    let integ: Rows<Integrals, 3> = id.block::<0, 0, 3, STATE>();
    let deriv: Rows<Derivatives, 3> = id.block::<3, 0, 3, STATE>();
    let rates: Rows<Rates, 3> = id.block::<6, 0, 3, STATE>();
    let plant_state: Rows<PlantState, 7> = id.block::<6, 0, 7, STATE>();
    let prev: Rows<Rates, 3> = id.block::<13, 0, 3, STATE>();
    let torques: Rows<Torques, 3> = id.block::<16, 0, 3, STATE>();
    let noise: Rows<GyroNoise, 6> = id.block::<19, 0, 6, STATE>();
    let noise_now: Rows<Rates, 3> = id.block::<19, 0, 3, STATE>();
    let measured = rates + noise_now;

    let leak = (tau_int - dt) / tau_int;
    let smooth = (tau_deriv - dt) / tau_deriv;
    let integral = gauge!(diag(leak) => rot, rot) * integ + gauge!(diag(dt) => rot, rot / s) * measured;
    let derivative = gauge!(diag(smooth) => rot / s ^ 2, rot / s ^ 2) * deriv
        + gauge!(diag(1.0 / tau_deriv) => rot / s ^ 2, rot / s) * (measured - prev);
    let c: Measurement = block_matrix![[integral], [derivative], [measured]];

    let plant_next = plant.ad * plant_state + plant.bt * torques;
    let tilt = disturbances.turbulence.map(|t| f64::from(-2.0 * core::f64::consts::PI * t.corner * dt).exp());
    let torque_next = gauge!(diag3(tilt.map(|x| quantity!(x, 1))) => rot / s ^ 2, rot / s ^ 2) * torques;
    let gyro = disturbances.gyro;
    let shaping = UniformUnitMatrix::<unit!(1), SMatrix<f64, 6, 6>>::from_fn(|i, j| match (i < 3, j < 3) {
        (true, true) if i == j => quantity!(gyro[i].a[0], 1),
        (true, false) if i == j - 3 => quantity!(gyro[i].a[1], 1),
        (false, true) if i - 3 == j => quantity!(1.0, 1),
        _ => quantity!(0.0, 1),
    });
    let noise_next = gauge!(shaping => rot / s, rot / s) * noise;
    let a = block_matrix![[integral], [derivative], [plant_next], [measured], [torque_next], [noise_next]];

    let b = block_matrix![
        [zeros![Concatenated<Integrals, Derivatives>, Volts]],
        [plant.bd],
        [zeros![Concatenated<Rates, DisturbanceStates>, Volts]],
    ];

    type Quiet = Concatenated<Applied, Concatenated<Motors, Rates>>;
    let turbulence = disturbances.turbulence;
    let torque_power = diag3(core::array::from_fn(|a| turbulence[a].rms * turbulence[a].rms * (1.0 - tilt[a] * tilt[a])));
    let innovation = UniformUnitMatrix::<unit!(rot ^ 2 / s ^ 2), SMatrix<f64, 6, 6>>::from_fn(|i, j| {
        if i == j && i < 3 { gyro[i].innovation * gyro[i].innovation } else { quantity!(0.0, rot ^ 2 / s ^ 2) }
    });
    let w = block_matrix![
        [zeros![Quiet, Dual<Quiet>], zeros![Quiet, Dual<Torques>], zeros![Quiet, Dual<GyroNoise>]],
        [zeros![Torques, Dual<Quiet>], gauge!(torque_power => rot / s ^ 2, s ^ 2 / rot), zeros![Torques, Dual<GyroNoise>]],
        [zeros![GyroNoise, Dual<Quiet>], zeros![GyroNoise, Dual<Torques>], gauge!(innovation => rot / s, s / rot)],
    ];
    System { a, b, c, w }
}

/// `x` on a 3×3 diagonal.
pub fn diag<U>(x: Quantity<U>) -> UniformUnitMatrix<U, SMatrix<f64, 3, 3>> {
    UniformUnitMatrix::from_diagonal(&UniformUnitMatrix::from_element(x))
}

/// `x`, model axis by axis, on a 3×3 diagonal.
pub fn diag3<U>(x: [Quantity<U>; 3]) -> UniformUnitMatrix<U, SMatrix<f64, 3, 3>> {
    let mut x = x.into_iter();
    UniformUnitMatrix::from_diagonal(&UniformUnitMatrix::<U, SMatrix<f64, 3, 1>>::from_fn(|_, _| x.next().unwrap()))
}

use whippyalgebra::Concatenated;
use whippyunits::Quantity;
