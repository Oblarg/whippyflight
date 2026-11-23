//! Gain blobs: the structured LQR gains over a grid of motor speeds, the
//! feedforward's reference model and the airframe constants the firmware
//! needs, packed as whippyflight-control's `.wfg`.

use whippyalgebra::nalgebra::{SMatrix, UniformUnitMatrix, block_matrix, gauge, zeros};
use whippyflight_control::gains::spec::BlobSpec;
use whippyflight_control::gains::{REF_CONST_FLOATS, REF_MOTOR_SLOTS};
use whippyunits::{qty, quantity, unit, value};

/// Rate of change of angular acceleration.
pub type Jerk = qty!(rot / s ^ 3);

use crate::disturbance::Disturbances;
use crate::fit::Fit;
use crate::lqr::{self, Costs, Weights};
use crate::par::{self, Control};
use crate::plant::{self, Discrete};
use crate::spaces::{DerivativeGain, Gain, IntegralGain, RateGain, Rates};
use crate::units::*;

/// The grid's lowest speed without a higher idle.
pub const MIN_GRID: Rate = quantity!(80.0, rot / s);
/// What the feedforward inverts the attitude response over by default.
pub const DEFAULT_FF_HORIZON: Time = quantity!(0.005, s);
/// The most grid points per motor the firmware's gain store holds.
pub const MAX_RESOLUTION: usize = 4;
/// The integral may cancel a pitch/roll imbalance as large as the hover
/// thrust acting this fraction of the arm off center.
const TRIM_CG_ARM_FRACTION: f64 = 0.25;
/// Fractions of the weakest pitch/roll authority the setpoint target may use.
const TARGET_SLEW_FRACTION: f64 = 0.83;
const TARGET_JERK_FRACTION: f64 = 0.63;

/// Speeds per motor, and the firmware's idle speed (0 for none).
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub resolution: usize,
    pub idle: Rate,
}

/// What a blob is solved for.
#[derive(Clone, Copy, Debug)]
pub struct Design {
    pub dt: Time,
    pub weights: Weights,
    /// What the feedforward inverts the attitude response over.
    pub ff_horizon: Time,
    pub disturbances: Disturbances,
    pub grid: GridSpec,
    /// Solve the gains once, at hover, for every grid point: a blob for
    /// scoring the loop at hover, not for flight.
    pub hover_only: bool,
}

/// The motor-speed grid.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub n: usize,
    pub min: Rate,
    pub max: Rate,
}

impl Grid {
    pub fn new(fit: &Fit, spec: GridSpec) -> Result<Grid, String> {
        if !(2..=MAX_RESOLUTION).contains(&spec.resolution) {
            return Err(format!("grid resolution must be 2–{MAX_RESOLUTION}, not {}", spec.resolution));
        }
        let min = if spec.idle > MIN_GRID { spec.idle } else { MIN_GRID };
        let step = quantity!(10.0, rot / s);
        let max = f64::from(fit.steady_speed(fit.full_battery()) / step).ceil() * step;
        if max <= min {
            return Err(format!(
                "the grid's bottom, {:.0} rev/s, is at or above the motors' full-battery speed, {:.0}",
                value!(min, rot / s),
                value!(max, rot / s)
            ));
        }
        Ok(Grid { n: spec.resolution, min, max })
    }

    pub fn speed(&self, i: usize) -> Rate {
        self.min + (i as f64 / (self.n - 1) as f64) * (self.max - self.min)
    }

    pub fn nodes(&self) -> usize {
        self.n.pow(4)
    }

    /// Node `node`'s grid index per slot (slot-major).
    pub fn indices(&self, node: usize) -> [usize; 4] {
        let n = self.n;
        [node / (n * n * n), node / (n * n) % n, node / n % n, node % n]
    }

    pub fn node(&self, indices: [usize; 4]) -> usize {
        indices.iter().fold(0, |acc, &i| acc * self.n + i)
    }

    /// The gains at slot speeds `omegas`, multilinearly interpolated as the
    /// firmware does (clamped to the grid).
    pub fn interpolate(&self, gains: &[Gain], omegas: [Rate; 4]) -> Gain {
        let pos: [(usize, f64); 4] = omegas.map(|w| {
            let x = f64::from((w - self.min) / (self.max - self.min)) * (self.n - 1) as f64;
            let x = x.clamp(0.0, (self.n - 1) as f64);
            let lower = (x.floor() as usize).min(self.n - 2);
            (lower, x - lower as f64)
        });
        let mut out: Gain = zeros![crate::spaces::Volts, crate::spaces::Applied];
        for corner in 0..16 {
            let mut weight = 1.0;
            let idx: [usize; 4] = core::array::from_fn(|m| {
                let upper = (corner >> m) & 1;
                weight *= if upper == 1 { pos[m].1 } else { 1.0 - pos[m].1 };
                pos[m].0 + upper
            });
            if weight > 0.0 {
                out += gains[self.node(idx)].scale(weight);
            }
        }
        out
    }
}

/// The LQR's voltage excursion: a volt per cell.
pub fn volt_excursion(fit: &Fit) -> Voltage {
    fit.cells * quantity!(1.0, V)
}

/// The gains at every node (slot-major), each solved from hover's, or
/// hover's for all, with the derivative filter at `tau_deriv`.
pub fn solve_grid(fit: &Fit, design: &Design, grid: &Grid, tau_deriv: Time, control: &Control) -> Result<Vec<Gain>, String> {
    let w = &design.weights;
    let costs = Costs::new(w.pitch_roll, w.yaw, volt_excursion(fit));
    let sample = |omegas: [Rate; 4]| Discrete::sample(&plant::effective_dynamics(fit, omegas), &plant::actuation(fit), design.dt);
    let system = |omegas: [Rate; 4]| plant::augmented(&sample(omegas), design.dt, w.tau_int, tau_deriv, &design.disturbances);
    let hover = lqr::solve(&system([fit.model.hover_rps; 4]), &costs, None).ok_or("no stabilizing gains at hover for these weights")?;
    if design.hover_only {
        return Ok(vec![hover; grid.nodes()]);
    }
    let nodes: Vec<usize> = (0..grid.nodes()).collect();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = nodes.len();
    let solved = par::map_with(&nodes, control, || (), |_, &node| {
        let omegas = grid.indices(node).map(|i| grid.speed(i));
        let gain = lqr::solve(&system(omegas), &costs, Some(&hover));
        let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if k.is_multiple_of((total / 10).max(1)) {
            control.say(&format!("  [{k}/{total}] operating points solved"));
        }
        gain.ok_or_else(|| {
            let speeds = omegas.map(|w| format!("{:.0}", value!(w, rot / s)));
            format!("no stabilizing gains at motor speeds {} rev/s", speeds.join(", "))
        })
    })
    .map_err(|c| c.to_string())?;
    solved.into_iter().collect()
}

/// `horizon` in whole ticks of `dt`, as the blob holds it.
pub fn horizon_ticks(horizon: Time, dt: Time) -> Result<u8, String> {
    let ticks = f64::from(horizon / dt).round();
    if !(1.0..=f64::from(u8::MAX)).contains(&ticks) {
        return Err(format!(
            "a {:.2} ms feedforward horizon is {ticks} ticks of {:.3} ms; it must be 1 to {}",
            value!(horizon, ms),
            value!(dt, ms),
            u8::MAX
        ));
    }
    Ok(ticks as u8)
}

/// The reference model at every motor at `speed`: one tick of the plant and
/// its response over the feedforward's `horizon` ticks.
struct Reference {
    ad: UnitMatrix<Dimensionless, 7, 7>,
    bd: UnitMatrix<RotPerSecondVolt, 7, 4>,
    fh: UnitMatrix<Dimensionless, 7, 7>,
    gh: UnitMatrix<RotPerSecondVolt, 7, 4>,
}

impl Reference {
    fn new(fit: &Fit, dt: Time, horizon: u8, speed: Rate) -> Self {
        let d = Discrete::sample(&plant::effective_dynamics(fit, [speed; 4]), &plant::actuation(fit), dt);
        let ad: UnitMatrix<Dimensionless, 7, 7> = d.ad.to_uniform();
        let bd: UnitMatrix<RotPerSecondVolt, 7, 4> = d.bd.to_uniform();
        let mut fh = UniformUnitMatrix::<unit!(1), SMatrix<f64, 7, 7>>::identity();
        let mut gh = UniformUnitMatrix::<unit!(rot / s / V), SMatrix<f64, 7, 4>>::from_element(quantity!(0.0, rot / s / V));
        for _ in 0..horizon {
            gh += fh * bd;
            fh = fh * ad;
        }
        Reference { ad, bd, fh, gh }
    }
}

/// FRD axis a's model axis.
fn model_axis(a: usize) -> usize {
    2 - a
}

/// Torque-free hover speed ratio per slot: ωᵢ ∝ 1/√(d_fbᵢ·d_lrᵢ), geometric
/// mean 1.
pub fn thrust_balance(fit: &Fit) -> [f64; 4] {
    let m = &fit.model;
    let fb = [m.w_front, m.w_back, m.w_front, m.w_back];
    let lr = [m.w_left, m.w_left, m.w_right, m.w_right];
    let b: [f64; 4] = core::array::from_fn(|s| 1.0 / (fb[s] * lr[s]).sqrt());
    let mean = (b.iter().map(|x| x.ln()).sum::<f64>() / 4.0).exp();
    b.map(|x| x / mean)
}

/// Hover speeds per slot.
fn hover_speeds(fit: &Fit) -> [Rate; 4] {
    thrust_balance(fit).map(|b| b * fit.model.hover_rps)
}

/// Setpoint target limits (slew, jerk) from the weakest pitch/roll
/// direction: from a torque-free hover the motors pushing that way go to
/// full speed on a full battery and the others to `idle`.
pub fn target_limits(fit: &Fit, idle: Rate) -> (AngAccel, Jerk) {
    let m = &fit.model;
    let hover = hover_speeds(fit);
    let v_hover = hover.map(|h| m.motor_vel * h + m.motor_vel_sq * h * h);
    let vbat = fit.full_battery();
    let full = fit.steady_speed(vbat);
    let pats = plant::patterns(fit);
    let props = plant::prop_coefficients(fit);
    let zero = quantity!(0.0, 1 / rot);
    let (mut slew, mut jerk) = (quantity!(f64::INFINITY, rot / s ^ 2), quantity!(f64::INFINITY, rot / s ^ 3));
    for axis in [1, 2] {
        for sign in [1.0, -1.0] {
            let s: [PerRot; 4] = core::array::from_fn(|i| sign * pats[axis][i] * props[axis]);
            let (mut a, mut j) = (quantity!(0.0, rot / s ^ 2), quantity!(0.0, rot / s ^ 3));
            for i in 0..4 {
                let up = s[i] > zero;
                let to = if up { full } else { idle };
                a += s[i] * (to * to - hover[i] * hover[i]);
                let step = if up { vbat - v_hover[i] } else { -v_hover[i] };
                j += s[i] * (2.0 * hover[i] * step / m.motor_accel);
            }
            if a < slew {
                slew = a;
            }
            if j < jerk {
                jerk = j;
            }
        }
    }
    (TARGET_SLEW_FRACTION * slew, TARGET_JERK_FRACTION * jerk)
}

/// The integral's voltage bound: the largest integral voltage on any motor
/// once the controller has trimmed a [`TRIM_CG_ARM_FRACTION`] CG offset on
/// pitch or on roll, from a torque-free hover, with the gains `gains`
/// interpolated at the trimmed speeds.
pub fn integral_limit(fit: &Fit, grid: &Grid, gains: &[Gain], tau_int: Time) -> Result<Voltage, String> {
    let m = &fit.model;
    let hover = hover_speeds(fit);
    let g = plant::linearizer(fit);
    let props = plant::prop_coefficients(fit);
    let volts = |w: Rate| m.motor_vel * w + m.motor_vel_sq * w * w.abs();
    let signed_sq = |w: [Rate; 4]| w.map(|x| x * x.abs());
    let linearized = |w: [Rate; 4]| {
        let sq = signed_sq(w);
        let v: [Voltage; 4] = core::array::from_fn(|i| (0..4).fold(quantity!(0.0, V), |acc, j| acc + g.get(i, j) * sq[j]));
        v
    };
    let lin_hover = linearized(hover);
    let v_c: [Voltage; 4] = core::array::from_fn(|i| volts(hover[i]) - lin_hover[i]);
    let hover_sq = hover.iter().fold(quantity!(0.0, rot ^ 2 / s ^ 2), |acc, &h| acc + h * h);
    let mut worst = quantity!(0.0, V);
    for axis in [1, 2] {
        let scale = props[axis] * hover_sq;
        for sign in [1.0, -1.0] {
            let mut want = [quantity!(0.0, rot / s ^ 2); 3];
            want[axis] = -sign * TRIM_CG_ARM_FRACTION * scale;
            let mut omega = hover;
            let mut e = [quantity!(0.0, rot / s); 3];
            let mut ki_tau = UniformUnitMatrix::<unit!(V * s / rot), SMatrix<f64, 4, 3>>::from_element(quantity!(0.0, V * s / rot));
            // The gains are the ones at the trimmed speeds: re-solve with them
            // there until the speeds settle (interpolation is only piecewise
            // smooth in ω, so they're held within each solve).
            for _ in 0..5 {
                let k = grid.interpolate(gains, omega);
                let ki: IntegralGain = k.block::<0, 0, 4, 3>().to_uniform();
                let kp: RateGain = k.block::<0, 6, 4, 3>().to_uniform();
                ki_tau = UniformUnitMatrix::from_fn(|i, j| ki.get(i, j) * tau_int);
                let feedback = UniformUnitMatrix::<unit!(V * s / rot), SMatrix<f64, 4, 3>>::from_fn(|i, j| ki_tau.get(i, j) + kp.get(i, j));
                (omega, e) = trim_solve(fit, &g, &feedback, &v_c, &want, scale, omega, e)?;
            }
            for i in 0..4 {
                let v = (0..3).fold(quantity!(0.0, V), |acc, a| acc + ki_tau.get(i, a) * e[a]).abs();
                if v > worst {
                    worst = v;
                }
            }
        }
    }
    Ok(worst)
}

/// Newton on the trimmed steady state: motor speeds `omega` and rate error
/// `e` with k_ω·ω + k_ω²·ω|ω| = feedback·e + G·ω|ω| + v_c and the speeds'
/// torques at `want` (normalized by `scale`).
#[allow(clippy::too_many_arguments)]
fn trim_solve(
    fit: &Fit,
    g: &plant::Linearizer,
    feedback: &RateGain,
    v_c: &[Voltage; 4],
    want: &[AngAccel; 3],
    scale: AngAccel,
    mut omega: [Rate; 4],
    mut e: [Rate; 3],
) -> Result<([Rate; 4], [Rate; 3]), String> {
    let m = &fit.model;
    let pats = plant::patterns(fit);
    let props = plant::prop_coefficients(fit);
    for _ in 0..100 {
        let sq = omega.map(|x| x * x.abs());
        let rv = UniformUnitMatrix::<unit!(V), SMatrix<f64, 4, 1>>::from_fn(|i, _| {
            m.motor_vel * omega[i] + m.motor_vel_sq * sq[i]
                - (0..3).fold(quantity!(0.0, V), |acc, a| acc + feedback.get(i, a) * e[a])
                - (0..4).fold(quantity!(0.0, V), |acc, j| acc + g.get(i, j) * sq[j])
                - v_c[i]
        });
        let rt = UniformUnitMatrix::<unit!(1), SMatrix<f64, 3, 1>>::from_fn(|a, _| {
            let torque = (0..4).fold(quantity!(0.0, rot / s ^ 2), |acc, j| acc + pats[a][j] * props[a] * sq[j]);
            (torque - want[a]) / scale
        });
        let dv_dw = UniformUnitMatrix::<unit!(V * s / rot), SMatrix<f64, 4, 4>>::from_fn(|i, j| {
            let slope = 2.0 * omega[j].abs();
            let own = if i == j { m.motor_vel + m.motor_vel_sq * slope } else { quantity!(0.0, V * s / rot) };
            own - g.get(i, j) * slope
        });
        let dv_de = UniformUnitMatrix::<unit!(V * s / rot), SMatrix<f64, 4, 3>>::from_fn(|i, a| -feedback.get(i, a));
        let dt_dw = UniformUnitMatrix::<unit!(s / rot), SMatrix<f64, 3, 4>>::from_fn(|a, j| pats[a][j] * props[a] * 2.0 * omega[j].abs() / scale);
        let jacobian = block_matrix![
            [gauge!(dv_dw => V, rot / s), gauge!(dv_de => V, rot / s)],
            [gauge!(dt_dw => 1, rot / s), zeros![whippyalgebra::dims![1, 1, 1], Rates]],
        ];
        let residual = block_matrix![[gauge!(rv => V, 1)], [gauge!(rt => 1, 1)]];
        let step = jacobian.try_inverse().ok_or("integral limit: singular trim Jacobian")? * residual;
        let dw: UnitMatrix<RotPerSecond, 4, 1> = step.block::<0, 0, 4, 1>().to_uniform();
        let de: UnitMatrix<RotPerSecond, 3, 1> = step.block::<4, 0, 3, 1>().to_uniform();
        let mut largest = 0.0f64;
        for (i, w) in omega.iter_mut().enumerate() {
            *w -= dw.get(i, 0);
            largest = largest.max(f64::from(dw.get(i, 0).abs() / w.abs()));
        }
        for (a, ea) in e.iter_mut().enumerate() {
            *ea -= de.get(a, 0);
        }
        if largest < 1e-12 {
            return Ok((omega, e));
        }
    }
    Err("integral limit: the trim solve didn't converge".into())
}

/// The blob for `design` on `fit`, and the derivative time constant it holds
/// ([`plant::derivative_tau`]).
pub fn generate(fit: &Fit, design: &Design, control: &Control) -> Result<(BlobSpec, Time), String> {
    let grid = Grid::new(fit, design.grid)?;
    let tau_deriv = plant::derivative_tau(fit, design.dt)?;
    let horizon = horizon_ticks(design.ff_horizon, design.dt)?;
    let n = grid.n;
    let m = &fit.model;
    let perm: [usize; 4] = core::array::from_fn(|i| fit.slot_of(i));
    control.say(&if design.hover_only {
        format!("Solving the structured LQR at hover for all {} grid points...", grid.nodes())
    } else {
        format!("Solving the structured LQR on {n}⁴ = {} grid points...", grid.nodes())
    });
    let gains = solve_grid(fit, design, &grid, tau_deriv, control)?;
    control.check().map_err(|c| c.to_string())?;

    let mut spec = BlobSpec::new(n as u8);
    spec.ff_horizon = horizon;
    spec.min_rps = value!(grid.min, rot / s) as f32;
    spec.max_rps = value!(grid.max, rot / s) as f32;
    spec.dt = value!(design.dt, s) as f32;
    spec.integral_tau = value!(design.weights.tau_int, s) as f32;
    spec.derivative_tau = value!(tau_deriv, s) as f32;

    let blocks: Vec<_> = gains
        .iter()
        .map(|k| {
            let ki: IntegralGain = k.block::<0, 0, 4, 3>().to_uniform();
            let kd: DerivativeGain = k.block::<0, 3, 4, 3>().to_uniform();
            let kp: RateGain = k.block::<0, 6, 4, 3>().to_uniform();
            (ki, kd, kp)
        })
        .collect();
    for firmware_node in 0..grid.nodes() {
        let by_motor = grid.indices(firmware_node);
        let mut by_slot = [0; 4];
        for i in 0..4 {
            by_slot[perm[i]] = by_motor[i];
        }
        let (ki, kd, kp) = &blocks[grid.node(by_slot)];
        for (motor, &s) in perm.iter().enumerate() {
            for a in 0..3 {
                let x = model_axis(a);
                let at = firmware_node * 12 + motor * 3 + a;
                spec.ki[at] = value!(ki.get(s, x), V / rot) as f32;
                spec.kd[at] = value!(kd.get(s, x), V * s ^ 2 / rot) as f32;
                spec.kp[at] = value!(kp.get(s, x), V * s / rot) as f32;
            }
        }
    }

    let origin = Reference::new(fit, design.dt, horizon, grid.speed(0));
    let mut ref_const = [0.0f32; REF_CONST_FLOATS];
    for a in 0..3 {
        let x = model_axis(a);
        ref_const[a] = f64::from(origin.ad.get(x, x)) as f32;
        ref_const[3 + a] = f64::from(origin.fh.get(x, x)) as f32;
    }
    spec.ref_const = ref_const;
    for i in 0..n {
        let r = if i == 0 { &origin } else { &Reference::new(fit, design.dt, horizon, grid.speed(i)) };
        for (motor, &s) in perm.iter().enumerate() {
            let mut slots = [0.0f32; REF_MOTOR_SLOTS];
            for a in 0..3 {
                let x = model_axis(a);
                slots[a] = f64::from(r.ad.get(x, 3 + s)) as f32;
                slots[4 + a] = value!(r.bd.get(x, s), rot / s / V) as f32;
                slots[8 + a] = f64::from(r.fh.get(x, 3 + s)) as f32;
                slots[11 + a] = value!(r.gh.get(x, s), rot / s / V) as f32;
            }
            slots[3] = f64::from(r.ad.get(3 + s, 3 + s)) as f32;
            slots[7] = value!(r.bd.get(3 + s, s), rot / s / V) as f32;
            let at = (i * 4 + motor) * REF_MOTOR_SLOTS;
            spec.ref_motor[at..at + REF_MOTOR_SLOTS].copy_from_slice(&slots);
        }
    }

    let g = plant::linearizer(fit);
    for i in 0..4 {
        for j in 0..4 {
            spec.affine[i * 4 + j] = value!(g.get(perm[i], perm[j]), V * s ^ 2 / rot ^ 2) as f32;
        }
    }
    spec.motor_drag = [value!(m.motor_vel, V * s / rot) as f32, value!(m.motor_vel_sq, V * s ^ 2 / rot ^ 2) as f32];
    let balance = thrust_balance(fit);
    spec.thrust_balance = core::array::from_fn(|i| balance[perm[i]] as f32);
    let d = fit.drag;
    spec.airframe_drag = if d.lp > quantity!(0.0, Hz) {
        [
            value!(d.lp, Hz) as f32,
            value!(d.roll_x, rot / m) as f32,
            value!(d.roll_y, rot / m) as f32,
            value!(d.pitch_x, rot / m) as f32,
            value!(d.pitch_y, rot / m) as f32,
        ]
    } else {
        [0.0; 5]
    };
    let (slew, jerk) = target_limits(fit, grid.min);
    spec.target_limits = [value!(slew, rot / s ^ 2) as f32, value!(jerk, rot / s ^ 3) as f32];
    control.say(&format!(
        "Target limits: slew {:.1} rot/s², jerk {:.0} rot/s³",
        value!(slew, rot / s ^ 2),
        value!(jerk, rot / s ^ 3)
    ));
    let bound = integral_limit(fit, &grid, &gains, design.weights.tau_int)?;
    control.say(&format!("Integral voltage bound: {:.3} V", value!(bound, V)));
    spec.hover = [value!(m.hover_rps, rot / s) as f32, value!(bound, V) as f32];
    let pats = plant::patterns(fit);
    let props = plant::prop_coefficients(fit);
    spec.thrust_torque = core::array::from_fn(|k| {
        let (axis, motor) = if k < 4 { (2, k) } else { (1, k - 4) };
        value!(pats[axis][perm[motor]] * props[axis], 1 / rot) as f32
    });
    spec.layout = m.layout;
    Ok((spec, tau_deriv))
}
