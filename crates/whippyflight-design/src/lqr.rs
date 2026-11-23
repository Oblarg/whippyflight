//! Structured discrete LQR: output feedback on the applied state as measured.
//!
//! Minimizes the steady J = E xᵀQx + uᵀRu over u = −K·C·x, C the applied
//! state as the firmware measures it, under the system's disturbances
//! (Levine–Athans: the stationarity condition iterated with a backtracking
//! step). Cold starts follow a discount homotopy from K = 0; warm starts take
//! any stabilizing gains (a neighbouring grid node's).

use std::cmp::Ordering;

use whippyalgebra::nalgebra::{UniformUnitMatrix, block_matrix, gauge, zeros};
use whippyalgebra::nalgebra::SMatrix;
use whippyunits::{quantity, unit};

use crate::spaces::*;
use crate::units::{Angle, Dimensionless, Rate, Time, UnitMatrix, Voltage};

/// Bryson's allowed excursions of one axis group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Excursions {
    pub rate: Rate,
    pub int: Angle,
}

/// The tradeoffs of a design: each axis group's excursions and the
/// integral's time constant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    pub pitch_roll: Excursions,
    pub yaw: Excursions,
    pub tau_int: Time,
}

impl Weights {
    pub fn group(&self, group: Group) -> Excursions {
        match group {
            Group::PitchRoll => self.pitch_roll,
            Group::Yaw => self.yaw,
        }
    }

    pub fn with_group(mut self, group: Group, excursions: Excursions) -> Self {
        match group {
            Group::PitchRoll => self.pitch_roll = excursions,
            Group::Yaw => self.yaw = excursions,
        }
        self
    }
}

/// The axes that share excursions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    PitchRoll,
    Yaw,
}

impl Group {
    pub const ALL: [Group; 2] = [Group::PitchRoll, Group::Yaw];

    /// Whether each model axis [yaw, pitch, roll] is in the group.
    pub fn axes(self) -> [bool; 3] {
        match self {
            Group::PitchRoll => [false, true, true],
            Group::Yaw => [true, false, false],
        }
    }
}

/// The LQR's costs.
pub struct Costs {
    pub q: StateCost,
    pub r: VoltCost,
}

impl Costs {
    /// Bryson's rule over `pitch_roll` and `yaw` excursions, the voltage
    /// allowed `volt`; the rest of the state is free.
    pub fn new(pitch_roll: Excursions, yaw: Excursions, volt: Voltage) -> Self {
        let axes = [yaw, pitch_roll, pitch_roll];
        let qi = UniformUnitMatrix::<unit!(1 / rot ^ 2), SMatrix<f64, 3, 3>>::from_fn(|i, j| {
            if i == j { 1.0 / (axes[i].int * axes[i].int) } else { quantity!(0.0, 1 / rot ^ 2) }
        });
        let qr = UniformUnitMatrix::<unit!(s ^ 2 / rot ^ 2), SMatrix<f64, 3, 3>>::from_fn(|i, j| {
            if i == j { 1.0 / (axes[i].rate * axes[i].rate) } else { quantity!(0.0, s ^ 2 / rot ^ 2) }
        });
        let applied: MixedUnitMatrix9 = block_matrix![
            [gauge!(qi => 1 / rot, rot), zeros![Dual<Integrals>, Derivatives], zeros![Dual<Integrals>, Rates]],
            [zeros![Dual<Derivatives>, Integrals], zeros![Dual<Derivatives>, Derivatives], zeros![Dual<Derivatives>, Rates]],
            [zeros![Dual<Rates>, Integrals], zeros![Dual<Rates>, Derivatives], gauge!(qr => s / rot, rot / s)],
        ];
        let q = block_matrix![
            [applied, zeros![Dual<Applied>, Unapplied]],
            [zeros![Dual<Unapplied>, Applied], zeros![Dual<Unapplied>, Unapplied]],
        ];
        let rv = UniformUnitMatrix::<unit!(1 / V ^ 2), SMatrix<f64, 4, 4>>::from_fn(|i, j| {
            if i == j { 1.0 / (volt * volt) } else { quantity!(0.0, 1 / V ^ 2) }
        });
        Costs { q, r: gauge!(rv => 1 / V, V) }
    }
}

type MixedUnitMatrix9 = whippyalgebra::nalgebra::MixedUnitMatrix<Dual<Applied>, Applied, SMatrix<f64, APPLIED, APPLIED>>;

/// One tick of the open loop, x ← A·x + B·u + w with E wwᵀ = W, and what of
/// it the firmware measures, C·x.
#[derive(Clone)]
pub struct System {
    pub a: StateMap,
    pub b: InputMap,
    pub c: Measurement,
    pub w: Covariance,
}

/// Squarings past which a loop counts as unstable (2³⁰ ticks).
const MAX_DOUBLINGS: usize = 30;
/// Relative change of the cost that ends the doubling.
const DOUBLING_TOLERANCE: f64 = 1e-15;
/// Relative rounding of a doubling's traces.
const ROUNDING: f64 = 1e-12;
/// Squarings after convergence (each squares the remaining tail).
const EXTRA_DOUBLINGS: usize = 1;

/// The closed loop's cost-to-go P, steady covariance L and cost J = tr(P·W).
pub struct Evaluation {
    pub j: f64,
    pub p: StateCost,
    pub l: Covariance,
}

fn full(sys: &System, gain: &Gain) -> MixedUnitMatrixVS {
    *gain * sys.c
}
type MixedUnitMatrixVS = whippyalgebra::nalgebra::MixedUnitMatrix<Volts, State, SMatrix<f64, 4, STATE>>;

/// tr(P·L), without forming the product.
fn trace_product(p: &StateCost, l: &Covariance) -> f64 {
    p.component_mul(&l.transpose()).as_uniform().iter().map(f64::from).sum()
}

/// The cost of `gain` on `sys`, by Smith doubling of both Stein equations;
/// None if the loop is unstable.
pub fn evaluate(sys: &System, costs: &Costs, gain: &Gain) -> Option<Evaluation> {
    let k = full(sys, gain);
    let mut f = sys.a - sys.b * k;
    let mut p = costs.q + k.transpose() * costs.r * k;
    let mut l = sys.w;
    let mut j = trace_product(&p, &sys.w);
    let mut s = trace_product(&p, &l);
    let mut extra = 0;
    for _ in 0..MAX_DOUBLINGS {
        let ft = f.transpose();
        p = p + ft * p * f;
        l = l + f * l * ft;
        f = f * f;
        let j_next = trace_product(&p, &sys.w);
        let s_next = trace_product(&p, &l);
        // Each doubling adds a positive semidefinite term; a fall past rounding
        // is a growing loop swamping it.
        let kept = |next: f64, last: f64| next.is_finite() && next >= last * (1.0 - ROUNDING);
        if !(j >= 0.0 && kept(j_next, j) && kept(s_next, s)) {
            return None;
        }
        let settled = j_next - j <= DOUBLING_TOLERANCE * j_next && s_next - s <= DOUBLING_TOLERANCE * s_next;
        j = j_next;
        s = s_next;
        if settled {
            extra += 1;
            if extra > EXTRA_DOUBLINGS {
                return Some(Evaluation { j, p, l });
            }
        }
    }
    None
}

/// `$m`, slots × applied states with rows `$rows` and columns `$cols`, with
/// the yaw axis's columns in the yaw pattern and the others out of it: each
/// axis group's feedback drives its own torques only.
macro_rules! decoupled {
    ($m:expr, $rows:ty, $cols:ty) => {{
        let norm: f64 = YAW_PATTERN.iter().map(|w| w * w).sum();
        let axes = Group::Yaw.axes();
        let yaw = MixedUnitMatrix::<$rows, $rows, SMatrix<f64, 4, 4>>::new(SMatrix::from_fn(|i, j| {
            YAW_PATTERN[i] * YAW_PATTERN[j] / norm
        }));
        let rest = MixedUnitMatrix::<$rows, $rows, SMatrix<f64, 4, 4>>::identity() - yaw;
        let columns = MixedUnitMatrix::<$cols, $cols, SMatrix<f64, APPLIED, APPLIED>>::new(SMatrix::from_fn(|i, j| {
            if i == j && axes[i % 3] { 1.0 } else { 0.0 }
        }));
        let others = MixedUnitMatrix::<$cols, $cols, SMatrix<f64, APPLIED, APPLIED>>::identity() - columns;
        yaw * $m * columns + rest * $m * others
    }};
}

/// Σ a(i,j)·b(i,j) of a gain step and a gradient, which pair to a plain number.
macro_rules! pair {
    ($a:expr, $b:expr) => {
        $a.component_mul($b).as_uniform().iter().map(f64::from).sum::<f64>()
    };
}

/// Smallest backtracking step before the iteration gives up.
const MIN_STEP: f64 = 1e-4;
/// Most iterations per solve.
const MAX_ITERATIONS: usize = 200;
/// Step/gradient-change pairs the quasi-Newton step remembers.
const MEMORY: usize = 8;
/// The sufficient decrease a step must make, of its first-order prediction.
const ARMIJO: f64 = 1e-4;

/// L-BFGS from the stabilizing `gain` until J improves by less than
/// `tolerance`·J, preconditioned by the Levine–Athans step −½·M⁻¹·∇J·(CLCᵀ)⁻¹
/// (M = R + BᵀPB, CLCᵀ the measured state's covariance), whose fixed point
/// is the stationarity condition.
fn descend(sys: &System, costs: &Costs, mut gain: Gain, mut ev: Evaluation, tolerance: f64) -> (Gain, Evaluation) {
    let bt = sys.b.transpose();
    let ct = sys.c.transpose();
    let mut history = std::collections::VecDeque::new();
    let mut previous = None;
    for _ in 0..MAX_ITERATIONS {
        let curvature = costs.r + bt * ev.p * sys.b;
        let seen = ev.l * ct;
        let slope = bt * ev.p * sys.a * seen;
        let spread = sys.c * seen;
        let grad = (curvature * gain * spread - slope).scale(2.0);
        let grad = decoupled!(grad, Dual<Volts>, Dual<Applied>);
        let (Some(curvature_inv), Some(spread_inv)) = (curvature.try_inverse(), spread.try_inverse()) else { break };
        macro_rules! precondition {
            ($q:expr) => {{
                let step: Gain = (curvature_inv.clone() * $q * spread_inv.clone()).scale(0.5);
                decoupled!(step, Volts, Applied)
            }};
        }
        if let Some((last_gain, last_grad)) = previous.take() {
            let s: Gain = gain - last_gain;
            let y = grad - last_grad;
            let sy = pair!(s, &y);
            if sy > 0.0 {
                history.push_back((s, y, 1.0 / sy));
                if history.len() > MEMORY {
                    history.pop_front();
                }
            }
        }
        let mut q = grad;
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let a = rho * pair!(s, &q);
            q -= y.scale(a);
            alphas.push(a);
        }
        let mut r: Gain = precondition!(q);
        if let Some((s, y, _)) = history.back() {
            let hy: Gain = precondition!(*y);
            let yhy = pair!(hy, y);
            if yhy > 0.0 {
                r = r.scale(pair!(s, y) / yhy);
            }
        }
        for ((s, y, rho), a) in history.iter().zip(alphas.iter().rev()) {
            let b = rho * pair!(r, y);
            r += s.scale(a - b);
        }
        let mut direction = r.scale(-1.0);
        let mut descent = pair!(direction, &grad);
        if descent.partial_cmp(&0.0) != Some(Ordering::Less) {
            history.clear();
            let la: Gain = precondition!(grad);
            direction = la.scale(-1.0);
            descent = pair!(direction, &grad);
        }
        let mut step = 1.0;
        let accepted = loop {
            if step <= MIN_STEP {
                break None;
            }
            let trial = gain + direction.scale(step);
            match evaluate(sys, costs, &trial) {
                Some(next) if next.j <= ev.j + ARMIJO * step * descent && next.j < ev.j => break Some((trial, next)),
                _ => step /= 2.0,
            }
        };
        let Some((next_gain, next)) = accepted else { break };
        // A short step's gain is judged per unit step.
        let done = ev.j - next.j < tolerance * step * ev.j;
        previous = Some((gain, grad));
        gain = next_gain;
        ev = next;
        if done {
            break;
        }
    }
    (gain, ev)
}

/// Discount of the homotopy's first stage.
const FIRST_DISCOUNT: f64 = 0.99;
/// Tolerance of the homotopy's intermediate stages.
const STAGE_TOLERANCE: f64 = 1e-6;
/// Tolerance of the final solve.
pub const TOLERANCE: f64 = 1e-9;
/// Most homotopy stages.
const MAX_STAGES: usize = 64;

fn discounted(sys: &System, discount: f64) -> System {
    System { a: sys.a.scale(discount), b: sys.b.scale(discount), ..sys.clone() }
}

/// Spectral radius of the closed loop of `gain` on `sys`, gauged to
/// dimensionless by each state's unit.
fn spectral_radius(sys: &System, gain: &Gain) -> f64 {
    let ones = |n| (0..n).map(|_| 1.0);
    let units = block_matrix![
        [gauge!(UniformUnitMatrix::<unit!(rot), SMatrix<f64, 3, 1>>::from_iterator(ones(3)) => rot, 1)],
        [gauge!(UniformUnitMatrix::<unit!(rot / s ^ 2), SMatrix<f64, 3, 1>>::from_iterator(ones(3)) => rot / s ^ 2, 1)],
        [gauge!(UniformUnitMatrix::<unit!(rot / s), SMatrix<f64, 10, 1>>::from_iterator(ones(10)) => rot / s, 1)],
        [gauge!(UniformUnitMatrix::<unit!(rot / s ^ 2), SMatrix<f64, 3, 1>>::from_iterator(ones(3)) => rot / s ^ 2, 1)],
        [gauge!(UniformUnitMatrix::<unit!(rot / s), SMatrix<f64, 6, 1>>::from_iterator(ones(6)) => rot / s, 1)],
    ];
    let scale = MixedUnitMatrix::from_diagonal(&units);
    let cl = sys.a - sys.b * full(sys, gain);
    let gauged: UnitMatrix<Dimensionless, STATE, STATE> =
        (scale.try_inverse().expect("unit references are nonzero") * cl * scale).to_uniform();
    crate::spectrum::spectral_radius(&Endomorphism::from_nalgebra(DMatrix::from_column_slice(
        STATE,
        STATE,
        gauged.nalgebra().as_slice(),
    )))
}

use crate::spectrum::Endomorphism;
use whippyalgebra::nalgebra::{DMatrix, MixedUnitMatrix};

/// The structured LQR gains on `sys` descended from `start`; None if `start`
/// doesn't stabilize it.
pub fn refine(sys: &System, costs: &Costs, start: &Gain) -> Option<Gain> {
    let start: Gain = decoupled!(*start, Volts, Applied);
    let ev = evaluate(sys, costs, &start)?;
    Some(descend(sys, costs, start, ev, TOLERANCE).0)
}

/// The structured LQR gains on `sys`, from the stabilizing `start` if given,
/// else by discount homotopy from zero gains; None if no stabilizing gains
/// were found.
pub fn solve(sys: &System, costs: &Costs, start: Option<&Gain>) -> Option<Gain> {
    if let Some(gain) = start.and_then(|start| refine(sys, costs, start)) {
        return Some(gain);
    }
    let mut gain: Gain = zeros![Volts, Applied];
    let mut discount = FIRST_DISCOUNT.min(0.99 / spectral_radius(sys, &gain).max(1e-300));
    for _ in 0..MAX_STAGES {
        let stage = discounted(sys, discount);
        let ev = evaluate(&stage, costs, &gain)?;
        gain = descend(&stage, costs, gain, ev, STAGE_TOLERANCE).0;
        let radius = spectral_radius(sys, &gain);
        // A radius a hair under 1 settles too slowly to evaluate.
        if radius < 1.0
            && let Some(ev) = evaluate(sys, costs, &gain)
        {
            return Some(descend(sys, costs, gain, ev, TOLERANCE).0);
        }
        discount = 0.5 * (discount + 1.0 / radius.max(1.0));
    }
    None
}
