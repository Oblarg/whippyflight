//! Regression rows: each axis's regressors and angular acceleration (or, for
//! the motors, voltage) over a flight's usable samples.
//!
//! Rows are central finite differences over a window of samples (a
//! duration, so logs at different rates difference over the same span). The
//! regressions run in Betaflight's motor order, in the physical frame turned
//! 180° about z (the frame lqr-prep/main.R solved in); `crate::frame`
//! converts to and from the canonical one.
//!
//! Units of the columns: rates rot/s, signed squared speeds rot²/s², motor
//! accelerations and the rate axes' y rot/s², specific force m/s², volts V.

use nalgebra::{DMatrix, DVector};
use whippyunits::{quantity, value};

use crate::flight::{Flight, VoltsMap};
use crate::num;
use crate::units::*;

/// main.R's windows, in samples of the log it was tuned on.
const R_LOG_HZ: f64 = 1609.0;
const EXCITATION_PERCENTILE: f64 = 99.5;
/// Accelerometer low-pass of the drag regressors (the firmware's filter).
pub const DRAG_LP: Freq = quantity!(5.0, Hz);
pub const DRAG_TERMS: [&str; 2] = ["accX", "accY"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    Yaw,
    Pitch,
    Roll,
    Motors,
}

pub const AXES: [Axis; 4] = [Axis::Yaw, Axis::Pitch, Axis::Roll, Axis::Motors];
pub const RATE_AXES: [Axis; 3] = [Axis::Yaw, Axis::Pitch, Axis::Roll];
pub const ARM_AXES: [Axis; 2] = [Axis::Pitch, Axis::Roll];

impl Axis {
    pub fn name(self) -> &'static str {
        match self {
            Axis::Yaw => "yaw",
            Axis::Pitch => "pitch",
            Axis::Roll => "roll",
            Axis::Motors => "motors",
        }
    }

    pub fn parse(s: &str) -> Option<Axis> {
        AXES.into_iter().find(|a| a.name() == s)
    }

    fn r_window(self) -> f64 {
        match self {
            Axis::Yaw => 50.0,
            Axis::Pitch | Axis::Roll => 20.0,
            Axis::Motors => 16.0,
        }
    }

    pub fn threshold(self) -> f64 {
        match self {
            Axis::Motors => 0.1,
            _ => 0.3,
        }
    }

    /// The gyro axis (FLU) of a rate axis.
    pub fn gyro(self) -> usize {
        match self {
            Axis::Roll => 0,
            Axis::Pitch => 1,
            Axis::Yaw => 2,
            Axis::Motors => unreachable!("the motors have no gyro axis"),
        }
    }

    /// (positive-side arm, negative-side arm) of a pitch/roll thrust regressor.
    pub fn arm_keys(self) -> (&'static str, &'static str) {
        match self {
            Axis::Pitch => ("dFront", "dBack"),
            Axis::Roll => ("dRight", "dLeft"),
            _ => unreachable!("only pitch and roll have arms"),
        }
    }
}

/// A flight's finite-difference window for `axis`, samples (even).
pub fn window(flight: &Flight, axis: Axis) -> usize {
    let hz = value!(flight.log_hz, Hz);
    (2 * num::round_even(axis.r_window() / R_LOG_HZ * hz / 2.0) as usize).max(2)
}

/// The regression-frame arm lengths; None where the fit finds the ratio.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Arms {
    pub d_front: Option<f64>,
    pub d_back: Option<f64>,
    pub d_left: Option<f64>,
    pub d_right: Option<f64>,
}

impl Arms {
    /// (positive side, negative side) of `axis`.
    pub fn of(&self, axis: Axis) -> (Option<f64>, Option<f64>) {
        match axis {
            Axis::Pitch => (self.d_front, self.d_back),
            Axis::Roll => (self.d_right, self.d_left),
            _ => (None, None),
        }
    }

    pub fn set(&mut self, axis: Axis, pos: f64, neg: f64) {
        match axis {
            Axis::Pitch => (self.d_front, self.d_back) = (Some(pos), Some(neg)),
            Axis::Roll => (self.d_right, self.d_left) = (Some(pos), Some(neg)),
            _ => {}
        }
    }

    /// Both arms of `axis`, if given.
    pub fn pinned(&self, axis: Axis) -> Option<(f64, f64)> {
        match self.of(axis) {
            (Some(p), Some(n)) => Some((p, n)),
            _ => None,
        }
    }
}

/// Regression rows: `x β ≈ y`, the regressors' names, each row's sample.
#[derive(Clone, Debug)]
pub struct Rows {
    pub x: DMatrix<f64>,
    pub y: DVector<f64>,
    pub names: Vec<String>,
    pub center: Vec<usize>,
}

impl Rows {
    pub fn len(&self) -> usize {
        self.y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
}

fn trunc<T: Copy>(x: &[T], n: usize) -> &[T] {
    &x[n / 2 - 1..x.len() - n / 2 - 1]
}

fn cfd(x: &[Rate], t: &[Time], n: usize) -> Vec<AngAccel> {
    (0..x.len() - n).map(|k| (x[k + n] - x[k]) / (t[k + n] - t[k])).collect()
}

fn sq(w: Rate) -> SpeedSq {
    w * w.abs()
}

fn sum_sq(flight: &Flight, plus: &[usize], minus: &[usize]) -> Vec<SpeedSq> {
    (0..flight.len())
        .map(|k| {
            let p = plus.iter().fold(quantity!(0.0, rot ^ 2 / s ^ 2), |a, &m| a + sq(flight.speeds[m][k]));
            minus.iter().fold(p, |a, &m| a - sq(flight.speeds[m][k]))
        })
        .collect()
}

fn rates(x: &[Rate]) -> impl Iterator<Item = f64> + '_ {
    x.iter().map(|&r| value!(r, rot / s))
}

fn speeds_sq(x: &[SpeedSq]) -> impl Iterator<Item = f64> + '_ {
    x.iter().map(|&w| value!(w, rot ^ 2 / s ^ 2))
}

fn accels(x: &[AngAccel]) -> impl Iterator<Item = f64> + '_ {
    x.iter().map(|&a| value!(a, rot / s ^ 2))
}

/// For each finite-difference row k (samples k..=k+n): every sample usable.
fn rows_ok(good: &[bool], n: usize) -> Vec<bool> {
    let mut bad = vec![0usize; good.len() + 1];
    for (i, &g) in good.iter().enumerate() {
        bad[i + 1] = bad[i] + usize::from(!g);
    }
    (0..good.len() - n).map(|k| bad[k + n + 1] == bad[k]).collect()
}

/// Row mask: window usable and, if `gated`, |gate| above `threshold` × the
/// excitation scale (a high percentile of |gate| over the usable rows).
fn gate(good: &[bool], gate: &[f64], threshold: f64, n: usize, gated: bool) -> Vec<bool> {
    let ok = rows_ok(good, n);
    if !gated || !ok.iter().any(|&o| o) {
        return ok;
    }
    let mut usable: Vec<f64> = gate.iter().zip(&ok).filter(|&(_, &o)| o).map(|(g, _)| g.abs()).collect();
    let scale = num::percentile_in_place(&mut usable, EXCITATION_PERCENTILE);
    ok.iter().zip(gate).map(|(&o, g)| o && g.abs() > scale * threshold).collect()
}

fn matrix(columns: Vec<Vec<f64>>) -> DMatrix<f64> {
    let rows = columns.first().map_or(0, Vec::len);
    let k = columns.len();
    DMatrix::from_vec(rows, k, columns.into_iter().flatten().collect())
}

fn keep_rows(x: DMatrix<f64>, y: Vec<f64>, names: Vec<String>, center: Vec<usize>, keep: &[bool]) -> Rows {
    let y = DVector::from_vec(y);
    let (x, y) = num::select_rows(&x, &y, keep);
    let center = center.into_iter().zip(keep).filter_map(|(c, &k)| k.then_some(c)).collect();
    Rows { x, y, names, center }
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Which rows: the excited ones, or every usable one; and the window.
#[derive(Clone, Copy, Debug)]
pub struct Select {
    pub gated: bool,
    pub window: Option<usize>,
}

impl Default for Select {
    fn default() -> Self {
        Select { gated: true, window: None }
    }
}

/// Pitch/roll rows with the two motor pairs as separate regressors:
/// [rate, Σω²(positive pair), Σω²(negative pair)], then with `drag` the
/// low-passed body x/y specific force.
pub fn split_rows(flight: &Flight, axis: Axis, drag: bool, sel: Select) -> Rows {
    let n = sel.window.unwrap_or_else(|| window(flight, axis));
    let rate = &flight.rates[axis.gyro()];
    let (pos, neg) = match axis {
        Axis::Pitch => (sum_sq(flight, &[0, 2], &[]), sum_sq(flight, &[1, 3], &[])),
        _ => (sum_sq(flight, &[2, 3], &[]), sum_sq(flight, &[0, 1], &[])),
    };
    let mut cols: Vec<Vec<f64>> =
        vec![rates(trunc(rate, n)).collect(), speeds_sq(trunc(&pos, n)).collect(), speeds_sq(trunc(&neg, n)).collect()];
    let mut names = vec![format!("{}Rate", axis.name()), "pos".into(), "neg".into()];
    if drag {
        let acc = flight.accel_lp(DRAG_LP).expect("drag terms need accSmooth");
        for (a, name) in acc.iter().zip(DRAG_TERMS) {
            cols.push(trunc(a, n).iter().map(|&v| value!(v, m / s ^ 2)).collect());
            names.push(name.into());
        }
    }
    let y: Vec<f64> = accels(&cfd(rate, &flight.t, n)).collect();
    let keep = gate(&flight.good, &cols[0], axis.threshold(), n, sel.gated);
    let center = (0..flight.len() - n).map(|k| k + n / 2 - 1).collect();
    keep_rows(matrix(cols), y, names, center, &keep)
}

/// main.R's thrust regressor 2·(d₊Σ₊ − d₋Σ₋)/(d₊ + d₋) from split columns.
pub fn arm_column(split: &DMatrix<f64>, d_pos: f64, d_neg: f64) -> DVector<f64> {
    (split.column(1) * d_pos - split.column(2) * d_neg) * (2.0 / (d_pos + d_neg))
}

/// Split rows with the thrust regressor at the given arms: [rate, thrust, drag…].
pub fn with_arms(split: &Rows, d_pos: f64, d_neg: f64) -> Rows {
    let k = split.x.ncols();
    let mut x = DMatrix::zeros(split.len(), k - 1);
    x.set_column(0, &split.x.column(0));
    x.set_column(1, &arm_column(&split.x, d_pos, d_neg));
    for j in 3..k {
        x.set_column(j - 1, &split.x.column(j));
    }
    let mut names = vec![split.names[0].clone(), "propVelSumSq".into()];
    names.extend(split.names[3..].iter().cloned());
    Rows { x, y: split.y.clone(), names, center: split.center.clone() }
}

/// An axis's regression rows. Rate axes keep only excited rows unless
/// `sel.gated` is false: then every usable one.
pub fn rows(flight: &Flight, axis: Axis, arms: &Arms, map: VoltsMap, drag: &[Axis], sel: Select) -> Rows {
    let n = sel.window.unwrap_or_else(|| window(flight, axis));
    let t = &flight.t;
    let w = &flight.speeds;
    match axis {
        Axis::Yaw => {
            let rate = &flight.rates[2];
            let torque = sum_sq(flight, &[2, 1], &[0, 3]);
            let (a0, a1, a2, a3) = (cfd(&w[0], t, n), cfd(&w[1], t, n), cfd(&w[2], t, n), cfd(&w[3], t, n));
            let accel: Vec<AngAccel> = (0..a0.len()).map(|k| -a0[k] + a2[k] + a1[k] - a3[k]).collect();
            let cols = vec![rates(trunc(rate, n)).collect(), speeds_sq(trunc(&torque, n)).collect::<Vec<_>>(), accels(&accel).collect()];
            let y: Vec<f64> = accels(&cfd(rate, t, n)).collect();
            let keep = gate(&flight.good, &cols[0], axis.threshold(), n, sel.gated);
            let center = (0..flight.len() - n).map(|k| k + n / 2 - 1).collect();
            keep_rows(matrix(cols), y, names(&["yawRate", "propVelSumSq", "propAccelSum"]), center, &keep)
        }
        Axis::Pitch | Axis::Roll => {
            let split = split_rows(flight, axis, drag.contains(&axis), sel);
            let (d_pos, d_neg) = arms.pinned(axis).expect("the arms are fitted before the rows");
            with_arms(&split, d_pos, d_neg)
        }
        Axis::Motors => {
            let volts = flight.volts(map);
            // main.R's motor order: FL, FR, BL, BR of its eRPM[0..3] naming.
            let order = [0, 2, 1, 3];
            let mut vel = vec![];
            let mut acc = vec![];
            let mut v = vec![];
            for &m in &order {
                vel.extend(rates(trunc(&w[m], n)));
                acc.extend(accels(&cfd(&w[m], t, n)));
                v.extend(trunc(&volts[m], n).iter().map(|&x| value!(x, V)));
            }
            let sq: Vec<f64> = vel.iter().map(|&x| x * x.abs()).collect();
            let ok1 = rows_ok(&flight.good, n);
            let ok: Vec<bool> = (0..4).flat_map(|_| ok1.iter().copied()).collect();
            let mut usable: Vec<f64> = vel.iter().zip(&ok).filter(|&(_, &o)| o).map(|(x, _)| x.abs()).collect();
            let scale = if usable.is_empty() { 0.0 } else { num::percentile_in_place(&mut usable, EXCITATION_PERCENTILE) };
            let keep: Vec<bool> = ok.iter().zip(&vel).map(|(&o, x)| o && x.abs() > scale * axis.threshold()).collect();
            let center1: Vec<usize> = (0..flight.len() - n).map(|k| k + n / 2 - 1).collect();
            let center = (0..4).flat_map(|_| center1.iter().copied()).collect();
            keep_rows(matrix(vec![vel, sq, acc]), v, names(&["motorVel", "motorVelSq", "motorAccel"]), center, &keep)
        }
    }
}

/// The three ways to split four motors into two pairs; for QUADX these are
/// the yaw (diagonals), pitch and roll groupings of the motor indices.
pub const PAIRINGS: [(Axis, [usize; 2], [usize; 2]); 3] =
    [(Axis::Yaw, [0, 3], [1, 2]), (Axis::Pitch, [0, 2], [1, 3]), (Axis::Roll, [2, 3], [0, 1])];

/// For each rate axis, R² of accel ~ rate + Σω²(pair A) + Σω²(pair B) under
/// each pairing. On a correctly indexed QUADX each axis is best explained by
/// its own grouping; a wrong layout shows up as a mismatch.
pub fn pairing_check(flight: &Flight) -> Vec<(Axis, Vec<(Axis, f64)>)> {
    let n = window(flight, Axis::Pitch);
    RATE_AXES
        .iter()
        .map(|&axis| {
            let rate = &flight.rates[axis.gyro()];
            let y = DVector::from_iterator(flight.len() - n, accels(&cfd(rate, &flight.t, n)));
            let r: Vec<f64> = rates(trunc(rate, n)).collect();
            let keep = gate(&flight.good, &r, 0.3, n, true);
            let scores = PAIRINGS
                .iter()
                .map(|&(name, a, b)| {
                    let pa = sum_sq(flight, &a, &[]);
                    let pb = sum_sq(flight, &b, &[]);
                    let x = matrix(vec![r.clone(), speeds_sq(trunc(&pa, n)).collect(), speeds_sq(trunc(&pb, n)).collect()]);
                    let (x, y) = num::select_rows(&x, &y, &keep);
                    (name, num::r2(&x, &y, &num::lstsq(&x, &y)))
                })
                .collect();
            (axis, scores)
        })
        .collect()
}

/// Rows of several flights, stacked.
pub fn stack(flights: &[Flight], axis: Axis, arms: &Arms, map: VoltsMap, drag: &[Axis]) -> Rows {
    let parts: Vec<Rows> = flights.iter().map(|f| rows(f, axis, arms, map, drag, Select::default())).collect();
    stack_rows(parts)
}

pub fn stack_rows(parts: Vec<Rows>) -> Rows {
    let names = parts.first().map(|p| p.names.clone()).unwrap_or_default();
    let x = num::vstack(&parts.iter().map(|p| p.x.clone()).collect::<Vec<_>>());
    let y = num::concat(&parts.iter().map(|p| p.y.clone()).collect::<Vec<_>>());
    let center = parts.into_iter().flat_map(|p| p.center).collect();
    Rows { x, y, names, center }
}
