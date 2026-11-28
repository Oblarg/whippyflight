//! The motor model V = motorAccel·ω̇ + motorVel·ω + motorVelSq·ω|ω|, fitted
//! by output error: the coefficients whose simulated speeds, from the
//! voltages some whole samples late, best match the telemetry. Regressing
//! voltage on a differenced speed biases motorAccel low (the noisy
//! derivative is a regressor); simulating differences nothing.

use nalgebra::{Matrix3, Vector3};
use whippyunits::{quantity, value};

use crate::flight::{Flight, VoltsMap};
use crate::units::*;

/// A fitted motor model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorModel {
    pub vel: MotorVel,
    pub vel_sq: MotorVelSq,
    pub accel: MotorAccel,
}

impl MotorModel {
    /// From (motorVel V·s/rot, motorVelSq V·s²/rot², motorAccel V·s²/rot).
    pub fn from_values(k: [f64; 3]) -> MotorModel {
        MotorModel {
            vel: quantity!(k[0], V * s / rot),
            vel_sq: quantity!(k[1], V * s ^ 2 / rot ^ 2),
            accel: quantity!(k[2], V * s ^ 2 / rot),
        }
    }

    pub fn values(&self) -> [f64; 3] {
        [value!(self.vel, V * s / rot), value!(self.vel_sq, V * s ^ 2 / rot ^ 2), value!(self.accel, V * s ^ 2 / rot)]
    }

    /// The motor speeds this model reaches from `flight`'s voltages `delay`
    /// late, restarting from the telemetry wherever a usable stretch begins:
    /// the speeds without telemetry noise.
    pub fn speeds(&self, flight: &Flight, map: VoltsMap, delay: Time) -> [Vec<Rate>; 4] {
        let run = Run::new(flight, map);
        let k = self.values();
        let n = flight.samples(delay);
        core::array::from_fn(|m| run.simulate(m, k, n, None).into_iter().map(|w| quantity!(w, rot / s)).collect())
    }
}

/// A flight's voltages and speed telemetry for simulating its motors.
/// Integration is in V, rot/s and s.
pub struct Run {
    volts: [Vec<f64>; 4],
    measured: [Vec<f64>; 4],
    good: Vec<bool>,
    restart: Vec<bool>,
    dt: Vec<f64>,
}

impl Run {
    pub fn new(flight: &Flight, map: VoltsMap) -> Run {
        let volts = flight.volts(map).map(|v| v.iter().map(|&x| value!(x, V)).collect());
        let measured = flight.speeds.clone().map(|w| w.iter().map(|&x| value!(x, rot / s)).collect());
        let good = flight.good.clone();
        let restart = (0..good.len()).map(|i| i == 0 || (good[i] && !good[i - 1])).collect();
        let dt = (0..flight.len()).map(|i| if i == 0 { 0.0 } else { value!(flight.t[i] - flight.t[i - 1], s) }).collect();
        Run { volts, measured, good, restart, dt }
    }

    /// Motor `m`'s Euler steps from the voltages `delay` samples late (the
    /// first held). With `sens`, also each step's derivative with respect to
    /// (motorVel, motorVelSq, ln motorAccel).
    fn simulate(&self, m: usize, k: [f64; 3], delay: usize, mut sens: Option<&mut Vec<[f64; 3]>>) -> Vec<f64> {
        let [kv, kq, ka] = k;
        let v = &self.volts[m];
        let meas = &self.measured[m];
        let n = meas.len();
        let mut out = Vec::with_capacity(n);
        let mut w = meas[0];
        let mut s = [0.0f64; 3];
        if let Some(sv) = sens.as_deref_mut() {
            sv.clear();
            sv.reserve(n);
        }
        for i in 0..n {
            if self.restart[i] {
                w = meas[i];
                s = [0.0; 3];
            } else {
                let volts = v[(i - 1).saturating_sub(delay)];
                let f = volts - kv * w - kq * w * w.abs();
                let h = self.dt[i] / ka;
                if sens.is_some() {
                    let dfdw = -kv - 2.0 * kq * w.abs();
                    s = [
                        s[0] + h * (dfdw * s[0] - w),
                        s[1] + h * (dfdw * s[1] - w * w.abs()),
                        s[2] + h * (dfdw * s[2]) - f * h,
                    ];
                }
                w += f * h;
            }
            out.push(w);
            if let Some(sv) = sens.as_deref_mut() {
                sv.push(s);
            }
        }
        out
    }

    fn rows(&self) -> usize {
        4 * self.good.iter().filter(|&&g| g).count()
    }
}

struct Normal {
    sse: f64,
    jtj: Matrix3<f64>,
    jtr: Vector3<f64>,
}

/// Over every run's usable rows: Σ r², and with `jacobian` JᵀJ and Jᵀr in
/// q = (motorVel, motorVelSq, ln motorAccel).
fn normal(runs: &[Run], q: Vector3<f64>, delay: usize, jacobian: bool) -> Normal {
    let k = [q[0], q[1], q[2].exp()];
    let mut out = Normal { sse: 0.0, jtj: Matrix3::zeros(), jtr: Vector3::zeros() };
    let mut sens = vec![];
    for run in runs {
        for m in 0..4 {
            let sim = run.simulate(m, k, delay, jacobian.then_some(&mut sens));
            for i in (0..sim.len()).filter(|&i| run.good[i]) {
                let r = sim[i] - run.measured[m][i];
                out.sse += r * r;
                if jacobian {
                    let j = Vector3::from(sens[i]);
                    out.jtj += j * j.transpose();
                    out.jtr += j * r;
                }
            }
        }
    }
    if !out.sse.is_finite() {
        out.sse = f64::INFINITY;
    }
    out
}

/// Levenberg–Marquardt from `q` for the voltages `delay` samples late:
/// (q, Σ r², JᵀJ at q).
fn solve(runs: &[Run], mut q: Vector3<f64>, delay: usize) -> (Vector3<f64>, f64, Matrix3<f64>) {
    const MAX_ITER: usize = 200;
    const FTOL: f64 = 1e-12;
    let mut lambda = 1e-3;
    let mut at = normal(runs, q, delay, true);
    for _ in 0..MAX_ITER {
        let d = Matrix3::from_diagonal(&at.jtj.diagonal().map(|x| x.max(f64::MIN_POSITIVE)));
        let mut improved = false;
        for _ in 0..30 {
            let Some(step) = (at.jtj + d * lambda).cholesky().map(|c| c.solve(&-at.jtr)) else {
                lambda *= 10.0;
                continue;
            };
            let trial = normal(runs, q + step, delay, false);
            if trial.sse < at.sse {
                let gain = (at.sse - trial.sse) / at.sse;
                q += step;
                at = normal(runs, q, delay, true);
                lambda = (lambda / 3.0).max(1e-12);
                improved = gain > FTOL;
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    (q, at.sse, at.jtj)
}

/// Sub-sample offset of the minimum of `cost` (by whole sample) around `best`.
pub fn parabola_min(cost: &[f64], best: usize) -> f64 {
    if best >= 1 && best + 1 < cost.len() {
        let (a, b, c) = (cost[best - 1], cost[best], cost[best + 1]);
        if a - 2.0 * b + c > 0.0 {
            return 0.5 * (a - c) / (a - 2.0 * b + c);
        }
    }
    0.0
}

/// Scan whole-sample delays 0, 1, … up to `max`, stopping two past the
/// cheapest so far: (costs by delay, the cheapest).
pub fn scan(max: usize, mut cost: impl FnMut(usize) -> Result<f64, String>) -> Result<(Vec<f64>, usize), String> {
    let mut costs = vec![];
    let mut best = 0;
    for n in 0..=max {
        costs.push(cost(n)?);
        best = (0..costs.len()).min_by(|&a, &b| costs[a].total_cmp(&costs[b])).unwrap();
        if n >= best + 2 {
            break;
        }
    }
    Ok((costs, best))
}

pub struct MotorFit {
    pub model: MotorModel,
    pub r2: f64,
    pub rows: usize,
    /// Standard errors of (motorVel, motorVelSq, motorAccel).
    pub se: [f64; 3],
    /// Voltage-to-telemetry delay.
    pub delay: Time,
    pub at_edge: bool,
}

/// The output-error fit from `start` (a regression's coefficients), over
/// whole-sample delays up to `max_delay`.
pub fn fit(
    flights: &[Flight],
    map: VoltsMap,
    start: [f64; 3],
    max_delay: Time,
    check: &dyn Fn() -> Result<(), String>,
) -> Result<MotorFit, String> {
    if start[2].is_nan() || start[2] <= 0.0 {
        return Err(format!("the motor regression's motorAccel {} isn't positive; the motors can't be fitted", start[2]));
    }
    let runs: Vec<Run> = flights.iter().map(|f| Run::new(f, map)).collect();
    let hz = flights.iter().map(|f| f.log_hz).fold(quantity!(0.0, Hz), |a, b| a.max(b));
    let max = crate::num::round_even(f64::from(max_delay * hz)) as usize;
    let mut q = Vector3::new(start[0], start[1], start[2].ln());
    let mut fits = vec![];
    let (costs, best) = scan(max, |n| {
        check()?;
        let (sol, sse, jtj) = solve(&runs, q, n);
        q = sol;
        fits.push((sol, jtj));
        Ok(sse)
    })?;
    let (sol, jtj) = fits[best];
    let rows: usize = runs.iter().map(Run::rows).sum();
    let k = [sol[0], sol[1], sol[2].exp()];
    let s2 = costs[best] / (rows as f64 - 3.0).max(1.0);
    let cov = jtj.try_inverse().map(|inv| inv * s2);
    let se = match cov {
        Some(c) => [c[(0, 0)].sqrt(), c[(1, 1)].sqrt(), c[(2, 2)].sqrt() * k[2]],
        None => [f64::NAN; 3],
    };
    let (mut sum, mut count) = (0.0, 0usize);
    for run in &runs {
        for m in 0..4 {
            for i in (0..run.good.len()).filter(|&i| run.good[i]) {
                sum += run.measured[m][i];
                count += 1;
            }
        }
    }
    let mean = sum / count as f64;
    let mut sst = 0.0;
    for run in &runs {
        for m in 0..4 {
            for i in (0..run.good.len()).filter(|&i| run.good[i]) {
                sst += (run.measured[m][i] - mean).powi(2);
            }
        }
    }
    let shift = parabola_min(&costs, best);
    Ok(MotorFit {
        model: MotorModel::from_values(k),
        r2: 1.0 - costs[best] / sst,
        rows,
        se,
        delay: (best as f64 + shift) / hz,
        at_edge: best == costs.len() - 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parabola_vertex() {
        assert!((parabola_min(&[4.0, 1.0, 0.0, 1.0], 2) - 0.0).abs() < 1e-12);
        assert!((parabola_min(&[2.0, 1.0, 2.0 + 1.0 / 3.0], 1) - (-0.5 * (1.0 / 3.0) / (2.0 + 1.0 / 3.0))).abs() < 1e-12);
        assert_eq!(parabola_min(&[1.0, 2.0], 0), 0.0);
    }

    #[test]
    fn scan_stops_two_past_the_minimum() {
        let costs = [5.0, 3.0, 4.0, 6.0, 1.0];
        let (c, best) = scan(4, |n| Ok(costs[n])).unwrap();
        assert_eq!((c.len(), best), (4, 1));
    }

    /// Sensitivities against finite differences of the simulation.
    #[test]
    fn sensitivities_match_finite_differences() {
        let n = 400;
        let dt = 1.0 / 1600.0;
        let volts: Vec<f64> = (0..n).map(|i| 2.0 + 0.8 * (i as f64 * 0.05).sin()).collect();
        let run = Run {
            volts: core::array::from_fn(|_| volts.clone()),
            measured: core::array::from_fn(|_| vec![300.0; n]),
            good: vec![true; n],
            restart: (0..n).map(|i| i == 0).collect(),
            dt: (0..n).map(|i| if i == 0 { 0.0 } else { dt }).collect(),
        };
        let q = [0.0039, 1.1e-6, 0.000_22f64.ln()];
        let k = |q: [f64; 3]| [q[0], q[1], q[2].exp()];
        let mut sens = vec![];
        let base = run.simulate(0, k(q), 1, Some(&mut sens));
        for p in 0..3 {
            let h = 1e-6 * q[p].abs().max(1e-9);
            let mut qp = q;
            qp[p] += h;
            let up = run.simulate(0, k(qp), 1, None);
            let fd = (up[n - 1] - base[n - 1]) / h;
            let an = sens[n - 1][p];
            assert!((fd - an).abs() <= 1e-4 * an.abs().max(1e-9), "param {p}: fd {fd} analytic {an}");
        }
    }
}
