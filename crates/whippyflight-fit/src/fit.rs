//! The plant fit: the lqr-prep/main.R regressions (same regressors and arm
//! weighting) over gated flights, with the motor model by output error and
//! the voltage-to-telemetry and voltage-to-airframe delays.
//!
//! The front/back and left/right arm ratios are fitted, not assumed: the
//! thrust regressor with a free ratio spans the same space as regressing the
//! two motor pairs separately, so that fit gives the R²-optimal ratio in
//! closed form. Given arms pin an axis instead.
//!
//! Airframe drag: tall frames carry rotor drag well above the CG, so airspeed
//! torques pitch and roll. The fit can add the body x/y accelerometer
//! (low-passed like the firmware) as pitch/roll regressors; auto keeps them on
//! an axis only if they raise its held-out R² by DRAG_MIN_GAIN to at least
//! DRAG_MIN_HELDOUT_R2. Held-out R² is pooled over folds that stay within one
//! flight: props and motors change between flights, and across flights the
//! drag terms absorb that instead.

use nalgebra::{DMatrix, DVector};
use whippyunits::{quantity, value};

use crate::flight::{Flight, VoltsMap};
use crate::json::J;
use crate::motors::{self, MotorModel};
use crate::num;
use crate::obj;
use crate::rows::{self, ARM_AXES, AXES, Arms, Axis, DRAG_LP, RATE_AXES, Rows, Select};
use crate::units::*;

/// The output's "motor_speed": the unit its motor speeds are in.
pub const MOTOR_SPEED_UNIT: &str = "shaft rev/s";
pub const MOTOR_NAMES: [&str; 3] = ["motorVel", "motorVelSq", "motorAccel"];

const DRAG_MIN_GAIN: f64 = 0.02;
const DRAG_MIN_HELDOUT_R2: f64 = 0.3;
const DRAG_FOLDS: usize = 4;
/// The longest voltage-to-telemetry and voltage-to-airframe delays tried.
const DELAY_SCAN: Time = quantity!(0.010, s);
const ARM_PROFILE: [f64; 5] = [0.8, 0.9, 1.0, 1.1, 1.25];

/// --drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragMode {
    Auto,
    On,
    Off,
}

impl DragMode {
    pub fn parse(s: &str) -> Result<DragMode, String> {
        match s {
            "auto" => Ok(DragMode::Auto),
            "on" => Ok(DragMode::On),
            "off" => Ok(DragMode::Off),
            _ => Err(format!("invalid drag mode: {s}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DragMode::Auto => "auto",
            DragMode::On => "on",
            DragMode::Off => "off",
        }
    }
}

/// Where a long step reports, and whether to stop.
pub struct Report<'a> {
    pub say: &'a dyn Fn(&str),
    pub check: &'a dyn Fn() -> Result<(), String>,
}

impl Report<'_> {
    pub fn quiet() -> Report<'static> {
        Report { say: &|_| {}, check: &|| Ok(()) }
    }
}

/// The physical arm a regressor-frame arm is (the frame is turned 180° about z).
fn physical(arm: &str) -> &'static str {
    match arm {
        "dFront" => "dBack",
        "dBack" => "dFront",
        "dLeft" => "dRight",
        _ => "dLeft",
    }
}

/// A coefficient's standard error relative to it; None where it couldn't be estimated.
pub fn rel_se(se: f64, value: f64) -> Option<f64> {
    let rel = se / value.abs();
    rel.is_finite().then(|| num::round_to(rel, 5))
}

fn percent(rel: Option<f64>) -> String {
    rel.map_or("?".into(), |r| format!("{:.2}%", r * 100.0))
}

/// Best-fit arm ratio d₋/d₊ of split rows, and its standard error by the
/// delta method: c₊Σ₊ + c₋Σ₋ gives d₋/d₊ = −c₋/c₊.
pub fn arm_ratio(x: &DMatrix<f64>, y: &DVector<f64>) -> (f64, f64) {
    let (beta, cov) = num::ols_cov(x, y);
    let (cp, cn) = (beta[1], beta[2]);
    let g = [cn / (cp * cp), -1.0 / cp];
    let var = g[0] * g[0] * cov[(1, 1)] + 2.0 * g[0] * g[1] * cov[(1, 2)] + g[1] * g[1] * cov[(2, 2)];
    (-cn / cp, var.sqrt())
}

/// Arm lengths not pinned in `arms` from the best-fit ratio; per-axis stats.
fn fit_arms(flights: &[Flight], arms: &Arms, drag: &[Axis], report: &Report) -> (Arms, J) {
    let mut arms = *arms;
    let mut stats = J::obj();
    for axis in ARM_AXES {
        if let Some((p, n)) = arms.pinned(axis) {
            stats.set(axis.name(), obj! {"ratio" => n / p, "source" => "given"});
            continue;
        }
        let split = rows::stack_rows(flights.iter().map(|f| rows::split_rows(f, axis, drag.contains(&axis), Select::default())).collect());
        let (ratio, se) = arm_ratio(&split.x, &split.y);
        arms.set(axis, 1.0, ratio);
        let profile: Vec<(f64, f64)> = ARM_PROFILE
            .iter()
            .map(|&scale| {
                let r = rows::with_arms(&split, 1.0, ratio * scale);
                (scale, num::r2(&r.x, &r.y, &num::lstsq(&r.x, &r.y)))
            })
            .collect();
        let mut prof = J::obj();
        for &(scale, r) in &profile {
            prof.set(&format!("{}x", json_g(scale)), num::round_to(r, 5));
        }
        stats.set(axis.name(), obj! {"ratio" => ratio, "se" => se, "source" => "fit", "r2_profile" => prof});
        let (pos, neg) = axis.arm_keys();
        let shown: Vec<String> = profile.iter().map(|(s, r)| format!("{}: {r:.4}", json_g(*s))).collect();
        (report.say)(&format!(
            "{:7} {}/{} = {ratio:.4} ± {se:.4}   R² at ratio × {}",
            axis.name(),
            physical(neg),
            physical(pos),
            shown.join("  ")
        ));
    }
    (arms, stats)
}

/// Python's `{:g}` for the profile's scales.
fn json_g(x: f64) -> String {
    let s = format!("{x}");
    s.strip_suffix(".0").map_or(s.clone(), str::to_string)
}

/// --drag resolved to the axes that get drag terms, with auto's evidence:
/// per axis, the held-out R² with and without the terms, pooled over folds
/// that each hold out one of DRAG_FOLDS contiguous, equal-row blocks of one
/// flight's rows (excitation is bursty: time quarters can leave a fold
/// training on a few dozen rows).
pub fn choose_drag(flights: &[Flight], mode: DragMode, arms: &Arms) -> Result<(Vec<Axis>, Option<J>), String> {
    if mode == DragMode::Off {
        return Ok((vec![], None));
    }
    if !flights.iter().all(Flight::has_accel) {
        if mode == DragMode::On {
            return Err("--drag on needs accSmooth in every log".into());
        }
        return Ok((vec![], Some(obj! {"reason" => "no accSmooth"})));
    }
    if mode == DragMode::On {
        return Ok((ARM_AXES.to_vec(), None));
    }
    let mut sse = [[0.0f64; 2]; 2];
    let mut sst = [0.0f64; 2];
    for f in flights {
        for (a, &axis) in ARM_AXES.iter().enumerate() {
            let split = rows::split_rows(f, axis, true, Select::default());
            let n = split.len();
            let fold: Vec<usize> = (0..n).map(|i| i * DRAG_FOLDS / n.max(1)).collect();
            for k in 0..DRAG_FOLDS {
                let held: Vec<bool> = fold.iter().map(|&x| x == k).collect();
                if !held.iter().any(|&h| h) || held.iter().all(|&h| h) {
                    continue;
                }
                let train: Vec<bool> = held.iter().map(|h| !h).collect();
                let yh = num::select_rows(&split.x, &split.y, &held).1;
                let m = yh.mean();
                sst[a] += yh.iter().map(|v| (v - m) * (v - m)).sum::<f64>();
                for on in [false, true] {
                    let s = if on { split.clone() } else { columns(&split, 3) };
                    let (d_pos, d_neg) = match arms.pinned(axis) {
                        Some(p) => p,
                        None => {
                            let (xt, yt) = num::select_rows(&s.x, &s.y, &train);
                            (1.0, arm_ratio(&xt, &yt).0)
                        }
                    };
                    let r = rows::with_arms(&s, d_pos, d_neg);
                    let (xt, yt) = num::select_rows(&r.x, &r.y, &train);
                    let beta = num::lstsq(&xt, &yt);
                    let (xh, yh) = num::select_rows(&r.x, &r.y, &held);
                    let resid = yh - xh * beta;
                    sse[a][usize::from(on)] += resid.dot(&resid);
                }
            }
        }
    }
    let heldout = |a: usize, on: usize| if sst[a] != 0.0 { 1.0 - sse[a][on] / sst[a] } else { 0.0 };
    let axes: Vec<Axis> = (0..2)
        .filter(|&a| heldout(a, 1) - heldout(a, 0) > DRAG_MIN_GAIN && heldout(a, 1) >= DRAG_MIN_HELDOUT_R2)
        .map(|a| ARM_AXES[a])
        .collect();
    let mut r2 = J::obj();
    for (a, axis) in ARM_AXES.iter().enumerate() {
        r2.set(axis.name(), obj! {"off" => num::round_to(heldout(a, 0), 4), "on" => num::round_to(heldout(a, 1), 4)});
    }
    let folds = format!("{DRAG_FOLDS} equal-row blocks per flight × {}", flights.len());
    Ok((axes, Some(obj! {"heldout_r2" => r2, "folds" => folds})))
}

/// The first `k` columns of split rows.
fn columns(r: &Rows, k: usize) -> Rows {
    Rows { x: r.x.columns(0, k).into_owned(), y: r.y.clone(), names: r.names[..k].to_vec(), center: r.center.clone() }
}

/// A fit: its JSON in the regressors' frame (without the airframe), and
/// what the diagnostics need.
pub struct Fitted {
    pub json: J,
    pub arms: Arms,
    pub drag: Vec<Axis>,
    pub map: VoltsMap,
    pub motors: MotorModel,
    /// Voltage-to-telemetry and voltage-to-airframe delays.
    pub telemetry_delay: Time,
    pub actuator_delay: Time,
    /// Per axis, the regressors' names and coefficients.
    pub coefficients: Vec<(Axis, Vec<(String, f64)>)>,
}

impl Fitted {
    pub fn beta(&self, axis: Axis) -> (Vec<String>, DVector<f64>) {
        let (_, c) = self.coefficients.iter().find(|(a, _)| *a == axis).expect("every axis is fitted");
        (c.iter().map(|(n, _)| n.clone()).collect(), DVector::from_iterator(c.len(), c.iter().map(|(_, v)| *v)))
    }
}

/// The gyro's low-pass stages as the sim models them, at a flight's rate:
/// each stage's PT1 gains.
fn gyro_gains(flight: &Flight) -> Result<Vec<f64>, String> {
    let stages: Vec<(String, Freq)> = crate::airframe::gyro_stages(&flight.header)
        .into_iter()
        .map(|(k, hz)| (k, quantity!(hz, Hz)))
        .collect();
    let cutoffs = if stages.is_empty() && flight.header.is_empty() {
        vec![quantity!(275.0, Hz), quantity!(550.0, Hz)]
    } else {
        whippyflight_sim::margin::stage_cutoffs(&stages, "gyro")?
    };
    Ok(cutoffs.into_iter().map(|c| whippyflight_design::plant::pt1_gain(c, 1.0 / flight.log_hz)).collect())
}

/// `x` through PT1 stages of the given gains.
fn filtered(x: &[Rate], gains: &[f64]) -> Vec<Rate> {
    let mut v: Vec<f64> = x.iter().map(|&w| value!(w, rot / s)).collect();
    for &k in gains {
        v = num::one_pole(&v, k);
    }
    v.into_iter().map(|w| quantity!(w, rot / s)).collect()
}

/// The voltage-to-airframe delay: the one whose motor-model speeds, from the
/// voltages that late and through the gyro's low-pass stages as the sim
/// models them, the rate axes' regressions fit best (summed 1 − R²), over
/// whole log samples up to DELAY_SCAN. (delay, at the scan's edge)
fn actuator_delay(flights: &[Flight], model: &MotorModel, arms: &Arms, map: VoltsMap, drag: &[Axis], report: &Report) -> Result<(Time, bool), String> {
    let gains: Vec<Vec<f64>> = flights.iter().map(gyro_gains).collect::<Result<_, _>>()?;
    let hz = flights.iter().map(|f| f.log_hz).fold(quantity!(0.0, Hz), |a, b| a.max(b));
    let max = num::round_even(f64::from(DELAY_SCAN * hz)) as usize;
    let (costs, best) = motors::scan(max, |n| {
        (report.check)()?;
        let late: Vec<Flight> = flights
            .iter()
            .zip(&gains)
            .map(|(f, g)| {
                let speeds = model.speeds(f, map, n as f64 / f.log_hz);
                f.with_speeds(speeds.map(|w| filtered(&w, g)))
            })
            .collect();
        Ok(RATE_AXES
            .iter()
            .map(|&axis| {
                let r = rows::stack(&late, axis, arms, map, drag);
                1.0 - num::ols(&r.x, &r.y).2
            })
            .sum())
    })?;
    Ok(((best as f64 + motors::parabola_min(&costs, best)) / hz, best == costs.len() - 1))
}

fn round_ms(t: Time) -> f64 {
    num::round_to(value!(t, ms), 2)
}

/// The pooled fit over `flights` in the regressors' frame (no airframe).
pub fn fit(flights: &[Flight], given: &Arms, map: VoltsMap, drag: &[Axis], report: &Report) -> Result<Fitted, String> {
    let (arms, arm_stats) = fit_arms(flights, given, drag, report);
    (report.check)()?;
    let mut out = obj! {"motor_speed" => MOTOR_SPEED_UNIT};
    let mut fit_stats = J::obj();
    let mut coefficients = vec![];
    let mut motor_model = None;
    let (mut telemetry, mut actuator) = (quantity!(0.0, s), quantity!(0.0, s));
    for axis in AXES {
        if axis == Axis::Motors {
            let r = rows::stack(flights, axis, &arms, map, drag);
            let b = num::lstsq(&r.x, &r.y);
            let m = motors::fit(flights, map, [b[0], b[1], b[2]], DELAY_SCAN, report.check)?;
            let (airframe, airframe_edge) = actuator_delay(flights, &m.model, &arms, map, drag, report)?;
            let k = m.model.values();
            let mut block = J::obj();
            let mut rel = J::obj();
            for (i, name) in MOTOR_NAMES.iter().enumerate() {
                block.set(name, k[i]);
                rel.set(name, rel_se(m.se[i], k[i]));
            }
            out.set("motors", block);
            out.set("latency", obj! {
                "actuator_ms" => round_ms(airframe),
                "telemetry_ms" => num::round_to(value!(m.delay - airframe, ms).max(0.0), 2),
                "voltage_to_telemetry_ms" => round_ms(m.delay),
            });
            fit_stats.set("motors", obj! {"r2" => num::round_to(m.r2, 4), "rows" => m.rows, "method" => "output error", "rel_se" => rel});
            if m.at_edge || airframe_edge {
                (report.say)("warning: a delay is at the edge of its scan; the latency is unreliable");
            }
            if airframe > m.delay {
                (report.say)(&format!(
                    "warning: the airframe answers {:.2} ms after the voltages, later than the telemetry's {:.2} ms; the gyro path has delay the model lacks, given to the motors",
                    value!(airframe, ms),
                    value!(m.delay, ms)
                ));
            }
            let shown: Vec<String> = MOTOR_NAMES
                .iter()
                .enumerate()
                .map(|(i, n)| format!("{n}={}±{}", g6(k[i]), percent(rel_se(m.se[i], k[i]))))
                .collect();
            (report.say)(&format!("{:7} R²={:.4} n={:7}  {}  (output error, speeds)", "motors", m.r2, m.rows, shown.join("  ")));
            coefficients.push((axis, MOTOR_NAMES.iter().zip(k).map(|(n, v)| (n.to_string(), v)).collect()));
            motor_model = Some(m.model);
            (telemetry, actuator) = (m.delay, airframe);
            continue;
        }
        (report.check)()?;
        let r = rows::stack(flights, axis, &arms, map, drag);
        let (beta, se, r2) = num::ols(&r.x, &r.y);
        let mut block = J::obj();
        let mut rel = J::obj();
        for (i, name) in r.names.iter().enumerate() {
            block.set(name, beta[i]);
            rel.set(name, rel_se(se[i], beta[i]));
        }
        out.set(axis.name(), block);
        fit_stats.set(axis.name(), obj! {"r2" => num::round_to(r2, 4), "rows" => r.len(), "rel_se" => rel});
        let shown: Vec<String> =
            r.names.iter().enumerate().map(|(i, n)| format!("{n}={}±{}", g6(beta[i]), percent(rel_se(se[i], beta[i])))).collect();
        (report.say)(&format!("{:7} R²={r2:.4} n={:7}  {}", axis.name(), r.len(), shown.join("  ")));
        coefficients.push((axis, r.names.iter().cloned().zip(beta.iter().copied()).collect()));
    }
    (report.say)(&format!(
        "latency: voltage to speed telemetry {:.2} ms (motors {:.2} ms, telemetry {:.2} ms)",
        round_ms(telemetry),
        round_ms(actuator),
        num::round_to(value!(telemetry - actuator, ms).max(0.0), 2)
    ));
    let (dfront, dback) = arms.pinned(Axis::Pitch).unwrap();
    let (dright, dleft) = arms.pinned(Axis::Roll).unwrap();
    let pitch = out.get_mut("pitch").unwrap();
    pitch.set("dFront", dfront);
    pitch.set("dBack", dback);
    let roll = out.get_mut("roll").unwrap();
    roll.set("dLeft", dleft);
    roll.set("dRight", dright);
    out.set("drag", obj! {"axes" => drag.iter().map(|a| a.name()).collect::<Vec<_>>(), "lp_hz" => value!(DRAG_LP, Hz)});
    let mut fit_block = obj! {
        "sources" => flights.iter().map(|f| f.label.clone()).collect::<Vec<_>>(),
        "volts" => map.name(),
        "select" => "robust",
        "arms" => arm_stats,
    };
    if let J::Obj(pairs) = fit_stats {
        for (k, v) in pairs {
            fit_block.set(&k, v);
        }
    }
    out.set("fit", fit_block);
    Ok(Fitted {
        json: out,
        arms,
        drag: drag.to_vec(),
        map,
        motors: motor_model.expect("the motors are fitted"),
        telemetry_delay: telemetry,
        actuator_delay: actuator,
        coefficients,
    })
}

/// Python's `{:.6g}`.
pub fn g6(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{:.5e}", x);
        let (m, e) = s.split_once('e').unwrap();
        let m = if m.contains('.') { m.trim_end_matches('0').trim_end_matches('.') } else { m };
        let e: i32 = e.parse().unwrap();
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let s = format!("{:.*}", (5 - exp).max(0) as usize, x);
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// A fit's regression-frame coefficients read back from a file (see
/// `frame::regressor_frame`): what the diagnostics, validation and noise
/// extraction evaluate.
pub struct Stored {
    pub json: J,
    pub arms: Arms,
    pub drag: Vec<Axis>,
    pub map: VoltsMap,
    pub motors: MotorModel,
    pub telemetry_delay: Time,
    pub actuator_delay: Time,
}

impl Stored {
    pub fn from_regressor_frame(c: &J) -> Result<Stored, String> {
        let num = |axis: &str, key: &str| c.get(axis).and_then(|b| b.get(key)).and_then(J::as_f64);
        let need = |axis: &str, key: &str| num(axis, key).ok_or_else(|| format!("the fit has no {axis}.{key}"));
        let arms = Arms {
            d_front: Some(need("pitch", "dFront")?),
            d_back: Some(need("pitch", "dBack")?),
            d_left: Some(need("roll", "dLeft")?),
            d_right: Some(need("roll", "dRight")?),
        };
        let drag = match c.get("drag").and_then(|d| d.get("axes")) {
            Some(J::Arr(a)) => a.iter().filter_map(|x| if let J::Str(s) = x { Axis::parse(s) } else { None }).collect(),
            _ => vec![],
        };
        let map = match c.get("fit").and_then(|f| f.get("volts")) {
            Some(J::Str(v)) => VoltsMap::parse(v)?,
            _ => VoltsMap::R,
        };
        let motors = MotorModel::from_values([need("motors", "motorVel")?, need("motors", "motorVelSq")?, need("motors", "motorAccel")?]);
        let ms = |key: &str| quantity!(1e-3 * num("latency", key).unwrap_or(0.0), s);
        Ok(Stored {
            json: c.clone(),
            arms,
            drag,
            map,
            motors,
            telemetry_delay: ms("voltage_to_telemetry_ms"),
            actuator_delay: ms("actuator_ms"),
        })
    }

    /// An axis's coefficients in the order of its rows' regressors.
    pub fn beta(&self, axis: Axis, names: &[String]) -> Result<DVector<f64>, String> {
        let block = self.json.get(axis.name()).ok_or_else(|| format!("the fit has no {}", axis.name()))?;
        let v: Vec<f64> = names
            .iter()
            .map(|n| block.get(n).and_then(J::as_f64).ok_or_else(|| format!("the fit has no {}.{n}", axis.name())))
            .collect::<Result<_, _>>()?;
        Ok(DVector::from_vec(v))
    }
}

impl From<&Fitted> for Stored {
    fn from(f: &Fitted) -> Stored {
        Stored {
            json: f.json.clone(),
            arms: f.arms,
            drag: f.drag.clone(),
            map: f.map,
            motors: f.motors,
            telemetry_delay: f.telemetry_delay,
            actuator_delay: f.actuator_delay,
        }
    }
}

/// Per-axis R² of a fit on one flight's rows (the motors' voltages at the
/// fit's delay).
pub fn evaluate(fit: &Stored, flight: &Flight) -> Result<Vec<(Axis, f64)>, String> {
    let lagged = flight.with_voltage_delay(fit.telemetry_delay);
    AXES.iter()
        .map(|&axis| {
            let f = if axis == Axis::Motors { &lagged } else { flight };
            let r = rows::rows(f, axis, &fit.arms, fit.map, &fit.drag, Select::default());
            let beta = fit.beta(axis, &r.names)?;
            Ok((axis, num::r2(&r.x, &r.y, &beta)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_g_formatting() {
        assert_eq!(g6(-1.512245366558469), "-1.51225");
        assert_eq!(g6(1.7525621493441606e-05), "1.75256e-05");
        assert_eq!(g6(0.0018356674400452904), "0.00183567");
        assert_eq!(g6(0.003907112490702572), "0.00390711");
        assert_eq!(g6(445.5), "445.5");
        assert_eq!(json_g(0.8), "0.8");
        assert_eq!(json_g(1.0), "1");
    }
}
