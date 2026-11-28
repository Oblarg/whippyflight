//! One armed flight's signals in model units, and which samples a fit may use.
//!
//! Rates are in rot/s in Betaflight's FLU body axes, motor speeds in shaft
//! rev/s (the eRPM telemetry over motor_poles / 2, as the firmware reads it)
//! by Betaflight motor index, volts the motor command × vbat.
//!
//! Gating works on uncurated flights: ground time is cut where motor speed
//! sits near idle (takeoff, landing, sitting armed after a crash), and
//! impacts are masked from the accelerometer and gyro saturation, padded.

use std::collections::HashMap;
use std::path::Path;

use whippyunits::{quantity, value};

use crate::bbl::{self, Log};
use crate::num;
use crate::units::*;

pub const MIN_DURATION: Time = quantity!(10.0, s);
const GROUND_FRACTION: f64 = 0.3;
const GROUND_MIN: Time = quantity!(0.5, s);
const IMPACT_G: f64 = 8.0;
const GYRO_SATURATION_DPS: f64 = 1950.0;
const IMPACT_PAD: (Time, Time) = (quantity!(0.25, s), quantity!(1.0, s));
const MIN_MOTOR_ERPM_CORR: f64 = 0.6;
const DEFAULT_ACC_1G: f64 = 2048.0;

const DSHOT_MIN: f64 = 48.0;
const DSHOT_MAX: f64 = 2047.0;

/// How the motor command maps to a fraction of the battery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoltsMap {
    /// motor / 2047.
    R,
    /// (motor − 48) / 1999, clamped: as the glue drives it.
    Glue,
}

impl VoltsMap {
    pub fn name(self) -> &'static str {
        match self {
            VoltsMap::R => "r",
            VoltsMap::Glue => "glue",
        }
    }

    pub fn parse(s: &str) -> Result<VoltsMap, String> {
        match s {
            "r" => Ok(VoltsMap::R),
            "glue" => Ok(VoltsMap::Glue),
            _ => Err(format!("unknown volts mapping {s}")),
        }
    }
}

/// What gating cut.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gating {
    pub ground: Time,
    pub impacts: usize,
    pub impact: Time,
}

#[derive(Clone)]
pub struct Flight {
    pub label: String,
    pub header: HashMap<String, String>,
    pub t: Vec<Time>,
    pub log_hz: Freq,
    /// Body rates, gyroADC (FLU).
    pub rates: [Vec<Rate>; 3],
    /// Stick setpoints (FLU), where logged.
    pub setpoint: Option<[Vec<Rate>; 3]>,
    /// Shaft speeds by Betaflight motor index.
    pub speeds: [Vec<Rate>; 4],
    /// Motor commands (DShot values) by motor index.
    pub commands: [Vec<f64>; 4],
    pub vbat: Vec<Voltage>,
    /// Body specific force (accSmooth, FLU), where logged.
    pub accel: Option<[Vec<Accel>; 3]>,
    pub motor_poles: u32,
    pub good: Vec<bool>,
    pub gating: Gating,
    /// Voltages are this many samples late (see [`Flight::with_voltage_delay`]).
    volts_delay: usize,
}

/// `[f(0), f(1), …]`.
fn each<const N: usize, T>(f: impl FnMut(usize) -> Result<T, String>) -> Result<[T; N], String> {
    let v: Vec<T> = (0..N).map(f).collect::<Result<_, _>>()?;
    v.try_into().map_err(|_| "length".to_string())
}

/// BF's eRPM/100 telemetry (the eRPM[i] log field) as shaft speed on a `poles`-pole motor.
pub fn shaft_speed(erpm: f64, poles: u32) -> Rate {
    quantity!(erpm * (100.0 / 60.0) / (poles as f64 / 2.0), rot / s)
}

impl Flight {
    /// A log's flight, gated. `motor_poles` overrides the log's.
    pub fn from_log(label: String, log: &Log, motor_poles: Option<u32>) -> Result<Flight, String> {
        let poles = motor_poles.unwrap_or_else(|| log.header_int("motor_poles").unwrap_or(0).max(0) as u32);
        if poles < 2 {
            return Err(format!("{label}: the log doesn't record motor_poles; pass --motor-poles"));
        }
        let time = log.require("time")?;
        let t: Vec<Time> = time.iter().map(|&us| quantity!((us - time[0]) * 1e-6, s)).collect();
        let span = value!(t[t.len() - 1] - t[0], s);
        let log_hz = quantity!((t.len() as f64 - 1.0) / span, Hz);
        let rates = each(|i| Ok(log.require(&format!("gyroADC[{i}]"))?.iter().map(|&d| quantity!(d / 360.0, rot / s)).collect()))?;
        let setpoint = (0..3).all(|i| log.has(&format!("setpoint[{i}]"))).then(|| {
            each(|i| Ok(log.require(&format!("setpoint[{i}]"))?.iter().map(|&d| quantity!(d / 360.0, rot / s)).collect()))
        });
        let setpoint = setpoint.transpose()?;
        let speeds = each(|i| Ok(log.require(&format!("eRPM[{i}]"))?.iter().map(|&e| shaft_speed(e, poles)).collect()))?;
        let commands = each(|i| log.require(&format!("motor[{i}]")))?;
        let vbat = log.require("vbatLatest")?.iter().map(|&cv| quantity!(cv / 100.0, V)).collect();
        let one_g = log.header_int("acc_1G").map_or(DEFAULT_ACC_1G, |g| g as f64);
        let accel = (0..3).all(|i| log.has(&format!("accSmooth[{i}]"))).then(|| {
            each(|i| {
                Ok(log.require(&format!("accSmooth[{i}]"))?.iter().map(|&a| quantity!(a / one_g * STANDARD_GRAVITY, m / s ^ 2)).collect())
            })
        });
        let accel = accel.transpose()?;
        let mut flight = Flight {
            label,
            header: log.header.clone(),
            t,
            log_hz,
            rates,
            setpoint,
            speeds,
            commands,
            vbat,
            accel,
            motor_poles: poles,
            good: vec![],
            gating: Gating::default(),
            volts_delay: 0,
        };
        flight.gate();
        Ok(flight)
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn duration(&self) -> Time {
        self.t[self.len() - 1] - self.t[0]
    }

    /// `d` in whole log samples, rounded half to even.
    pub fn samples(&self, d: Time) -> usize {
        num::round_even(f64::from(d * self.log_hz)).max(0.0) as usize
    }

    pub fn kept(&self) -> Time {
        self.good.iter().filter(|&&g| g).count() as f64 / self.log_hz
    }

    pub fn has_accel(&self) -> bool {
        self.accel.is_some()
    }

    fn gate(&mut self) {
        let n = self.len();
        let secs: Vec<f64> = self.t.iter().map(|&t| value!(t, s)).collect();
        let mut good: Vec<bool> = (0..n).map(|k| self.speeds.iter().all(|w| w[k] > quantity!(0.0, rot / s))).collect();

        let speed: Vec<f64> = (0..n).map(|k| self.speeds.iter().map(|w| value!(w[k], rot / s)).sum::<f64>() / 4.0).collect();
        let (idle, typical) = (num::percentile(&speed, 2.0), num::median(&speed));
        let ground: Vec<bool> = speed.iter().map(|&w| w < idle + GROUND_FRACTION * (typical - idle)).collect();
        let mut ground_mask = vec![false; n];
        for (a, b) in runs(&ground) {
            if a == 0 || b == n || secs[b - 1] - secs[a] > value!(GROUND_MIN, s) {
                ground_mask[a..b].fill(true);
            }
        }

        let saturation = quantity!(GYRO_SATURATION_DPS / 360.0, rot / s);
        let mut impact: Vec<bool> = (0..n).map(|k| self.rates.iter().any(|r| r[k].abs() > saturation)).collect();
        if let Some(acc) = &self.accel {
            for (k, hit) in impact.iter_mut().enumerate() {
                let g = (acc.iter().map(|a| value!(a[k], m / s ^ 2).powi(2)).sum::<f64>()).sqrt() / STANDARD_GRAVITY;
                *hit |= g > IMPACT_G;
            }
        }
        let impact_mask = dilate(&impact, &secs, value!(IMPACT_PAD.0, s), value!(IMPACT_PAD.1, s));
        for k in 0..n {
            good[k] &= !ground_mask[k] && !impact_mask[k];
        }
        let count = |m: &[bool]| m.iter().filter(|&&x| x).count() as f64 / self.log_hz;
        self.gating = Gating { ground: count(&ground_mask), impacts: runs(&impact).len(), impact: count(&impact_mask) };
        self.good = good;
    }

    /// Battery fractions by motor index, `delay` samples late.
    pub fn duty(&self, map: VoltsMap) -> [Vec<f64>; 4] {
        core::array::from_fn(|i| {
            let m = &self.commands[i];
            let d = self.volts_delay;
            (0..m.len())
                .map(|k| {
                    let x = m[k.saturating_sub(d)];
                    match map {
                        VoltsMap::Glue => ((x - DSHOT_MIN) / (DSHOT_MAX - DSHOT_MIN)).clamp(0.0, 1.0),
                        VoltsMap::R => x / DSHOT_MAX,
                    }
                })
                .collect()
        })
    }

    /// Motor voltages by motor index.
    pub fn volts(&self, map: VoltsMap) -> [Vec<Voltage>; 4] {
        let d = self.volts_delay;
        self.duty(map).map(|duty| duty.iter().enumerate().map(|(k, &f)| f * self.vbat[k.saturating_sub(d)]).collect())
    }

    /// This flight with its voltages `delay` late (whole samples, the first
    /// held): what the telemetry answers.
    pub fn with_voltage_delay(&self, delay: Time) -> Flight {
        let mut out = self.clone();
        out.volts_delay = self.samples(delay);
        out
    }

    /// This flight with `speeds` for its motor speeds.
    pub fn with_speeds(&self, speeds: [Vec<Rate>; 4]) -> Flight {
        Flight { speeds, ..self.clone() }
    }

    /// Body x/y specific force, one-pole low-passed at `lp` (causal, as on the FC).
    pub fn accel_lp(&self, lp: Freq) -> Option<[Vec<Accel>; 2]> {
        let acc = self.accel.as_ref()?;
        let k = 1.0 - (-2.0 * core::f64::consts::PI * f64::from(lp / self.log_hz)).exp();
        Some(core::array::from_fn(|i| {
            let raw: Vec<f64> = acc[i].iter().map(|&a| value!(a, m / s ^ 2)).collect();
            num::one_pole(&raw, k).into_iter().map(|a| quantity!(a, m / s ^ 2)).collect()
        }))
    }

    /// Why this flight can't be fitted, or None.
    pub fn unusable(&self) -> Option<String> {
        let n = self.len();
        let zero = quantity!(0.0, rot / s);
        let spinning = (0..n).filter(|&k| self.speeds.iter().all(|w| w[k] > zero)).count();
        if (spinning as f64) < 0.5 * n as f64 {
            return Some("no eRPM telemetry for most of the flight".into());
        }
        let corr: Vec<f64> = (0..4)
            .map(|i| {
                let w: Vec<f64> = self.speeds[i].iter().map(|&w| value!(w, rot / s)).collect();
                num::corrcoef(&self.commands[i], &w)
            })
            .collect();
        if num::median(&corr) < MIN_MOTOR_ERPM_CORR {
            let shown: Vec<String> = corr.iter().map(|c| format!("{:.2}", c)).collect();
            return Some(format!(
                "logged motor[] doesn't track eRPM (corr [{}]); motor output was not what the ESCs received",
                shown.join(", ")
            ));
        }
        if self.kept() < MIN_DURATION {
            return Some(format!("only {:.1} s airborne and impact-free", value!(self.kept(), s)));
        }
        None
    }

    /// Cut into `parts` contiguous, equal-duration flights.
    pub fn split(&self, parts: usize) -> Vec<Flight> {
        let (t0, t1) = (value!(self.t[0], s), value!(self.t[self.len() - 1], s));
        (0..parts)
            .map(|k| {
                let lo = t0 + (t1 - t0) * k as f64 / parts as f64;
                let hi = t0 + (t1 - t0) * (k + 1) as f64 / parts as f64;
                let keep: Vec<usize> =
                    (0..self.len()).filter(|&i| (lo..=hi).contains(&value!(self.t[i], s))).collect();
                let pick = |v: &[Rate]| keep.iter().map(|&i| v[i]).collect::<Vec<_>>();
                Flight {
                    label: format!("{}/{}", self.label, k + 1),
                    t: keep.iter().map(|&i| self.t[i]).collect(),
                    rates: core::array::from_fn(|a| pick(&self.rates[a])),
                    setpoint: self.setpoint.as_ref().map(|sp| core::array::from_fn(|a| pick(&sp[a]))),
                    speeds: core::array::from_fn(|m| pick(&self.speeds[m])),
                    commands: core::array::from_fn(|m| keep.iter().map(|&i| self.commands[m][i]).collect()),
                    vbat: keep.iter().map(|&i| self.vbat[i]).collect(),
                    accel: self
                        .accel
                        .as_ref()
                        .map(|acc| core::array::from_fn(|a| keep.iter().map(|&i| acc[a][i]).collect())),
                    good: keep.iter().map(|&i| self.good[i]).collect(),
                    ..self.clone()
                }
            })
            .collect()
    }
}

/// (start, stop) of each run of `true`.
pub fn runs(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut start = None;
    for (i, &m) in mask.iter().enumerate() {
        match (m, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, mask.len()));
    }
    out
}

/// Samples within [event − before, event + after] of any `true` sample.
fn dilate(mask: &[bool], t: &[f64], before: f64, after: f64) -> Vec<bool> {
    let events: Vec<f64> = t.iter().zip(mask).filter_map(|(&t, &m)| m.then_some(t)).collect();
    if events.is_empty() {
        return vec![false; mask.len()];
    }
    t.iter()
        .map(|&ti| {
            let lo = events.partition_point(|&e| e < ti - after);
            let hi = events.partition_point(|&e| e <= ti + before);
            hi > lo
        })
        .collect()
}

/// A flight spec: `PATH` (every log in it) or `PATH#N` (its N-th, 1-based).
pub fn parse_spec(spec: &str) -> Result<(&str, Option<usize>), String> {
    match spec.rsplit_once('#') {
        Some((path, n)) if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) => {
            Ok((path, Some(n.parse().map_err(|e| format!("{spec}: {e}"))?)))
        }
        _ => Ok((spec, None)),
    }
}

/// `<parent dir>/<file name>`, as flights are labelled.
pub fn file_label(path: &Path) -> String {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let parent = abs.parent().and_then(|p| p.file_name()).map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let name = abs.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    if parent.is_empty() { name } else { format!("{parent}/{name}") }
}

/// A flight spec's usable flights, gated, and what was skipped and why.
pub fn load_flights(spec: &str, motor_poles: Option<u32>, skipped: &mut Vec<String>) -> Result<Vec<Flight>, String> {
    let (path, which) = parse_spec(spec)?;
    let path = Path::new(path);
    let logs = bbl::read_logs(path)?;
    let label = file_label(path);
    let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
    let mut short = 0;
    let mut out = vec![];
    for (n, log) in logs.iter().enumerate().map(|(i, l)| (i + 1, l)) {
        if which.is_some_and(|w| w != n) {
            continue;
        }
        let time = log.require("time")?;
        if (time[time.len() - 1] - time[0]) * 1e-6 < value!(MIN_DURATION, s) {
            short += 1;
            continue;
        }
        let flight = Flight::from_log(format!("{label}#{n}"), log, motor_poles)?;
        match flight.unusable() {
            Some(reason) => skipped.push(format!("skipping {}: {reason}", flight.label)),
            None => out.push(flight),
        }
    }
    if short > 0 {
        skipped.push(format!("skipping {short} log(s) in {name} shorter than {} s", value!(MIN_DURATION, s)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_and_dilation() {
        let m = [false, true, true, false, true];
        assert_eq!(runs(&m), vec![(1, 3), (4, 5)]);
        let t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let hit = [false, false, true, false, false, false];
        assert_eq!(dilate(&hit, &t, 1.0, 2.0), vec![false, true, true, true, true, false]);
    }

    #[test]
    fn specs() {
        assert_eq!(parse_spec("a/b.bbl#3").unwrap(), ("a/b.bbl", Some(3)));
        assert_eq!(parse_spec("a/b.bbl").unwrap(), ("a/b.bbl", None));
    }

    #[test]
    fn shaft_speed_from_erpm() {
        assert!((value!(shaft_speed(600.0, 12), rot / s) - 166.666_666_666_666_66).abs() < 1e-9);
    }
}
