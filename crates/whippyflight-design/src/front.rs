//! The Pareto front of an axis group's LQR weights: gain blobs over a grid of
//! σ_rate and σ_int, for the Configurator to score and draw.
//!
//! Everything is found on the loop at hover linearized as the sim flies it
//! ([`crate::hover`]), with the other group held at its current weights.
//! The blobs hold the gains solved at hover at every grid point, for scoring
//! at hover, not for flight, all with the airframe's derivative time
//! constant ([`crate::plant::derivative_tau`]); the range searches start each
//! solve from the current weights' gains.
//!
//! σ_rate runs from the stiffest that changes how fast a knock
//! dies out by no more than SEARCH_TOLERANCE, on the loop with integral
//! weight alone at the loose end of σ_int's range at that σ_rate (looser
//! changes nothing), to where the loop with rate weight alone has a delay
//! margin of twice the fit's measured loop latency. At each σ_rate level
//! σ_int runs from where the loop's delay margin is twice that latency, or
//! where a knock dies out fastest if that is looser, to where the loop goes
//! unstable. Levels are midpoints of equal steps in ln σ.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::atomic::{self, AtomicUsize};

use whippyunits::{quantity, value};

use crate::blob::{self, Design, GridSpec};
use crate::fit::Fit;
use crate::hover::HoverLoop;
use crate::lqr::{Excursions, Group, Weights};
use crate::par::{self, Control};
use crate::units::*;

/// Relative width the grid's ends are found to.
pub const TOLERANCE: f64 = 1e-3;
/// How much a knock's recovery may change for σ_rate to count as unfelt.
pub const SEARCH_TOLERANCE: f64 = 1e-2;
/// Most rounds of σ_rate's loose end's search, each re-anchoring σ_int at the last σ_rate.
const LOOSE_RATE_ROUNDS: usize = 16;
/// The delay margin σ_rate's stiff end and σ_int's loose end keep, in the fit's measured loop latencies.
const MARGIN_LATENCIES: f64 = 2.0;
fn loosen() -> f64 {
    2f64.ln()
}
fn loosest() -> f64 {
    1e12f64.ln()
}

const RATE: Rate = quantity!(1.0, rot / s);
const ANGLE: Angle = quantity!(1.0, rot);
const SECOND: Time = quantity!(1.0, s);
const NO_RATE: Rate = quantity!(f64::INFINITY, rot / s);
const NO_INT: Angle = quantity!(f64::INFINITY, rot);

/// What a front is computed at.
#[derive(Clone, Copy, Debug)]
pub struct FrontSpec {
    pub dt: Time,
    /// The current weights: the other group is held at them, and τ_int.
    pub weights: Weights,
    /// The feedforward's horizon in every blob.
    pub ff_horizon: Time,
    pub grid: GridSpec,
    pub rate_levels: usize,
    pub int_levels: usize,
    /// Whether the LQR weighs the fit's gyro noise.
    pub gyro_noise: bool,
}

#[derive(Clone, Debug)]
pub struct Point {
    pub sigma_rate: Rate,
    pub sigma_int: Angle,
    /// [σ_rate level, σ_int level].
    pub grid: [usize; 2],
    pub wfg: String,
}

#[derive(Clone, Debug)]
pub struct Front {
    pub group: Group,
    /// The other group's excursions, held.
    pub held: Excursions,
    /// Every blob's derivative time constant.
    pub tau_deriv: Time,
    pub points: Vec<Point>,
}

impl Front {
    /// As the `front.json` index of `spec`'s search: weights in rot/s, rot and s.
    pub fn to_json(&self, spec: &FrontSpec) -> serde_json::Value {
        serde_json::json!({
            "group": match self.group { Group::PitchRoll => "pitch_roll", Group::Yaw => "yaw" },
            "dt": value!(spec.dt, s),
            "tau_deriv": value!(self.tau_deriv, s),
            "tau_int": value!(spec.weights.tau_int, s),
            "ff_horizon": value!(spec.ff_horizon, s),
            "resolution": spec.grid.resolution,
            "idle_rps": value!(spec.grid.idle, rot / s),
            "structured": true,
            "hover_only": true,
            "gyro_noise": spec.gyro_noise,
            "held": {
                "sigma_rate": value!(self.held.rate, rot / s),
                "sigma_int": value!(self.held.int, rot),
            },
            "points": self.points.iter().map(|p| serde_json::json!({
                "sigma_rate": value!(p.sigma_rate, rot / s),
                "sigma_int": value!(p.sigma_int, rot),
                "grid": p.grid,
                "wfg": p.wfg,
            })).collect::<Vec<_>>(),
        })
    }
}

fn other(group: Group) -> Group {
    match group {
        Group::PitchRoll => Group::Yaw,
        Group::Yaw => Group::PitchRoll,
    }
}

fn group_name(group: Group) -> &'static str {
    match group {
        Group::PitchRoll => "pitch/roll",
        Group::Yaw => "yaw",
    }
}

/// Why `g`'s search can't succeed from `base`, if the other group's weights
/// held there are at fault: the loop at `base` is unstable at hover, or keeps
/// under MARGIN_LATENCIES of the fit's loop latency of delay margin on the
/// held group's axes.
fn held_fault(lp: &mut HoverLoop, base: &Weights, g: Group) -> Option<String> {
    let held = other(g);
    let w = base.group(held);
    let at = format!(
        "the {} weights held (σ_rate {:.4} rot/s, σ_int {:.4} rot)",
        group_name(held),
        value!(w.rate, rot / s),
        value!(w.int, rot)
    );
    let target = MARGIN_LATENCIES * lp.latency;
    let why = match lp.delay_margin(base, held, TOLERANCE) {
        None => format!("at {at} the loop is unstable at hover"),
        Some(m) if m < target => format!(
            "{at} keep only {:.3} ms of delay margin on their axes, under the {:.3} ms the search needs",
            value!(m, ms),
            value!(target, ms)
        ),
        Some(_) => return None,
    };
    Some(format!("{why}; reset the weights, or pick {}'s from its own front", group_name(held)))
}

fn with(base: &Weights, group: Group, rate: Rate, int: Angle) -> Weights {
    base.with_group(group, Excursions { rate, int })
}

fn secs(t: Time) -> f64 {
    f64::from(t / SECOND)
}

/// The x in [`lo`, `hi`] where `cost` is least, to `width` (golden section).
fn golden(mut cost: impl FnMut(f64) -> f64, mut lo: f64, mut hi: f64, width: f64) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut ca, mut cb) = (cost(a), cost(b));
    while hi - lo > width {
        if ca <= cb {
            (hi, b, cb) = (b, a, ca);
            a = hi - ratio * (hi - lo);
            ca = cost(a);
        } else {
            (lo, a, ca) = (a, b, cb);
            b = lo + ratio * (hi - lo);
            cb = cost(b);
        }
    }
    0.5 * (lo + hi)
}

/// The weight 1/σ where `stable(1/σ)` turns false, from stable at 0.
fn boundary(mut stable: impl FnMut(f64) -> bool, tolerance: f64) -> Result<f64, String> {
    if !stable(0.0) {
        return Ok(0.0);
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    while stable(hi) {
        (lo, hi) = (hi, 2.0 * hi);
        if hi > 1e12 {
            return Err("the loop never goes unstable: no range to derive".into());
        }
    }
    while hi - lo > tolerance * hi {
        let mid = 0.5 * (lo + hi);
        if stable(mid) { lo = mid } else { hi = mid }
    }
    Ok(lo)
}

/// Midpoints of n equal steps in ln σ from `loose` to `stiff` (same unit).
fn levels(loose: f64, stiff: f64, n: usize) -> Vec<f64> {
    let (lo, hi) = (loose.ln(), stiff.ln());
    (0..n).map(|i| (lo + (hi - lo) * (i as f64 + 0.5) / n as f64).exp()).collect()
}

/// (loose, stiff) σ_int at `rate`: where the loop's delay margin reaches
/// MARGIN_LATENCIES of the fit's loop latency, or a knock dies out fastest if
/// that is looser, and where the loop goes unstable; None if it is unstable
/// without integral weight too.
fn int_range(lp: &mut HoverLoop, base: &Weights, g: Group, rate: Rate, tolerance: f64) -> Result<Option<(Angle, Angle)>, String> {
    let edge = boundary(|u| lp.stable(&with(base, g, rate, 1.0 / (u / ANGLE)), g, 0.0), tolerance)?;
    if edge <= 0.0 {
        return Ok(None);
    }
    let start = (1.0 / edge).ln();
    let at = |x: f64| with(base, g, rate, x.exp() * ANGLE);
    let kick = |lp: &mut HoverLoop, x: f64| secs(lp.kick_iae(&at(x), g));
    let target = MARGIN_LATENCIES * lp.latency;
    let spared = |lp: &mut HoverLoop, x: f64| lp.delay_margin(&at(x), g, tolerance).is_some_and(|m| m >= target);
    let (mut x, mut here) = (start, kick(lp, start));
    loop {
        let looser = kick(lp, x + loosen());
        if looser.partial_cmp(&here) != Some(Ordering::Less) {
            break;
        }
        (x, here) = (x + loosen(), looser);
        if x - start > loosest() {
            return Err("integral weight never stops speeding a knock's recovery".into());
        }
    }
    let mut lo = golden(|x| kick(lp, x), start.max(x - loosen()), x + loosen(), tolerance);
    if !spared(lp, lo) {
        let mut hi = lo + loosen();
        while !spared(lp, hi) {
            (lo, hi) = (hi, hi + loosen());
            if hi - start > loosest() {
                return Err(format!(
                    "the loop's delay margin never reaches {:.3} ms",
                    value!(target, ms)
                ));
            }
        }
        while hi - lo > tolerance {
            let mid = 0.5 * (lo + hi);
            if spared(lp, mid) { hi = mid } else { lo = mid }
        }
        lo = hi;
    }
    Ok(Some((lo.exp() * ANGLE, (1.0 / edge) * ANGLE)))
}

/// (loose, stiff) σ_rate: [`loose_rate`], looser than which the rate weight
/// changes nothing, and where the loop with rate weight alone has a delay
/// margin of MARGIN_LATENCIES of the fit's loop latency, stiffer than which
/// no σ_int range keeps it.
fn rate_range(lp: &mut HoverLoop, base: &Weights, g: Group, tolerance: f64) -> Result<(Rate, Rate), String> {
    let loose = loose_rate(lp, base, g, tolerance)?;
    let at = |x: f64| with(base, g, x.exp() * RATE, NO_INT);
    let target = MARGIN_LATENCIES * lp.latency;
    let spared = |lp: &mut HoverLoop, x: f64| lp.delay_margin(&at(x), g, tolerance).is_some_and(|m| m >= target);
    let top = f64::from(loose / RATE).ln();
    if !spared(lp, top) {
        return Err(format!(
            "the loop's delay margin is under {:.3} ms with rate weight {:.3} rot/s alone",
            value!(target, ms),
            value!(loose, rot / s)
        ));
    }
    let (mut lo, mut hi) = (top - loosen(), top);
    while spared(lp, lo) {
        (lo, hi) = (lo - loosen(), lo);
        if top - lo > loosest() {
            return Err("rate weight never spends the loop's delay margin".into());
        }
    }
    while hi - lo > tolerance {
        let mid = 0.5 * (lo + hi);
        if spared(lp, mid) { hi = mid } else { lo = mid }
    }
    Ok((loose, hi.exp() * RATE))
}

/// The σ_rate on `g` that is the stiffest to change a knock's recovery by no
/// more than SEARCH_TOLERANCE on the loop with integral weight alone at the
/// loose end of σ_int's range at that σ_rate; from `base`'s.
fn loose_rate(lp: &mut HoverLoop, base: &Weights, g: Group, tolerance: f64) -> Result<Rate, String> {
    let mut rate = base.group(g).rate;
    let mut why = String::from("σ_rate's search doesn't settle");
    for _ in 0..LOOSE_RATE_ROUNDS {
        let sigma_int = match int_range(lp, base, g, rate, tolerance) {
            Ok(Some((loose, _))) => loose,
            Ok(None) => {
                rate = stable_rate(lp, base, g, rate)?;
                continue;
            }
            // Too stiff a rate weight spends the delay margin σ_int's range needs.
            Err(e) => {
                why = e;
                rate = loosen().exp() * rate;
                continue;
            }
        };
        let next = unfelt_rate(lp, base, g, sigma_int, tolerance)?;
        if f64::from(next / rate).ln().abs() <= tolerance {
            return Ok(next);
        }
        rate = next;
    }
    Err(why)
}

/// The σ_rate nearest `from` in ln σ (stiffer first) whose loop is stable
/// without integral weight on `g`.
fn stable_rate(lp: &mut HoverLoop, base: &Weights, g: Group, from: Rate) -> Result<Rate, String> {
    let mut stable = |rate: Rate| lp.stable(&with(base, g, rate, NO_INT), g, 0.0);
    let mut step = 0.0;
    while step <= loosest() {
        for rate in [(-step).exp() * from, step.exp() * from] {
            if stable(rate) {
                return Ok(rate);
            }
        }
        step += loosen();
    }
    Err("the loop is unstable at hover at every σ_rate without integral weight".into())
}

/// The stiffest σ_rate on `g` that changes a knock's recovery by no more than
/// SEARCH_TOLERANCE on the loop with integral weight `sigma_int` alone.
fn unfelt_rate(lp: &mut HoverLoop, base: &Weights, g: Group, sigma_int: Angle, tolerance: f64) -> Result<Rate, String> {
    let none = secs(lp.kick_iae(&with(base, g, NO_RATE, sigma_int), g));
    if !none.is_finite() {
        return Err(format!("the loop is unstable at hover with integral weight {:.4} rot alone", value!(sigma_int, rot)));
    }
    let unchanged =
        |lp: &mut HoverLoop, x: f64| (secs(lp.kick_iae(&with(base, g, x.exp() * RATE, sigma_int), g)) / none - 1.0).abs() <= SEARCH_TOLERANCE;
    let (mut lo, mut hi) = (-loosen(), 0.0);
    if unchanged(lp, hi) {
        while unchanged(lp, lo) {
            (lo, hi) = (lo - loosen(), lo);
            if -lo > loosest() {
                return Err("rate weight never changes a knock's recovery".into());
            }
        }
    } else {
        (lo, hi) = (hi, hi + loosen());
        while !unchanged(lp, hi) {
            (lo, hi) = (hi, hi + loosen());
            if hi > loosest() {
                return Err("rate weight never stops changing a knock's recovery".into());
            }
        }
    }
    while hi - lo > tolerance {
        let mid = 0.5 * (lo + hi);
        if unchanged(lp, mid) { hi = mid } else { lo = mid }
    }
    Ok(hi.exp() * RATE)
}

fn rate_levels(range: (Rate, Rate), n: usize) -> Vec<Rate> {
    levels(f64::from(range.0 / RATE), f64::from(range.1 / RATE), n).into_iter().map(|x| x * RATE).collect()
}

/// [(rate level, int level, σ_rate, σ_int)] over σ_rates `rates` and each
/// one's σ_int range.
fn grid(rates: &[Rate], spans: &[Option<(Angle, Angle)>], int_levels: usize) -> Vec<(usize, usize, Rate, Angle)> {
    let mut out = Vec::new();
    for (i, (&sr, span)) in rates.iter().zip(spans).enumerate() {
        let Some((loose, stiff)) = span else { continue };
        for (j, si) in levels(f64::from(*loose / ANGLE), f64::from(*stiff / ANGLE), int_levels).into_iter().enumerate() {
            out.push((i, j, sr, si * ANGLE));
        }
    }
    out
}

/// The index [`explore`] writes beside its blobs.
pub const INDEX: &str = "front.json";

/// The front of group `g` on `fit`, disturbed as measured beside it in
/// `fit_dir`, its blobs and [`INDEX`] written to `out`.
pub fn explore(fit: &Fit, fit_dir: &Path, g: Group, spec: &FrontSpec, out: &Path, control: &Control) -> Result<Front, String> {
    if spec.rate_levels < 1 || spec.int_levels < 1 {
        return Err("each weight needs at least one level".into());
    }
    if spec.weights.tau_int <= spec.dt {
        return Err("τ_int must be longer than a loop tick".into());
    }
    let fail = |c: par::Cancelled| c.to_string();
    let mut lp = HoverLoop::new(fit, fit_dir, spec.dt)?;
    if !spec.gyro_noise {
        lp.disturbances = lp.disturbances.without_gyro_noise();
    }
    let base = spec.weights;
    lp.anchor(&base)?;
    let name = group_name(g);
    control.say(&format!("Finding the {name} weights' ranges at hover..."));
    let ranges = rate_range(&mut lp, &base, g, TOLERANCE).and_then(|rate_range| {
        let rates = rate_levels(rate_range, spec.rate_levels);
        let spans = par::map_with(&rates, control, || lp.clone(), |lp, &sr| int_range(lp, &base, g, sr, TOLERANCE))
            .map_err(fail)?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok((rate_range, rates, spans))
    });
    let (rate_range, rates, spans) = ranges.map_err(|e| held_fault(&mut lp, &base, g).unwrap_or(e))?;
    let spans_text: Vec<String> = spans
        .iter()
        .map(|s| match s {
            Some((a, b)) => format!("{:.4} to {:.4}", value!(*a, rot), value!(*b, rot)),
            None => "none".into(),
        })
        .collect();
    control.say(&format!(
        "  σ_rate from {:.4} to {:.4}; σ_int from {}",
        value!(rate_range.0, rot / s),
        value!(rate_range.1, rot / s),
        spans_text.join(", ")
    ));
    let points_grid = grid(&rates, &spans, spec.int_levels);
    control.say(&format!("Solving gains at {} weightings...", points_grid.len()));
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let quiet = |_: &str| {};
    let inner = Control { cancel: control.cancel, progress: &quiet };
    let done = AtomicUsize::new(0);
    let total = points_grid.len();
    let indexed: Vec<(usize, (usize, usize, Rate, Angle))> = points_grid.into_iter().enumerate().collect();
    let points = par::map_with(&indexed, control, || (), |_, &(n, (i, j, sr, si))| {
        let design = Design {
            dt: spec.dt,
            weights: with(&base, g, sr, si),
            ff_horizon: spec.ff_horizon,
            disturbances: lp.disturbances,
            grid: spec.grid,
            hover_only: true,
        };
        let result = blob::generate(fit, &design, &inner).and_then(|(b, _)| {
            let file = format!("{n}.wfg");
            let path: PathBuf = out.join(&file);
            std::fs::write(&path, b.encode()).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(file)
        });
        let k = done.fetch_add(1, atomic::Ordering::Relaxed) + 1;
        let label = format!("σ_rate {:.4}, σ_int {:.4}", value!(sr, rot / s), value!(si, rot));
        match result {
            Ok(wfg) => {
                control.say(&format!("  [{k}/{total}] {label}"));
                Some(Point { sigma_rate: sr, sigma_int: si, grid: [i, j], wfg })
            }
            Err(e) => {
                control.say(&format!("  [{k}/{total}] {label}: failed: {e}"));
                None
            }
        }
    })
    .map_err(fail)?;
    let points: Vec<Point> = points.into_iter().flatten().collect();
    if points.is_empty() {
        return Err("no gains could be computed".into());
    }
    let front = Front { group: g, held: base.group(other(g)), tau_deriv: lp.tau_deriv, points };
    let index = out.join(INDEX);
    let text = serde_json::to_string_pretty(&front.to_json(spec)).map_err(|e| e.to_string())?;
    std::fs::write(&index, text).map_err(|e| format!("{}: {e}", index.display()))?;
    control.say(&format!("Wrote {} blobs and {INDEX} to {}", front.points.len(), out.display()));
    Ok(front)
}
