//! Fit-quality diagnostics, for judging whether test flights excited the
//! airframe enough and whether the model explains them; what the
//! Configurator's WhippyFlight tab draws.
//!
//! Per axis (yaw, pitch, roll: angular acceleration in rot/s²; motors: volts):
//!  * R² of the pooled fit on its own rows, and held out: leave one flight
//!    out with two or more flights, else each quarter of the lone flight
//!    scored by a fit to the other three;
//!  * coefficients with relative standard errors;
//!  * measured vs predicted on a random sample of rows, and the residual
//!    histogram;
//!  * a time-series excerpt: the most heavily excited window of the longest
//!    flight, measured and predicted.
//!
//! Per channel (rate axes and motors), the noise split: where the spectrum
//! of what the model leaves unexplained over every usable row, on the
//! shortest finite difference, stops falling (unmodeled dynamics) and starts
//! rising (sensor noise, differentiated): its smoothed minimum, Hz. The rate
//! axes' residuals are of the motor model's speeds rather than the
//! telemetry, so what rises is the gyro's noise; the motors' is of the
//! telemetry, whose noise it measures. Below the split the residual is the
//! airframe's, above it the sensor's; noise extraction and the turbulence
//! band divide there.
//!
//! Per rate axis, turbulence: that residual from TURBULENCE_LOW to the
//! axis's split, as the one-pole spectrum with the same power and the same
//! division about the band's middle: its RMS (rot/s²) and corner (Hz), as
//! the flick sim's turbulence takes them.
//!
//! Per flight: kept time, gating, the pooled fit's R² on that flight alone,
//! and the motor-pairing check. `warnings` lists what looks wrong, as stable
//! codes with parameters, so a UI can phrase them.

use std::collections::HashSet;
use std::f64::consts::PI;

use nalgebra::DVector;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use whippyunits::value;

use crate::fit::{Report, Stored};
use crate::flight::Flight;
use crate::json::J;
use crate::num;
use crate::obj;
use crate::rows::{self, AXES, Axis, RATE_AXES, Rows, Select};

const MAX_POINTS: usize = 1500;
const HIST_BINS: usize = 41;
const EXCERPT_S: f64 = 6.0;
const HELDOUT_FOLDS: usize = 4;

const MAX_HELDOUT_DROP: f64 = 0.1;
const MIN_TOTAL_KEPT_S: f64 = 30.0;

/// Slower than this is a bias over one flick (a 0.3 s hold between its rise
/// and release), the CG offset's and the pilot's to stand for, not turbulence.
const TURBULENCE_LOW_HZ: f64 = 2.0;
/// The noise split is searched from here up to SPLIT_MAX of the log rate, on
/// third-octave bands smoothed over SPLIT_SMOOTH of them.
const SPLIT_MIN_HZ: f64 = 5.0;
const SPLIT_MAX: f64 = 0.25;
const SPLIT_SMOOTH: usize = 3;
/// The residual's finite difference: two samples, its sinc null at the log rate.
const SPLIT_WINDOW: usize = 2;
/// Welch segments: as long as a run allows up to the longer, none from runs
/// shorter than the shorter.
const SEGMENT_S: (f64, f64) = (2.0, 4.0);

fn min_r2(axis: Axis) -> f64 {
    if axis == Axis::Motors { 0.9 } else { 0.5 }
}

fn unit(axis: Axis) -> &'static str {
    if axis == Axis::Motors { "V" } else { "rot/s²" }
}

fn round5(v: impl IntoIterator<Item = f64>) -> Vec<f64> {
    v.into_iter().map(|x| num::round_to(x, 5)).collect()
}

/// One-sided PSD by Welch's method as scipy's defaults: periodic Hann,
/// half overlap, each segment's mean removed, density scaling.
pub fn welch(x: &[f64], fs: f64, nperseg: usize) -> (Vec<f64>, Vec<f64>) {
    let n = nperseg.min(x.len());
    let step = n - n / 2;
    let w: Vec<f64> = (0..n).map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()).collect();
    let scale = 1.0 / (fs * w.iter().map(|v| v * v).sum::<f64>());
    let bins = n / 2 + 1;
    let fft = FftPlanner::new().plan_fft_forward(n);
    let mut psd = vec![0.0; bins];
    let segments = (x.len() - n) / step + 1;
    let mut buf = vec![Complex::new(0.0, 0.0); n];
    for s in 0..segments {
        let seg = &x[s * step..s * step + n];
        let m = num::mean(seg);
        for (b, (&v, &wi)) in buf.iter_mut().zip(seg.iter().zip(&w)) {
            *b = Complex::new((v - m) * wi, 0.0);
        }
        fft.process(&mut buf);
        for (p, b) in psd.iter_mut().zip(&buf) {
            *p += b.norm_sqr() * scale;
        }
    }
    let last = if n.is_multiple_of(2) { bins - 1 } else { bins };
    for (k, p) in psd.iter_mut().enumerate() {
        *p /= segments as f64;
        if k > 0 && k < last {
            *p *= 2.0;
        }
    }
    ((0..bins).map(|k| k as f64 * fs / n as f64).collect(), psd)
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) }
}

/// A residual spectrum: frequencies, sinc-corrected PSD, and its rows.
struct Spectrum {
    freqs: Vec<f64>,
    psd: Vec<f64>,
    rows: usize,
}

fn residual(r: &Rows, beta: &DVector<f64>) -> DVector<f64> {
    &r.y - &r.x * beta
}

/// The model's residual over every usable row on the SPLIT_WINDOW
/// difference, per contiguous run (each motor's apart).
fn residual_spectra(flights: &[Flight], axis: Axis, fit: &Stored) -> Result<Vec<Spectrum>, String> {
    let mut out = vec![];
    for f in flights {
        let r = rows::rows(f, axis, &fit.arms, fit.map, &fit.drag, Select { gated: false, window: Some(SPLIT_WINDOW) });
        let resid = residual(&r, &fit.beta(axis, &r.names)?);
        let hz = value!(f.log_hz, Hz);
        let (shortest, longest) = ((SEGMENT_S.0 * hz) as usize, (SEGMENT_S.1 * hz) as usize);
        let mut start = 0;
        for end in 1..=r.len() {
            if end < r.len() && r.center[end] == r.center[end - 1] + 1 {
                continue;
            }
            if end - start >= shortest {
                let (freqs, mut psd) = welch(&resid.as_slice()[start..end], hz, longest.min(end - start));
                for (p, &fr) in psd.iter_mut().zip(&freqs) {
                    *p /= sinc(fr * SPLIT_WINDOW as f64 / hz).powi(2);
                }
                out.push(Spectrum { freqs, psd, rows: end - start });
            }
            start = end;
        }
    }
    Ok(out)
}

/// Mean over rows of the power in [lo, hi).
fn band_power(spectra: &[Spectrum], lo: f64, hi: f64) -> f64 {
    let rows: usize = spectra.iter().map(|s| s.rows).sum();
    let total: f64 = spectra
        .iter()
        .map(|s| {
            let df = s.freqs[1] - s.freqs[0];
            let p: f64 = s.freqs.iter().zip(&s.psd).filter(|&(&fr, _)| fr >= lo && fr < hi).map(|(_, &p)| p).sum();
            p * df * s.rows as f64
        })
        .sum();
    total / rows as f64
}

/// The smoothed third-octave minimum of the residual's spectrum, Hz.
fn split(spectra: &[Spectrum], log_hz: f64) -> Option<f64> {
    let bands = (3.0 * (SPLIT_MAX * log_hz / SPLIT_MIN_HZ).log2()) as i64 + 1;
    let edges: Vec<f64> = (0..bands.max(0)).map(|k| SPLIT_MIN_HZ * 2f64.powf(k as f64 / 3.0)).collect();
    let density: Vec<f64> = edges.windows(2).map(|e| band_power(spectra, e[0], e[1]) / (e[1] - e[0])).collect();
    if density.len() < SPLIT_SMOOTH || !density.iter().all(|&d| d > 0.0) {
        return None;
    }
    let logs: Vec<f64> = density.iter().map(|d| d.ln()).collect();
    let smooth: Vec<f64> = logs.windows(SPLIT_SMOOTH).map(|w| w.iter().sum::<f64>() / SPLIT_SMOOTH as f64).collect();
    let k = smooth.iter().enumerate().fold(0, |b, (i, &v)| if v < smooth[b] { i } else { b }) + SPLIT_SMOOTH / 2;
    Some((edges[k] * edges[k + 1]).sqrt())
}

/// ∫ 1/(1 + (f/hz)²) df over [lo, hi].
fn one_pole_power(lo: f64, hi: f64, hz: f64) -> f64 {
    hz * ((hi / hz).atan() - (lo / hz).atan())
}

/// Per rate axis, {rms, hz} of the one-pole spectrum with the residual's
/// power from TURBULENCE_LOW to the axis's split and its division about the
/// band's geometric middle; null if too little data, no split, or a spectrum
/// no one-pole has.
fn turbulence(spectra: &[(Axis, Vec<Spectrum>)], splits: &[(Axis, Option<f64>)]) -> J {
    let mut out = J::obj();
    for &axis in &RATE_AXES {
        let s = &spectra.iter().find(|(a, _)| *a == axis).unwrap().1;
        let lo = TURBULENCE_LOW_HZ;
        let hi = splits.iter().find(|(a, _)| *a == axis).and_then(|(_, h)| *h);
        let value = (|| {
            let hi = hi.filter(|&h| !s.is_empty() && h > 2.0 * lo)?;
            let mid = (lo * hi).sqrt();
            let (low, high) = (band_power(s, lo, mid), band_power(s, mid, hi));
            // A one-pole spectrum's division runs from the flat (mid − lo)/(hi − mid)
            // at a high corner to the f⁻² (1/lo − 1/mid)/(1/mid − 1/hi) at a low one.
            let division = |hz: f64| one_pole_power(lo, mid, hz) / one_pole_power(mid, hi, hz) - low / high;
            if !(low > 0.0 && high > 0.0) || division(1e-3) * division(1e4) >= 0.0 {
                return None;
            }
            let hz = num::brent(division, 1e-3, 1e4);
            let s0 = (low + high) / one_pole_power(lo, hi, hz);
            Some(obj! {"rms" => num::round_to((s0 * hz * PI / 2.0).sqrt(), 5), "hz" => num::round_to(hz, 3)})
        })();
        out.set(axis.name(), value.unwrap_or(J::Null));
    }
    out
}

/// Per-axis held-out R², pooled over folds as 1 − ΣSSE/ΣSST. Folds
/// partition the pooled fit's own rows (by flight, or a lone flight's rows
/// into contiguous blocks of equal count) rather than re-gating pieces of a
/// flight, so every fold is scored on rows the deployed fit was made from.
/// Blocks are equal in rows, not duration: excitation comes in bursts, and
/// equal-duration folds can leave a fold trained on a handful of rows.
fn heldout_r2(flights: &[Flight], fit: &Stored, axis: Axis) -> Option<f64> {
    let parts: Vec<Rows> = flights.iter().map(|f| rows::rows(f, axis, &fit.arms, fit.map, &fit.drag, Select::default())).collect();
    let mut fold = vec![];
    for (k, p) in parts.iter().enumerate() {
        let n = p.len();
        if flights.len() >= 2 {
            fold.extend(std::iter::repeat_n(k, n));
        } else {
            fold.extend((0..n).map(|i| i * HELDOUT_FOLDS / n.max(1)));
        }
    }
    let all = rows::stack_rows(parts);
    let mut folds: Vec<usize> = fold.clone();
    folds.sort_unstable();
    folds.dedup();
    let (mut sse, mut sst) = (0.0, 0.0);
    for k in folds {
        let held: Vec<bool> = fold.iter().map(|&f| f == k).collect();
        if held.iter().all(|&h| h) {
            continue;
        }
        let rest: Vec<bool> = held.iter().map(|h| !h).collect();
        let (xt, yt) = num::select_rows(&all.x, &all.y, &rest);
        let (xh, yh) = num::select_rows(&all.x, &all.y, &held);
        let (a, b) = num::sse_sst(&xh, &yh, &num::lstsq(&xt, &yt));
        sse += a;
        sst += b;
    }
    (sst != 0.0).then(|| num::round_to(1.0 - sse / sst, 4))
}

/// A deterministic SplitMix64 stream for the plots' row samples.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `k` of 0..n without replacement, sorted (all of them if n ≤ k).
fn sample(n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    if n <= k {
        return (0..n).collect();
    }
    let mut picked = HashSet::with_capacity(k);
    for j in n - k..n {
        let t = rng.below(j + 1);
        if !picked.insert(t) {
            picked.insert(j);
        }
    }
    let mut v: Vec<usize> = picked.into_iter().collect();
    v.sort_unstable();
    v
}

/// numpy's histogram over [-lim, lim]: (edges, counts).
fn histogram(x: &[f64], bins: usize, lim: f64) -> (Vec<f64>, Vec<usize>) {
    let (lo, hi) = if lim > 0.0 { (-lim, lim) } else { (-lim - 0.5, lim + 0.5) };
    let edges: Vec<f64> = (0..=bins).map(|i| lo + (hi - lo) * i as f64 / bins as f64).collect();
    let mut counts = vec![0usize; bins];
    for &v in x {
        if v < lo || v > hi || !v.is_finite() {
            continue;
        }
        let i = (((v - lo) / (hi - lo) * bins as f64) as usize).min(bins - 1);
        counts[i] += 1;
    }
    (edges, counts)
}

/// Measured and predicted over the EXCERPT_S window with the most rows.
fn excerpt(flight: &Flight, axis: Axis, fit: &Stored, beta: &DVector<f64>) -> J {
    let r = rows::rows(flight, axis, &fit.arms, fit.map, &fit.drag, Select::default());
    // Rows are the four motors' in turn; the first motor's run ends where the samples wrap.
    let end = if axis == Axis::Motors { (1..r.len()).find(|&i| r.center[i] < r.center[i - 1]).unwrap_or(r.len()) } else { r.len() };
    if end == 0 {
        return J::Null;
    }
    let t: Vec<f64> = r.center[..end].iter().map(|&i| value!(flight.t[i], s)).collect();
    let mut best = (0, 0);
    let mut stop = 0;
    for (k, &tk) in t.iter().enumerate() {
        while stop < t.len() && t[stop] < tk + EXCERPT_S {
            stop += 1;
        }
        if stop - k > best.1 {
            best = (k, stop - k);
        }
    }
    let (start, count) = best;
    let pick: Vec<usize> = sample(count, MAX_POINTS, &mut Rng(0)).into_iter().map(|i| start + i).collect();
    let pred = &r.x * beta;
    obj! {
        "flight" => flight.label.clone(),
        "window_s" => EXCERPT_S,
        "t" => pick.iter().map(|&i| num::round_to(t[i] - t[start], 4)).collect::<Vec<_>>(),
        "measured" => round5(pick.iter().map(|&i| r.y[i])),
        "predicted" => round5(pick.iter().map(|&i| pred[i])),
    }
}

fn names_of(fit: &Stored, flight: &Flight, axis: Axis) -> Vec<String> {
    rows::rows(flight, axis, &fit.arms, fit.map, &fit.drag, Select::default()).names
}

/// The diagnostics of `fit` (regression frame) over the flights it was made from.
pub fn diagnostics(flights: &[Flight], fit: &Stored, report: &Report) -> Result<J, String> {
    let clean: Vec<Flight> = flights.iter().map(|f| f.with_speeds(fit.motors.speeds(f, fit.map, fit.actuator_delay))).collect();
    // The motor rows' voltages at the fit's delay (only the motor rows read them).
    let lagged: Vec<Flight> = flights.iter().map(|f| f.with_voltage_delay(fit.telemetry_delay)).collect();
    (report.check)()?;
    let log_hz = flights.iter().map(|f| value!(f.log_hz, Hz)).fold(f64::INFINITY, f64::min);
    let spectra: Vec<(Axis, Vec<Spectrum>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = AXES
            .iter()
            .map(|&axis| {
                let source = if axis == Axis::Motors { &lagged } else { &clean };
                scope.spawn(move || residual_spectra(source, axis, fit).map(|s| (axis, s)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("spectra")).collect::<Result<_, _>>()
    })?;
    (report.check)()?;
    let splits: Vec<(Axis, Option<f64>)> = spectra
        .iter()
        .map(|(a, s)| (*a, if s.is_empty() { None } else { split(s, log_hz) }.filter(|&h| h != 0.0).map(|h| num::round_to(h, 1))))
        .collect();
    let heldout: Vec<Option<f64>> = std::thread::scope(|scope| {
        let lagged = &lagged;
        let handles: Vec<_> = AXES.iter().map(|&axis| scope.spawn(move || heldout_r2(lagged, fit, axis))).collect();
        handles.into_iter().map(|h| h.join().expect("held-out")).collect()
    });
    (report.check)()?;
    let mut rng = Rng(0);
    let longest = lagged.iter().fold(&lagged[0], |b, f| if f.kept() > b.kept() { f } else { b });
    let rel_se_of = |axis: Axis, name: &str| -> J {
        fit.json.get("fit").and_then(|f| f.get(axis.name())).and_then(|a| a.get("rel_se")).and_then(|r| r.get(name)).cloned().unwrap_or(J::Null)
    };
    let mut axes = J::obj();
    let mut warnings: Vec<J> = vec![];
    for (i, &axis) in AXES.iter().enumerate() {
        let names = names_of(fit, &flights[0], axis);
        let beta = fit.beta(axis, &names)?;
        let r = rows::stack(&lagged, axis, &fit.arms, fit.map, &fit.drag);
        let pred: DVector<f64> = &r.x * &beta;
        let resid: Vec<f64> = r.y.iter().zip(pred.iter()).map(|(y, p)| y - p).collect();
        let (sse, sst) = num::sse_sst(&r.x, &r.y, &beta);
        let r2 = if sst != 0.0 { 1.0 - sse / sst } else { 0.0 };
        let pick = sample(r.len(), MAX_POINTS, &mut rng);
        let abs: Vec<f64> = resid.iter().map(|v| v.abs()).collect();
        let lim = if abs.is_empty() { 1.0 } else { num::percentile(&abs, 99.5) };
        let (edges, counts) = histogram(&resid, HIST_BINS, lim);
        let coefficients: Vec<J> = names
            .iter()
            .zip(beta.iter())
            .map(|(n, &b)| obj! {"name" => n.clone(), "value" => b, "rel_se" => rel_se_of(axis, n)})
            .collect();
        if r2 < min_r2(axis) {
            warnings.push(obj! {"code" => "low_r2", "axis" => axis.name(), "value" => num::round_to(r2, 3), "limit" => min_r2(axis)});
        }
        if let Some(h) = heldout[i]
            && r2 - h > MAX_HELDOUT_DROP
        {
            warnings.push(obj! {"code" => "overfit", "axis" => axis.name(), "r2" => num::round_to(r2, 3), "heldout_r2" => h});
        }
        let y = r.y.as_slice();
        axes.set(axis.name(), obj! {
            "unit" => unit(axis),
            "rows" => r.len(),
            "r2" => num::round_to(r2, 4),
            "heldout_r2" => heldout[i],
            "measured_std" => num::round_to(num::std(y), 5),
            "residual_std" => num::round_to(num::std(&resid), 5),
            "coefficients" => coefficients,
            "scatter" => obj! {
                "measured" => round5(pick.iter().map(|&k| y[k])),
                "predicted" => round5(pick.iter().map(|&k| pred[k])),
            },
            "residual_hist" => obj! {"edges" => round5(edges), "counts" => counts},
            "excerpt" => excerpt(longest, axis, fit, &beta),
        });
        (report.check)()?;
    }
    let mut per_flight = vec![];
    for f in flights {
        let pairing = rows::pairing_check(f);
        let mismatched: Vec<&str> = pairing
            .iter()
            .filter(|(a, s)| s.iter().fold(&s[0], |b, x| if x.1 > b.1 { x } else { b }).0 != *a)
            .map(|(a, _)| a.name())
            .collect();
        let mut r2 = J::obj();
        for (a, v) in crate::fit::evaluate(fit, f)? {
            r2.set(a.name(), num::round_to(v, 4));
        }
        let mut pairs = J::obj();
        for (a, s) in &pairing {
            let mut o = J::obj();
            for (b, v) in s {
                o.set(b.name(), num::round_to(*v, 4));
            }
            pairs.set(a.name(), o);
        }
        per_flight.push(obj! {
            "label" => f.label.clone(),
            "duration_s" => num::round_to(value!(f.t[f.len() - 1] - f.t[0], s), 1),
            "kept_s" => num::round_to(value!(f.kept(), s), 1),
            "log_hz" => num::round_to(value!(f.log_hz, Hz), 1),
            "gating" => obj! {"ground_s" => value!(f.gating.ground, s), "impacts" => f.gating.impacts, "impact_s" => value!(f.gating.impact, s)},
            "r2" => r2,
            "pairing" => pairs,
            "pairing_mismatch" => mismatched.clone(),
        });
        if !mismatched.is_empty() {
            warnings.push(obj! {"code" => "pairing_mismatch", "flight" => f.label.clone(), "axes" => mismatched});
        }
    }
    let kept: f64 = flights.iter().map(|f| value!(f.kept(), s)).sum();
    if kept < MIN_TOTAL_KEPT_S {
        warnings.push(obj! {"code" => "little_data", "kept_s" => num::round_to(kept, 1), "limit" => MIN_TOTAL_KEPT_S});
    }
    let mut split_json = J::obj();
    for (a, h) in &splits {
        split_json.set(a.name(), *h);
    }
    Ok(obj! {
        "version" => 1usize,
        "heldout_method" => if flights.len() >= 2 { "leave one flight out".to_string() } else { format!("{HELDOUT_FOLDS} blocks of one flight") },
        "kept_s" => num::round_to(kept, 1),
        "drag_axes" => fit.drag.iter().map(|a| a.name()).collect::<Vec<_>>(),
        "axes" => axes,
        "noise_split_hz" => split_json,
        "turbulence" => turbulence(&spectra, &splits),
        "flights" => per_flight,
        "warnings" => warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welch_of_a_sine_has_its_power_at_its_frequency() {
        let fs = 1000.0;
        let x: Vec<f64> = (0..4000).map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin()).collect();
        let (f, p) = welch(&x, fs, 1000);
        let df = f[1] - f[0];
        let total: f64 = p.iter().sum::<f64>() * df;
        assert!((total - 0.5).abs() < 0.01, "{total}");
        let peak = p.iter().enumerate().fold(0, |b, (i, &v)| if v > p[b] { i } else { b });
        assert_eq!(f[peak], 50.0);
    }

    #[test]
    fn samples_are_distinct_and_sorted() {
        let s = sample(10_000, 1500, &mut Rng(0));
        assert_eq!(s.len(), 1500);
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(sample(5, 10, &mut Rng(0)), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn histogram_puts_the_upper_edge_in_the_last_bin() {
        let (e, c) = histogram(&[-1.0, 0.0, 1.0, 2.0], 4, 1.0);
        assert_eq!(e, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        assert_eq!(c, vec![1, 0, 1, 1]);
    }
}
