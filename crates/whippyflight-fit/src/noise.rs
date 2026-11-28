//! Sensor noise from a flight, as the recording whippyflight-sim's evals
//! replay (`noise.bin`; the format is whippyflight_sim::noise's).
//!
//! Noise is what the controller would see on top of the true motion: BF's
//! filtered gyro (gyroADC, after its filters), and the eRPM telemetry in
//! shaft rev/s. With a fit, the motion its model explains is taken out
//! first: the body rates its angular accelerations integrate to, and the
//! motor speeds its motor model reaches from the logged voltages. What
//! remains is zero-phase high-passed at each channel's noise split from the
//! fit's diagnostics (above it the fit's residual is the sensors', below it
//! the airframe's), or without them at the given cut-offs. Content below the
//! cut-offs is not represented. With a fit it also measures how the
//! motor-speed noise grows with the motor's speed over the whole flight
//! (`speed_exponent`), for the sim to scale each motor's replay by
//! (speed / the window's)^exponent.
//!
//! The window is the stretch of flight (sticks within `max_stick`, all
//! motors above idle) whose mean motor speed is closest to `hover` (default:
//! the median of those samples).
//!
//! Also BF's own motor-voltage roughness over that window, computed like the
//! evals' `roughness_v` (one-pole high-pass at ROUGH_HP), as a budget
//! reference: that much ripple is known to fly on this airframe.

use std::f64::consts::PI;

use whippyflight_sim::noise::MAX_SPEED_EXPONENT;
use whippyunits::{quantity, value};

use crate::fit::Stored;
use crate::flight::Flight;
use crate::json::J;
use crate::num;
use crate::obj;
use crate::rows::{self, Axis, Select};
use crate::units::{Freq, Rate, Time};

const GYRO_LSB: Rate = quantity!(1e-5, rot / s);
const ERPM_LSB: Rate = quantity!(1e-3, rot / s);
const ROUGH_HP: Freq = quantity!(50.0, Hz);
const LAW_BLOCK: Time = quantity!(0.1, s);
const LAW_TRIM: Time = quantity!(0.2, s);
const WINDOW_PAD: Time = quantity!(0.2, s);
/// The window's motors: all above this fraction of the median spinning speed.
const IDLE_FRACTION: f64 = 0.4;
const DSHOT_MAX: f64 = 2047.0;
/// The gyro's axes in its (and the frames') order: BF's FLU roll, pitch, yaw.
const GYRO_AXES: [Axis; 3] = [Axis::Roll, Axis::Pitch, Axis::Yaw];
const FRD_FROM_FLU: [f64; 3] = [1.0, -1.0, -1.0];

/// How to pick the window and cut the noise.
#[derive(Clone, Debug)]
pub struct Options {
    pub seconds: Time,
    pub hover: Option<Rate>,
    pub max_stick: Rate,
    /// Cut-offs where the diagnostics have no noise split.
    pub gyro_hp: Freq,
    pub erpm_hp: Freq,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            seconds: quantity!(3.0, s),
            hover: None,
            max_stick: quantity!(0.55, rot / s),
            gyro_hp: quantity!(80.0, Hz),
            erpm_hp: quantity!(40.0, Hz),
        }
    }
}

/// How motor-speed noise grows with speed: (reference speed, exponent).
#[derive(Clone, Copy, Debug)]
pub struct SpeedLaw {
    pub reference: Rate,
    pub exponent: f64,
}

/// An extracted recording.
pub struct Noise {
    pub fs: Freq,
    /// Gyro [roll, pitch, yaw] FRD in GYRO_LSB, motors 0..3 in ERPM_LSB.
    pub frames: Vec<[i16; 7]>,
    pub window: (Time, Time),
    pub gyro_hp: [Freq; 3],
    pub erpm_hp: Freq,
    pub fitted: bool,
    pub gyro_rms: [Rate; 3],
    pub erpm_rms: [Rate; 4],
    pub mean_speed: Rate,
    pub mean_vbat: f64,
    pub bf_rough_v: f64,
    pub speed_law: Option<SpeedLaw>,
}

/// A 4th-order Butterworth high-pass as two biquads: [b0, b1, b2, a1, a2].
fn butter4_highpass(fc: f64, fs: f64) -> [[f64; 5]; 2] {
    let k = (PI * fc / fs).tan();
    [1.0 / (2.0 * (PI / 8.0).cos()), 1.0 / (2.0 * (3.0 * PI / 8.0).cos())].map(|q| {
        let norm = 1.0 / (1.0 + k / q + k * k);
        [norm, -2.0 * norm, norm, 2.0 * (k * k - 1.0) * norm, (1.0 - k / q + k * k) * norm]
    })
}

/// A biquad's state for a constant input of 1 at steady state (scipy's lfilter_zi).
fn biquad_zi(s: &[f64; 5]) -> [f64; 2] {
    let [b0, b1, b2, a1, a2] = *s;
    let (r0, r1) = (b1 - a1 * b0, b2 - a2 * b0);
    // [[1 + a1, -1], [a2, 1]] z = r
    let det = 1.0 + a1 + a2;
    [(r0 + r1) / det, (r1 * (1.0 + a1) - a2 * r0) / det]
}

fn sosfilt(sos: &[[f64; 5]], x: &mut [f64], x0: f64) {
    let mut scale = 1.0;
    for s in sos {
        let [b0, b1, b2, a1, a2] = *s;
        let zi = biquad_zi(s);
        let (mut z0, mut z1) = (zi[0] * scale * x0, zi[1] * scale * x0);
        scale *= (b0 + b1 + b2) / (1.0 + a1 + a2);
        for v in x.iter_mut() {
            let y = b0 * *v + z0;
            z0 = b1 * *v - a1 * y + z1;
            z1 = b2 * *v - a2 * y;
            *v = y;
        }
    }
}

/// scipy's sosfiltfilt: zero-phase, odd extension of 3·(2·sections + 1) samples.
pub fn sosfiltfilt(sos: &[[f64; 5]], x: &[f64]) -> Vec<f64> {
    let pad = 3 * (2 * sos.len() + 1);
    let n = x.len();
    if n <= pad {
        return x.to_vec();
    }
    let mut ext = Vec::with_capacity(n + 2 * pad);
    ext.extend((1..=pad).rev().map(|i| 2.0 * x[0] - x[i]));
    ext.extend_from_slice(x);
    ext.extend((1..=pad).map(|i| 2.0 * x[n - 1] - x[n - 1 - i]));
    let first = ext[0];
    sosfilt(sos, &mut ext, first);
    ext.reverse();
    let first = ext[0];
    sosfilt(sos, &mut ext, first);
    ext.reverse();
    ext[pad..pad + n].to_vec()
}

fn highpass(x: &[f64], fc: Freq, fs: Freq) -> Vec<f64> {
    sosfiltfilt(&butter4_highpass(value!(fc, Hz), value!(fs, Hz)), x)
}

/// x minus its one-pole low-pass at `fc`, from rest at zero (the evals' roughness filter).
fn one_pole_hp(x: &[f64], fc: Freq, fs: Freq) -> Vec<f64> {
    let k = 2.0 * PI * f64::from(fc / fs);
    let k = k / (k + 1.0);
    let mut lp = 0.0;
    x.iter()
        .map(|&v| {
            lp = k * v + (1.0 - k) * lp;
            v - lp
        })
        .collect()
}

fn rot_s(x: &[Rate]) -> Vec<f64> {
    x.iter().map(|&w| value!(w, rot / s)).collect()
}

/// The motion `fit`'s model explains over samples [lo, hi): body rates
/// (BF's FLU) integrated from the angular accelerations its modeled speeds
/// drive as the airframe feels them, and motor speeds as the telemetry
/// reports them. None where a row is unusable.
/// Body rates [roll, pitch, yaw] and motor speeds, rot/s.
type Motion = ([Vec<f64>; 3], [Vec<f64>; 4]);

fn explained(flight: &Flight, fit: &Stored, lo: usize, hi: usize) -> Result<Option<Motion>, String> {
    let clean = flight.with_speeds(fit.motors.speeds(flight, fit.map, fit.actuator_delay));
    let t: Vec<f64> = flight.t[lo..hi].iter().map(|&t| value!(t, s)).collect();
    let mut rates: [Vec<f64>; 3] = Default::default();
    for (a, &axis) in GYRO_AXES.iter().enumerate() {
        let r = rows::rows(&clean, axis, &fit.arms, fit.map, &fit.drag, Select { gated: false, window: None });
        let pred = &r.x * fit.beta(axis, &r.names)?;
        let mut alpha = vec![f64::NAN; flight.len()];
        for (&c, &p) in r.center.iter().zip(pred.iter()) {
            alpha[c] = p;
        }
        let alpha = &alpha[lo..hi];
        if alpha.iter().any(|v| v.is_nan()) {
            return Ok(None);
        }
        let mut acc = 0.0;
        rates[a] = std::iter::once(0.0)
            .chain((1..alpha.len()).map(|k| {
                acc += 0.5 * (alpha[k] + alpha[k - 1]) * (t[k] - t[k - 1]);
                acc
            }))
            .collect();
    }
    let reported = fit.motors.speeds(flight, fit.map, fit.telemetry_delay);
    Ok(Some((rates, reported.map(|w| rot_s(&w[lo..hi])))))
}

/// How the motor-speed noise grows with the motor's speed over the whole
/// flight: the slope of log RMS against log mean speed, per motor, over
/// LAW_BLOCK blocks of every usable stretch, the noise taken as in the
/// window (the motor model's speeds out, high-passed at `hp`). Blocks below
/// a tenth of `reference` are left out; None without enough spread.
fn speed_exponent(flight: &Flight, fit: &Stored, fs: Freq, hp: Freq, reference: Rate) -> Option<f64> {
    let model = fit.motors.speeds(flight, fit.map, fit.telemetry_delay);
    let erpm: [Vec<f64>; 4] = flight.speeds.clone().map(|w| rot_s(&w));
    let sos = butter4_highpass(value!(hp, Hz), value!(fs, Hz));
    let (block, trim) = (f64::from(LAW_BLOCK * fs) as usize, f64::from(LAW_TRIM * fs) as usize);
    let (mut speed, mut rms) = (vec![], vec![]);
    for (s, e) in crate::flight::runs(&flight.good) {
        if e - s < 2 * trim + block + 30 {
            continue;
        }
        let noise: Vec<Vec<f64>> = (0..4)
            .map(|m| {
                let d: Vec<f64> = (s..e).map(|k| erpm[m][k] - value!(model[m][k], rot / s)).collect();
                sosfiltfilt(&sos, &d)
            })
            .collect();
        let mut b = trim;
        while b + block + trim <= e - s {
            for m in 0..4 {
                speed.push(num::mean(&erpm[m][s + b..s + b + block]));
                rms.push(num::std(&noise[m][b..b + block]));
            }
            b += block;
        }
    }
    let r = value!(reference, rot / s);
    let (xs, ys): (Vec<f64>, Vec<f64>) =
        speed.iter().zip(&rms).filter(|&(&w, &n)| w > 0.1 * r && n > 0.0).map(|(w, n)| (w.ln(), n.ln())).unzip();
    if xs.len() < 20 {
        return None;
    }
    let kept: Vec<f64> = xs.iter().map(|x| x.exp()).collect();
    if num::percentile(&kept, 90.0) < 1.5 * num::percentile(&kept, 10.0) {
        return None;
    }
    let (mx, my) = (num::mean(&xs), num::mean(&ys));
    let (sxy, sxx) = xs.iter().zip(&ys).fold((0.0, 0.0), |(a, b), (x, y)| (a + (x - mx) * (y - my), b + (x - mx) * (x - mx)));
    Some((sxy / sxx).clamp(0.0, MAX_SPEED_EXPONENT))
}

/// The noise of `flight`, its fit's motion taken out if given (regression
/// frame), high-passed at `splits` (the diagnostics' noise_split_hz) where
/// they have a channel.
pub fn extract(flight: &Flight, fit: Option<&Stored>, splits: Option<&J>, opts: &Options) -> Result<Noise, String> {
    let setpoint = flight.setpoint.as_ref().ok_or("the log has no setpoint")?;
    let secs: Vec<f64> = flight.t.iter().map(|&t| value!(t, s)).collect();
    let steps: Vec<f64> = secs.windows(2).map(|w| w[1] - w[0]).collect();
    let fs: Freq = 1.0 / quantity!(num::median(&steps), s);
    let n = f64::from(opts.seconds * fs) as usize;
    let len = flight.len();
    if n == 0 || n > len {
        return Err(format!("the flight is shorter than {:.1} s", value!(opts.seconds, s)));
    }
    let erpm: [Vec<f64>; 4] = flight.speeds.clone().map(|w| rot_s(&w));
    let mean_speed: Vec<f64> = (0..len).map(|k| (0..4).map(|m| erpm[m][k]).sum::<f64>() / 4.0).collect();
    let spinning: Vec<f64> = mean_speed.iter().copied().filter(|&w| w > 0.0).collect();
    let idle = IDLE_FRACTION * num::median(&spinning);
    let stick = value!(opts.max_stick, rot / s);
    let ok: Vec<bool> = (0..len)
        .map(|k| (0..4).all(|m| erpm[m][k] > idle) && (0..3).all(|a| value!(setpoint[a][k], rot / s).abs() < stick))
        .collect();
    let (mut okc, mut sum) = (vec![0usize; len + 1], vec![0.0; len + 1]);
    for k in 0..len {
        okc[k + 1] = okc[k] + usize::from(ok[k]);
        sum[k + 1] = sum[k] + mean_speed[k];
    }
    let window_mean = |k: usize| (sum[k + n] - sum[k]) / n as f64;
    let starts: Vec<usize> = (0..=len - n).filter(|&k| okc[k + n] - okc[k] == n).collect();
    if starts.is_empty() {
        return Err(format!("no {:.1} s window meets the criteria", value!(opts.seconds, s)));
    }
    let hover = match opts.hover {
        Some(h) => value!(h, rot / s),
        None => num::median(&(0..len).filter(|&k| ok[k]).map(|k| mean_speed[k]).collect::<Vec<_>>()),
    };
    let a = starts.iter().copied().fold(starts[0], |b, k| if (window_mean(k) - hover).abs() < (window_mean(b) - hover).abs() { k } else { b });
    let pad = f64::from(WINDOW_PAD * fs) as usize;
    let (lo, hi) = (a.saturating_sub(pad), (a + n + pad).min(len));

    let split = |key: &str, default: Freq| {
        splits.and_then(|s| s.get(key)).and_then(J::as_f64).filter(|&h| h != 0.0).map_or(default, |h| quantity!(h, Hz))
    };
    let gyro_hp = GYRO_AXES.map(|a| split(a.name(), opts.gyro_hp));
    let erpm_hp = split("motors", opts.erpm_hp);
    let mut gyro: [Vec<f64>; 3] = core::array::from_fn(|i| rot_s(&flight.rates[i][lo..hi]));
    let mut motors: [Vec<f64>; 4] = core::array::from_fn(|m| erpm[m][lo..hi].to_vec());
    let mut speed_law = None;
    if let Some(fit) = fit {
        let reference = quantity!(window_mean(a), rot / s);
        speed_law = speed_exponent(flight, fit, fs, erpm_hp, reference).map(|exponent| SpeedLaw { reference, exponent });
        let (rates, speeds) = explained(flight, fit, lo, hi)?.ok_or("the window has rows the fit can't use; pick another flight")?;
        for (g, r) in gyro.iter_mut().zip(&rates) {
            g.iter_mut().zip(r).for_each(|(g, r)| *g -= r);
        }
        for (w, m) in motors.iter_mut().zip(&speeds) {
            w.iter_mut().zip(m).for_each(|(w, m)| *w -= m);
        }
    }
    let window = |x: Vec<f64>| x[a - lo..a - lo + n].to_vec();
    let gyro: [Vec<f64>; 3] = core::array::from_fn(|i| window(highpass(&gyro[i], gyro_hp[i], fs)).into_iter().map(|v| v * FRD_FROM_FLU[i]).collect());
    let motors: [Vec<f64>; 4] = core::array::from_fn(|m| window(highpass(&motors[m], erpm_hp, fs)));
    let (glsb, elsb) = (value!(GYRO_LSB, rot / s), value!(ERPM_LSB, rot / s));
    let mut frames = Vec::with_capacity(n);
    for k in 0..n {
        let mut f = [0i16; 7];
        for c in 0..7 {
            let v = if c < 3 { gyro[c][k] / glsb } else { motors[c - 3][k] / elsb };
            let v = v.round_ties_even();
            if v.abs() > 32767.0 {
                return Err("noise exceeds the i16 range".into());
            }
            f[c] = v as i16;
        }
        frames.push(f);
    }
    let volts: Vec<Vec<f64>> =
        (0..4).map(|m| (lo..hi).map(|k| flight.commands[m][k] / DSHOT_MAX * value!(flight.vbat[k], V)).collect()).collect();
    let rough_sq: f64 = volts.iter().map(|v| window(one_pole_hp(v, ROUGH_HP, fs)).iter().map(|r| r * r).sum::<f64>()).sum();
    Ok(Noise {
        fs,
        frames,
        window: (flight.t[a], flight.t[a + n - 1]),
        gyro_hp,
        erpm_hp,
        fitted: fit.is_some(),
        gyro_rms: core::array::from_fn(|i| quantity!(num::std(&gyro[i]), rot / s)),
        erpm_rms: core::array::from_fn(|m| quantity!(num::std(&motors[m]), rot / s)),
        mean_speed: quantity!(window_mean(a), rot / s),
        mean_vbat: num::mean(&flight.vbat[a..a + n].iter().map(|&v| value!(v, V)).collect::<Vec<_>>()),
        bf_rough_v: (rough_sq / (4 * n) as f64).sqrt(),
        speed_law,
    })
}

fn g(x: f64) -> String {
    crate::fit::g6(x)
}

impl Noise {
    /// The recording as whippyflight_sim::noise reads it.
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 14 * self.frames.len());
        out.extend((value!(self.fs, Hz) as f32).to_le_bytes());
        out.extend((self.frames.len() as u32).to_le_bytes());
        for f in &self.frames {
            for v in f {
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    /// noise.json: the window, BF's roughness and the speed law, for the
    /// Configurator and the sim.
    pub fn info(&self, log: &str) -> J {
        obj! {
            "log" => log,
            "window_s" => vec![value!(self.window.0, s), value!(self.window.1, s)],
            "bf_rough_v" => self.bf_rough_v,
            "motor_noise_speed" => self.speed_law.map(|l| obj! {"reference_rps" => value!(l.reference, rot / s), "exponent" => l.exponent}),
        }
    }

    /// What the extraction found, as lines.
    pub fn summary(&self) -> Vec<String> {
        let hp: Vec<String> = self.gyro_hp.iter().map(|&h| g(value!(h, Hz))).collect();
        let n = self.frames.len();
        let mut out = vec![
            format!(
                "high-passed at gyro {} Hz, motor speed {} Hz{}",
                hp.join("/"),
                g(value!(self.erpm_hp, Hz)),
                if self.fitted { ", the fit's motion taken out" } else { "" }
            ),
            format!(
                "window {:.2}–{:.2} s at {:.0} Hz, mean motor speed {:.0} rev/s, vbat {:.2} V",
                value!(self.window.0, s),
                value!(self.window.1, s),
                value!(self.fs, Hz),
                value!(self.mean_speed, rot / s),
                self.mean_vbat
            ),
            format!("gyro noise rms (rot/s) roll/pitch/yaw: {}", self.gyro_rms.map(|r| format!("{:.5}", value!(r, rot / s))).join(" ")),
            format!("motor speed noise rms (rev/s) motors 0-3: {}", self.erpm_rms.map(|r| format!("{:.3}", value!(r, rot / s))).join(" ")),
        ];
        if let Some(l) = self.speed_law {
            out.push(format!("motor speed noise grows as speed^{:.2} about {:.0} rev/s", l.exponent, value!(l.reference, rot / s)));
        }
        out.push(format!(
            "BF roughness_v (rms of {:.0} Hz high-passed motor volts, mean over motors): {:.4} V",
            value!(ROUGH_HP, Hz),
            self.bf_rough_v
        ));
        out.push(format!("{} frames, {} bytes", n, 8 + 14 * n));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highpass_blocks_dc_and_passes_fast_content() {
        let fs = 1600.0;
        let x: Vec<f64> = (0..3200).map(|i| 3.0 + (2.0 * PI * 400.0 * i as f64 / fs).sin()).collect();
        let y = sosfiltfilt(&butter4_highpass(20.0, fs), &x);
        let mid = &y[800..2400];
        assert!(num::mean(mid).abs() < 1e-3);
        assert!((num::std(mid) - 0.5f64.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn steady_state_states_leave_a_constant_unchanged() {
        let sos = butter4_highpass(30.0, 1600.0);
        let lp = [[0.25, 0.5, 0.25, -0.2, 0.1]];
        let mut x = vec![2.0; 50];
        sosfilt(&lp, &mut x, 2.0);
        let gain = 1.0 / (1.0 - 0.2 + 0.1);
        assert!(x.iter().all(|&v| (v - 2.0 * gain).abs() < 1e-12));
        let mut x = vec![2.0; 50];
        sosfilt(&sos, &mut x, 2.0);
        assert!(x.iter().all(|&v| v.abs() < 1e-12));
    }
}
