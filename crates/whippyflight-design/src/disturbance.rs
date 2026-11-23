//! What disturbs the loop, as a fit measured it: turbulence, the one-pole
//! torque spectrum per axis its diagnostics estimate, and gyro noise, its
//! recording's (whippyflight-sim noise.rs) as an AR(2) process per axis at
//! the loop rate. Model order throughout.

use std::path::Path;

use whippyflight_sim::noise::Noise;
use whippyunits::quantity;

use crate::units::*;

/// The fit's diagnostics, beside its coefficients.
pub const DIAGNOSTICS: &str = "diagnostics.json";

/// One-pole turbulence on an axis: white angular acceleration low-passed at
/// `corner` to an RMS of `rms`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turbulence {
    pub rms: AngAccel,
    pub corner: Freq,
}

/// Noise n[k] = a[0]·n[k−1] + a[1]·n[k−2] + e[k], e white of RMS `innovation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Autoregression {
    pub a: [f64; 2],
    pub innovation: Rate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disturbances {
    pub turbulence: [Turbulence; 3],
    pub gyro: [Autoregression; 3],
}

const RATE: Rate = quantity!(1.0, rot / s);
const AXES: [&str; 3] = ["yaw", "pitch", "roll"];

impl Disturbances {
    /// The disturbances measured beside the fit in `fit_dir`, the noise
    /// sampled every `dt`.
    pub fn load(fit_dir: &Path, dt: Time) -> Result<Self, String> {
        let path = fit_dir.join(DIAGNOSTICS);
        let text = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let diag: serde_json::Value = serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut turbulence = [Turbulence { rms: quantity!(0.0, rot / s ^ 2), corner: quantity!(0.0, Hz) }; 3];
        for (t, axis) in turbulence.iter_mut().zip(AXES) {
            let spec = &diag["turbulence"][axis];
            let (Some(rms), Some(hz)) = (spec["rms"].as_f64(), spec["hz"].as_f64()) else {
                return Err(format!("the fit measured no {axis} turbulence; fit it again from flights with more hover"));
            };
            if !(rms > 0.0 && hz > 0.0) {
                return Err(format!("the fit's {axis} turbulence is degenerate: {rms} rot/s² at {hz} Hz"));
            }
            *t = Turbulence { rms: quantity!(rms, rot / s ^ 2), corner: quantity!(hz, Hz) };
        }
        let noise = Noise::load_fit(fit_dir)?
            .0
            .ok_or("the fit has no recorded sensor noise; extract it from one of its flights")?;
        let ticks = f64::from(noise.duration() / dt) as usize;
        let mut series = [vec![0.0; ticks], vec![0.0; ticks], vec![0.0; ticks]];
        for k in 0..ticks {
            let frd = noise.at(k as f64 * dt, [quantity!(0.0, rot / s).lossy_into(); 4]).gyro;
            for (a, s) in series.iter_mut().enumerate() {
                let r: Rate = frd[2 - a].lossy_into();
                s[k] = f64::from(r / RATE);
            }
        }
        let mut gyro = [Autoregression { a: [0.0; 2], innovation: quantity!(0.0, rot / s) }; 3];
        for ((g, s), axis) in gyro.iter_mut().zip(&series).zip(AXES) {
            *g = yule_walker(s).ok_or_else(|| format!("the recorded {axis} gyro noise is too short or flat to model"))?;
        }
        Ok(Disturbances { turbulence, gyro })
    }

    /// These disturbances with no gyro noise.
    pub fn without_gyro_noise(self) -> Self {
        let gyro = self.gyro.map(|g| Autoregression { innovation: quantity!(0.0, rot / s), ..g });
        Disturbances { gyro, ..self }
    }
}

/// The AR(2) of `x` (rot/s) by the Yule–Walker equations on its biased
/// autocovariance, which keep it stationary.
fn yule_walker(x: &[f64]) -> Option<Autoregression> {
    let n = x.len();
    if n < 3 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let r = |lag: usize| x[lag..].iter().zip(x).map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>() / n as f64;
    let (r0, r1, r2) = (r(0), r(1), r(2));
    let det = r0 * r0 - r1 * r1;
    if !(r0 > 0.0 && det > 0.0) {
        return None;
    }
    let a = [(r1 * r0 - r1 * r2) / det, (r0 * r2 - r1 * r1) / det];
    let innovation = r0 - a[0] * r1 - a[1] * r2;
    (innovation > 0.0).then(|| Autoregression { a, innovation: innovation.sqrt() * RATE })
}
