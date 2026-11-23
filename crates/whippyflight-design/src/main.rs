//! whippyflight-design gen FIT.json OUT.wfg [options]
//! whippyflight-design front FIT.json OUT_DIR [options]
//!
//! Options (rot/s, rot, s, rev/s):
//!   --sigma-rate R --sigma-int I        pitch/roll excursions
//!   --yaw-sigma-rate R --yaw-sigma-int I
//!   --tau-int S --dt S --ff-horizon S
//!   --no-gyro-noise                     the LQR ignores the fit's gyro noise
//!   --resolution N --idle-rps W --hover-only (gen)
//!   --yaw --rate-levels N --int-levels N (front)

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use whippyflight_design::blob::{self, Design, GridSpec};
use whippyflight_design::fit::Fit;
use whippyflight_design::front::{self, FrontSpec};
use whippyflight_design::hover::HoverLoop;
use whippyflight_design::lqr::{Excursions, Group, Weights};
use whippyflight_design::par::Control;
use whippyflight_design::plant;
use whippyunits::{quantity, value};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [command, fit_path, out, rest @ ..] = args.as_slice() else {
        return Err("usage: whippyflight-design gen|front FIT.json OUT [options]".into());
    };
    let mut flags = HashMap::new();
    let mut switches = Vec::new();
    let mut it = rest.iter();
    while let Some(flag) = it.next() {
        let name = flag.strip_prefix("--").ok_or_else(|| format!("unexpected argument {flag}"))?;
        if name == "hover-only" || name == "yaw" || name == "no-gyro-noise" {
            switches.push(name.to_string());
            continue;
        }
        let value: f64 = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))?;
        flags.insert(name.to_string(), value);
    }
    let get = |name: &str, default: f64| flags.get(name).copied().unwrap_or(default);
    let fit_path = std::path::PathBuf::from(fit_path);
    let fit_dir = fit_path.parent().ok_or("the fit has no directory")?;
    let fit = Fit::load(&fit_path)?;
    let pitch_roll = Excursions {
        rate: quantity!(get("sigma-rate", 5.0), rot / s),
        int: quantity!(get("sigma-int", 0.06), rot),
    };
    let yaw = Excursions {
        rate: flags.get("yaw-sigma-rate").map_or(pitch_roll.rate, |&r| quantity!(r, rot / s)),
        int: flags.get("yaw-sigma-int").map_or(pitch_roll.int, |&i| quantity!(i, rot)),
    };
    let weights = Weights { pitch_roll, yaw, tau_int: quantity!(get("tau-int", 5.0), s) };
    let dt = flags.get("dt").map_or(1.0 / fit.loop_rate, |&dt| quantity!(dt, s));
    let ff_horizon = flags.get("ff-horizon").map_or(blob::DEFAULT_FF_HORIZON, |&h| quantity!(h, s));
    let grid = GridSpec { resolution: get("resolution", 3.0) as usize, idle: quantity!(get("idle-rps", 0.0), rot / s) };
    let cancel = AtomicBool::new(false);
    let start = std::time::Instant::now();
    let say = |line: &str| eprintln!("{:7.1} {line}", start.elapsed().as_secs_f64());
    let control = Control { cancel: &cancel, progress: &say };
    let gyro_noise = !switches.iter().any(|s| s == "no-gyro-noise");
    match command.as_str() {
        "gen" => {
            let mut lp = HoverLoop::new(&fit, fit_dir, dt)?;
            if !gyro_noise {
                lp.disturbances = lp.disturbances.without_gyro_noise();
            }
            lp.flies(&weights)?;
            let design = Design {
                dt,
                weights,
                ff_horizon,
                disturbances: lp.disturbances,
                grid,
                hover_only: switches.iter().any(|s| s == "hover-only"),
            };
            let t = std::time::Instant::now();
            let (spec, tau_deriv) = blob::generate(&fit, &design, &control)?;
            for delay in plant::derivative_delays(&fit)? {
                say(&format!("τ_deriv: {}", delay.describe()));
            }
            let bytes = spec.encode();
            std::fs::write(out, &bytes).map_err(|e| format!("{out}: {e}"))?;
            eprintln!(
                "Wrote {} bytes to {out} in {:.2?}: τ_deriv {:.3} ms",
                bytes.len(),
                t.elapsed(),
                value!(tau_deriv, ms)
            );
            Ok(())
        }
        "front" => {
            let group = if switches.iter().any(|s| s == "yaw") { Group::Yaw } else { Group::PitchRoll };
            let spec = FrontSpec {
                dt,
                weights,
                ff_horizon,
                grid,
                rate_levels: get("rate-levels", 6.0) as usize,
                int_levels: get("int-levels", 7.0) as usize,
                gyro_noise,
            };
            front::explore(&fit, fit_dir, group, &spec, std::path::Path::new(out), &control)?;
            Ok(())
        }
        _ => Err(format!("unknown command {command}")),
    }
}
