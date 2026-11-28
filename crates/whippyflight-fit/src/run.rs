//! The fit as the CLI and the Configurator run it: flight specs in, the
//! canonical coefficient JSON (and its diagnostics) out.

use std::path::{Path, PathBuf};

use whippyunits::value;

use crate::airframe::{self, Dump};
use crate::fit::{self, DragMode, Fitted, Report};
use crate::flight::{self, Flight, VoltsMap};
use crate::frame;
use crate::json::J;
use crate::rows::{Arms, Axis};

/// What to fit and how.
#[derive(Clone, Debug)]
pub struct Options {
    /// Physical arms to pin (both of an axis), else fitted.
    pub d_front: Option<f64>,
    pub d_back: Option<f64>,
    pub d_left: Option<f64>,
    pub d_right: Option<f64>,
    pub map: VoltsMap,
    pub drag: DragMode,
    /// Overrides the logs' motor_poles.
    pub motor_poles: Option<u32>,
    /// A Betaflight CLI dump: checks the motor layout and records it.
    pub dump: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Options { d_front: None, d_back: None, d_left: None, d_right: None, map: VoltsMap::R, drag: DragMode::Auto, motor_poles: None, dump: None }
    }
}

impl Options {
    /// The pinned arms in the regressors' frame (turned 180° about z).
    pub fn given_arms(&self) -> Arms {
        Arms { d_front: self.d_back, d_back: self.d_front, d_left: self.d_right, d_right: self.d_left }
    }
}

/// Every flight spec's usable flights, decoded in parallel, in order; what
/// was skipped is reported.
pub fn load_all(specs: &[String], motor_poles: Option<u32>, report: &Report) -> Result<Vec<Flight>, String> {
    type Loaded = Result<(Vec<Flight>, Vec<String>), String>;
    let results: Vec<Loaded> = std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .iter()
            .map(|spec| {
                scope.spawn(move || {
                    let mut skipped = vec![];
                    flight::load_flights(spec, motor_poles, &mut skipped).map(|f| (f, skipped))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("a log decoder panicked".into()))).collect()
    });
    let mut flights = vec![];
    for r in results {
        let (f, skipped) = r?;
        for line in skipped {
            (report.say)(&line);
        }
        flights.extend(f);
    }
    for f in &flights {
        let g = f.gating;
        (report.say)(&format!(
            "{}: {:.0}/{:.0} s kept; ground {:.1} s, {} impact(s) masking {:.1} s",
            f.label,
            value!(f.kept(), s),
            value!(f.duration(), s),
            value!(g.ground, s),
            g.impacts,
            value!(g.impact, s)
        ));
    }
    Ok(flights)
}

/// A fit's results.
pub struct Output {
    /// The coefficient JSON, canonical frame.
    pub coefficients: J,
    pub fitted: Fitted,
    pub flights: Vec<Flight>,
}

fn file_name(p: &Path) -> String {
    p.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned())
}

/// Fit flights already loaded.
pub fn fit_flights(flights: Vec<Flight>, opts: &Options, report: &Report) -> Result<Output, String> {
    if flights.is_empty() {
        return Err("no usable flights".into());
    }
    let dump: Option<Dump> = opts.dump.as_deref().map(airframe::read_dump).transpose()?;
    if let Some(d) = &dump {
        airframe::check_layout(d)?;
    }
    let given = opts.given_arms();
    let (drag, evidence) = fit::choose_drag(&flights, opts.drag, &given)?;
    let names: Vec<&str> = drag.iter().map(|a: &Axis| a.name()).collect();
    (report.say)(&format!(
        "airframe drag: {} (--drag {}{})",
        if names.is_empty() { "off".to_string() } else { names.join(", ") },
        opts.drag.name(),
        evidence.as_ref().map_or(String::new(), |e| format!("; {}", e.to_value()))
    ));
    let mut fitted = fit::fit(&flights, &given, opts.map, &drag, report)?;
    let out = &mut fitted.json;
    if let Some(e) = evidence {
        out.get_mut("drag").unwrap().set("selection", e);
    }
    let mut air = airframe::airframe(&flights)?;
    let poles = air.get("motor_poles").and_then(J::as_f64).map(|p| p as u32);
    if let (Some(d), Some(p)) = (&dump, poles)
        && let Some(dp) = d.motor_poles
        && dp != p
    {
        (report.say)(&format!(
            "warning: {} has motor_poles {dp}, the fit {p}; the flight controller must be set to the fit's",
            file_name(opts.dump.as_deref().unwrap())
        ));
    }
    match (&dump, &opts.dump) {
        (Some(d), Some(path)) => {
            air.set("mixer", d.mixer.clone());
            air.set("yaw_motors_reversed", d.yaw_motors_reversed);
            air.set("dump", file_name(path));
        }
        _ => (report.say)(
            "warning: no --dump; the fit records props in (yaw_motors_reversed OFF), and the firmware refuses its gains if the quad is props out",
        ),
    }
    out.set("airframe", air);
    let coefficients = frame::canonical(out)?;
    Ok(Output { coefficients, fitted, flights })
}

/// Decode and fit flight specs.
pub fn fit_logs(specs: &[String], opts: &Options, report: &Report) -> Result<Output, String> {
    let flights = load_all(specs, opts.motor_poles, report)?;
    fit_flights(flights, opts, report)
}

impl Output {
    /// The fit's diagnostics over its flights.
    pub fn diagnostics(&self, report: &Report) -> Result<J, String> {
        crate::diagnostics::diagnostics(&self.flights, &fit::Stored::from(&self.fitted), report)
    }
}

/// Write `json` to `path`, reporting it.
pub fn write_json(path: &Path, json: &J, report: &Report) -> Result<(), String> {
    std::fs::write(path, json.pretty()).map_err(|e| format!("{}: {e}", path.display()))?;
    (report.say)(&format!("wrote {}", path.display()));
    Ok(())
}
