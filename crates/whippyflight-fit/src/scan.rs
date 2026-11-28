//! The flight index: every log in a blackbox file, usable or not, with what
//! a user picking flights to fit wants to know: when, how long, how much of
//! it is airborne and impact-free, and how many seconds of each axis clear
//! the excitation gate (the rows a fit gets).

use std::path::Path;

use whippyunits::value;

use crate::bbl;
use crate::fit::Report;
use crate::flight::{self, Flight, MIN_DURATION};
use crate::json::J;
use crate::num::round_to;
use crate::obj;
use crate::rows::{self, Arms, Axis, Select};

/// Every log in the file at `path`.
pub fn scan_log(path: &Path) -> Result<Vec<J>, String> {
    let label = flight::file_label(path);
    let mut out = vec![];
    for (n, log) in bbl::read_logs(path)?.iter().enumerate().map(|(i, l)| (i + 1, l)) {
        let time = log.require("time")?;
        let duration = round_to((time[time.len() - 1] - time[0]) * 1e-6, 1);
        let mut entry = obj! {"index" => n, "duration_s" => duration};
        if let Some(start) = log.header_str("Log start datetime")
            && !start.is_empty()
            && !start.starts_with("0000")
        {
            entry.set("start", start);
        }
        if duration < value!(MIN_DURATION, s) {
            entry.set("reason", format!("shorter than {} s", value!(MIN_DURATION, s)));
            out.push(entry);
            continue;
        }
        if log.header_int("motor_poles").unwrap_or(0) < 2 {
            entry.set("reason", "the log doesn't record motor_poles");
            out.push(entry);
            continue;
        }
        let f = Flight::from_log(format!("{label}#{n}"), log, None)?;
        let hz = value!(f.log_hz, Hz);
        let mut excited = J::obj();
        for axis in [Axis::Yaw, Axis::Pitch, Axis::Roll] {
            let r = if axis == Axis::Yaw {
                rows::rows(&f, axis, &Arms::default(), flight::VoltsMap::R, &[], Select::default())
            } else {
                rows::split_rows(&f, axis, false, Select::default())
            };
            excited.set(axis.name(), round_to(r.len() as f64 / hz, 1));
        }
        let g = f.gating;
        entry.set("kept_s", round_to(value!(f.kept(), s), 1));
        entry.set("log_hz", round_to(hz, 1));
        entry.set("gating", obj! {"ground_s" => round_to(value!(g.ground, s), 1), "impacts" => g.impacts, "impact_s" => round_to(value!(g.impact, s), 1)});
        entry.set("excited_s", excited);
        entry.set("reason", f.unusable());
        out.push(entry);
    }
    Ok(out)
}

/// `index` (keyed by file name) with each of `paths` scanned into it; a file
/// that can't be read gets an `error` rather than sinking the others.
pub fn index(paths: &[&Path], mut index: J, report: &Report) -> Result<J, String> {
    if !matches!(index, J::Obj(_)) {
        index = J::obj();
    }
    let scanned: Vec<Result<Vec<J>, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = paths.iter().map(|&p| scope.spawn(move || scan_log(p))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("the scan panicked".into()))).collect()
    });
    for (path, result) in paths.iter().zip(scanned) {
        (report.check)()?;
        let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        let size = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
        let entry = match result {
            Ok(flights) => obj! {"size" => size, "flights" => J::Arr(flights)},
            Err(e) => obj! {"size" => size, "flights" => J::Arr(vec![]), "error" => e},
        };
        let (total, usable) = match entry.get("flights") {
            Some(J::Arr(f)) => (f.len(), f.iter().filter(|e| matches!(e.get("reason"), None | Some(J::Null))).count()),
            _ => (0, 0),
        };
        (report.say)(&format!("{name}: {total} log(s), {usable} usable"));
        index.set(&name, entry);
    }
    Ok(index)
}
