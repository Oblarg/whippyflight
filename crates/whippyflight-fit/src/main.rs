//! whippyflight-fit: the WhippyFlight plant fit from Betaflight blackbox logs.
//!
//!   fit LOG... -o COEFFICIENTS.json [--diagnostics DIAG.json] [--dump CLI.txt] [--drag auto|on|off]
//!   validate COEFFICIENTS.json LOG...      per-flight R², the given fit's and each flight's own
//!   crossval LOG... [--split N] [-o OUT.json] [--dump CLI.txt] [--drag auto|on|off]
//!   flights FILE... [-o INDEX.json]        index every log in each file (merged into -o)
//!   noise LOG OUT.bin [--json INFO.json] [--fit COEFFICIENTS.json --diagnostics DIAG.json]
//!       [--seconds 3] [--hover-rps W] [--max-stick 0.55] [--gyro-hp 80] [--erpm-hp 40]
//!   speeds LOG... -o SPEEDS.bin            gated motor speeds, for plot_occupancy
//!
//! Common: [--volts r|glue] [--motor-poles N] [--d-front A --d-back B] [--d-left A --d-right B].
//! LOG is a blackbox file (every log in it) or FILE#N (its N-th log, 1-based).

use std::path::{Path, PathBuf};

use whippyflight_fit::fit::{self, DragMode, Report, Stored};
use whippyflight_fit::flight::{Flight, VoltsMap};
use whippyflight_fit::json::J;
use whippyflight_fit::num;
use whippyflight_fit::rows::{AXES, Arms, Axis, RATE_AXES, pairing_check};
use whippyflight_fit::{airframe, frame, noise, occupancy, run, scan};
use whippyunits::quantity;

fn main() {
    if let Err(e) = cli() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

struct Args {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Args, String> {
        let mut out = Args { positional: vec![], flags: vec![] };
        let mut it = raw.iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--").or_else(|| a.strip_prefix('-').filter(|n| n.len() == 1)) {
                let v = it.next().ok_or_else(|| format!("{a} needs a value"))?;
                out.flags.push((name.to_string(), v.clone()));
            } else {
                out.positional.push(a.clone());
            }
        }
        Ok(out)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.iter().rev().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    fn output(&self) -> Option<&Path> {
        self.get("o").or(self.get("output")).map(Path::new)
    }

    fn num(&self, name: &str) -> Result<Option<f64>, String> {
        self.get(name).map(|v| v.parse().map_err(|e| format!("--{name}: {e}"))).transpose()
    }

    fn check(&self, known: &[&str]) -> Result<(), String> {
        match self.flags.iter().find(|(k, _)| !known.contains(&k.as_str())) {
            Some((k, _)) => Err(format!("unknown option --{k}")),
            None => Ok(()),
        }
    }

    fn logs(&self) -> Result<&[String], String> {
        if self.positional.is_empty() { Err("no logs given".into()) } else { Ok(&self.positional) }
    }
}

fn options(args: &Args) -> Result<run::Options, String> {
    Ok(run::Options {
        d_front: args.num("d-front")?,
        d_back: args.num("d-back")?,
        d_left: args.num("d-left")?,
        d_right: args.num("d-right")?,
        map: args.get("volts").map_or(Ok(VoltsMap::R), VoltsMap::parse)?,
        drag: args.get("drag").map_or(Ok(DragMode::Auto), DragMode::parse)?,
        motor_poles: args.num("motor-poles")?.map(|p| p as u32),
        dump: args.get("dump").map(PathBuf::from),
    })
}

const COMMON: [&str; 6] = ["volts", "motor-poles", "d-front", "d-back", "d-left", "d-right"];

fn known(extra: &[&'static str]) -> Vec<&'static str> {
    COMMON.iter().chain(extra).copied().collect()
}

fn cli() -> Result<(), String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let Some((command, rest)) = raw.split_first() else {
        return Err("usage: whippyflight-fit fit|validate|crossval|flights ... (see the crate docs)".into());
    };
    let args = Args::parse(rest)?;
    let say = |line: &str| eprintln!("{line}");
    let check = || Ok(());
    let report = Report { say: &say, check: &check };
    match command.as_str() {
        "fit" => {
            args.check(&known(&["o", "output", "dump", "diagnostics", "drag"]))?;
            let out = run::fit_logs(args.logs()?, &options(&args)?, &report)?;
            if let Some(path) = args.output() {
                run::write_json(path, &out.coefficients, &report)?;
            }
            if let Some(path) = args.get("diagnostics") {
                run::write_json(path.as_ref(), &out.diagnostics(&report)?, &report)?;
            }
            Ok(())
        }
        "validate" => {
            args.check(&known(&[]))?;
            let (coefficients, logs) = args.positional.split_first().ok_or("validate COEFFICIENTS.json LOG...")?;
            if logs.is_empty() {
                return Err("no logs given".into());
            }
            validate(Path::new(coefficients), logs, &options(&args)?, &report)
        }
        "crossval" => {
            args.check(&known(&["o", "output", "dump", "drag", "split"]))?;
            let parts = args.num("split")?.map_or(1, |n| n as usize);
            crossval(args.logs()?, parts, args.output(), &options(&args)?, &report)
        }
        "flights" => {
            args.check(&["o", "output"])?;
            let paths: Vec<&Path> = args.logs()?.iter().map(Path::new).collect();
            let existing = match args.output() {
                Some(p) if p.exists() => read_json(p)?,
                _ => J::obj(),
            };
            let index = scan::index(&paths, existing, &report)?;
            match args.output() {
                Some(p) => run::write_json(p, &index, &report),
                None => {
                    print!("{}", index.pretty());
                    Ok(())
                }
            }
        }
        "noise" => {
            args.check(&["json", "fit", "diagnostics", "seconds", "hover-rps", "max-stick", "gyro-hp", "erpm-hp", "motor-poles"])?;
            let [log, out] = args.positional.as_slice() else {
                return Err("noise LOG OUT.bin [--json INFO.json] [--fit COEFFICIENTS.json --diagnostics DIAG.json]".into());
            };
            noise(log, Path::new(out), &args, &report)
        }
        "speeds" => {
            args.check(&["o", "output", "motor-poles"])?;
            let out = args.output().ok_or("speeds LOG... -o SPEEDS.bin")?;
            let flights = run::load_all(args.logs()?, args.num("motor-poles")?.map(|p| p as u32), &report)?;
            if flights.is_empty() {
                return Err("no usable flights".into());
            }
            std::fs::write(out, occupancy::speeds(&flights)).map_err(|e| format!("{}: {e}", out.display()))?;
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        _ => Err(format!("unknown command {command}")),
    }
}

fn noise(log: &str, out: &Path, args: &Args, report: &Report) -> Result<(), String> {
    let defaults = noise::Options::default();
    let opts = noise::Options {
        seconds: args.num("seconds")?.map_or(defaults.seconds, |v| quantity!(v, s)),
        hover: args.num("hover-rps")?.map(|v| quantity!(v, rot / s)),
        max_stick: args.num("max-stick")?.map_or(defaults.max_stick, |v| quantity!(v, rot / s)),
        gyro_hp: args.num("gyro-hp")?.map_or(defaults.gyro_hp, |v| quantity!(v, Hz)),
        erpm_hp: args.num("erpm-hp")?.map_or(defaults.erpm_hp, |v| quantity!(v, Hz)),
    };
    let mut skipped = vec![];
    let flights = whippyflight_fit::flight::load_flights(log, args.num("motor-poles")?.map(|p| p as u32), &mut skipped)?;
    for line in skipped {
        (report.say)(&line);
    }
    let flight = flights.first().ok_or_else(|| format!("no usable flight in {log}"))?;
    let fit = args.get("fit").map(|p| Stored::from_regressor_frame(&frame::regressor_frame(&read_json(Path::new(p))?)?)).transpose()?;
    let diagnostics = args.get("diagnostics").map(|p| read_json(Path::new(p))).transpose()?;
    let splits = diagnostics.as_ref().and_then(|d| d.get("noise_split_hz"));
    let noise = noise::extract(flight, fit.as_ref(), splits, &opts)?;
    for line in noise.summary() {
        println!("{line}");
    }
    std::fs::write(out, noise.bytes()).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("wrote {}", out.display());
    if let Some(path) = args.get("json") {
        run::write_json(path.as_ref(), &noise.info(log), report)?;
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<J, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(J::from_value(&v))
}

/// A flight's own pooled fit, scored on it.
fn own_r2(flight: &Flight, given: &Arms, map: VoltsMap, drag: &[Axis]) -> Result<(Stored, Vec<(Axis, f64)>), String> {
    let own = Stored::from(&fit::fit(std::slice::from_ref(flight), given, map, drag, &Report::quiet())?);
    let r2 = fit::evaluate(&own, flight)?;
    Ok((own, r2))
}

/// A flight's label, its R² per axis under a fit, and under its own.
type R2Row = (String, Vec<(Axis, f64)>, Vec<(Axis, f64)>);

fn print_r2_table(rows: &[R2Row], label: &str) {
    let axes: Vec<String> = AXES.iter().map(|a| format!("{:>15}", a.name())).collect();
    println!("{:30} {}", "flight", axes.join(" "));
    let heads: Vec<String> = AXES.iter().map(|_| format!("{:>15}", format!("{label} / own"))).collect();
    println!("{:30} {}", "", heads.join(" "));
    for (name, given, own) in rows {
        let cells: Vec<String> = given.iter().zip(own).map(|((_, g), (_, o))| format!("{g:7.3} /{o:6.3}")).collect();
        println!("{name:30} {}", cells.join(" "));
    }
}

fn validate(path: &Path, logs: &[String], opts: &run::Options, report: &Report) -> Result<(), String> {
    let coeffs = read_json(path)?;
    if !matches!(coeffs.get("motor_speed"), Some(J::Str(u)) if u == fit::MOTOR_SPEED_UNIT) {
        return Err(format!("{} was fitted before motor speeds were in {}; fit it again", path.display(), fit::MOTOR_SPEED_UNIT));
    }
    let mut stored = Stored::from_regressor_frame(&frame::regressor_frame(&coeffs)?)?;
    let given = opts.given_arms();
    let a = &mut stored.arms;
    a.d_front = given.d_front.or(a.d_front);
    a.d_back = given.d_back.or(a.d_back);
    a.d_left = given.d_left.or(a.d_left);
    a.d_right = given.d_right.or(a.d_right);
    let mut rows = vec![];
    for f in run::load_all(logs, opts.motor_poles, report)? {
        let (_, own) = own_r2(&f, &Arms::default(), stored.map, &stored.drag)?;
        rows.push((f.label.clone(), fit::evaluate(&stored, &f)?, own));
    }
    print_r2_table(&rows, "given");
    Ok(())
}

/// Leave-one-out within an airframe: each flight scored by a fit to all the
/// others, next to its own best fit; then per-flight coefficient spread and
/// the motor-pairing check.
fn crossval(logs: &[String], parts: usize, output: Option<&Path>, opts: &run::Options, report: &Report) -> Result<(), String> {
    let mut flights = run::load_all(logs, opts.motor_poles, report)?;
    if parts > 1 {
        flights = flights.iter().flat_map(|f| f.split(parts)).collect();
    }
    if flights.len() < 2 {
        return Err("crossval needs at least two usable flights (try --split)".into());
    }
    if let Some(d) = &opts.dump {
        airframe::check_layout(&airframe::read_dump(d)?)?;
    }
    let given = opts.given_arms();
    let (drag, _) = fit::choose_drag(&flights, opts.drag, &given)?;
    let mut rows = vec![];
    let mut singles = vec![];
    for (i, f) in flights.iter().enumerate() {
        let others: Vec<Flight> = flights.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, g)| g.clone()).collect();
        let loo = Stored::from(&fit::fit(&others, &given, opts.map, &drag, &Report::quiet())?);
        let (own, own_r2) = own_r2(f, &given, opts.map, &drag)?;
        singles.push(frame::canonical(&own.json)?);
        rows.push((f.label.clone(), fit::evaluate(&loo, f)?, own_r2));
    }
    print_r2_table(&rows, "loo");

    println!("\nper-flight fits (canonical frame): median, and spread as (p90 − p10)/|median|");
    for axis in AXES {
        let Some(J::Obj(block)) = singles[0].get(axis.name()) else { continue };
        for (name, _) in block {
            let vals: Vec<f64> = singles.iter().filter_map(|s| s.get(axis.name()).and_then(|b| b.get(name)).and_then(J::as_f64)).collect();
            let med = num::median(&vals);
            let spread = if med != 0.0 { (num::percentile(&vals, 90.0) - num::percentile(&vals, 10.0)) / med.abs() } else { f64::NAN };
            println!("  {:7} {name:13} {}  spread {:6.1}%", axis.name(), signed_g6(med), spread * 100.0);
        }
    }

    println!("\npairing check (R² of each axis under each grouping; own grouping should win):");
    for axis in RATE_AXES {
        let mut bad = vec![];
        for f in &flights {
            let checks = pairing_check(f);
            let (_, scores) = checks.iter().find(|(a, _)| *a == axis).unwrap();
            let best = scores.iter().fold(&scores[0], |b, x| if x.1 > b.1 { x } else { b }).0;
            if best != axis {
                let shown: Vec<String> = scores.iter().map(|(a, v)| format!("{}: {:.3}", a.name(), v)).collect();
                bad.push(format!("\n      {}: best {} {{{}}}", f.label, best.name(), shown.join(", ")));
            }
        }
        println!("  {:5}: {}/{} flights best fit by their own grouping{}", axis.name(), flights.len() - bad.len(), flights.len(), bad.concat());
    }

    println!("\npooled:");
    let out = run::fit_flights(flights, opts, report)?;
    if let Some(path) = output {
        run::write_json(path, &out.coefficients, report)?;
    }
    Ok(())
}

fn signed_g6(x: f64) -> String {
    let s = fit::g6(x);
    if s.starts_with('-') { s } else { format!("+{s}") }
}
