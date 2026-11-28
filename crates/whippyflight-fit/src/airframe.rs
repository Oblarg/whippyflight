//! The airframe a fit was logged on: from the log header (the settings the
//! gains will fly with) and the Betaflight CLI dump (the motor layout).

use std::collections::HashMap;
use std::path::Path;

use whippyunits::value;

use crate::flight::Flight;
use crate::json::J;
use crate::num;
use crate::obj;

/// The firmware's whippyflight_rpm_lpf_hz default, for logs that predate it
/// (stock BF logs lack it too): what the gains will fly with.
const RPM_LPF_HZ_DEFAULT: i64 = 50;
const RATE_KEYS: [&str; 4] = ["rc_rates", "rates", "rc_expo", "rate_limits"];

fn ints(h: &HashMap<String, String>, key: &str, default: &str) -> Vec<i64> {
    h.get(key).map_or(default, String::as_str).split(',').map(|x| x.trim().parse().unwrap_or(0)).collect()
}

fn int(h: &HashMap<String, String>, key: &str, default: i64) -> i64 {
    h.get(key).and_then(|v| v.split(',').next()?.trim().parse().ok()).unwrap_or(default)
}

fn lpf_type(code: i64) -> &'static str {
    match code {
        1 => "BIQUAD",
        2 => "PT2",
        3 => "PT3",
        _ => "PT1",
    }
}

/// BF's `prefix` ("gyro" or "dterm") low-pass stages from the log header, in
/// order, as {type, hz[, max_hz]}. LPF1 is dynamic (cutoff rising with
/// throttle from hz to max_hz) when its dynamic minimum is set, else static;
/// a 0 Hz stage is off. Notches and the RPM filter are not recorded.
pub fn lpf_stages(h: &HashMap<String, String>, prefix: &str) -> Vec<J> {
    let num = |key: &str, default: &str| ints(h, &format!("{prefix}_{key}"), default);
    let mut dyn_hz = num("lpf1_dyn_hz", "0,0");
    dyn_hz.resize(2.max(dyn_hz.len()), 0);
    let (dyn_lo, dyn_hi) = (dyn_hz[0], dyn_hz[1]);
    let lpf1 = num("lpf1_static_hz", "0")[0];
    let kind = lpf_type(num("lpf1_type", "0")[0]);
    let mut stages = vec![];
    if dyn_lo > 0 {
        stages.push(obj! {"type" => kind, "hz" => dyn_lo, "max_hz" => dyn_hi});
    } else if lpf1 > 0 {
        stages.push(obj! {"type" => kind, "hz" => lpf1});
    }
    let lpf2 = num("lpf2_static_hz", "0")[0];
    if lpf2 > 0 {
        stages.push(obj! {"type" => lpf_type(num("lpf2_type", "0")[0]), "hz" => lpf2});
    }
    stages
}

/// The gyro low-pass stages as (type, Hz), the sim's model of them.
pub fn gyro_stages(h: &HashMap<String, String>) -> Vec<(String, f64)> {
    lpf_stages(h, "gyro")
        .iter()
        .map(|s| {
            let J::Str(kind) = s.get("type").unwrap() else { unreachable!() };
            (kind.clone(), s.get("hz").and_then(J::as_f64).unwrap())
        })
        .collect()
}

/// The rate profile as BF logs it: `rates_type`, and [roll, pitch, yaw]
/// `rc_rates`, `rates` and `rc_expo` in hundredths and `rate_limits` in
/// deg/s. None if the header lacks any of them.
fn stick_rates(h: &HashMap<String, String>) -> J {
    if !RATE_KEYS.iter().all(|k| h.contains_key(*k)) {
        return J::Null;
    }
    let mut out = obj! {"rates_type" => int(h, "rates_type", 0)};
    for k in RATE_KEYS {
        out.set(k, ints(h, k, "").into_iter().take(3).collect::<Vec<_>>());
    }
    out
}

/// The airframe block from the flights' headers; None without any header.
pub fn airframe(flights: &[Flight]) -> Result<J, String> {
    let Some(h) = flights.iter().map(|f| &f.header).find(|h| !h.is_empty()) else {
        return Ok(J::obj());
    };
    let mut poles: Vec<u32> = flights.iter().map(|f| f.motor_poles).collect();
    poles.sort_unstable();
    poles.dedup();
    if poles.len() > 1 {
        return Err(format!(
            "the flights were logged with different motor_poles {poles:?}; fit them separately or pass --motor-poles"
        ));
    }
    let looptime = int(h, "looptime", 0);
    let denom = int(h, "pid_process_denom", 1);
    let mut vbat = vec![];
    let mut speed = vec![];
    for f in flights {
        for k in (0..f.len()).filter(|&k| f.good[k]) {
            vbat.push(value!(f.vbat[k], V));
            speed.push(f.speeds.iter().map(|w| value!(w[k], rot / s)).sum::<f64>() / 4.0);
        }
    }
    let cellv = ints(h, "vbatcellvoltage", "0");
    let vmax = flights.iter().flat_map(|f| f.vbat.iter().map(|&v| value!(v, V))).fold(f64::NEG_INFINITY, f64::max);
    let top = *cellv.last().unwrap_or(&0);
    let cells = (top > 0).then(|| (vmax / (top as f64 / 100.0)).ceil() as i64);
    let s = |k: &str| h.get(k).cloned().unwrap_or_default();
    let rx = int(h, "rc_smoothing_rx_smoothed", 0);
    Ok(obj! {
        "vbat_v" => num::round_to(num::median(&vbat), 3),
        "hover_rps" => num::round_to(num::median(&speed), 1),
        "cells" => cells,
        "craft" => s("Craft name"),
        "board" => s("Board information"),
        "firmware" => s("Firmware revision"),
        "motor_poles" => poles[0],
        "loop_hz" => (looptime > 0).then(|| num::round_to(1e6 / (looptime * denom) as f64, 1)),
        "rx_hz" => (rx != 0).then_some(rx),
        "vbatcellvoltage" => cellv,
        "gyro_lpf" => J::Arr(lpf_stages(h, "gyro")),
        "dterm_lpf" => J::Arr(lpf_stages(h, "dterm")),
        "rpm_lpf_hz" => int(h, "whippyflight_rpm_lpf_hz", RPM_LPF_HZ_DEFAULT),
        "brake_margin_pct" => int(h, "whippyflight_brake_margin", 0),
        "idle_rps" => int(h, "whippyflight_idle_rps", 0),
        "stick_rates" => stick_rates(h),
    })
}

/// Motor-layout facts from a Betaflight CLI `diff all` / `dump`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dump {
    pub mixer: Option<String>,
    pub mmix: Vec<Vec<f64>>,
    pub yaw_motors_reversed: bool,
    pub motor_poles: Option<u32>,
    pub name: Option<String>,
}

pub fn parse_dump(text: &str) -> Dump {
    let mut out = Dump::default();
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["mixer", m, ..] => out.mixer = Some(m.to_string()),
            ["mmix", rest @ ..] if rest.len() == 5 => {
                out.mmix.push(rest.iter().map(|w| w.parse().unwrap_or(f64::NAN)).collect());
            }
            ["#", "name:", rest @ ..] => out.name = Some(rest.join(" ")),
            ["set", "yaw_motors_reversed", _, v, ..] => out.yaw_motors_reversed = *v == "ON",
            ["set", "motor_poles", _, v, ..] => out.motor_poles = v.parse().ok(),
            _ => {}
        }
    }
    out
}

pub fn read_dump(path: &Path) -> Result<Dump, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(parse_dump(&String::from_utf8_lossy(&bytes)))
}

/// The regressors assume Betaflight's QUADX motor indices (0 rear-right,
/// 1 front-right, 2 rear-left, 3 front-left). Prop direction and gyro or
/// board alignment don't matter: BF aligns gyroADC before logging, and spin
/// direction only flips signs the fit absorbs. A different mixer or a custom
/// mmix reorders which indices pair up, which the model can't absorb.
pub fn check_layout(dump: &Dump) -> Result<(), String> {
    if dump.mixer.as_deref() != Some("QUADX") || !dump.mmix.is_empty() {
        let mmix = if dump.mmix.is_empty() { "reset".to_string() } else { format!("{:?}", dump.mmix) };
        return Err(format!(
            "unsupported motor layout: mixer {}, mmix {mmix}; the regressors assume QUADX motor indices",
            dump.mixer.as_deref().unwrap_or("None")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn low_pass_stages() {
        let h = header(&[("gyro_lpf1_dyn_hz", "250,500"), ("gyro_lpf1_type", "0"), ("gyro_lpf2_static_hz", "500"), ("gyro_lpf2_type", "2")]);
        let stages = lpf_stages(&h, "gyro");
        assert_eq!(stages, vec![obj! {"type" => "PT1", "hz" => 250i64, "max_hz" => 500i64}, obj! {"type" => "PT2", "hz" => 500i64}]);
        assert_eq!(gyro_stages(&h), vec![("PT1".to_string(), 250.0), ("PT2".to_string(), 500.0)]);
        assert!(lpf_stages(&header(&[]), "dterm").is_empty());
    }

    #[test]
    fn dump_layout() {
        let d = parse_dump("# name: bluebird\nmixer QUADX\nset yaw_motors_reversed = ON\nset motor_poles = 12\n");
        assert_eq!(d.name.as_deref(), Some("bluebird"));
        assert!(d.yaw_motors_reversed && d.motor_poles == Some(12));
        assert!(check_layout(&d).is_ok());
        assert!(check_layout(&parse_dump("mixer QUADX\nmmix 0 1 -1 1 -1\n")).is_err());
        assert!(check_layout(&parse_dump("mixer HEX6\n")).is_err());
    }
}
