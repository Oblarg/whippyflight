//! The fit's frames. The regressions run in Betaflight's motor order in the
//! physical frame turned 180° about z, drag on BF's FLU accelerometer; the
//! files hold whippyflight's canonical frame (physical arms, FRD drag, and a
//! `frame` block of BF's QUADX motors with the dump's yaw_motors_reversed).

use crate::json::J;
use crate::obj;

/// Betaflight's QUADX mixer (mixerQuadX), by motor index, and its default
/// (props-in) spins; yaw_motors_reversed flips every spin.
pub const BETAFLIGHT_QUADX: [&str; 4] = ["rear_right", "front_right", "rear_left", "front_left"];
const BETAFLIGHT_PROPS_IN: [&str; 4] = ["cw", "ccw", "ccw", "cw"];

/// Drag terms whose sign differs between BF's FLU axes and FRD.
const FLU_DRAG: [(&str, &str); 2] = [("roll", "accY"), ("pitch", "accX")];
/// Arm pairs, which swap between the frames.
const ARM_PAIRS: [(&str, &str, &str); 2] = [("pitch", "dFront", "dBack"), ("roll", "dLeft", "dRight")];

/// The `frame` block of a Betaflight QUADX airframe.
pub fn betaflight_frame(airframe: &J) -> Result<J, String> {
    let mixer = match airframe.get("mixer") {
        Some(J::Str(m)) => m.as_str(),
        _ => "QUADX",
    };
    if mixer != "QUADX" {
        return Err(format!("mixer {mixer}: only QUADX is supported"));
    }
    let reversed = matches!(airframe.get("yaw_motors_reversed"), Some(J::Bool(true)));
    let motors = BETAFLIGHT_QUADX
        .iter()
        .zip(BETAFLIGHT_PROPS_IN)
        .map(|(&p, s)| {
            let spin = if reversed { if s == "cw" { "ccw" } else { "cw" } } else { s };
            obj! {"position" => p, "spin" => spin}
        })
        .collect::<Vec<_>>();
    Ok(obj! {"axes" => "FRD", "motors" => J::Arr(motors)})
}

fn swap_arms(c: &mut J) {
    for (axis, a, b) in ARM_PAIRS {
        let Some(block) = c.get_mut(axis) else { continue };
        if let (Some(va), Some(vb)) = (block.get(a).cloned(), block.get(b).cloned()) {
            block.set(a, vb);
            block.set(b, va);
        }
    }
}

fn flip_drag(c: &mut J) {
    for (axis, term) in FLU_DRAG {
        if let Some(block) = c.get_mut(axis)
            && let Some(v) = block.get(term).and_then(J::as_f64)
        {
            block.set(term, -v);
        }
    }
}

/// A fit in the regressors' frame as the canonical one (what files hold).
pub fn canonical(fit: &J) -> Result<J, String> {
    let mut out = fit.clone();
    swap_arms(&mut out);
    flip_drag(&mut out);
    let frame = betaflight_frame(out.get("airframe").unwrap_or(&J::Null))?;
    out.set("frame", frame);
    Ok(out)
}

/// A canonical coefficient file in the regressors' frame: the inverse of
/// [`canonical`]. Legacy fits (no `frame`) are already in it.
pub fn regressor_frame(coeffs: &J) -> Result<J, String> {
    let Some(frame) = coeffs.get("frame") else {
        return Ok(coeffs.clone());
    };
    let positions: Vec<&str> = match frame.get("motors") {
        Some(J::Arr(m)) => m.iter().filter_map(|m| if let Some(J::Str(p)) = m.get("position") { Some(p.as_str()) } else { None }).collect(),
        _ => vec![],
    };
    if positions != BETAFLIGHT_QUADX {
        return Err(format!("motors {positions:?} are not Betaflight's QUADX order {BETAFLIGHT_QUADX:?}"));
    }
    if !matches!(frame.get("axes"), Some(J::Str(a)) if a == "FRD") {
        return Err("frame.axes is not FRD".into());
    }
    let mut out = coeffs.clone();
    if let J::Obj(pairs) = &mut out {
        pairs.retain(|(k, _)| k != "frame");
    }
    swap_arms(&mut out);
    flip_drag(&mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_round_trips() {
        let fit = obj! {
            "pitch" => obj! {"pitchRate" => 1.0, "accX" => 2.0, "dFront" => 1.0, "dBack" => 1.2},
            "roll" => obj! {"rollRate" => 1.0, "accY" => 3.0, "dLeft" => 1.0, "dRight" => 0.9},
            "airframe" => obj! {"mixer" => "QUADX", "yaw_motors_reversed" => true},
        };
        let c = canonical(&fit).unwrap();
        assert_eq!(c.get("pitch").unwrap().get("dFront").unwrap().as_f64(), Some(1.2));
        assert_eq!(c.get("pitch").unwrap().get("accX").unwrap().as_f64(), Some(-2.0));
        assert_eq!(c.get("roll").unwrap().get("accY").unwrap().as_f64(), Some(-3.0));
        let spins: Vec<J> = match c.get("frame").unwrap().get("motors").unwrap() {
            J::Arr(m) => m.iter().map(|m| m.get("spin").unwrap().clone()).collect(),
            _ => panic!(),
        };
        assert_eq!(spins, vec![J::from("ccw"), J::from("cw"), J::from("cw"), J::from("ccw")]);
        assert_eq!(regressor_frame(&c).unwrap(), fit);
    }
}
