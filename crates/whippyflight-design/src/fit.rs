//! A plant fit (whippyflight-fit's coefficient JSON) and the airframe it was
//! logged on.

use whippyflight_sim::plant::Model;
use whippyunits::quantity;

use crate::units::*;

/// A fully charged cell.
pub const FULL_CELL: Voltage = quantity!(4.2, V);

/// Airframe drag: angular acceleration per low-passed body specific force.
#[derive(Clone, Copy, Debug, Default)]
pub struct Drag {
    /// The accelerometer low-pass; 0 with drag off.
    pub lp: Freq,
    pub roll_x: PerMeter,
    pub roll_y: PerMeter,
    pub pitch_x: PerMeter,
    pub pitch_y: PerMeter,
}

/// Which of the gyro's and the D-term's low-pass stages the fit recorded; the
/// model's defaults stand in for those it didn't.
#[derive(Clone, Debug, Default)]
pub struct Recorded {
    pub gyro_lpf: bool,
    pub dterm_lpf: bool,
    /// Each gyro and D-term low-pass stage's dynamic ceiling, where it has
    /// one; the model holds the stage at its floor.
    pub gyro_lpf_max: Vec<Option<Freq>>,
    pub dterm_lpf_max: Vec<Option<Freq>>,
}

pub struct Fit {
    pub model: Model,
    pub cells: f64,
    pub loop_rate: Freq,
    pub drag: Drag,
    pub recorded: Recorded,
}

impl Fit {
    pub fn from_json(text: &str, source: &str) -> Result<Fit, String> {
        let model = Model::from_json(text, source)?;
        let c: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{source}: {e}"))?;
        let airframe = &c["airframe"];
        if airframe.is_null() {
            return Err(format!("{source}: no airframe block; fit it again"));
        }
        let cells = airframe["cells"].as_f64().ok_or_else(|| format!("{source}: no airframe.cells"))?;
        let loop_hz = airframe["loop_hz"].as_f64().ok_or_else(|| format!("{source}: no airframe.loop_hz"))?;
        let legacy = c.get("frame").is_none();
        let axes: Vec<&str> = c["drag"]["axes"].as_array().map_or(Vec::new(), |a| a.iter().filter_map(|x| x.as_str()).collect());
        let drag = if axes.is_empty() {
            Drag::default()
        } else {
            // Legacy drag was fitted FLU on FLU: FRD negates roll←y and pitch←x.
            let term = |axis: &str, name: &str, flip: bool| {
                let v = if axes.contains(&axis) { c[axis][name].as_f64().unwrap_or(0.0) } else { 0.0 };
                PerMeter::new(if legacy && flip { -v } else { v })
            };
            Drag {
                lp: Freq::new(c["drag"]["lp_hz"].as_f64().ok_or_else(|| format!("{source}: no drag.lp_hz"))?),
                roll_x: term("roll", "accX", false),
                roll_y: term("roll", "accY", true),
                pitch_x: term("pitch", "accX", true),
                pitch_y: term("pitch", "accY", false),
            }
        };
        let maxima = |key: &str| -> Option<Vec<Option<Freq>>> {
            airframe[key].as_array().map(|stages| stages.iter().map(|s| s["max_hz"].as_f64().map(Freq::new)).collect())
        };
        let (gyro_lpf_max, dterm_lpf_max) = (maxima("gyro_lpf"), maxima("dterm_lpf"));
        let recorded = Recorded {
            gyro_lpf: gyro_lpf_max.is_some(),
            dterm_lpf: dterm_lpf_max.is_some(),
            gyro_lpf_max: gyro_lpf_max.unwrap_or_default(),
            dterm_lpf_max: dterm_lpf_max.unwrap_or_default(),
        };
        Ok(Fit { model, cells, loop_rate: Freq::new(loop_hz), drag, recorded })
    }

    pub fn load(path: &std::path::Path) -> Result<Fit, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&text, &path.display().to_string())
    }

    /// A full battery.
    pub fn full_battery(&self) -> Voltage {
        self.cells * FULL_CELL
    }

    /// The motor speed the motor fit settles at on `volts`.
    pub fn steady_speed(&self, volts: Voltage) -> Rate {
        let (kw, kq) = (self.model.motor_vel, self.model.motor_vel_sq);
        (-kw + (kw * kw + 4.0 * kq * volts).sqrt()) / (2.0 * kq)
    }

    /// Firmware motor i's slot.
    pub fn slot_of(&self, motor: usize) -> usize {
        self.model.layout.position[motor] as usize
    }
}
