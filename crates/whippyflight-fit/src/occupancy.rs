//! Flights' motor speeds over their airborne, impact-free samples, for
//! whippyflight_betaflight.plot_occupancy.
//!
//! File format, little-endian: `u32` byte length of the flights' labels,
//! the labels as UTF-8 joined by newlines, `u32` sample count, then per
//! sample `f32` flight time (s) and `f32` motor speeds [0..4] by BF motor
//! index (shaft rev/s).

use whippyunits::value;

use crate::flight::Flight;

/// The speed samples of `flights`, pooled.
pub fn speeds(flights: &[Flight]) -> Vec<u8> {
    let labels = flights.iter().map(|f| f.label.as_str()).collect::<Vec<_>>().join("\n");
    let rows: usize = flights.iter().map(|f| f.good.iter().filter(|&&g| g).count()).sum();
    let mut out = Vec::with_capacity(8 + labels.len() + 20 * rows);
    out.extend((labels.len() as u32).to_le_bytes());
    out.extend(labels.as_bytes());
    out.extend((rows as u32).to_le_bytes());
    for f in flights {
        let dt = (1.0 / value!(f.log_hz, Hz)) as f32;
        for k in (0..f.len()).filter(|&k| f.good[k]) {
            out.extend(dt.to_le_bytes());
            for m in 0..4 {
                out.extend((value!(f.speeds[m][k], rot / s) as f32).to_le_bytes());
            }
        }
    }
    out
}
