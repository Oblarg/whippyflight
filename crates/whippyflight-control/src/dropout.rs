//! Smoothing of RX dropouts, ahead of the setpoint estimator.
//!
//! A packet that follows k missed ones may be far from the last: the stick
//! moved through frames that never arrived.  Rather than step there, the
//! rate setpoint ramps to it over the k missed periods, capped at the
//! configured limit:
//!
//! ```text
//!   packet after k missed:  δ = out − r,   ramp = min(k·T, cap)
//!   each packet at t:       out = r + δ·(1 − (t − t_dropout)/ramp)⁺
//! ```
//!
//! Packets arriving on schedule pass through unchanged.

use crate::setpoint::elapsed;
use crate::types::*;

pub struct DropoutRamp {
    cap: Seconds,
    seeded: bool,
    packet_us: Micros,
    /// The rate the last packet put out.
    out: [RotPerSec; 3],
    /// What the ramp still adds to the incoming rate at its start.
    offset: [RotPerSec; 3],
    start_us: Micros,
    ramp: Seconds,
}

impl DropoutRamp {
    pub const fn new() -> Self {
        Self {
            cap: Seconds::new(0.0),
            seeded: false,
            packet_us: Micros::new(0),
            out: [RotPerSec::new(0.0); 3],
            offset: [RotPerSec::new(0.0); 3],
            start_us: Micros::new(0),
            ramp: Seconds::new(0.0),
        }
    }

    /// The longest ramp after a dropout; 0 steps to the new packet.
    pub fn set_cap(&mut self, cap: Seconds) {
        self.cap = cap;
    }

    /// An RX packet's rate setpoint, its arrival time, and the RX period:
    /// the rate to hand the estimator.
    pub fn packet(&mut self, rate: [RotPerSec; 3], time: Micros, period: Seconds) -> [RotPerSec; 3] {
        let gap = elapsed(self.packet_us, time);
        self.packet_us = time;
        if !self.seeded || period <= Seconds::new(0.0) {
            self.seeded = true;
            self.ramp = Seconds::new(0.0);
            self.out = rate;
            return rate;
        }
        let missed = (libm::roundf(f32::from(gap / period)) - 1.0).max(0.0);
        if missed >= 1.0 {
            self.ramp = (missed * period).min(self.cap);
            self.start_us = time;
            self.offset = core::array::from_fn(|i| self.out[i] - rate[i]);
        }
        let left = if self.ramp > Seconds::new(0.0) {
            (1.0 - f32::from(elapsed(self.start_us, time) / self.ramp)).max(0.0)
        } else {
            0.0
        };
        self.out = core::array::from_fn(|i| rate[i] + left * self.offset[i]);
        self.out
    }
}

impl Default for DropoutRamp {
    fn default() -> Self {
        Self::new()
    }
}
