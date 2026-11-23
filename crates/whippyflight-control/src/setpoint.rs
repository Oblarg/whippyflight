//! RX setpoint estimator for the reference feedforward.
//!
//! The feedforward needs a per-tick rate setpoint, but the stick is only
//! sampled at the RX packet rate.  Each axis follows the uniform quadratic
//! B-spline whose control points are the packets, over the packet period T:
//!
//! ```text
//!   tick at t, u = (t − t_n)/T clamped to [0, 1]:
//!     r(t) = ½(1−u)²·r_{n−2} + (½ + u − u²)·r_{n−1} + ½u²·r_n
//!     r_target = r(t)                                (slew- and jerk-limited)
//! ```
//!
//! Each segment runs between packet midpoints, so r(t) trails the stick by
//! 1.5 T, and its slope is continuous across packets.  r(t) never leaves the
//! range of the three packets it blends, so a stick that stops is never
//! passed: extrapolating it (an α-β tracker) bought back the lag on ramps
//! but ran past every stop, and holding or linearly interpolating packets
//! steps the slope at each one, which the jerk limit turns into voltage
//! roughness.
//!
//! The control points are slots one period apart.  A period that passes
//! without a packet fills its slot with the last value, as a repeated
//! packet would; a packet within half a period of that slot takes it over.
//! So the curve keeps its time base through a late or missing packet
//! instead of stalling and catching up.  An early packet starts the next
//! segment where this one would have ended, a jump of at most the rest of
//! it.
//!
//! The target follows its goal with bounded slope (angular acceleration)
//! and bounded change of slope (jerk), braking so it stops on the goal
//! instead of past it.  Stops plan with a fraction of the jerk bound
//! ([`SetpointEstimator::set_brake_margin`]) where the airframe brakes worse
//! than its model.  Both bounds sit inside what the airframe can do:
//! the feedforward only inverts the next h ticks, so a target that needs
//! more acceleration — or a faster stop — than the motors have saturates
//! them, and the motors' spin-down lag then carries the airframe past it.
//!
//! The bounds are fixed, though, and what the motors can do isn't (battery
//! sag, throttle, a heavier airframe).  So whenever allocation saturates,
//! the target is conditioned: pulled back to what the delivered voltage
//! reaches over the horizon ([`SetpointEstimator::condition`]).  Otherwise
//! it runs ahead, the feedforward spends the motors' authority catching up
//! with it, and the motor-speed difference built doing so carries the
//! airframe past the stop.  Unsaturated, conditioning does nothing.

use crate::gains::GainStorage;
use crate::types::*;

use whippyunits::api::rescale_f32;
use whippyunits::{quantity, value};

/// Target slew limit (an angular-acceleration bound), used when the gain
/// blob carries none.  `gen_wfg` sets the blob's from the airframe:
/// 0.83 × the weakest pitch/roll direction's peak angular acceleration
/// (opposing motors at full battery and idle, from hover), so it only clips
/// targets the airframe cannot follow.  This is that value for the legacy
/// 1S regression airframe (weakest 120 rot/s²).
pub const DEFAULT_TARGET_SLEW_RPS2: RotPerSecSq = RotPerSecSq::new(100.0);
/// Target jerk limit, used when the gain blob carries none.  The blob's is
/// 0.63 × the weakest pitch/roll jerk authority at hover (motor slew ×
/// thrust slope); the fraction was tuned with
/// `tests/eval_feedforward.rs` (1–4 rot/s flicks within 3% overshoot) on the
/// legacy airframe (8000 rot/s³), which this default is.
pub const DEFAULT_TARGET_JERK_RPS3: RotPerSecCu = RotPerSecCu::new(5000.0);

/// Largest [`SetpointEstimator::set_brake_margin`]: a stop needs some jerk.
pub const MAX_BRAKE_MARGIN: f32 = 0.9;

/// Past this many RX periods without a packet the next packet reseeds the
/// estimate (link loss, not a drop).
const MAX_HOLD_PERIODS: u8 = 3;

pub struct SetpointEstimator {
    max_slew: RotPerSecSq,
    max_jerk: RotPerSecCu,
    /// Fraction of `max_jerk` stops plan with.
    brake: f32,
    period_s: Seconds,
    /// The last three slots per axis, oldest first.
    packets: [[RotPerSec; 3]; 3],
    packet_us: Micros,
    /// Slots since the last packet that it filled.
    held: u8,
    /// The last packet's rate, as received.
    raw: [RotPerSec; 3],
    seeded: bool,
    target: [RotPerSec; 3],
    target_valid: bool,
    /// d(target)/dt.
    target_v: [RotPerSecSq; 3],
}

impl SetpointEstimator {
    pub const fn new() -> Self {
        Self {
            max_slew: DEFAULT_TARGET_SLEW_RPS2,
            max_jerk: DEFAULT_TARGET_JERK_RPS3,
            brake: 1.0,
            period_s: Seconds::new(0.0),
            packets: [[RotPerSec::new(0.0); 3]; 3],
            packet_us: Micros::new(0),
            held: 0,
            raw: [RotPerSec::new(0.0); 3],
            seeded: false,
            target: [RotPerSec::new(0.0); 3],
            target_valid: false,
            target_v: [RotPerSecSq::new(0.0); 3],
        }
    }

    /// The target's slew and jerk limits.  A jerk limit ≤ 0 leaves only the
    /// slew limit.
    pub fn configure(&mut self, max_target_slew: RotPerSecSq, max_target_jerk: RotPerSecCu) {
        self.max_slew = max_target_slew;
        self.max_jerk = max_target_jerk;
    }

    /// The gain blob's target limits, falling back to the defaults for any
    /// the blob doesn't set.
    pub fn configure_for(&mut self, gains: &GainStorage) {
        let (slew, jerk) = target_limits(gains);
        self.configure(slew, jerk);
    }

    /// Hold back `margin` (0..1) of the jerk limit when easing the target's
    /// slope off, so it starts onto its goal earlier and lands more gently;
    /// it still leaves at the full limit.  Out-of-range values clamp.
    pub fn set_brake_margin(&mut self, margin: f32) {
        self.brake = 1.0 - margin.clamp(0.0, MAX_BRAKE_MARGIN);
    }

    /// A new RX packet: raw (unsmoothed) setpoint [roll, pitch, yaw], its
    /// arrival time, and the current RX period estimate.
    pub fn packet(&mut self, rate: [RotPerSec; 3], time: Micros, rx_period: Seconds) {
        self.raw = rate;
        if rx_period <= Seconds::new(0.0) {
            return;
        }
        let gap = elapsed(self.packet_us, time);
        let held = self.held as f32 * self.period_s;
        if !self.seeded || gap > MAX_HOLD_PERIODS as f32 * self.period_s {
            self.packets = [rate; 3];
            self.seeded = true;
        } else if self.held > 0 && gap - held < 0.5 * self.period_s {
            self.packets[2] = rate;
        } else {
            self.packets = [self.packets[1], self.packets[2], rate];
        }
        self.period_s = rx_period;
        self.packet_us = time;
        self.held = 0;
    }

    /// The last packet's rate setpoint, as received.
    pub fn raw(&self) -> [RotPerSec; 3] {
        self.raw
    }

    /// Per-tick setpoint r(t) and slew/jerk-limited feedforward target.
    pub fn tick(&mut self, now: Micros, dt: Seconds) -> ([RotPerSec; 3], [RotPerSec; 3]) {
        if !self.seeded {
            return Default::default();
        }
        let since: f32 = (elapsed(self.packet_us, now) / self.period_s).into();
        while since - self.held as f32 >= 1.0 && self.held < MAX_HOLD_PERIODS {
            self.packets = [self.packets[1], self.packets[2], self.packets[2]];
            self.held += 1;
        }
        let u = (since - self.held as f32).clamp(0.0, 1.0);
        let [old, mid, new] = self.packets;
        let mut rate = [RotPerSec::new(0.0); 3];
        let mut target = [RotPerSec::new(0.0); 3];
        let step = self.max_slew * dt;
        for i in 0..3 {
            rate[i] = 0.5 * (1.0 - u) * (1.0 - u) * old[i] + (0.5 + u - u * u) * mid[i] + 0.5 * u * u * new[i];
            let wanted = rate[i];
            target[i] = if !self.target_valid {
                wanted
            } else if self.max_jerk > RotPerSecCu::new(0.0) {
                // Head for `wanted` at the slope from which the jerk limit
                // still brakes to a stop on it; land on it once reached.
                // Every jump of the goal goes through this, small ones too:
                // the feedforward inverts the target over h ticks, so a
                // target whose slope steps turns into a voltage spike.
                let err = wanted - self.target[i];
                let brake = (2.0 * self.brake * self.max_jerk * err.abs()).sqrt().clamp(RotPerSecSq::new(0.0), self.max_slew);
                let dv = self.max_jerk * dt;
                let ease = self.brake * dv;
                let toward = if err < RotPerSec::new(0.0) { -brake } else { brake };
                let (lo, hi) = if self.target_v[i] > RotPerSecSq::new(0.0) { (-ease, dv) } else { (-dv, ease) };
                self.target_v[i] += (toward - self.target_v[i]).clamp(lo, hi);
                let next = self.target[i] + self.target_v[i] * dt;
                if (next - wanted) * (self.target[i] - wanted) <= RotSqPerSecSq::new(0.0) {
                    let slope = ((u - 1.0) * old[i] + (1.0 - 2.0 * u) * mid[i] + u * new[i]) / self.period_s;
                    self.target_v[i] = slope.clamp(-self.target_v[i].abs(), self.target_v[i].abs());
                    wanted
                } else {
                    next
                }
            } else {
                self.target[i] + (wanted - self.target[i]).clamp(-step, step)
            };
        }
        self.target = target;
        self.target_valid = true;
        (rate, target)
    }

    /// Pull the target back to what the motors can reach
    /// ([`ControllerState::reachable_target`]), on each axis where it has
    /// run ahead of it in its direction of travel.  Call after every
    /// controller update; `None` (not saturated) leaves the target alone.
    ///
    /// [`ControllerState::reachable_target`]: crate::controller::ControllerState::reachable_target
    pub fn condition(&mut self, reachable: Option<[RotPerSec; 3]>) {
        let Some(reachable) = reachable else { return };
        if !self.target_valid {
            return;
        }
        let zero = RotPerSecSq::new(0.0);
        for ((target, v), reach) in self.target.iter_mut().zip(self.target_v).zip(reachable) {
            let ahead = *target - reach;
            if (v > zero && ahead > RotPerSec::new(0.0)) || (v < zero && ahead < RotPerSec::new(0.0)) {
                *target = reach;
            }
        }
    }
}

/// The target `(slew, jerk)` limits a gain blob selects.
pub fn target_limits(gains: &GainStorage) -> (RotPerSecSq, RotPerSecCu) {
    let limits = gains.target_limits;
    (
        if limits.slew > RotPerSecSq::new(0.0) { limits.slew } else { DEFAULT_TARGET_SLEW_RPS2 },
        if limits.jerk > RotPerSecCu::new(0.0) { limits.jerk } else { DEFAULT_TARGET_JERK_RPS3 },
    )
}

/// Time between two readings of the wrapping microsecond clock.
pub(crate) fn elapsed(from: Micros, to: Micros) -> Seconds {
    let us = value!(to, us, u32).wrapping_sub(value!(from, us, u32));
    rescale_f32(quantity!(us as f32, us, f32))
}

impl Default for SetpointEstimator {
    fn default() -> Self {
        Self::new()
    }
}
