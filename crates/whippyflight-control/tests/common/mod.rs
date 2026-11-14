//! Shared test helpers.
#![allow(dead_code, unused_imports)]

pub use whippyflight_control::gains::static_blob;
use whippyflight_control::types::{Micros, Seconds};
use whippyunits::value;

/// The controller's microsecond clock at time `t` (truncated).
pub fn micros(t: Seconds) -> Micros {
    Micros::new(value!(t, us, f32) as u32)
}
