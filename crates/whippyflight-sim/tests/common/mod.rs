//! Shared test helpers.
#![allow(dead_code)]

pub mod noise;
pub mod plant;
pub use whippyflight_sim::units;

#[allow(unused_imports)]
pub use whippyflight_sim::flick::micros as micros64;
pub use whippyflight_control::gains::static_blob;
use whippyflight_control::gains::{
    crc32, DRAG_FLOATS, GAIN_VERSION, HOVER_FLOATS, LAYOUT_OFFSET, TARGET_FLOATS,
};
use whippyflight_control::types::{Hertz, Micros, RotPerMeter, Seconds};
use whippyunits::value;

/// Write `value` into a blob at byte offset `at`, in its storage unit (as
/// `load_gains` reads it back).
pub fn put<Q: bytemuck::Pod>(buf: &mut [u8], at: usize, value: Q) {
    let bytes = bytemuck::bytes_of(&value);
    buf[at..at + bytes.len()].copy_from_slice(bytes);
}

/// A copy of `blob` with its airframe drag block replaced by low-pass `lp`
/// and coefficients `c` (`[roll←ax, roll←ay, pitch←ax, pitch←ay]`), CRC
/// updated.
pub fn with_airframe_drag(blob: &[u8], lp: Hertz, c: [RotPerMeter; 4]) -> Vec<u8> {
    assert_eq!(u16::from_le_bytes([blob[4], blob[5]]), GAIN_VERSION);
    // The drag block comes right before the target, hover and layout blocks.
    let at = LAYOUT_OFFSET - 4 * (DRAG_FLOATS + TARGET_FLOATS + HOVER_FLOATS);
    let mut out = blob[..blob.len() - 4].to_vec();
    put(&mut out, at, lp);
    for (i, c) in c.into_iter().enumerate() {
        put(&mut out, at + 4 * (1 + i), c);
    }
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// The controller's microsecond clock at time `t` (truncated).
pub fn micros(t: Seconds) -> Micros {
    Micros::new(value!(t, us, f32) as u32)
}