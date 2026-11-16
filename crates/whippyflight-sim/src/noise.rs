//! Replay of measured sensor noise, as a firmware's noise extractor (e.g.
//! whippyflight_betaflight.extract_noise) writes it from a flight
//! log: the firmware's filtered gyro (FRD) and the motor speed telemetry
//! (shaft rev/s, by firmware motor index), high-passed to remove the body's
//! own motion, looped.  The gyro noise was taken after the firmware's
//! filters, so it belongs on top of the gyro the controller is handed.
//! Motor-speed noise grows with the motor's speed (telemetry times its
//! commutation, so timing jitter weighs more at speed): with a [`SpeedLaw`]
//! each motor's channel follows that motor's speed.
//!
//! File format, little-endian: `f32` sample rate (Hz), `u32` frame count,
//! then per frame `i16` gyro [roll, pitch, yaw] in 1e-5 rot/s and `i16`
//! motor speeds [0..4] in 1e-3 rot/s.

use crate::units::{Freq, Rate, Time};
use whippyflight_control::types::RotPerSec;
use whippyunits::{quantity, value};

const GYRO_LSB: RotPerSec = RotPerSec::new(1e-5);
const ERPM_LSB: RotPerSec = RotPerSec::new(1e-3);
const CHANNELS: usize = 7;
const HEADER: usize = 8;

/// How motor-speed noise grows with the motor's speed: its recorded RMS at
/// `reference` (the recording's mean motor speed), times
/// (speed / `reference`)^`exponent`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SpeedLaw {
    pub reference: Rate,
    pub exponent: f64,
}

/// The steepest [`SpeedLaw::exponent`] taken: jitter in the commutation
/// timing gives a speed error growing as speed².
pub const MAX_SPEED_EXPONENT: f64 = 2.0;

pub struct Noise {
    fs: Freq,
    /// Gyro [roll, pitch, yaw] then motor speeds [0..4].
    frames: Vec<[RotPerSec; CHANNELS]>,
    offset: Time,
    gyro_scale: f32,
    motor_scale: f32,
    speed_law: Option<SpeedLaw>,
}

pub struct NoiseSample {
    /// Gyro, FRD [roll, pitch, yaw].
    pub gyro: [RotPerSec; 3],
    /// Motor speed telemetry, shaft rev/s.
    pub motor: [RotPerSec; 4],
}

impl Noise {
    /// Parse extract_noise.py's output, starting `offset` into the recording
    /// (it loops), with the gyro and motor-speed noise scaled separately.
    pub fn from_bytes(
        bytes: &[u8],
        offset: Time,
        gyro_scale: f32,
        motor_scale: f32,
    ) -> Result<Self, String> {
        if bytes.len() < HEADER {
            return Err("noise file too short".into());
        }
        let rate = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if !rate.is_finite() || rate <= 0.0 || n == 0 || bytes.len() != HEADER + 2 * CHANNELS * n {
            return Err(format!(
                "not a noise file: {rate} Hz, {n} frames in {} bytes",
                bytes.len()
            ));
        }
        let frames = (0..n)
            .map(|k| {
                core::array::from_fn(|c| {
                    let at = HEADER + 2 * (k * CHANNELS + c);
                    let lsb = if c < 3 { GYRO_LSB } else { ERPM_LSB };
                    i16::from_le_bytes([bytes[at], bytes[at + 1]]) as f32 * lsb
                })
            })
            .collect();
        Ok(Self {
            fs: Freq::new(rate as f64),
            frames,
            offset,
            gyro_scale,
            motor_scale,
            speed_law: None,
        })
    }

    /// This recording with the motor-speed noise following `law`; None
    /// replays it as recorded at every speed.
    pub fn with_speed_law(self, law: Option<SpeedLaw>) -> Result<Self, String> {
        if let Some(SpeedLaw { reference, exponent }) = law {
            let reference = value!(reference, rot / s);
            if !(reference.is_finite() && reference > 0.0 && (0.0..=MAX_SPEED_EXPONENT).contains(&exponent)) {
                return Err(format!("bad motor noise speed law: {reference} rot/s, exponent {exponent}"));
            }
        }
        Ok(Self { speed_law: law, ..self })
    }

    pub fn speed_law(&self) -> Option<SpeedLaw> {
        self.speed_law
    }

    /// This recording with the gyro and motor-speed noise scaled by
    /// `gyro_scale` and `motor_scale` instead.
    pub fn scaled(&self, gyro_scale: f32, motor_scale: f32) -> Self {
        Self {
            fs: self.fs,
            frames: self.frames.clone(),
            offset: self.offset,
            gyro_scale,
            motor_scale,
            speed_law: self.speed_law,
        }
    }

    /// RMS of the recording as logged, before scaling: the gyro over its
    /// three axes, and the motor speed over the four motors (at the
    /// [`SpeedLaw::reference`]).
    pub fn rms(&self) -> (Rate, Rate) {
        let rms = |channels: core::ops::Range<usize>| -> Rate {
            let count = (self.frames.len() * channels.len()) as f64;
            let sum = self
                .frames
                .iter()
                .flat_map(|f| f[channels.clone()].iter())
                .fold(quantity!(0.0, rot ^ 2 / s ^ 2), |sum, &x| {
                    let x: Rate = x.lossy_into();
                    sum + x * x
                });
            (sum / count).sqrt()
        };
        (rms(0..3), rms(3..CHANNELS))
    }

    pub fn duration(&self) -> Time {
        self.frames.len() as f64 / self.fs
    }

    /// Noise at time `t`, linearly interpolated from the log rate, with the
    /// motors at `motor_rps` (shaft rev/s, by firmware motor index).
    pub fn at(&self, t: Time, motor_rps: [RotPerSec; 4]) -> NoiseSample {
        let n = self.frames.len();
        let x = f64::from((t + self.offset) * self.fs).rem_euclid(n as f64);
        let (i, f) = (x as usize % n, (x - x.floor()) as f32);
        let (a, b) = (self.frames[i], self.frames[(i + 1) % n]);
        let motor_scale = |m: usize| match self.speed_law {
            None => self.motor_scale,
            Some(law) => {
                let speed: Rate = motor_rps[m].lossy_into();
                self.motor_scale * f64::from(speed.abs() / law.reference).powf(law.exponent) as f32
            }
        };
        let v: [RotPerSec; CHANNELS] = core::array::from_fn(|c| {
            (if c < 3 { self.gyro_scale } else { motor_scale(c - 3) }) * (a[c] + f * (b[c] - a[c]))
        });
        NoiseSample {
            gyro: [v[0], v[1], v[2]],
            motor: [v[3], v[4], v[5], v[6]],
        }
    }
}
