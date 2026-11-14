//! Gain data storage and binary format parsing.
//!
//! # The `.wfg` gain blob (version 9)
//!
//! Every value is little-endian.  The header:
//!
//! | Offset | Size | Field          | Description                          |
//! |--------|------|----------------|--------------------------------------|
//! | 0x00   | 4    | magic          | `0x4C515247` ("LQRG")                |
//! | 0x04   | 2    | version        | 9                                    |
//! | 0x06   | 1    | resolution     | N (grid points per motor axis)       |
//! | 0x07   | 1    | ff_horizon     | Feedforward horizon h (ticks, ≥ 1)   |
//! | 0x08   | 4    | min_rps        | `f32` — grid's lowest motor speed    |
//! | 0x0C   | 4    | max_rps        | `f32` — grid's highest motor speed   |
//! | 0x10   | 4    | dt             | `f32` — loop period (s)              |
//! | 0x14   | 4    | integral_tau   | `f32` — integral time constant       |
//! | 0x18   | 4    | derivative_tau | `f32` — derivative smoothing time    |
//! | 0x1C   | 4    | total_size     | `u32` — total blob size in bytes     |
//!
//! then the body:
//!
//! | Size       | Field          | Description                                   |
//! |------------|----------------|-----------------------------------------------|
//! | 4 × 96     | gain quant     | per gain table (ki, kd, kp, binv): `f32[12]` lo, then `f32[12]` step |
//! | 24         | ref constants  | `f32[6]` ad_rr, fh_rr (speed-free)            |
//! | 352        | ref quant      | `f32[44]` lo, then `f32[44]` step             |
//! | 64         | affine         | `f32[4×4]` row-major linearizer G             |
//! | 8          | motor          | `f32[2]` motor drag [k_ω, k_ω²]               |
//! | 16         | balance        | `f32[4]` hover speed ratio per motor          |
//! | 20         | drag           | `f32[5]` airframe drag, see [`DRAG_FLOATS`]   |
//! | 8          | target         | `f32[2]` setpoint target limits               |
//! | 8          | hover          | `f32[2]` hover rps, integral limit V          |
//! | 8          | layout         | `u8[4]` motor positions, `u8[4]` spins, see [`MotorLayout`] |
//! | N⁴·24 × 4  | gain tables    | per gain table: `u16[N⁴ × 12]` q              |
//! | N·88       | ref table      | `u16[N × 4 × 11]` q                           |
//! | 4          | crc32          | CRC32 over every byte before it               |
//!
//! The tables are quantized to `u16` to halve their size (they are copied
//! to RAM, which is scarce on F411-class targets).  Each table slot (one of
//! the values per grid point) is mapped linearly over its own range across
//! the grid, value = lo + step·q.  Every section is a multiple of 4 bytes,
//! so the blob stays 4-byte aligned throughout.
//!
//! A gain table is indexed `[((m0·N + m1)·N + m2)·N + m3] · 12 + motor · 3
//! + axis`, m_i being motor i's grid index.  The reference model's blocks
//! each depend on at most one motor's speed (the rate and motor-speed
//! dynamics are diagonal and the plant is block-triangular, so column m of
//! every rate ← speed or rate ← voltage block, and motor m's own decay and
//! gain, follow motor m alone), so it is a 1D table over each motor's speed,
//! indexed `[grid index · 4 + motor] · REF_MOTOR_SLOTS + slot` (see
//! [`REF_MOTOR_SLOTS`]).  Multilinear interpolation of such a table is
//! exactly the 1D interpolation, so nothing is approximated.  At N = 4 the
//! blob is 25 856 bytes.
//!
//! # Conventions
//!
//! Axes are the body's FRD rates `[roll, pitch, yaw]` and motors are the
//! firmware's own motor indices, whose positions and spins the layout block
//! records (see [`crate::types`]).  The blocks whose values may be zero
//! say what zero means: no airframe drag model, the setpoint estimator's
//! default target limits, no hover speed (no takeoff/landing detection) and
//! no trim-sized integral bound.

use core::ops::{Add, Mul};

use crate::types::*;
use nalgebra::SMatrix;
use whippyalgebra::nalgebra::UniformUnitMatrix;
use whippyunits::qty;

/// Magic bytes: "LQRG" in little-endian.
pub const GAIN_MAGIC: u32 = 0x4C51_5247;

/// The format version this crate reads and writes.  Versions 1–8 were in
/// Betaflight's axes and are not read.
pub const GAIN_VERSION: u16 = 9;

/// Floats per grid point in the reference model, assembled from
/// [`REF_CONST_FLOATS`] and each motor's [`REF_MOTOR_SLOTS`].
///
/// The blocks are the pieces of the discrete (ZOH) model in deviation
/// coordinates, δx = [r (3 rates); δω (4 motor speeds)], u = 4 voltages:
///
/// ```text
///   One tick:   δx[k+1] = Ad·δx[k] + Bd·u[k]
///
///   ┌  r'  ┐   ┌ diag(ad_rr)    ad_rw     ┐ ┌ r  ┐   ┌   bd_r     ┐
///   │      │ = │   (3×3)        (3×4)     │ │    │ + │   (3×4)    │ u
///   │  δω' │   │     0       diag(ad_ww)  │ │ δω │   │ diag(bd_w) │
///   └      ┘   └   (4×3)        (4×4)     ┘ └    ┘   └   (4×4)    ┘
///
///   h ticks, free response (attitude rows of Ad^h):
///
///   F_h = [ diag(fh_rr)   fh_rw ]
///           (3×3)        (3×4)
/// ```
///
/// The rate→rate and motor→motor blocks are diagonal (rates and motors
/// each decay independently), and motor speeds don't depend on rates, so
/// the lower-left block is zero.  Only the non-zero blocks are kept,
/// diagonals as vectors:
///
/// | Offset | Block   | Shape | Meaning                                 |
/// |--------|---------|-------|-----------------------------------------|
/// | 0      | `ad_rr` | 3     | one-tick rate decay e^{ρ·dt}            |
/// | 3      | `ad_rw` | 3×4   | one-tick rate ← motor speed             |
/// | 15     | `ad_ww` | 4     | one-tick motor-speed decay              |
/// | 19     | `bd_r`  | 3×4   | one-tick rate ← voltage                 |
/// | 31     | `bd_w`  | 4     | one-tick motor speed ← voltage          |
/// | 35     | `fh_rr` | 3     | h-tick free response, rate ← rate       |
/// | 38     | `fh_rw` | 3×4   | h-tick free response, rate ← motor speed|
pub const REF_FLOATS: usize = 50;

/// Speed-independent reference-model floats: `ad_rr` then `fh_rr`.
pub const REF_CONST_FLOATS: usize = 6;

/// Reference-model values per motor per grid speed, from motor m's
/// columns of the blocks in [`REF_FLOATS`]:
///
/// | Slot | Value          |
/// |------|----------------|
/// | 0–2  | `ad_rw[0..3][m]` |
/// | 3    | `ad_ww[m]`     |
/// | 4–6  | `bd_r[0..3][m]` |
/// | 7    | `bd_w[m]`      |
/// | 8–10 | `fh_rw[0..3][m]` |
pub const REF_MOTOR_SLOTS: usize = 11;

/// Width of the reference table's rows: all motors at one grid speed.
pub const REF_MOTOR_WIDTH: usize = 4 * REF_MOTOR_SLOTS;

/// Assemble the [`REF_FLOATS`] slots from the speed-free values and each
/// motor's [`REF_MOTOR_SLOTS`] at its own speed.
pub fn ref_slots_from_motors(
    constant: &[f32; REF_CONST_FLOATS],
    motors: &[[f32; REF_MOTOR_SLOTS]; 4],
) -> [f32; REF_FLOATS] {
    let mut r = [0.0f32; REF_FLOATS];
    r[0..3].copy_from_slice(&constant[0..3]);
    r[35..38].copy_from_slice(&constant[3..6]);
    for (m, s) in motors.iter().enumerate() {
        for i in 0..3 {
            r[3 + 4 * i + m] = s[i];
            r[19 + 4 * i + m] = s[4 + i];
            r[38 + 4 * i + m] = s[8 + i];
        }
        r[15 + m] = s[3];
        r[31 + m] = s[7];
    }
    r
}

/// The reference-model blocks of [`REF_FLOATS`] at one operating point,
/// with their units.  Motor speeds and rates are both rot/s, so the
/// rate ← speed blocks and every decay are dimensionless.
#[derive(Clone, Copy)]
pub struct ReferenceModel {
    pub ad_rr: [f32; 3],
    pub ad_rw: RateFromSpeedMatrix,
    pub ad_ww: [f32; 4],
    pub bd_r: RateFromVoltMatrix,
    pub bd_w: [RpsPerVolt; 4],
    pub fh_rr: [f32; 3],
    pub fh_rw: RateFromSpeedMatrix,
}

impl ReferenceModel {
    /// Decode the slots (see [`REF_FLOATS`] for the layout).
    pub fn from_slots(r: &[f32; REF_FLOATS]) -> Self {
        let rate_from_speed = |o: usize| {
            UniformUnitMatrix::<_, SMatrix<f32, 3, 4>>::from_fn(|i, j| {
                <qty!(1, f32)>::new(r[o + 4 * i + j])
            })
            .gauge::<AttitudeAxes, MotorSpeedAxes>()
        };
        Self {
            ad_rr: [r[0], r[1], r[2]],
            ad_rw: rate_from_speed(3),
            ad_ww: [r[15], r[16], r[17], r[18]],
            bd_r: UniformUnitMatrix::<_, SMatrix<f32, 3, 4>>::from_fn(|i, j| {
                RpsPerVolt::new(r[19 + 4 * i + j])
            })
            .gauge::<AttitudeAxes, MotorAxes>(),
            bd_w: [31, 32, 33, 34].map(|o| RpsPerVolt::new(r[o])),
            fh_rr: [r[35], r[36], r[37]],
            fh_rw: rate_from_speed(38),
        }
    }
}

/// Floats in the affine linearizer block.
pub const AFFINE_FLOATS: usize = 16;

/// Floats in the motor block.
pub const MOTOR_FLOATS: usize = 2;

/// Steady-state motor drag, V = k_ω·ω + k_ω²·ω|ω| (the motor fit without
/// its k_α·ω̇ term).
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct MotorDrag {
    pub k_w: VoltPerRps,
    pub k_w2: VoltPerRpsSq,
}

impl MotorDrag {
    pub const ZERO: Self = Self { k_w: VoltPerRps::new(0.0), k_w2: VoltPerRpsSq::new(0.0) };
}

/// Floats in the thrust-balance block.
pub const BALANCE_FLOATS: usize = 4;

/// Floats in the airframe drag block: `[lp_hz, roll←ax, roll←ay,
/// pitch←ax, pitch←ay]`.
///
/// Rotor drag on tall frames acts well above the CG, so airspeed torques
/// pitch and roll.  In flight the accelerometer reads thrust (body z) plus
/// that drag, so its body x/y components, one-pole low-passed at `lp_hz`,
/// measure it; the coefficients (rot/m, i.e. rot/s² per m/s², FRD axes)
/// give the angular acceleration it causes.  `lp_hz = 0` turns the mode off.
pub const DRAG_FLOATS: usize = 5;

/// The airframe drag block, see [`DRAG_FLOATS`]: angular acceleration per
/// unit of body specific force.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct AirframeDrag {
    pub lp: Hertz,
    /// Roll angular acceleration per body [x, y] specific force.
    pub roll: [RotPerMeter; 2],
    /// Pitch angular acceleration per body [x, y] specific force.
    pub pitch: [RotPerMeter; 2],
}

impl AirframeDrag {
    pub const OFF: Self = Self {
        lp: Hertz::new(0.0),
        roll: [RotPerMeter::new(0.0); 2],
        pitch: [RotPerMeter::new(0.0); 2],
    };
}

/// Floats in the target-limit block: the setpoint estimator's `[slew
/// (rot/s²), jerk (rot/s³)]` bounds on the feedforward target, set from the
/// airframe's pitch/roll authority (see [`crate::setpoint`]).  A value ≤ 0
/// selects the estimator's default.
pub const TARGET_FLOATS: usize = 2;

/// The target-limit block, see [`TARGET_FLOATS`].
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct TargetLimits {
    pub slew: RotPerSecSq,
    pub jerk: RotPerSecCu,
}

impl TargetLimits {
    pub const DEFAULT: Self = Self { slew: RotPerSecSq::new(0.0), jerk: RotPerSecCu::new(0.0) };
}

/// Floats in the hover block: [hover rps, integral limit V] — the fitted
/// airframe's median hover motor speed (shaft rev/s), for takeoff/landing
/// detection, and the per-motor integral voltage that trims a generous CG
/// offset at hover (gen_wfg `build_integral_limit`).  0 when unknown.
pub const HOVER_FLOATS: usize = 2;

/// Bytes in the layout block: a [`MotorPosition`] per motor, then a
/// [`Spin`] per motor.
pub const LAYOUT_BYTES: usize = 8;

/// Where a motor sits on a quad-X frame.  The codes are the gain model's
/// own motor order.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum MotorPosition {
    FrontLeft = 0,
    RearLeft = 1,
    FrontRight = 2,
    RearRight = 3,
}

impl MotorPosition {
    pub const ALL: [Self; 4] = [Self::FrontLeft, Self::RearLeft, Self::FrontRight, Self::RearRight];

    pub fn from_code(code: u8) -> Option<Self> {
        Self::ALL.get(code as usize).copied()
    }

    /// The motor diagonally across the frame, which spins the same way.
    pub fn diagonal(self) -> Self {
        match self {
            Self::FrontLeft => Self::RearRight,
            Self::RearRight => Self::FrontLeft,
            Self::FrontRight => Self::RearLeft,
            Self::RearLeft => Self::FrontRight,
        }
    }
}

/// A propeller's direction of rotation, viewed from above.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum Spin {
    Clockwise = 0,
    CounterClockwise = 1,
}

impl Spin {
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Clockwise),
            1 => Some(Self::CounterClockwise),
            _ => None,
        }
    }
}

/// Each motor's position and spin, by the firmware's motor index: the
/// airframe the gains were solved for.  A firmware adapter checks it
/// against its own mixer before flying the gains
/// ([`GainLoadResult::LayoutMismatch`]).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct MotorLayout {
    pub position: [MotorPosition; 4],
    pub spin: [Spin; 4],
}

impl MotorLayout {
    /// Decode the layout block: every position once, and diagonal motors
    /// spinning the same way, against the other pair.
    pub fn from_codes(codes: &[u8; LAYOUT_BYTES]) -> Option<Self> {
        let position = [0, 1, 2, 3].map(|m| MotorPosition::from_code(codes[m]));
        let spin = [4, 5, 6, 7].map(|m| Spin::from_code(codes[m]));
        if position.iter().any(Option::is_none) || spin.iter().any(Option::is_none) {
            return None;
        }
        let layout = Self { position: position.map(Option::unwrap), spin: spin.map(Option::unwrap) };
        let motor_at = |p: MotorPosition| layout.position.iter().position(|&q| q == p);
        for m in 0..4 {
            let across = motor_at(layout.position[m].diagonal())?;
            let beside = motor_at(match layout.position[m] {
                MotorPosition::FrontLeft => MotorPosition::FrontRight,
                MotorPosition::FrontRight => MotorPosition::FrontLeft,
                MotorPosition::RearLeft => MotorPosition::RearRight,
                MotorPosition::RearRight => MotorPosition::RearLeft,
            })?;
            if layout.spin[across] != layout.spin[m] || layout.spin[beside] == layout.spin[m] {
                return None;
            }
        }
        Some(layout)
    }

    pub fn to_codes(&self) -> [u8; LAYOUT_BYTES] {
        core::array::from_fn(|i| if i < 4 { self.position[i] as u8 } else { self.spin[i - 4] as u8 })
    }
}

/// Maximum supported resolution (N).  This bounds the format, not the
/// target: the blob is copied to RAM, and at N=5 it is ≈ 63 KB.
pub const MAX_RESOLUTION: usize = 5;

/// Floats per grid point in each of the 4 gain tables (4 motors × 3 axes).
pub const GAIN_SLOTS: usize = 12;

/// Total values per gain table at a given resolution: N⁴ × 4 motors × 3 axes.
const fn table_values(n: usize) -> usize {
    n * n * n * n * GAIN_SLOTS
}

/// Binary header size in bytes.
pub const HEADER_SIZE: usize = 0x20;

/// Bytes between the header and the table data.
const PARAMS_BYTES: usize = 4
    * (4 * 2 * GAIN_SLOTS + REF_CONST_FLOATS + 2 * REF_MOTOR_WIDTH + AFFINE_FLOATS + MOTOR_FLOATS
        + BALANCE_FLOATS + DRAG_FLOATS + TARGET_FLOATS + HOVER_FLOATS)
    + LAYOUT_BYTES;

/// Byte offset of the layout block, the last before the table data.
pub const LAYOUT_OFFSET: usize = HEADER_SIZE + PARAMS_BYTES - LAYOUT_BYTES;

/// Size of a blob of resolution `n`.
pub const fn blob_size(n: usize) -> usize {
    HEADER_SIZE + PARAMS_BYTES + 2 * (4 * table_values(n) + n * REF_MOTOR_WIDTH) + 4
}

// Tables are read in place as native f32/u16, which must match the blob's
// little-endian encoding.
#[cfg(target_endian = "big")]
compile_error!("the .wfg tables are little-endian and are read in place");

/// A value a lookup table can hold: a quantity of the table's unit, or a
/// bare `f32` for the reference-model table, whose slots differ in unit
/// (decoded by [`ReferenceModel::from_slots`]).
pub trait TableEntry: bytemuck::Pod + Default + Add<Output = Self> + Mul<f32, Output = Self> {}

impl<T: bytemuck::Pod + Default + Add<Output = T> + Mul<f32, Output = T>> TableEntry for T {}

/// A quantized lookup table read in place from the blob, `lo.len()` values
/// per grid point, indexed [grid point] · width + slot:
/// value[i] = lo[i % W] + step[i % W] · q[i].
#[derive(Clone, Copy)]
pub struct GainTable<E: 'static> {
    q: &'static [u16],
    lo: &'static [E],
    step: &'static [E],
}

impl<E: TableEntry> GainTable<E> {
    pub const EMPTY: Self = Self { q: &[], lo: &[], step: &[] };

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start of the table's per-point data within the blob.
    pub fn data_ptr(&self) -> *const u8 {
        self.q.as_ptr() as *const u8
    }

    /// Decoded value at flat index `i`.
    pub fn get(&self, i: usize) -> E {
        let slot = i % self.lo.len();
        self.lo[slot] + self.step[slot] * self.q[i] as f32
    }

    /// Weighted sum over grid points of the `W` values at each, i.e.
    /// Σ_v weight[v] · table[point[v]·W + i].  The weights must sum to 1:
    /// the raw codes are summed and lo/step applied once per slot.
    pub fn blend<const W: usize>(&self, point: &[usize], weight: &[f32]) -> [E; W] {
        self.blend_slots::<W>(W, 0, point, weight)
    }

    /// [`Self::blend`] of the `K` slots from `first` in rows of `width`.
    pub fn blend_slots<const K: usize>(&self, width: usize, first: usize, point: &[usize], weight: &[f32]) -> [E; K] {
        let mut code = [0.0f32; K];
        for (&p, &w) in point.iter().zip(weight) {
            let row = &self.q[p * width + first..p * width + first + K];
            for (o, &x) in code.iter_mut().zip(row) {
                *o += w * x as f32;
            }
        }
        core::array::from_fn(|i| self.lo[first + i] + self.step[first + i] * code[i])
    }
}

/// Parsed gain header.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct GainHeader {
    pub resolution: u8,
    pub min_rps: RotPerSec,
    pub max_rps: RotPerSec,
    pub dt: Seconds,
    pub integral_tau: Seconds,
    pub derivative_tau: Seconds,
    /// Feedforward inversion horizon in ticks.
    pub ff_horizon: u8,
}

/// Validated view of a gain blob.  The lookup tables are slices into the
/// blob itself (the firmware's RAM copy on target), so the blob must
/// outlive the controller and be 4-byte aligned; see [`load_gains`].
pub struct GainStorage {
    pub header: GainHeader,
    pub loaded: bool,

    // The 4 gain tables, each N⁴ × 12 values.  Indexed as
    // [((m0·N + m1)·N + m2)·N + m3] * 12 + motor * 3 + axis
    pub ki: GainTable<VoltPerRot>,
    pub kd: GainTable<VoltPerRpsPerSec>,
    pub kp: GainTable<VoltPerRps>,
    pub binv: GainTable<VoltPerRps>,

    /// Reference-model table, per motor (see [`REF_MOTOR_SLOTS`]).
    pub ref_model: GainTable<f32>,
    /// The reference model's speed-free values, see [`REF_CONST_FLOATS`].
    pub ref_const: [f32; REF_CONST_FLOATS],

    /// Affine linearizer G, row-major [motor][signed-square motor]:
    /// u_aff = G · ω|ω|.  Constant over the grid.
    pub affine: [VoltPerRpsSq; AFFINE_FLOATS],

    /// Steady-state motor drag.
    pub motor_drag: MotorDrag,

    /// Per-motor hover speed ratio for a torque-free hover, geometric mean 1
    /// (1/√(d_fb·d_lr) from the arm lengths).
    pub thrust_balance: [f32; BALANCE_FLOATS],

    /// Airframe drag model, see [`DRAG_FLOATS`].
    pub airframe_drag: AirframeDrag,

    /// Setpoint target limits, see [`TARGET_FLOATS`].
    pub target_limits: TargetLimits,

    /// Hover motor speed (shaft rev/s), see [`HOVER_FLOATS`].
    pub hover_rps: RotPerSec,
    /// Per-motor integral voltage bound, see [`HOVER_FLOATS`].
    pub integral_limit_v: Volt,

    /// The airframe's motors, by the firmware's motor index.
    pub layout: MotorLayout,
}

impl GainStorage {
    pub const fn new() -> Self {
        Self {
            header: GainHeader {
                resolution: 0,
                min_rps: RotPerSec::new(0.0),
                max_rps: RotPerSec::new(0.0),
                dt: Seconds::new(0.0),
                integral_tau: Seconds::new(0.0),
                derivative_tau: Seconds::new(0.0),
                ff_horizon: 0,
            },
            loaded: false,
            ki: GainTable::EMPTY,
            kd: GainTable::EMPTY,
            kp: GainTable::EMPTY,
            binv: GainTable::EMPTY,
            ref_model: GainTable::EMPTY,
            ref_const: [0.0; REF_CONST_FLOATS],
            affine: [VoltPerRpsSq::new(0.0); AFFINE_FLOATS],
            motor_drag: MotorDrag::ZERO,
            thrust_balance: [1.0; BALANCE_FLOATS],
            airframe_drag: AirframeDrag::OFF,
            target_limits: TargetLimits::DEFAULT,
            hover_rps: RotPerSec::new(0.0),
            integral_limit_v: Volt::new(0.0),
            layout: MotorLayout {
                position: MotorPosition::ALL,
                spin: [Spin::Clockwise, Spin::CounterClockwise, Spin::CounterClockwise, Spin::Clockwise],
            },
        }
    }

    /// The affine linearizer G as a matrix.
    pub fn affine_matrix(&self) -> AffineMatrix {
        UniformUnitMatrix::<_, SMatrix<f32, 4, 4>>::from_fn(|i, j| self.affine[4 * i + j])
            .gauge::<MotorAxes, MotorSpeedSqAxes>()
    }
}

impl Default for GainStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Sequential typed reader over an aligned blob.  Tables are viewed in
/// place: blob alignment is checked in `load_gains` and every section
/// offset is a multiple of 4.
struct Cursor {
    data: &'static [u8],
    offset: usize,
}

impl Cursor {
    fn take<E: bytemuck::Pod>(&mut self, count: usize) -> &'static [E] {
        let bytes = &self.data[self.offset..self.offset + count * core::mem::size_of::<E>()];
        self.offset += bytes.len();
        debug_assert!(bytes.as_ptr() as usize % core::mem::align_of::<E>() == 0);
        // SAFETY: `bytes` is in bounds and aligned (see the struct docs), and
        // `E: Pod` accepts any bit pattern.  Not `bytemuck::cast_slice`: its
        // formatting panic path pulls unwinding symbols into the SITL link.
        unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<E>(), count) }
    }

    fn one<E: bytemuck::Pod>(&mut self) -> E {
        self.take::<E>(1)[0]
    }

    /// A quantized table's per-slot (lo, step).
    fn quant<E: bytemuck::Pod>(&mut self, width: usize) -> (&'static [E], &'static [E]) {
        (self.take(width), self.take(width))
    }
}

/// Read a little-endian `u32` from a byte slice at the given offset.
fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

/// Read a little-endian `u16` from a byte slice at the given offset.
fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Read a value stored as a little-endian `f32` at the given offset.
fn read_f32<E: bytemuck::Pod>(data: &[u8], offset: usize) -> E {
    bytemuck::pod_read_unaligned(&data[offset..offset + 4])
}

/// Simple CRC32 (ISO 3309 / ITU-T V.42 polynomial).
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Copy `bytes` into leaked, 4-byte-aligned memory, as [`load_gains`]
/// needs: on a host, where the blob comes from a file rather than a static
/// RAM buffer.
#[cfg(feature = "std")]
pub fn static_blob(bytes: &[u8]) -> &'static [u8] {
    let words: &'static mut [u32] = std::vec![0u32; bytes.len().div_ceil(4)].leak();
    // SAFETY: the u32 buffer covers at least `bytes.len()` bytes and is leaked.
    let buf = unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, bytes.len()) };
    buf.copy_from_slice(bytes);
    buf
}

/// Parse and validate a gain binary blob, pointing `storage` at its tables.
///
/// The tables are not copied: `data` must stay valid and unmodified for the
/// life of the program (on target, the firmware's static RAM copy) and must
/// be 4-byte aligned.
pub fn load_gains(data: &'static [u8], storage: &mut GainStorage) -> GainLoadResult {
    storage.loaded = false;

    if data.len() < HEADER_SIZE + 4 {
        return GainLoadResult::TooShort;
    }
    if data.as_ptr() as usize % core::mem::align_of::<f32>() != 0 {
        return GainLoadResult::Misaligned;
    }
    if read_u32(data, 0x00) != GAIN_MAGIC {
        return GainLoadResult::BadMagic;
    }
    if read_u16(data, 0x04) != GAIN_VERSION {
        return GainLoadResult::BadVersion;
    }
    let resolution = data[0x06] as usize;
    if resolution > MAX_RESOLUTION || resolution == 0 {
        return GainLoadResult::ResolutionTooLarge;
    }

    let total_size = read_u32(data, 0x1C) as usize;
    if data.len() < total_size || total_size != blob_size(resolution) {
        return GainLoadResult::TooShort;
    }
    let expected_crc = read_u32(data, total_size - 4);
    if crc32(&data[..total_size - 4]) != expected_crc {
        return GainLoadResult::BadChecksum;
    }

    let header = GainHeader {
        resolution: resolution as u8,
        min_rps: read_f32(data, 0x08),
        max_rps: read_f32(data, 0x0C),
        dt: read_f32(data, 0x10),
        integral_tau: read_f32(data, 0x14),
        derivative_tau: read_f32(data, 0x18),
        ff_horizon: data[0x07],
    };

    let mut cur = Cursor { data, offset: HEADER_SIZE };
    let ki_q = cur.quant(GAIN_SLOTS);
    let kd_q = cur.quant(GAIN_SLOTS);
    let kp_q = cur.quant(GAIN_SLOTS);
    let binv_q = cur.quant(GAIN_SLOTS);
    let ref_const: &[f32] = cur.take(REF_CONST_FLOATS);
    let ref_q = cur.quant(REF_MOTOR_WIDTH);
    let affine: &[VoltPerRpsSq] = cur.take(AFFINE_FLOATS);
    let motor_drag = MotorDrag { k_w: cur.one(), k_w2: cur.one() };
    let thrust_balance: &[f32] = cur.take(BALANCE_FLOATS);
    let airframe_drag = AirframeDrag {
        lp: cur.one(),
        roll: [cur.one(), cur.one()],
        pitch: [cur.one(), cur.one()],
    };
    let target_limits = TargetLimits { slew: cur.one(), jerk: cur.one() };
    let hover_rps = cur.one();
    let integral_limit_v = cur.one();
    let codes: &[u8] = cur.take(LAYOUT_BYTES);
    let Some(layout) = MotorLayout::from_codes(&core::array::from_fn(|i| codes[i])) else {
        return GainLoadResult::BadLayout;
    };
    fn table<E: TableEntry>(cur: &mut Cursor, (lo, step): (&'static [E], &'static [E]), count: usize) -> GainTable<E> {
        GainTable { q: cur.take(count), lo, step }
    }
    let n = table_values(resolution);
    storage.ki = table(&mut cur, ki_q, n);
    storage.kd = table(&mut cur, kd_q, n);
    storage.kp = table(&mut cur, kp_q, n);
    storage.binv = table(&mut cur, binv_q, n);
    storage.ref_model = table(&mut cur, ref_q, resolution * REF_MOTOR_WIDTH);
    storage.ref_const.copy_from_slice(ref_const);
    storage.affine.copy_from_slice(affine);
    storage.thrust_balance.copy_from_slice(thrust_balance);
    storage.header = header;
    storage.motor_drag = motor_drag;
    storage.airframe_drag = airframe_drag;
    storage.target_limits = target_limits;
    storage.hover_rps = hover_rps;
    storage.integral_limit_v = integral_limit_v;
    storage.layout = layout;

    storage.loaded = true;
    GainLoadResult::Ok
}

/// Host-side encoder of the blob format, for tests and tools that build
/// blobs from values rather than from a fit (gen_wfg packs the same way).
#[cfg(feature = "std")]
pub mod spec {
    use super::*;
    use std::vec::Vec;

    /// Every value of a blob, unquantized; tables in their blob order.
    #[derive(Clone)]
    pub struct BlobSpec {
        pub resolution: u8,
        pub ff_horizon: u8,
        pub min_rps: f32,
        pub max_rps: f32,
        pub dt: f32,
        pub integral_tau: f32,
        pub derivative_tau: f32,
        /// N⁴ × 12 each (see the module docs for the index).
        pub ki: Vec<f32>,
        pub kd: Vec<f32>,
        pub kp: Vec<f32>,
        pub binv: Vec<f32>,
        pub ref_const: [f32; REF_CONST_FLOATS],
        /// N × 4 × [`REF_MOTOR_SLOTS`].
        pub ref_motor: Vec<f32>,
        pub affine: [f32; AFFINE_FLOATS],
        pub motor_drag: [f32; MOTOR_FLOATS],
        pub thrust_balance: [f32; BALANCE_FLOATS],
        pub airframe_drag: [f32; DRAG_FLOATS],
        pub target_limits: [f32; TARGET_FLOATS],
        pub hover: [f32; HOVER_FLOATS],
        pub layout: MotorLayout,
    }

    impl BlobSpec {
        /// Zero gains on an N-point grid over 100–500 rev/s at 8 kHz, with
        /// a reference model whose rates hold (ad_rr = fh_rr = 1) and that
        /// nothing drives.
        pub fn new(resolution: u8) -> Self {
            let n = resolution as usize;
            Self {
                resolution,
                ff_horizon: 2,
                min_rps: 100.0,
                max_rps: 500.0,
                dt: 0.000125,
                integral_tau: 0.1,
                derivative_tau: 0.01,
                ki: std::vec![0.0; table_values(n)],
                kd: std::vec![0.0; table_values(n)],
                kp: std::vec![0.0; table_values(n)],
                binv: std::vec![0.0; table_values(n)],
                ref_const: [1.0; REF_CONST_FLOATS],
                ref_motor: std::vec![0.0; n * REF_MOTOR_WIDTH],
                affine: [0.0; AFFINE_FLOATS],
                motor_drag: [0.0; MOTOR_FLOATS],
                thrust_balance: [1.0; BALANCE_FLOATS],
                airframe_drag: [0.0; DRAG_FLOATS],
                target_limits: [0.0; TARGET_FLOATS],
                hover: [0.0; HOVER_FLOATS],
                layout: GainStorage::new().layout,
            }
        }

        /// Fill a gain table with `f(motor, axis)` at every grid point.
        pub fn fill(table: &mut [f32], f: impl Fn(usize, usize) -> f32) {
            for (i, v) in table.iter_mut().enumerate() {
                *v = f((i % GAIN_SLOTS) / 3, i % 3);
            }
        }

        /// The blob, quantized and checksummed.
        pub fn encode(&self) -> Vec<u8> {
            let n = self.resolution as usize;
            let quantized = [
                quantize(&self.ki, GAIN_SLOTS),
                quantize(&self.kd, GAIN_SLOTS),
                quantize(&self.kp, GAIN_SLOTS),
                quantize(&self.binv, GAIN_SLOTS),
                quantize(&self.ref_motor, REF_MOTOR_WIDTH),
            ];
            let mut out = Vec::with_capacity(blob_size(n));
            out.extend_from_slice(&GAIN_MAGIC.to_le_bytes());
            out.extend_from_slice(&GAIN_VERSION.to_le_bytes());
            out.extend_from_slice(&[self.resolution, self.ff_horizon]);
            let f32s = |out: &mut Vec<u8>, values: &[f32]| {
                values.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()))
            };
            f32s(&mut out, &[self.min_rps, self.max_rps, self.dt, self.integral_tau, self.derivative_tau]);
            out.extend_from_slice(&(blob_size(n) as u32).to_le_bytes());
            for (k, (lo, step, _)) in quantized.iter().enumerate() {
                if k == 4 {
                    f32s(&mut out, &self.ref_const);
                }
                f32s(&mut out, lo);
                f32s(&mut out, step);
            }
            f32s(&mut out, &self.affine);
            f32s(&mut out, &self.motor_drag);
            f32s(&mut out, &self.thrust_balance);
            f32s(&mut out, &self.airframe_drag);
            f32s(&mut out, &self.target_limits);
            f32s(&mut out, &self.hover);
            out.extend_from_slice(&self.layout.to_codes());
            for (_, _, q) in &quantized {
                q.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes()));
            }
            let crc = crc32(&out);
            out.extend_from_slice(&crc.to_le_bytes());
            assert_eq!(out.len(), blob_size(n), "table sizes don't match the resolution");
            out
        }
    }

    /// Quantize each of a table's `width` slots over its own range:
    /// value = lo + step·q.  Returns (lo, step, q).
    fn quantize(table: &[f32], width: usize) -> (Vec<f32>, Vec<f32>, Vec<u16>) {
        let rows = table.len() / width;
        let column = |s: usize| (0..rows).map(move |r| table[r * width + s] as f64);
        let lo: Vec<f32> = (0..width).map(|s| column(s).fold(f64::INFINITY, f64::min) as f32).collect();
        let step: Vec<f32> = (0..width)
            .map(|s| ((column(s).fold(f64::NEG_INFINITY, f64::max) - lo[s] as f64) / 65535.0) as f32)
            .collect();
        let q = table
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let s = i % width;
                if step[s] > 0.0 {
                    ((v as f64 - lo[s] as f64) / step[s] as f64).round().clamp(0.0, 65535.0) as u16
                } else {
                    0
                }
            })
            .collect();
        (lo, step, q)
    }
}
