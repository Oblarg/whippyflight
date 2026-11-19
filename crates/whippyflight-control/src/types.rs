//! Core types for the WhippyFlight control stack.
//!
//! Internal types use whippyunits quantities and whippyalgebra matrices for
//! compile-time dimensional safety.  The few `#[repr(C)]` ones can cross a
//! firmware adapter's FFI as they are.
//!
//! # Units
//!
//! SI-adjacent units throughout: rates and motor shaft speeds in rot/s,
//! voltages in V, specific force in m/s², time in s.  Throttle is a
//! fraction [0, 1].  A firmware adapter converts its own units on the way
//! in and out.
//!
//! Motor output is duty [0, 1]: applied voltage over
//! [`SensorData::battery_voltage`], the ceiling allocation saturates against.
//!
//! # Axes
//!
//! Every 3-axis array is in the body's **FRD** frame (x forward, y right,
//! z down; right-handed):
//!
//!   **\[roll=0, pitch=1, yaw=2\]**, i.e. \[p, q, r\]
//!
//! so positive roll lowers the right side, positive pitch raises the nose,
//! and positive yaw turns the nose right.  Body specific force is `[x, y,
//! z]` in the same frame (−g₀ on z when level and at rest).
//!
//! # Motors
//!
//! Every 4-motor array is by the **firmware's motor index**, whatever order
//! its mixer numbers them in.  The gain blob records where each index sits
//! and which way it spins ([`crate::gains::MotorLayout`]), so the gains and
//! the airframe agree, and an adapter checks the layout against its mixer.
//!
//! Key layout guarantee: `Quantity<U, f32>` is `#[repr(transparent)]` over
//! `f32`, and `MixedUnitMatrix` carries only `PhantomData` alongside its inner
//! `SMatrix`.  Everything erases to raw floats at compile time, so the
//! `#[repr(C)]` structs below have the same layout as their C declarations
//! (plain `float` arrays) even though every field is a typed quantity.
//!
//! Dimensionless values (duty, throttle, scales, ratios, filter
//! coefficients) stay plain `f32`.  Accelerometers commonly read in standard
//! gravities g₀; g₀ = 9.80665 m/s² is not a power-of-2·3·5·π multiple of
//! m/s², so it is a non-storage unit (`g_n`, stored as 0.980665 dam/s²), and
//! such a reading is declared in it on the way in ([`accel_from_g`]).

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{MixedUnitMatrix, UniformUnitMatrix};
use nalgebra::SMatrix;
use nalgebra::{Const, Matrix, U1};
use whippyalgebra::nalgebra::__backend::RawStorage;
use whippyalgebra::{CollapseUniform, DimList, FromRaw, GaugeReproduces, Uniform};
use whippyunits::api::rescale_f32;
use whippyunits::{qty, quantity, DivUnit, Quantity, UnitDiv};

// Re-export for use in other modules
pub use whippyalgebra::nalgebra as nalgebra;

// ---------------------------------------------------------------------------
// Dimension lists
// ---------------------------------------------------------------------------

/// Attitude rate axes: [roll, pitch, yaw] in rot/s.
pub type AttitudeAxes = dims![rot / s, rot / s, rot / s];

/// Integrated attitude error axes: [roll, pitch, yaw] in rot.
pub type IntegralAxes = dims![rot, rot, rot];

/// Derivative error axes: [roll, pitch, yaw] in rot/s².
pub type DerivativeAxes = dims![rot / s ^ 2, rot / s ^ 2, rot / s ^ 2];

/// Motor voltage outputs, by motor index, in V.
pub type MotorAxes = dims![V, V, V, V];

/// Motor speeds: one per motor, in shaft rot/s.
pub type MotorSpeedAxes = dims![rot / s, rot / s, rot / s, rot / s];

/// Signed squared motor speeds ω|ω|: one per motor, in rot²/s².
pub type MotorSpeedSqAxes = dims![rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2];

/// Dimensionless column (single column vector tag).
pub type Scalar = dims![1];

// ---------------------------------------------------------------------------
// Gain matrix types
// ---------------------------------------------------------------------------

/// Proportional gain Kp: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s) = V·s/rot.
pub type KpMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

/// Integral gain Ki: 4 motors × 3 axes.
/// Entry (i,j): V / rot.
pub type KiMatrix = MixedUnitMatrix<MotorAxes, IntegralAxes, SMatrix<f32, 4, 3>>;

/// Derivative gain Kd: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s²) = V·s²/rot.
pub type KdMatrix = MixedUnitMatrix<MotorAxes, DerivativeAxes, SMatrix<f32, 4, 3>>;

/// Feedforward gain B⁻¹: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s) = V·s/rot.
pub type BInvMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

/// Reference-model rate ← motor-speed coupling: 3 axes × 4 motors.
/// Entry (i,j): (rot/s) / (rot/s) — dimensionless.
pub type RateFromSpeedMatrix = MixedUnitMatrix<AttitudeAxes, MotorSpeedAxes, SMatrix<f32, 3, 4>>;

/// Reference-model rate ← voltage over one tick: 3 axes × 4 motors.
/// Entry (i,j): (rot/s) / V.
pub type RateFromVoltMatrix = MixedUnitMatrix<AttitudeAxes, MotorAxes, SMatrix<f32, 3, 4>>;

/// Affine linearizer G = −B⁺·P: 4 motors × 4 signed squared motor speeds.
/// Entry (i,j): V / (rot²/s²) = V·s²/rot².
pub type AffineMatrix = MixedUnitMatrix<MotorAxes, MotorSpeedSqAxes, SMatrix<f32, 4, 4>>;

// ---------------------------------------------------------------------------
// State vector types
// ---------------------------------------------------------------------------

/// Proportional error vector: 3 attitude rates in rot/s.
pub type PropErrorVec = MixedUnitMatrix<AttitudeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Integral error vector: 3 integrated errors in rot.
pub type IntegralErrorVec = MixedUnitMatrix<IntegralAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Derivative error vector: 3 rate derivatives in rot/s².
pub type DerivErrorVec = MixedUnitMatrix<DerivativeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Setpoint diff vector: 3 rate changes in rot/s.
pub type SetpointDiffVec = MixedUnitMatrix<AttitudeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Motor voltage vector: 4 motor voltages in V.
pub type MotorVoltageVec = MixedUnitMatrix<MotorAxes, Scalar, SMatrix<f32, 4, 1>>;

/// Motor speed deviation vector in shaft rot/s.
pub type MotorSpeedVec = MixedUnitMatrix<MotorSpeedAxes, Scalar, SMatrix<f32, 4, 1>>;

/// Signed squared motor speed vector ω|ω| in rot²/s².
pub type MotorSpeedSqVec = MixedUnitMatrix<MotorSpeedSqAxes, Scalar, SMatrix<f32, 4, 1>>;

// ---------------------------------------------------------------------------
// Scalar quantity aliases
// ---------------------------------------------------------------------------

/// A rotation rate in rot/s (also motor shaft speed).
pub type RotPerSec = qty!(rot / s, f32);
/// A rotation (integrated angle) in rot.
pub type Rotation = qty!(rot, f32);
/// Angular acceleration in rot/s².
pub type RotPerSecSq = qty!(rot / s ^ 2, f32);
/// Angular jerk in rot/s³.
pub type RotPerSecCu = qty!(rot / s ^ 3, f32);
/// A signed squared motor speed ω|ω| in rot²/s².
pub type RotSqPerSecSq = qty!(rot ^ 2 / s ^ 2, f32);
/// A voltage in V.
pub type Volt = qty!(V, f32);
/// A time duration in s.
pub type Seconds = qty!(s, f32);
/// A timestamp on the firmware's free-running microsecond clock (wraps).
pub type Micros = qty!(us, u32);
/// A frequency in Hz.
pub type Hertz = qty!(Hz, f32);
/// Voltage per rotation rate, V·s/rot (Kp, G_h⁺, motor drag k_ω).
pub type VoltPerRps = qty!(V * s / rot, f32);
/// Voltage per signed squared speed, V·s²/rot² (G, motor drag k_ω²).
pub type VoltPerRpsSq = qty!(V * s ^ 2 / rot ^ 2, f32);
/// Voltage per rotation, V/rot (Ki).
pub type VoltPerRot = qty!(V / rot, f32);
/// Voltage per angular acceleration, V·s²/rot (Kd).
pub type VoltPerRpsPerSec = qty!(V * s ^ 2 / rot, f32);
/// Rotation rate per volt over one tick, rot/(s·V) (the reference model's Bd).
pub type RpsPerVolt = qty!(rot / s / V, f32);
/// Specific force (what an accelerometer measures) in m/s².
pub type MetersPerSecSq = qty!(m / s ^ 2, f32);
/// What a `g_n` declaration stores: 1 g₀ is 0.980665 dam/s².
pub type DecaMetersPerSecSq = qty!(dam / s ^ 2, f32);
/// Angular acceleration per specific force, rot/m (airframe drag).
pub type RotPerMeter = qty!(rot / m, f32);
/// Angular acceleration per signed squared speed, 1/rot (thrust torque).
pub type PerRot = qty!(1 / rot, f32);

/// Standard gravity g₀.
pub const STANDARD_GRAVITY: MetersPerSecSq = mps2(quantity!(1.0, g_n, f32));

/// m/s² of a quantity declared in g₀.
pub const fn mps2(a: DecaMetersPerSecSq) -> MetersPerSecSq {
    rescale_f32(a)
}

/// Specific force from an accelerometer reading in g₀.
pub const fn accel_from_g(g: f32) -> MetersPerSecSq {
    mps2(quantity!(g, g_n, f32))
}

// ---------------------------------------------------------------------------
// Vector ↔ array conversion
// ---------------------------------------------------------------------------

/// A uniform column vector (e.g. [`PropErrorVec`], [`MotorVoltageVec`]) from
/// its entries; the expected type picks the dimension lists, whose entry
/// unit must be the entries'.
pub fn from_array<U, Rows, Cols, const N: usize>(
    entries: [Quantity<U, f32>; N],
) -> MixedUnitMatrix<Rows, Cols, SMatrix<f32, N, 1>>
where
    U: Clone,
    Rows: DimList + CollapseUniform,
    Cols: DimList + CollapseUniform,
    Uniform<Rows>: UnitDiv<Uniform<Cols>>,
    DivUnit<Uniform<Rows>, Uniform<Cols>>: GaugeReproduces<U>,
{
    UniformUnitMatrix::<U, SMatrix<f32, N, 1>>::from_fn(|i, _| entries[i]).gauge()
}

/// The entries of a uniform column vector.
pub fn to_array<Rows, Cols, S, const N: usize>(
    v: &MixedUnitMatrix<Rows, Cols, Matrix<f32, Const<N>, U1, S>>,
) -> [Quantity<DivUnit<Uniform<Rows>, Uniform<Cols>>, f32>; N]
where
    Rows: CollapseUniform,
    Cols: CollapseUniform,
    Uniform<Rows>: UnitDiv<Uniform<Cols>>,
    S: RawStorage<f32, Const<N>, U1>,
    Quantity<DivUnit<Uniform<Rows>, Uniform<Cols>>, f32>: FromRaw<f32>,
{
    let u = v.as_uniform();
    core::array::from_fn(|i| u.get(i, 0))
}

// ---------------------------------------------------------------------------
// Controller I/O; the `#[repr(C)]` ones cross the FFI
// ---------------------------------------------------------------------------

/// Sensor data in WhippyFlight's units, as the controller consumes it.
///
/// FRD axes, motors by index.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SensorData {
    /// Gyro rates: [roll, pitch, yaw].
    pub gyro_rps: [RotPerSec; 3],
    /// Motor shaft speeds.
    pub motor_rps: [RotPerSec; 4],
    /// Battery voltage.
    pub battery_voltage: Volt,
    /// Body-frame specific force: [x, y, z].
    pub accel: [MetersPerSecSq; 3],
}

/// Per-tick setpoint for [`crate::controller::ControllerState`].
///
/// FRD axes.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SetpointData {
    /// Desired attitude rates in rot/s: [roll, pitch, yaw].
    ///
    /// On the firmware path this is the [`crate::setpoint`] estimate r(t),
    /// built from the pilot's raw setpoint packets.
    pub rate_rps: [RotPerSec; 3],
    /// Throttle as a fraction [0, 1].
    pub throttle: f32,
}

/// Motor output of the control stack, by motor index.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct ControlOutput {
    /// Motor duty cycle fractions [0, 1], which the adapter maps onto its
    /// ESC protocol's output range.
    pub motor_duty: [f32; 4],
}

/// The controller's internal state for one tick, in WhippyFlight's units,
/// for an adapter to log.  FRD axes, motors by index.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct LqrTelemetry {
    /// Per-motor feedback voltage.
    pub feedback_v: [Volt; 4],
    /// Per-motor feedforward voltage.
    pub feedforward_v: [Volt; 4],
    /// Per-motor affine linearizer voltage.
    pub affine_v: [Volt; 4],
    /// Proportional error: [roll, pitch, yaw] in rot/s.
    pub error_prop: [RotPerSec; 3],
    /// Integral error state: [roll, pitch, yaw] in rot.
    pub error_integral: [Rotation; 3],
    /// The integral state over its time constant, e_int/τ_int: the rate error
    /// low-passed over the integral's time constant, in rot/s.
    pub error_integral_rate: [RotPerSec; 3],
    /// Derivative error state: [roll, pitch, yaw] in rot/s².
    pub error_derivative: [RotPerSecSq; 3],
    /// Interpolated Kp row of motor 0: [roll, pitch, yaw].
    pub kp_motor0: [VoltPerRps; 3],
    /// Collective along the thrust-balanced direction (V, motor mean).
    pub collective_v: Volt,
    /// Attitude command scale applied by the allocator; 1 unless saturated.
    pub command_scale: f32,
    /// Rate setpoint the controller tracked: [roll, pitch, yaw] in rot/s.
    pub setpoint_rps: [RotPerSec; 3],
    /// Feedforward's h-tick-ahead target: [roll, pitch, yaw] in rot/s.
    pub ff_target_rps: [RotPerSec; 3],
    /// Status: 0 = no gains, 1 = gains loaded, 2 = gains corrupt.
    pub status: u8,
}

/// Result of gain loading.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainLoadResult {
    Ok = 0,
    TooShort = 1,
    BadMagic = 2,
    BadChecksum = 3,
    BadVersion = 4,
    ResolutionTooLarge = 5,
    /// Blob is not 4-byte aligned; tables are read from it in place.
    Misaligned = 6,
    /// Blob is larger than the target's RAM buffer (reported by the adapter).
    TooLarge = 7,
    /// No storage for gains on this target (reported by the adapter).
    NoStore = 8,
    /// The blob's motor layout block is not a quad-X layout.
    BadLayout = 9,
    /// The blob's motor layout is not the airframe's (reported by the adapter).
    LayoutMismatch = 10,
}
