//! The unit spaces of the design: the plant's, and the LQR's augmented state.
//!
//! Model order throughout: rates [yaw, pitch, roll], motors by slot
//! [front left, rear left, front right, rear right], all FRD; motor speeds
//! in shaft rev/s (rot/s).

use whippyalgebra::nalgebra::{MixedUnitMatrix, SMatrix, UniformUnitMatrix};
use whippyalgebra::{Concatenated, Mapped, Reciprocal, dims};
use whippyunits::unit;

/// The dual (element-wise reciprocal) of a dimension list.
pub type Dual<D> = Mapped<Reciprocal, D>;

/// Body rates, model order.
pub type Rates = dims![rot / s, rot / s, rot / s];
/// Motor speeds, by slot.
pub type Motors = dims![rot / s, rot / s, rot / s, rot / s];
/// Motor voltages, by slot.
pub type Volts = dims![V, V, V, V];
/// The controller's integral of each rate error.
pub type Integrals = dims![rot, rot, rot];
/// The controller's filtered derivative of each rate error.
pub type Derivatives = dims![rot / s ^ 2, rot / s ^ 2, rot / s ^ 2];

/// External angular accelerations, model order.
pub type Torques = dims![rot / s ^ 2, rot / s ^ 2, rot / s ^ 2];
/// The gyro noise's shaping states: this tick's noise, then last tick's.
pub type GyroNoise = Concatenated<Rates, Rates>;
/// What disturbs the loop: the turbulence's torques and the gyro's noise.
pub type DisturbanceStates = Concatenated<Torques, GyroNoise>;
/// The units of what the firmware feeds back: the integral, derivative and
/// rate errors. The state starts with the integral and derivative states and
/// the body rates, in these units.
pub type Applied = Concatenated<Concatenated<Integrals, Derivatives>, Rates>;
/// The rest: the motor speeds (the operating point the gains are scheduled
/// at, so δω ≡ 0 to the feedback), the last measured rate (the derivative's
/// one-tick delay) and the disturbances.
pub type Unapplied = Concatenated<Concatenated<Motors, Rates>, DisturbanceStates>;
/// The LQR's state: the plant augmented with the controller's filters and
/// the disturbances' shaping filters.
pub type State = Concatenated<Applied, Unapplied>;

pub const APPLIED: usize = 9;
pub const STATE: usize = 25;

/// A state endomorphism (one tick of the loop).
pub type StateMap = MixedUnitMatrix<State, State, SMatrix<f64, STATE, STATE>>;
/// Voltage into the state.
pub type InputMap = MixedUnitMatrix<State, Volts, SMatrix<f64, STATE, 4>>;
/// What the firmware feeds back, from the state: the integral and derivative
/// after this tick's update, and the measured rate.
pub type Measurement = MixedUnitMatrix<Applied, State, SMatrix<f64, APPLIED, STATE>>;
/// A quadratic state cost.
pub type StateCost = MixedUnitMatrix<Dual<State>, State, SMatrix<f64, STATE, STATE>>;
/// A quadratic voltage cost.
pub type VoltCost = MixedUnitMatrix<Dual<Volts>, Volts, SMatrix<f64, 4, 4>>;
/// A state covariance.
pub type Covariance = MixedUnitMatrix<State, Dual<State>, SMatrix<f64, STATE, STATE>>;
/// The feedback gains on the applied state: u = −K·x.
pub type Gain = MixedUnitMatrix<Volts, Applied, SMatrix<f64, 4, APPLIED>>;
/// A voltage endomorphism (a projection of the motors' commands).
pub type VoltMap = MixedUnitMatrix<Volts, Volts, SMatrix<f64, 4, 4>>;
/// An applied-state endomorphism.
pub type AppliedMap = MixedUnitMatrix<Applied, Applied, SMatrix<f64, APPLIED, APPLIED>>;

/// Each slot's weight in the yaw torque of the speeds' signed squares.
pub const YAW_PATTERN: [f64; 4] = [1.0, -1.0, -1.0, 1.0];

/// Gain blocks, by slot × model axis.
pub type IntegralGain = UniformUnitMatrix<unit!(V / rot), SMatrix<f64, 4, 3>>;
pub type DerivativeGain = UniformUnitMatrix<unit!(V * s ^ 2 / rot), SMatrix<f64, 4, 3>>;
pub type RateGain = UniformUnitMatrix<unit!(V * s / rot), SMatrix<f64, 4, 3>>;
