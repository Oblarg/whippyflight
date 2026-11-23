//! f64 quantities of the design.

use whippyalgebra::nalgebra::{SMatrix, UniformUnitMatrix};
use whippyunits::unit;

pub use whippyflight_sim::units::*;

/// An `R`×`C` matrix whose entries are all in unit `U`.
pub type UnitMatrix<U, const R: usize, const C: usize> = UniformUnitMatrix<U, SMatrix<f64, R, C>>;

pub type Dimensionless = unit!(1);
pub type Seconds = unit!(s);
pub type RotPerSecond = unit!(rot / s);
pub type RotPerSecondVolt = unit!(rot / s / V);
pub type VoltSecondPerRot = unit!(V * s / rot);
