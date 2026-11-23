//! Spectral radius by repeated squaring (Gelfand: ‖Mᵏ‖^(1/k) → ρ), which
//! needs no eigendecomposition and so cannot stall on a non-normal matrix.

use whippyalgebra::nalgebra::{DMatrix, UniformUnitMatrix};
use whippyunits::unit;

/// A dimensionless endomorphism.
pub type Endomorphism = UniformUnitMatrix<unit!(1), DMatrix<f64>>;

/// Most squarings: the estimate's bias, ln‖C‖/2ᵏ for a transient gain C,
/// is then far below any margin the callers resolve.
const MAX_SQUARINGS: usize = 48;
/// Squarings before the estimate may be taken as settled.
const MIN_SQUARINGS: usize = 8;
/// Relative change of ln ρ that settles the estimate.
const SETTLED: f64 = 1e-3;

/// The spectral radius of `m`.
pub fn spectral_radius(m: &Endomorphism) -> f64 {
    let norm = f64::from(m.norm());
    if norm == 0.0 || !norm.is_finite() {
        return if norm == 0.0 { 0.0 } else { f64::INFINITY };
    }
    let mut power = m.scale(1.0 / norm);
    let mut log_norm = norm.ln();
    let mut estimate = log_norm;
    for k in 1..=MAX_SQUARINGS {
        power = power.clone() * power;
        let n = f64::from(power.norm());
        if n == 0.0 {
            return 0.0;
        }
        power = power.scale(1.0 / n);
        log_norm = 2.0 * log_norm + n.ln();
        let next = log_norm / 2f64.powi(k as i32);
        let change = (next - estimate).abs();
        estimate = next;
        if k >= MIN_SQUARINGS && change <= SETTLED * estimate.abs() {
            break;
        }
    }
    estimate.exp()
}
