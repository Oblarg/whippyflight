//! 4D multilinear interpolation of gain matrices over the motor-speed hypercube.
//!
//! The gain lookup tables are indexed by the 4 motor speeds (by motor
//! index).  At runtime we interpolate across the 16 vertices of the
//! surrounding 4-cube to get smooth gain variation.
//!
//! Each table holds quantities of its gain's unit (see [`GainTable::blend`]);
//! the interpolated entries are gauged into typed `MixedUnitMatrix` values,
//! which checks at compile time that every table feeds the matrix of its unit.

use crate::gains::{
    ref_slots_from_motors, GainStorage, GainTable, ReferenceModel, TableEntry, REF_FLOATS, REF_MOTOR_SLOTS,
    REF_MOTOR_WIDTH,
};
use crate::types::*;

use nalgebra::SMatrix;
use whippyalgebra::nalgebra::UniformUnitMatrix;

/// Interpolated gain set at a single operating point, fully typed.
pub struct InterpolatedGains {
    pub ki: KiMatrix,
    pub kd: KdMatrix,
    pub kp: KpMatrix,
    /// Horizon inverse G_h⁺, computed from the interpolated G_h.
    pub binv: BInvMatrix,
    /// Reference-model terms.
    pub ref_model: ReferenceModel,
}

impl InterpolatedGains {
    pub fn zero() -> Self {
        Self {
            ki: KiMatrix::from_nalgebra(SMatrix::zeros()),
            kd: KdMatrix::from_nalgebra(SMatrix::zeros()),
            kp: KpMatrix::from_nalgebra(SMatrix::zeros()),
            binv: BInvMatrix::from_nalgebra(SMatrix::zeros()),
            ref_model: ReferenceModel::from_slots(&[0.0; REF_FLOATS]),
        }
    }
}

/// The 16 corners of the grid cell around an operating point, as flat
/// grid-point indices and multilinear weights.
struct Vertices {
    points: [usize; 16],
    weights: [f32; 16],
}

impl Vertices {
    /// Bit m of the corner number picks motor m's lower (0) or upper (1)
    /// grid index; the corner's weight is the product over motors of
    /// `1 − frac` or `frac` respectively, so the weights sum to 1.
    fn new(n: usize, lower: &[usize; 4], frac: &[f32; 4]) -> Self {
        let mut points = [0usize; 16];
        let mut weights = [0.0f32; 16];
        for corner in 0..16usize {
            let mut idx = [0usize; 4];
            let mut weight = 1.0f32;
            for m in 0..4 {
                let upper = (corner >> m) & 1 == 1;
                idx[m] = lower[m] + upper as usize;
                weight *= if upper { frac[m] } else { 1.0 - frac[m] };
            }
            points[corner] = ((idx[0] * n + idx[1]) * n + idx[2]) * n + idx[3];
            weights[corner] = weight;
        }
        Self { points, weights }
    }

    /// A table's rows (`W` slots each) interpolated at the operating point.
    fn interpolate<E: TableEntry, const W: usize>(&self, table: &GainTable<E>) -> [E; W] {
        table.blend::<W>(&self.points, &self.weights)
    }
}

/// A 4×3 (motor × axis) table's interpolated entries as a uniform matrix,
/// ready to gauge into the gain's mixed-unit type.
macro_rules! interpolate_4x3 {
    ($vertices:expr, $table:expr) => {{
        let entries = $vertices.interpolate::<_, 12>(&$table);
        UniformUnitMatrix::<_, SMatrix<f32, 4, 3>>::from_fn(|i, j| entries[3 * i + j]).gauge()
    }};
}

/// Interpolate gains at the given motor speeds (shaft rev/s).
pub fn interpolate_gains(
    motor_rps: &[RotPerSec; 4],
    storage: &GainStorage,
) -> InterpolatedGains {
    let header = &storage.header;
    let n = header.resolution as usize;
    let step = (header.max_rps - header.min_rps) / (n as f32 - 1.0);

    // Each motor's position on the grid: the cell's lower index and the
    // fraction of the way to the upper one.  Speeds off the grid clamp to
    // its edge, and the top grid point is the last cell at frac = 1.
    let mut lower = [0usize; 4];
    let mut frac = [0.0f32; 4];
    for m in 0..4 {
        let clamped = motor_rps[m].clamp(header.min_rps, header.max_rps);
        let position: f32 = ((clamped - header.min_rps) / step).into();
        lower[m] = (position as usize).min(n - 2);
        frac[m] = position - lower[m] as f32;
    }

    let vertices = Vertices::new(n, &lower, &frac);

    let ref_model = ReferenceModel::from_slots(&reference_model(storage, &lower, &frac));
    InterpolatedGains {
        ki: interpolate_4x3!(vertices, storage.ki),
        kd: interpolate_4x3!(vertices, storage.kd),
        kp: interpolate_4x3!(vertices, storage.kp),
        binv: ref_model.horizon_inverse(),
        ref_model,
    }
}

/// The reference model: each motor's slots interpolated along its own speed.
///
/// The table's row at grid index i holds every motor's slots at speed i,
/// motor m's at `m · REF_MOTOR_SLOTS`, so motor m interpolates its block
/// between the rows at its lower and upper index.
fn reference_model(storage: &GainStorage, lower: &[usize; 4], frac: &[f32; 4]) -> [f32; REF_FLOATS] {
    let motors = core::array::from_fn(|m| {
        storage.ref_model.blend_slots::<REF_MOTOR_SLOTS>(
            REF_MOTOR_WIDTH,
            m * REF_MOTOR_SLOTS,
            &[lower[m], lower[m] + 1],
            &[1.0 - frac[m], frac[m]],
        )
    });
    ref_slots_from_motors(&storage.ref_const, &motors)
}
