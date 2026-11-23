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
    fixed_weight, ref_slots_from_motors, GainStorage, GainTable, ReferenceModel, TableEntry, REF_FLOATS,
    REF_MOTOR_SLOTS, REF_MOTOR_WIDTH,
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

/// Slots per grid point in a feedback table: motor × axis, 4×3.
const FEEDBACK_SLOTS: usize = 12;

/// The 16 corners of the grid cell around an operating point, as flat
/// grid-point indices and multilinear weights in [`fixed_weight`] units.
struct Vertices {
    points: [usize; 16],
    weights: [u32; 16],
    /// The reciprocal of the rounded weights' sum.
    per_weight: f32,
}

impl Vertices {
    /// Bit m of the corner number picks motor m's lower (0) or upper (1)
    /// grid index; the corner's weight is the product over motors of
    /// `1 − frac` or `frac` respectively, so the weights sum to 1.  Each
    /// motor in turn doubles the corners built so far into its lower and
    /// upper halves.
    fn new(n: usize, lower: &[usize; 4], frac: &[f32; 4]) -> Self {
        let strides = [n * n * n, n * n, n, 1];
        let mut points = [0usize; 16];
        let mut weights = [0.0f32; 16];
        points[0] = (0..4).map(|m| lower[m] * strides[m]).sum();
        weights[0] = 1.0;
        for m in 0..4 {
            let half = 1 << m;
            for corner in 0..half {
                points[corner + half] = points[corner] + strides[m];
                weights[corner + half] = weights[corner] * frac[m];
                weights[corner] *= 1.0 - frac[m];
            }
        }
        let weights = weights.map(fixed_weight);
        let per_weight = 1.0 / weights.iter().sum::<u32>().max(1) as f32;
        Self { points, weights, per_weight }
    }

    /// A 4×3 (motor × axis) feedback table interpolated at the operating point.
    fn interpolate<E: TableEntry>(&self, table: &GainTable<E>) -> [E; FEEDBACK_SLOTS] {
        let codes = table.codes();
        let mut sums = [0u32; FEEDBACK_SLOTS];
        for (&point, &w) in self.points.iter().zip(&self.weights) {
            let start = point * FEEDBACK_SLOTS;
            for (sum, &q) in sums.iter_mut().zip(&codes[start..start + FEEDBACK_SLOTS]) {
                *sum += w * q as u32;
            }
        }
        table.decode(0, &sums, self.per_weight)
    }
}

/// A 4×3 (motor × axis) table interpolated at the operating point as a
/// uniform matrix, ready to gauge into the gain's mixed-unit type.
macro_rules! interpolate_4x3 {
    ($vertices:expr, $table:expr) => {{
        let entries = $vertices.interpolate(&$table);
        UniformUnitMatrix::<_, SMatrix<f32, 4, 3>>::from_fn(|i, j| entries[3 * i + j]).gauge()
    }};
}

/// The feedback gains alone at an operating point.
pub struct FeedbackGains {
    pub ki: KiMatrix,
    pub kd: KdMatrix,
    pub kp: KpMatrix,
}

/// Each motor's position on the grid: the cell's lower index and the
/// fraction of the way to the upper one.  Speeds off the grid clamp to its
/// edge, and the top grid point is the last cell at frac = 1.
fn grid_position(motor_rps: &[RotPerSec; 4], storage: &GainStorage) -> ([usize; 4], [f32; 4]) {
    let header = &storage.header;
    let n = header.resolution as usize;
    let step = (header.max_rps - header.min_rps) / (n as f32 - 1.0);
    let mut lower = [0usize; 4];
    let mut frac = [0.0f32; 4];
    for m in 0..4 {
        let clamped = motor_rps[m].clamp(header.min_rps, header.max_rps);
        let position: f32 = ((clamped - header.min_rps) / step).into();
        lower[m] = (position as usize).min(n - 2);
        frac[m] = position - lower[m] as f32;
    }
    (lower, frac)
}

/// The first half of [`interpolate_gains`]: the operating point and Ki, Kd
/// there.  [`Self::finish`] interpolates the rest at the same point, so the
/// work can be spread over two ticks.
pub struct ScheduleStart {
    lower: [usize; 4],
    frac: [f32; 4],
    vertices: Vertices,
    ki: KiMatrix,
    kd: KdMatrix,
}

impl ScheduleStart {
    pub fn new(motor_rps: &[RotPerSec; 4], storage: &GainStorage) -> Self {
        let (lower, frac) = grid_position(motor_rps, storage);
        let vertices = Vertices::new(storage.header.resolution as usize, &lower, &frac);
        let ki = interpolate_4x3!(vertices, storage.ki);
        let kd = interpolate_4x3!(vertices, storage.kd);
        Self { lower, frac, vertices, ki, kd }
    }

    /// Kp, the reference model and G_h⁺ at the same operating point.
    pub fn finish(self, storage: &GainStorage) -> InterpolatedGains {
        let ref_model = ReferenceModel::from_slots(&reference_model(storage, &self.lower, &self.frac));
        InterpolatedGains {
            ki: self.ki,
            kd: self.kd,
            kp: interpolate_4x3!(self.vertices, storage.kp),
            binv: ref_model.horizon_inverse(),
            ref_model,
        }
    }
}

/// Interpolate gains at the given motor speeds (shaft rev/s).
pub fn interpolate_gains(motor_rps: &[RotPerSec; 4], storage: &GainStorage) -> InterpolatedGains {
    ScheduleStart::new(motor_rps, storage).finish(storage)
}

/// [`interpolate_gains`]' feedback gains only.
pub fn interpolate_feedback(motor_rps: &[RotPerSec; 4], storage: &GainStorage) -> FeedbackGains {
    let (lower, frac) = grid_position(motor_rps, storage);
    let vertices = Vertices::new(storage.header.resolution as usize, &lower, &frac);
    FeedbackGains {
        ki: interpolate_4x3!(vertices, storage.ki),
        kd: interpolate_4x3!(vertices, storage.kd),
        kp: interpolate_4x3!(vertices, storage.kp),
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
