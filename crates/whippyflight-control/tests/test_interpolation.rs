//! Tests for 4D multilinear gain interpolation.

mod common;

use whippyflight_control::gains::spec::BlobSpec;
use whippyflight_control::gains::{GainStorage, load_gains, REF_MOTOR_SLOTS, REF_MOTOR_WIDTH};
use whippyflight_control::interpolation::interpolate_gains;
use whippyflight_control::types::*;

/// Fill a table with a gradient: entry (motor, axis) at vertex
/// (i0, i1, i2, i3) = base + i0 + i1*2 + i2*4 + i3*8 + motor*0.01 + axis*0.001.
fn gradient_table(table: &mut [f32], n: usize, base: f32) {
    for (i, v) in table.iter_mut().enumerate() {
        let point = i / 12;
        let (motor, axis) = ((i % 12) / 3, i % 3);
        let idx = [point / (n * n * n), point / (n * n) % n, point / n % n, point % n];
        let vertex_val = idx[0] + idx[1] * 2 + idx[2] * 4 + idx[3] * 8;
        *v = base + vertex_val as f32 + motor as f32 * 0.01 + axis as f32 * 0.001;
    }
}

/// Build a gain blob at resolution N=2 where each table has a known gradient:
/// the value at each grid vertex encodes its position in the hypercube.
///
/// For a 4-motor × 3-axis matrix at vertex (i0, i1, i2, i3),
/// entry (motor, axis) = table_base + i0 + i1*2 + i2*4 + i3*8 + motor*0.01 + axis*0.001
///
/// This lets us verify that:
/// - At grid corners, interpolation returns exact vertex values
/// - At midpoints, interpolation returns the average of vertices
///
/// The reference model's motor-speed gain bd_w[m] is 1 + m + 10·(grid index).
fn make_gradient_blob() -> Vec<u8> {
    let n = 2usize;
    let mut spec = BlobSpec::new(n as u8);
    spec.max_rps = 200.0;
    gradient_table(&mut spec.ki, n, 0.0);
    gradient_table(&mut spec.kd, n, 100.0);
    gradient_table(&mut spec.kp, n, 200.0);
    for g in 0..n {
        for m in 0..4 {
            spec.ref_motor[g * REF_MOTOR_WIDTH + m * REF_MOTOR_SLOTS + 7] = (1 + m + 10 * g) as f32;
        }
    }
    spec.encode()
}

fn load_gradient_gains() -> GainStorage {
    let blob = make_gradient_blob();
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

#[test]
fn interpolation_at_min_corner() {
    let storage = load_gradient_gains();

    // All motors at min_rps (100.0) → vertex (0,0,0,0)
    let interp = interpolate_gains(&[RotPerSec::new(100.0); 4], &storage);

    // At vertex (0,0,0,0): vertex_val = 0
    // kp (table_base=200.0): entry (0,0) = 200.0 + 0 + 0*0.01 + 0*0.001 = 200.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4), "kp(0,0) = {kp:?}");

    // ki (table_base=0.0): entry (1,2) = 0.0 + 0 + 1*0.01 + 2*0.001 = 0.012
    let ki = interp.ki.as_uniform().get(1, 2);
    assert!((ki - VoltPerRot::new(0.012)).abs() < VoltPerRot::new(1e-4), "ki(1,2) = {ki:?}");
}

#[test]
fn interpolation_at_max_corner() {
    let storage = load_gradient_gains();

    // All motors at max_rps (200.0) → vertex (1,1,1,1)
    let interp = interpolate_gains(&[RotPerSec::new(200.0); 4], &storage);

    // vertex (1,1,1,1): vertex_val = 1 + 2 + 4 + 8 = 15
    // kp (base=200.0): entry (0,0) = 200.0 + 15 = 215.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(215.0)).abs() < VoltPerRps::new(1e-3), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_at_midpoint() {
    let storage = load_gradient_gains();

    // All motors at midpoint (150.0) → frac = [0.5, 0.5, 0.5, 0.5]
    // Result should be average of all 16 vertices
    let interp = interpolate_gains(&[RotPerSec::new(150.0); 4], &storage);

    // Average of vertex_val over all 16 vertices:
    // vertices have vals 0..15, average = 7.5
    // kp (base=200.0): entry (0,0) = 200.0 + 7.5 = 207.5
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!(
        (kp - VoltPerRps::new(207.5)).abs() < VoltPerRps::new(1e-2),
        "kp(0,0) at midpoint = {kp:?}"
    );
}

#[test]
fn interpolation_clamps_below_min() {
    let storage = load_gradient_gains();

    // Motor speeds below min_rps should clamp to min corner
    let interp = interpolate_gains(&[RotPerSec::new(0.0); 4], &storage);

    // Should match vertex (0,0,0,0) — same as min corner test
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_clamps_above_max() {
    let storage = load_gradient_gains();

    // Motor speeds above max_rps should clamp to max corner
    let interp = interpolate_gains(&[RotPerSec::new(9999.0); 4], &storage);

    // Should match vertex (1,1,1,1)
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(215.0)).abs() < VoltPerRps::new(1e-3), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_distinguishes_tables() {
    let storage = load_gradient_gains();

    let interp = interpolate_gains(&[RotPerSec::new(100.0); 4], &storage); // min corner

    // At vertex (0,0,0,0), entry (0,0):
    // ki (base=0):   0.0
    // kd (base=100): 100.0
    // kp (base=200): 200.0
    let ki = interp.ki.as_uniform().get(0, 0);
    let kd = interp.kd.as_uniform().get(0, 0);
    let kp = interp.kp.as_uniform().get(0, 0);

    assert!((ki - VoltPerRot::new(0.0)).abs() < VoltPerRot::new(1e-4));
    assert!((kd - VoltPerRpsPerSec::new(100.0)).abs() < VoltPerRpsPerSec::new(1e-4));
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4));
}

#[test]
fn interpolation_asymmetric_motors() {
    let storage = load_gradient_gains();

    // Motor 1 at max, others at min: each motor axis interpolates independently
    let motor_rps = [100.0, 200.0, 100.0, 100.0].map(RotPerSec::new);
    let interp = interpolate_gains(&motor_rps, &storage);

    // frac = [0.0, 1.0, 0.0, 0.0]
    // Active vertices: (0, 1, 0, 0) with weight 1.0
    // vertex_val = 0 + 1*2 + 0 + 0 = 2
    // kp entry (0,0) = 200.0 + 2 = 202.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!(
        (kp - VoltPerRps::new(202.0)).abs() < VoltPerRps::new(1e-3),
        "kp(0,0) with asymmetric motors = {kp:?}"
    );
}

#[test]
fn reference_model_follows_each_motors_own_speed() {
    let storage = load_gradient_gains();

    // Grid positions 0, 1, 0.5, 0.25 → bd_w[m] = 1 + m + 10·position.
    let motor_rps = [100.0, 200.0, 150.0, 125.0].map(RotPerSec::new);
    let interp = interpolate_gains(&motor_rps, &storage);
    for (m, position) in [0.0, 1.0, 0.5, 0.25].into_iter().enumerate() {
        let want = RpsPerVolt::new(1.0 + m as f32 + 10.0 * position);
        let got = interp.ref_model.bd_w[m];
        assert!((got - want).abs() < RpsPerVolt::new(1e-3), "bd_w[{m}] = {got:?}, want {want:?}");
    }
    // The speed-free blocks come through as stored.
    assert_eq!(interp.ref_model.ad_rr, [1.0; 3]);
}

#[test]
fn horizon_inverse_is_the_least_norm_inverse_of_the_interpolated_gh() {
    // gh[i][m] = (i == m ? 2 : 0.3·(i + 1)) · (1 + grid index): motor 3 drives
    // every axis, so the inverse spreads over all four motors.
    let n = 2usize;
    let mut spec = BlobSpec::new(n as u8);
    spec.max_rps = 200.0;
    for g in 0..n {
        for m in 0..4 {
            for i in 0..3 {
                let v = if i == m { 2.0 } else { 0.3 * (i + 1) as f32 };
                spec.ref_motor[g * REF_MOTOR_WIDTH + m * REF_MOTOR_SLOTS + 11 + i] = v * (1 + g) as f32;
            }
        }
    }
    let blob = spec.encode();
    let mut storage = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(&blob), &mut storage), GainLoadResult::Ok);

    let interp = interpolate_gains(&[100.0, 200.0, 150.0, 125.0].map(RotPerSec::new), &storage);
    let gh = |i, m| interp.ref_model.gh.as_uniform().get(i, m).unsafe_value;
    let binv = |m, i| interp.binv.as_uniform().get(m, i).unsafe_value;
    assert!((gh(1, 1) - 4.0).abs() < 1e-3, "gh follows motor 1's speed: {}", gh(1, 1));

    // G·G⁺ = I, and G⁺·G is symmetric (G⁺ lies in G's row space).
    for i in 0..3 {
        for k in 0..3 {
            let p: f32 = (0..4).map(|m| gh(i, m) * binv(m, k)).sum();
            let want = if i == k { 1.0 } else { 0.0 };
            assert!((p - want).abs() < 1e-4, "(G·G⁺)[{i}][{k}] = {p}");
        }
    }
    for a in 0..4 {
        for b in 0..4 {
            let pab: f32 = (0..3).map(|i| binv(a, i) * gh(i, b)).sum();
            let pba: f32 = (0..3).map(|i| binv(b, i) * gh(i, a)).sum();
            assert!((pab - pba).abs() < 1e-4, "(G⁺·G)[{a}][{b}] = {pab}, [{b}][{a}] = {pba}");
        }
    }
}

#[test]
fn horizon_inverse_is_zero_without_a_reference_model() {
    let interp = interpolate_gains(&[RotPerSec::new(150.0); 4], &load_gradient_gains());
    for m in 0..4 {
        for i in 0..3 {
            assert_eq!(interp.binv.as_uniform().get(m, i), VoltPerRps::new(0.0));
        }
    }
}
