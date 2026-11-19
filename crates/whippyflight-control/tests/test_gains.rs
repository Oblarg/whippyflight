//! Tests for gain binary parsing and validation.

mod common;

use whippyflight_control::gains::spec::BlobSpec;
use whippyflight_control::gains::*;
use whippyflight_control::types::*;

/// A valid blob of the given resolution, every gain table filled with
/// `fill_value` (in its own unit).
fn make_gain_blob(resolution: u8, fill_value: f32) -> Vec<u8> {
    let mut spec = BlobSpec::new(resolution);
    for table in [&mut spec.ki, &mut spec.kd, &mut spec.kp] {
        table.fill(fill_value);
    }
    spec.encode()
}

/// Recompute the CRC after editing `blob`'s payload.
fn refresh_crc(blob: &mut [u8]) {
    let payload_size = blob.len() - 4;
    let crc = crc32(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());
}

/// A quantity's stored value, in the unit the blob holds it in.
fn raw<Q: bytemuck::Pod>(q: Q) -> f32 {
    bytemuck::cast(q)
}

fn load(blob: &[u8]) -> (GainLoadResult, GainStorage) {
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(blob), &mut storage);
    (result, storage)
}

// =========================================================================
// Gain parsing tests
// =========================================================================

#[test]
fn load_valid_gains_resolution_2() {
    let blob = make_gain_blob(2, 1.0);
    assert_eq!(blob.len(), blob_size(2));
    let (result, storage) = load(&blob);

    assert_eq!(result, GainLoadResult::Ok);
    assert!(storage.loaded);
    assert_eq!(storage.header.resolution, 2);
    assert_eq!(storage.header.min_rps, RotPerSec::new(100.0));
    assert_eq!(storage.header.max_rps, RotPerSec::new(500.0));
    let tol = Seconds::new(1e-9);
    assert!((storage.header.dt - Seconds::new(0.000125)).abs() < tol);
    assert!((storage.header.integral_tau - Seconds::new(0.1)).abs() < tol);
    assert!((storage.header.derivative_tau - Seconds::new(0.01)).abs() < tol);

    // A constant table quantizes exactly.
    let n = 2usize;
    let floats = n * n * n * n * 4 * 3;
    for i in 0..floats {
        assert_eq!(storage.ki.get(i), VoltPerRot::new(1.0), "ki[{i}]");
        assert_eq!(storage.kd.get(i), VoltPerRpsPerSec::new(1.0), "kd[{i}]");
        assert_eq!(storage.kp.get(i), VoltPerRps::new(1.0), "kp[{i}]");
    }
}

#[test]
fn load_valid_gains_resolution_5() {
    let (result, storage) = load(&make_gain_blob(5, 0.5));
    assert_eq!(result, GainLoadResult::Ok);
    assert!(storage.loaded);
    assert_eq!(storage.header.resolution, 5);
}

#[test]
fn blocks_round_trip() {
    let mut spec = BlobSpec::new(3);
    spec.ff_horizon = 4;
    spec.ref_const = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
    spec.affine = core::array::from_fn(|i| i as f32 * 0.25);
    spec.motor_drag = [0.01, 0.0002];
    spec.thrust_balance = [1.1, 0.9, 1.05, 0.95];
    spec.airframe_drag = [5.0, 0.1, -0.2, 0.3, -0.4];
    spec.target_limits = [30.0, 900.0];
    spec.hover = [220.0, 0.6];
    spec.thrust_torque = [-1.3e-4, -1.3e-4, 1.2e-4, 1.2e-4, -1.0e-4, 1.1e-4, -1.0e-4, 1.1e-4];
    spec.layout = MotorLayout {
        position: [
            MotorPosition::RearRight,
            MotorPosition::FrontRight,
            MotorPosition::RearLeft,
            MotorPosition::FrontLeft,
        ],
        spin: [Spin::CounterClockwise, Spin::Clockwise, Spin::Clockwise, Spin::CounterClockwise],
    };
    let (result, s) = load(&spec.encode());
    assert_eq!(result, GainLoadResult::Ok);
    assert_eq!(s.header.ff_horizon, 4);
    assert_eq!(s.ref_const, spec.ref_const);
    assert_eq!(s.affine.map(raw), spec.affine);
    assert_eq!([raw(s.motor_drag.k_w), raw(s.motor_drag.k_w2)], spec.motor_drag);
    assert_eq!(s.thrust_balance, spec.thrust_balance);
    let d = s.airframe_drag;
    assert_eq!(
        [raw(d.lp), raw(d.roll[0]), raw(d.roll[1]), raw(d.pitch[0]), raw(d.pitch[1])],
        spec.airframe_drag
    );
    assert_eq!([raw(s.target_limits.slew), raw(s.target_limits.jerk)], spec.target_limits);
    assert_eq!([raw(s.hover_rps), raw(s.integral_limit_v)], spec.hover);
    let t = s.thrust_torque;
    assert_eq!([t.roll.map(raw), t.pitch.map(raw)].concat(), spec.thrust_torque);
    assert_eq!(s.layout, spec.layout);
}

#[test]
fn quantized_tables_stay_within_a_step() {
    let mut spec = BlobSpec::new(3);
    for (i, v) in spec.kp.iter_mut().enumerate() {
        *v = (i as f32 * 0.37).sin() * (1.0 + (i % 12) as f32);
    }
    for (i, v) in spec.ref_motor.iter_mut().enumerate() {
        *v = (i as f32 * 0.11).cos();
    }
    let (result, s) = load(&spec.encode());
    assert_eq!(result, GainLoadResult::Ok);
    for (i, &v) in spec.kp.iter().enumerate() {
        let err = (raw(s.kp.get(i)) - v).abs();
        assert!(err <= 13.0 / 65535.0, "kp[{i}]: {err}");
    }
    for (i, &v) in spec.ref_motor.iter().enumerate() {
        assert!((s.ref_model.get(i) - v).abs() <= 2.0 / 65535.0, "ref[{i}]");
    }
}

#[test]
fn reject_empty_buffer() {
    let mut storage = GainStorage::new();
    let result = load_gains(&[], &mut storage);
    assert_eq!(result, GainLoadResult::TooShort);
    assert!(!storage.loaded);
}

#[test]
fn reject_truncated_header() {
    let (result, storage) = load(&[0u8; 16]);
    assert_eq!(result, GainLoadResult::TooShort);
    assert!(!storage.loaded);
}

#[test]
fn reject_bad_magic() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[0] = 0xFF;
    // Recompute CRC so we test magic check, not CRC check
    refresh_crc(&mut blob);
    assert_eq!(load(&blob).0, GainLoadResult::BadMagic);
}

#[test]
fn reject_other_versions() {
    for version in [1u16, 8, 9, 10, 12, 99] {
        let mut blob = make_gain_blob(2, 1.0);
        blob[4..6].copy_from_slice(&version.to_le_bytes());
        refresh_crc(&mut blob);
        assert_eq!(load(&blob).0, GainLoadResult::BadVersion, "version {version}");
    }
}

#[test]
fn reject_resolution_zero() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 0;
    refresh_crc(&mut blob);
    assert_eq!(load(&blob).0, GainLoadResult::ResolutionTooLarge);
}

#[test]
fn reject_resolution_too_large() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 6; // > MAX_RESOLUTION(5)
    refresh_crc(&mut blob);
    assert_eq!(load(&blob).0, GainLoadResult::ResolutionTooLarge);
}

#[test]
fn reject_size_of_another_resolution() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 3;
    refresh_crc(&mut blob);
    assert_eq!(load(&blob).0, GainLoadResult::TooShort);
}

#[test]
fn reject_corrupted_crc() {
    let mut blob = make_gain_blob(2, 1.0);
    // Corrupt a data byte without updating CRC
    blob[0x20] ^= 0xFF;
    assert_eq!(load(&blob).0, GainLoadResult::BadChecksum);
}

#[test]
fn reject_truncated_data() {
    let blob = make_gain_blob(2, 1.0);
    // total_size > data.len()
    assert_eq!(load(&blob[..blob.len() / 2]).0, GainLoadResult::TooShort);
}

#[test]
fn reject_bad_layouts() {
    let cw = Spin::Clockwise as u8;
    let ccw = Spin::CounterClockwise as u8;
    let bad: [[u8; LAYOUT_BYTES]; 5] = [
        // Two motors at front left.
        [0, 0, 2, 3, cw, ccw, ccw, cw],
        // No such position.
        [0, 1, 2, 4, cw, ccw, ccw, cw],
        // No such spin.
        [0, 1, 2, 3, cw, ccw, ccw, 2],
        // Diagonals (FL & RR) disagree.
        [0, 1, 2, 3, cw, ccw, ccw, ccw],
        // All one way.
        [0, 1, 2, 3, cw, cw, cw, cw],
    ];
    for codes in bad {
        let mut blob = make_gain_blob(2, 1.0);
        blob[LAYOUT_OFFSET..LAYOUT_OFFSET + LAYOUT_BYTES].copy_from_slice(&codes);
        refresh_crc(&mut blob);
        let (result, storage) = load(&blob);
        assert_eq!(result, GainLoadResult::BadLayout, "{codes:?}");
        assert!(!storage.loaded);
    }
}

#[test]
fn layout_codes_round_trip() {
    let layout = BlobSpec::new(2).layout;
    assert_eq!(MotorLayout::from_codes(&layout.to_codes()), Some(layout));
    // Props-out is as valid as props-in.
    let flipped = MotorLayout {
        spin: layout.spin.map(|s| if s == Spin::Clockwise { Spin::CounterClockwise } else { Spin::Clockwise }),
        ..layout
    };
    assert_eq!(MotorLayout::from_codes(&flipped.to_codes()), Some(flipped));
}

#[test]
fn load_then_reload_clears_previous() {
    let mut storage = GainStorage::new();

    let blob1 = make_gain_blob(2, 1.0);
    assert_eq!(load_gains(common::static_blob(&blob1), &mut storage), GainLoadResult::Ok);
    assert_eq!(storage.header.resolution, 2);
    assert_eq!(storage.ki.get(0), VoltPerRot::new(1.0));

    let blob2 = make_gain_blob(3, 2.0);
    assert_eq!(load_gains(common::static_blob(&blob2), &mut storage), GainLoadResult::Ok);
    assert_eq!(storage.header.resolution, 3);
    assert_eq!(storage.ki.get(0), VoltPerRot::new(2.0));
}

#[test]
fn failed_load_does_not_set_loaded() {
    let blob = make_gain_blob(2, 1.0);
    let mut storage = GainStorage::new();

    load_gains(common::static_blob(&blob), &mut storage);
    assert!(storage.loaded);

    load_gains(common::static_blob(&[0u8; 10]), &mut storage);
    assert!(!storage.loaded);
}

#[test]
fn reject_misaligned_blob() {
    // Tables are read in place, so the blob must be 4-byte aligned.
    let blob = make_gain_blob(2, 1.0);
    let mut padded = vec![0u8];
    padded.extend_from_slice(&blob);
    let misaligned = &common::static_blob(&padded)[1..];
    let mut storage = GainStorage::new();
    assert_eq!(load_gains(misaligned, &mut storage), GainLoadResult::Misaligned);
    assert!(!storage.loaded);
}

#[test]
fn tables_are_views_into_blob() {
    let blob = common::static_blob(&make_gain_blob(2, 1.0));
    let mut storage = GainStorage::new();
    assert_eq!(load_gains(blob, &mut storage), GainLoadResult::Ok);
    let range = blob.as_ptr_range();
    let tables = [
        storage.ki.data_ptr(),
        storage.kd.data_ptr(),
        storage.kp.data_ptr(),
        storage.ref_model.data_ptr(),
    ];
    for ptr in tables {
        assert!(range.contains(&ptr), "table was copied");
    }
}

#[test]
fn with_feedback_zeroes_only_the_terms_left_out() {
    let mut spec = BlobSpec::new(2);
    for table in [&mut spec.ki, &mut spec.kd, &mut spec.kp] {
        table.fill(1.0);
    }
    spec.ref_motor.fill(0.5);
    let blob = spec.encode();
    let p_only = FeedbackTerms { p: true, i: false, d: false };
    let id = FeedbackTerms { p: false, i: true, d: true };
    for keep in [FeedbackTerms::NONE, FeedbackTerms::ALL, p_only, id] {
        let (result, storage) = load(&with_feedback(&blob, keep));
        assert_eq!(result, GainLoadResult::Ok, "{keep:?}");
        for i in 0..storage.kp.len() {
            let kept = |on: bool| if on { 1.0 } else { 0.0 };
            assert!((storage.ki.get(i) - VoltPerRot::new(kept(keep.i))).abs() < VoltPerRot::new(1e-3), "{keep:?} ki[{i}]");
            assert!(
                (storage.kd.get(i) - VoltPerRpsPerSec::new(kept(keep.d))).abs() < VoltPerRpsPerSec::new(1e-3),
                "{keep:?} kd[{i}]"
            );
            assert!((storage.kp.get(i) - VoltPerRps::new(kept(keep.p))).abs() < VoltPerRps::new(1e-3), "{keep:?} kp[{i}]");
        }
        assert!((storage.ref_model.get(0) - 0.5).abs() < 1e-6);
    }
}
