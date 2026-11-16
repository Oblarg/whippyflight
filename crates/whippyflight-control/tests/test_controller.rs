//! Tests for the LQR controller logic.

mod common;

use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::spec::BlobSpec;
use whippyflight_control::gains::{GainStorage, REF_MOTOR_SLOTS, REF_MOTOR_WIDTH, load_gains};
use whippyflight_control::types::*;

const DT: Seconds = Seconds::new(0.000125);
const BATTERY: Volt = Volt::new(12.0);

/// Diagonal gains for [`make_identity_gains`].
#[derive(Clone, Copy)]
struct Diag {
    kp: VoltPerRps,
    ki: VoltPerRot,
    kd: VoltPerRpsPerSec,
    gh: RpsPerVolt,
}

impl Diag {
    const ZERO: Self = Self {
        kp: VoltPerRps::new(0.0),
        ki: VoltPerRot::new(0.0),
        kd: VoltPerRpsPerSec::new(0.0),
        gh: RpsPerVolt::new(0.0),
    };
}

/// Build a gain blob with identity-like gains for testing.
/// Kp = diag(kp), Ki = diag(ki), Kd = diag(kd), G_h = diag(gh)
/// (motor i maps to axis i for i < 3, motor 3 = zeros), so G_h⁺ = diag(1/gh).
/// The reference model holds its rate and nothing drives it, so it stays
/// where it was seeded: the first tick's setpoint.
fn make_identity_gains(resolution: u8, gains: &Diag, dt: Seconds) -> Vec<u8> {
    fn diag<Q: bytemuck::Pod>(table: &mut [f32], q: Q) {
        let v: f32 = bytemuck::cast(q);
        BlobSpec::fill(table, |motor, axis| if motor == axis { v } else { 0.0 });
    }
    let mut spec = BlobSpec::new(resolution);
    spec.dt = bytemuck::cast(dt);
    diag(&mut spec.ki, gains.ki);
    diag(&mut spec.kd, gains.kd);
    diag(&mut spec.kp, gains.kp);
    let gh: f32 = bytemuck::cast(gains.gh);
    for g in 0..resolution as usize {
        for m in 0..3 {
            spec.ref_motor[g * REF_MOTOR_WIDTH + m * REF_MOTOR_SLOTS + 11 + m] = gh;
        }
    }
    spec.encode()
}

fn load_identity_gains(gains: Diag) -> GainStorage {
    let blob = make_identity_gains(2, &gains, DT);
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

fn make_sensors(gyro: [RotPerSec; 3], battery_v: Volt) -> SensorData {
    SensorData {
        gyro_rps: gyro,
        motor_rps: [RotPerSec::new(200.0); 4],
        battery_voltage: battery_v,
        accel: [MetersPerSecSq::new(0.0); 3],
    }
}

fn make_setpoint(rates: [RotPerSec; 3], throttle: f32) -> SetpointData {
    SetpointData {
        rate_rps: rates,
        throttle,
    }
}

// =========================================================================
// Controller tests
// =========================================================================

#[test]
fn no_gains_returns_zero_output() {
    let mut ctrl = ControllerState::new();
    let storage = GainStorage::new(); // loaded = false

    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);

    let output = ctrl.update(&sensors, &setpoint, &storage);
    assert_eq!(output.motor_duty, [0.0; 4]);
    assert_eq!(ctrl.telemetry().status, 0);
}

#[test]
fn zero_error_produces_throttle_only() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // Gyro matches setpoint exactly → zero error → zero feedback/feedforward
    let sensors = make_sensors([1.0, 0.0, 0.0].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);

    let output = ctrl.update(&sensors, &setpoint, &gains);

    // Zero control voltage: duty is the throttle (collective = throttle·Vbat
    // along a uniform direction; no motor drag → no idle floor).
    let expected_duty = 0.5;
    for i in 0..4 {
        assert!(
            (output.motor_duty[i] - expected_duty).abs() < 1e-4,
            "motor[{i}] = {} (expected {})",
            output.motor_duty[i],
            expected_duty
        );
    }
    assert_eq!(ctrl.telemetry().status, 1);
}

#[test]
fn proportional_error_generates_feedback() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // Setpoint = [1, 0, 0], gyro = [0, 0, 0] → error = [1, 0, 0]
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);

    let _output = ctrl.update(&sensors, &setpoint, &gains);

    // Kp is identity-like: motor 0 gets error[0]*kp = 1.0*1.0 = 1.0 V
    let telem = ctrl.telemetry();
    assert!(
        telem.feedback_v[0].abs() > Volt::new(0.5),
        "feedback_v[0] should be nonzero, got {:?}",
        telem.feedback_v[0]
    );
    assert!(
        telem.error_prop[0] > RotPerSec::new(0.9),
        "error_prop[0] should be ~1.0, got {:?}",
        telem.error_prop[0]
    );
}

#[test]
fn opposite_error_reduces_duty() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // Negative error: setpoint = [0,0,0], gyro = [1,0,0] → error = [-1,0,0]
    let sensors = make_sensors([1.0, 0.0, 0.0].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([0.0, 0.0, 0.0].map(RotPerSec::new), 0.5);

    let _output = ctrl.update(&sensors, &setpoint, &gains);

    // Motor 0 feedback voltage should be negative
    let telem = ctrl.telemetry();
    assert!(
        telem.feedback_v[0] < Volt::new(-0.5),
        "negative error should produce negative feedback, got {:?}",
        telem.feedback_v[0]
    );
}

#[test]
fn integrator_accumulates_over_iterations() {
    let gains = load_identity_gains(Diag { ki: VoltPerRot::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);

    // Run several iterations to let the leaky integrator accumulate
    let mut prev_integral = Rotation::new(0.0);
    for i in 0..50 {
        ctrl.update(&sensors, &setpoint, &gains);
        let current = ctrl.telemetry().error_integral[0];
        if i > 0 {
            assert!(
                current >= prev_integral,
                "integral should grow: iter {i}, prev={prev_integral:?}, cur={current:?}"
            );
        }
        prev_integral = current;
    }

    assert!(
        prev_integral > RotPerSec::new(0.01) * DT,
        "integral should have accumulated, got {:?}",
        prev_integral
    );
}

#[test]
fn integrator_clamps() {
    let gains = load_identity_gains(Diag { ki: VoltPerRot::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([100.0, 0.0, 0.0].map(RotPerSec::new), 0.5); // huge error

    // Run many iterations — integrator should clamp at ±0.25 rot/s·dt
    for _ in 0..10000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let integral = ctrl.telemetry().error_integral[0];
    let limit = RotPerSec::new(0.25) * DT;
    assert!(
        (integral - limit).abs() <= Rotation::new(1e-9),
        "integral should clamp at {limit:?}, got {:?}",
        integral
    );
}

#[test]
fn integral_voltage_bounded() {
    let gains = load_identity_gains(Diag { ki: VoltPerRot::new(1e6), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    for _ in 0..10000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let fb = ctrl.telemetry().feedback_v;
    let limit = 0.2 * BATTERY;
    assert!(fb.iter().all(|v| v.abs() <= limit * (1.0 + 1e-5)), "integral voltage {fb:?} exceeds {limit:?}");
    assert!(fb.iter().any(|v| v.abs() > 0.5 * limit), "integral didn't build: {fb:?}");
}

#[test]
fn derivative_responds_to_error_change() {
    let gains = load_identity_gains(Diag { kd: VoltPerRpsPerSec::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // First iteration: setpoint=[0,0,0], gyro=[0,0,0] → error=[0,0,0]
    let setpoint = make_setpoint([0.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    ctrl.update(&make_sensors([0.0; 3].map(RotPerSec::new), BATTERY), &setpoint, &gains);

    // Second iteration: the gyro jumps, so the error does → derivative should be nonzero
    ctrl.update(&make_sensors([-1.0, 0.0, 0.0].map(RotPerSec::new), BATTERY), &setpoint, &gains);

    let deriv = ctrl.telemetry().error_derivative[0];
    assert!(
        deriv.abs() > RotPerSecSq::new(0.1),
        "derivative should respond to error change, got {:?}",
        deriv
    );
}

#[test]
fn feedforward_responds_to_setpoint_change() {
    let gains = load_identity_gains(Diag { gh: RpsPerVolt::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // First iteration: setpoint=[0,0,0]
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint0 = make_setpoint([0.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    ctrl.update(&sensors, &setpoint0, &gains);

    // Second iteration: setpoint jumps to [1,0,0]
    let setpoint1 = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    ctrl.update(&sensors, &setpoint1, &gains);

    let ff = ctrl.telemetry().feedforward_v[0];
    assert!(
        ff.abs() > Volt::new(0.01),
        "feedforward should respond to setpoint step, got {:?}",
        ff
    );
}

#[test]
fn reset_clears_state() {
    let gains = load_identity_gains(Diag {
        kp: VoltPerRps::new(1.0),
        ki: VoltPerRot::new(1.0),
        kd: VoltPerRpsPerSec::new(1.0),
        gh: RpsPerVolt::new(1.0),
    });
    let mut ctrl = ControllerState::new();

    // Run several iterations to build up integrator/derivative state
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([1.0, 0.5, 0.0].map(RotPerSec::new), 0.5);
    for _ in 0..100 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    // State should be nonzero
    let integral_before = ctrl.telemetry().error_integral[0];
    assert!(integral_before.abs() > RotPerSec::new(0.001) * DT);

    // Reset
    ctrl.reset();

    // Run one update with zero error to read telemetry
    let sensors_match = make_sensors([1.0, 0.5, 0.0].map(RotPerSec::new), BATTERY);
    ctrl.update(&sensors_match, &setpoint, &gains);

    // After reset, integral should be near zero (only one iter of accumulation)
    let integral_after = ctrl.telemetry().error_integral[0];
    assert!(
        integral_after.abs() < integral_before.abs() * 0.1,
        "integral should reset: before={integral_before:?}, after={integral_after:?}"
    );
}

#[test]
fn output_clamped_to_unit_interval() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(100.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // Enormous error with huge Kp should produce enormous voltage,
    // but output duty should clamp to [0, 1]
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let setpoint = make_setpoint([100.0, 100.0, 100.0].map(RotPerSec::new), 1.0);
    let output = ctrl.update(&sensors, &setpoint, &gains);

    for i in 0..4 {
        assert!(
            output.motor_duty[i] >= 0.0 && output.motor_duty[i] <= 1.0,
            "motor[{i}] = {} should be in [0, 1]",
            output.motor_duty[i]
        );
    }
}

#[test]
fn low_battery_guard() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let mut ctrl = ControllerState::new();

    // Battery voltage = 0 (should be guarded to 1.0 to prevent division by zero)
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), Volt::new(0.0));
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    let output = ctrl.update(&sensors, &setpoint, &gains);

    // Should not panic or produce NaN/Inf
    for i in 0..4 {
        assert!(
            output.motor_duty[i].is_finite(),
            "motor[{i}] should be finite with zero battery, got {}",
            output.motor_duty[i]
        );
    }
}

#[test]
fn throttle_maps_to_collective() {
    let gains = load_identity_gains(Diag::ZERO);
    let mut ctrl = ControllerState::new();

    // All gains zero, zero error → output = pure throttle
    let sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);

    let out_low = ctrl.update(&sensors, &make_setpoint([0.0; 3].map(RotPerSec::new), 0.0), &gains);
    ctrl.reset();
    let out_mid = ctrl.update(&sensors, &make_setpoint([0.0; 3].map(RotPerSec::new), 0.5), &gains);
    ctrl.reset();
    let out_high = ctrl.update(&sensors, &make_setpoint([0.0; 3].map(RotPerSec::new), 1.0), &gains);

    assert!((out_low.motor_duty[0] - 0.00).abs() < 1e-4);
    assert!((out_mid.motor_duty[0] - 0.50).abs() < 1e-4);
    assert!((out_high.motor_duty[0] - 1.00).abs() < 1e-4);
}
