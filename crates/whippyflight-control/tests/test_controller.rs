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

/// Kp = diag(kp·(1 + the vertex's grid indices summed)), so the gains
/// climb with every motor's speed over the grid's 100–500 rev/s.
fn load_scheduled_gains(kp: VoltPerRps) -> GainStorage {
    let n = 2usize;
    let mut spec = BlobSpec::new(n as u8);
    spec.dt = bytemuck::cast(DT);
    let kp: f32 = bytemuck::cast(kp);
    for (i, v) in spec.kp.iter_mut().enumerate() {
        let point = i / 12;
        let (motor, axis) = ((i % 12) / 3, i % 3);
        let level: usize = (0..4u32).map(|d| point / n.pow(d) % n).sum();
        *v = if motor == axis { kp * (1 + level) as f32 } else { 0.0 };
    }
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&spec.encode()), &mut storage);
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

/// After the first tick, each interpolation starts on one tick at its
/// speeds and is taken up on the next, so the gains fly speeds one or two
/// ticks old; a reset interpolates in full on its first tick.
#[test]
fn gains_follow_speeds_over_two_ticks() {
    let gains = load_scheduled_gains(VoltPerRps::new(1.0));
    let mut ctrl = ControllerState::new();
    let setpoint = make_setpoint([0.0; 3].map(RotPerSec::new), 0.5);
    let mut sensors = make_sensors([0.0; 3].map(RotPerSec::new), BATTERY);
    let mut kp_at = |ctrl: &mut ControllerState, speed: f32| {
        sensors.motor_rps = [RotPerSec::new(speed); 4];
        ctrl.update(&sensors, &setpoint, &gains);
        ctrl.telemetry().kp_motor0[0]
    };

    let slow = kp_at(&mut ctrl, 100.0);
    // Starts at 500 rev/s, still flying the gains at 100.
    assert_eq!(kp_at(&mut ctrl, 500.0), slow);
    // Finishes at 500 rev/s, whatever this tick's speed.
    let fast = kp_at(&mut ctrl, 100.0);
    assert!(fast > slow, "{fast:?} vs {slow:?}");
    // Starts at 100 rev/s, still flying the gains at 500.
    assert_eq!(kp_at(&mut ctrl, 100.0), fast);
    assert_eq!(kp_at(&mut ctrl, 100.0), slow);

    ctrl.reset();
    assert_eq!(kp_at(&mut ctrl, 500.0), fast);
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

    // Run many iterations — integrator should clamp at ±0.25 rot/s·τ_int
    for _ in 0..10000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let integral = ctrl.telemetry().error_integral[0];
    let limit = RotPerSec::new(0.25) * gains.header.integral_tau;
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
fn feedback_splits_into_its_terms() {
    let gains = load_identity_gains(Diag {
        kp: VoltPerRps::new(1.0),
        ki: VoltPerRot::new(50.0),
        kd: VoltPerRpsPerSec::new(1e-3),
        ..Diag::ZERO
    });
    let mut ctrl = ControllerState::new();
    let setpoint = make_setpoint([1.0, 0.5, -0.5].map(RotPerSec::new), 0.5);
    for k in 0..20 {
        let gyro = [0.01 * k as f32, 0.0, 0.0].map(RotPerSec::new);
        ctrl.update(&make_sensors(gyro, BATTERY), &setpoint, &gains);
    }
    let t = ctrl.telemetry();
    for m in 0..3 {
        assert!(t.feedback_p_v[m].abs() > Volt::new(0.1), "P {:?}", t.feedback_p_v);
        assert!(t.feedback_i_v[m].abs() > Volt::new(1e-4), "I {:?}", t.feedback_i_v);
    }
    // Only roll's error moves.
    assert!(t.feedback_d_v[0].abs() > Volt::new(0.0), "D {:?}", t.feedback_d_v);
    for m in 0..4 {
        let sum = t.feedback_p_v[m] + t.feedback_i_v[m] + t.feedback_d_v[m];
        assert!((sum - t.feedback_v[m]).abs() < Volt::new(1e-6), "motor {m}: {sum:?} vs {:?}", t.feedback_v[m]);
        assert!(t.applied_v[m] >= Volt::new(0.0) && t.applied_v[m] <= BATTERY);
        assert_eq!(t.filtered_rps[m], RotPerSec::new(200.0));
    }
}

#[test]
fn schedule_jitter_traced() {
    let gains = load_scheduled_gains(VoltPerRps::new(1.0));
    let mut ctrl = ControllerState::new();
    let setpoint = make_setpoint([1.0, 0.0, 0.0].map(RotPerSec::new), 0.5);
    let at = |rps: f32| SensorData {
        motor_rps: [RotPerSec::new(rps); 4],
        ..make_sensors([0.0; 3].map(RotPerSec::new), BATTERY)
    };
    let jitter = |ctrl: &ControllerState| ctrl.telemetry().schedule_jitter_v[0];

    // Untraced, the jitter reads zero even as the speeds jump.
    ctrl.update(&at(300.0), &setpoint, &gains);
    ctrl.update(&at(400.0), &setpoint, &gains);
    assert_eq!(jitter(&ctrl), Volt::new(0.0));

    // Traced at steady speeds, the gains don't move.
    ctrl.set_effort_trace(true);
    for _ in 0..100 {
        ctrl.update(&at(300.0), &setpoint, &gains);
    }
    let steady = ctrl.telemetry().feedback_v[0];
    assert!(jitter(&ctrl).abs() < Volt::new(1e-6), "{:?}", jitter(&ctrl));

    // A step in the speeds reaches the gains within three ticks (they fly
    // speeds one or two ticks old): almost all of the feedback's change is
    // jitter at first, and it fades as the reference schedule follows.
    let mut rise = Volt::new(0.0);
    for _ in 0..3 {
        ctrl.update(&at(400.0), &setpoint, &gains);
        rise = ctrl.telemetry().feedback_v[0] - steady;
        if rise > Volt::new(0.5) {
            break;
        }
    }
    assert!(rise > Volt::new(0.5), "{rise:?}");
    assert!(jitter(&ctrl) > 0.95 * rise && jitter(&ctrl) <= rise, "{:?} of {rise:?}", jitter(&ctrl));
    for _ in 0..4000 {
        ctrl.update(&at(400.0), &setpoint, &gains);
    }
    assert!(jitter(&ctrl).abs() < 1e-3 * rise, "{:?}", jitter(&ctrl));
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
fn battery_floor_stops_compensation() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let setpoint = make_setpoint([0.5, 0.0, 0.0].map(RotPerSec::new), 0.5);
    let duty = |reading: f32, floor: f32| {
        let mut ctrl = ControllerState::new();
        ctrl.set_min_battery(Volt::new(floor));
        ctrl.update(&make_sensors([0.0; 3].map(RotPerSec::new), Volt::new(reading)), &setpoint, &gains)
            .motor_duty
    };
    // Above the floor the reading is compensated; below it, the floor is used.
    assert_ne!(duty(12.0, 10.0), duty(11.0, 10.0));
    assert_eq!(duty(9.0, 10.0), duty(10.0, 10.0));
    // No floor compensates any reading.
    assert_ne!(duty(9.0, 0.0), duty(10.0, 0.0));
}

#[test]
fn battery_lpf_lags_a_sag() {
    let gains = load_identity_gains(Diag { kp: VoltPerRps::new(1.0), ..Diag::ZERO });
    let setpoint = make_setpoint([0.5, 0.0, 0.0].map(RotPerSec::new), 0.5);
    let duty_after = |second: f32, cutoff: f32| {
        let mut ctrl = ControllerState::new();
        ctrl.set_battery_lpf(Hertz::new(cutoff));
        let sensors = |v: f32| make_sensors([0.0; 3].map(RotPerSec::new), Volt::new(v));
        ctrl.update(&sensors(12.0), &setpoint, &gains);
        ctrl.update(&sensors(second), &setpoint, &gains).motor_duty[0]
    };
    let (sagged, held) = (duty_after(10.0, 0.0), duty_after(12.0, 0.0));
    let filtered = duty_after(10.0, 20.0);
    assert!((filtered - held) * (sagged - filtered) > 0.0, "{held} {filtered} {sagged}");
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
