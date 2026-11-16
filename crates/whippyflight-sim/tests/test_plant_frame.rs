//! Coefficient JSON frames: a legacy Betaflight fit loads as the same plant
//! as its canonical conversion (`python -m whippyflight.coefficients`).

use whippyflight_control::gains::{MotorLayout, MotorPosition, Spin};
use whippyflight_sim::plant::Model;

static CANONICAL: &str = include_str!("../../../fixtures/airframes/meteor65.json");
static LEGACY: &str = include_str!("../../../fixtures/legacy/meteor65_betaflight.json");

fn model(text: &str) -> Model {
    Model::from_json(text, "test").unwrap()
}

#[test]
fn legacy_fit_converts_to_canonical() {
    let (a, b) = (model(CANONICAL), model(LEGACY));
    assert_eq!(a.layout, b.layout);
    assert_eq!(a.torque_patterns(), b.torque_patterns());
    assert_eq!([a.w_front, a.w_back, a.w_left, a.w_right], [b.w_front, b.w_back, b.w_left, b.w_right]);
}

#[test]
fn legacy_fit_is_betaflight_quadx_props_out() {
    // meteor65 flies yaw_motors_reversed (props out).
    let layout = model(LEGACY).layout;
    assert_eq!(layout, MotorLayout {
        position: [MotorPosition::RearRight, MotorPosition::FrontRight, MotorPosition::RearLeft, MotorPosition::FrontLeft],
        spin: [Spin::CounterClockwise, Spin::Clockwise, Spin::Clockwise, Spin::CounterClockwise],
    });
}

#[test]
fn legacy_arms_swap_into_the_physical_frame() {
    // The legacy fit's longer pitch arm is its back, the physical front.
    let m = model(LEGACY);
    assert!(m.w_front > m.w_back && m.w_right > m.w_left);
}

#[test]
fn latency_loads_and_delays_whole_ticks() {
    use whippyflight_sim::plant::DelayLine;
    use whippyflight_sim::units::Time;
    assert_eq!(model(CANONICAL).actuator_delay, Time::new(0.0), "none recorded");
    let mut c: serde_json::Value = serde_json::from_str(CANONICAL).unwrap();
    c["latency"] = serde_json::json!({"actuator_ms": 0.31, "telemetry_ms": 3.23});
    let m = model(&c.to_string());
    assert!((m.actuator_delay - Time::new(0.31e-3)).abs() < Time::new(1e-12));
    assert!((m.telemetry_delay - Time::new(3.23e-3)).abs() < Time::new(1e-12));
    c["latency"]["actuator_ms"] = (-1.0).into();
    assert!(Model::from_json(&c.to_string(), "test").is_err());

    let mut line = DelayLine::new(Time::new(1e-3), Time::new(0.3e-3));
    let out: Vec<i32> = (1..=6).map(|x| line.push(x)).collect();
    assert_eq!(out, [1, 1, 1, 1, 2, 3], "3 ticks, filled with the first value");
    let mut none = DelayLine::new(Time::new(0.0), Time::new(0.3e-3));
    assert_eq!(none.push(7), 7);
    let flown = DelayLine::<i32>::flown(Time::new(1e-3), Time::new(0.3e-3));
    assert!((flown - Time::new(0.9e-3)).abs() < Time::new(1e-12));
}

#[test]
fn bad_frames_are_rejected() {
    let mut c: serde_json::Value = serde_json::from_str(CANONICAL).unwrap();
    c["frame"]["motors"][0]["position"] = "front_left".into();
    assert!(Model::from_json(&c.to_string(), "test").err().unwrap().contains("quad-X"));
    let mut c: serde_json::Value = serde_json::from_str(CANONICAL).unwrap();
    c["frame"]["axes"] = "FLU".into();
    assert!(Model::from_json(&c.to_string(), "test").is_err());
    let mut c: serde_json::Value = serde_json::from_str(LEGACY).unwrap();
    c["airframe"]["mixer"] = "HEX6X".into();
    assert!(Model::from_json(&c.to_string(), "test").is_err());
}
