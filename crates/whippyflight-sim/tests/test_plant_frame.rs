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
