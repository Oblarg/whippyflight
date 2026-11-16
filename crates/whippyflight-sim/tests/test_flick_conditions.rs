//! What a flick's [`Conditions`] do to it: gyro filters, sensor noise in the
//! measured flight, a CG offset and turbulence, on test_gains.wfg and the
//! plant it was generated from.

mod common;

use common::plant::model;
use common::units::{AngAccel, Freq, Rate, Time};
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::types::*;
use whippyflight_sim::flick::{measure, Conditions, Flick, Limits, Turbulence};
use whippyflight_sim::margin::pt1_cutoffs;

static TEST_WFG: &[u8] = include_bytes!("../../../fixtures/test_gains.wfg");

fn load() -> GainStorage {
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(TEST_WFG), &mut gains), GainLoadResult::Ok);
    gains
}

/// A 1 rot/s roll flick rising over 30 ms.
fn roll(gains: &GainStorage, conditions: &Conditions) -> Flick {
    measure(gains, model(), 0, Rate::new(1.0), Time::new(0.030), Limits::default(), conditions)
}

fn max_diff(a: &[Rate], b: &[Rate]) -> Rate {
    a.iter().zip(b).fold(Rate::new(0.0), |m, (&x, &y)| m.max((x - y).abs()))
}

#[test]
fn gyro_filters_reach_the_feedback() {
    let gains = load();
    let filter = pt1_cutoffs(model()).unwrap();
    assert!(!filter.is_empty(), "the fixture has gyro filters");
    let ideal = roll(&gains, &Conditions::default());
    let filtered = roll(&gains, &Conditions::of(&filter));
    let moved = max_diff(&filtered.response, &ideal.response);
    println!("the gyro filters move the response by up to {moved:?}");
    assert!(moved > Rate::new(1e-4));
}

#[test]
fn noise_reaches_the_response_only_in_noisy_flights() {
    let gains = load();
    let noise = common::noise::fixture_noise(Time::new(0.0), 1.0, 1.0);
    let calm = roll(&gains, &Conditions::default());
    let quiet = roll(&gains, &Conditions { noise: Some(&noise), ..Conditions::default() });
    let noisy = roll(&gains, &Conditions { noise: Some(&noise), noisy_flights: true, ..Conditions::default() });
    assert_eq!(quiet.response, calm.response);
    let shown = max_diff(&noisy.response, &calm.response);
    println!("noise moves the response by up to {shown:?}");
    assert!(shown > Rate::new(1e-4));
    // The roughness is what the noise adds either way.
    let (a, b) = (quiet.metrics.rough_v.unwrap(), noisy.metrics.rough_v.unwrap());
    assert!((a - b).abs() < 1e-6 * a, "{a:?} vs {b:?}");
}

#[test]
fn cg_offset_is_trimmed_before_the_stick() {
    let gains = load();
    let calm = roll(&gains, &Conditions::default());
    let shifted = roll(&gains, &Conditions { cg_shift: [0.05, 0.0], ..Conditions::default() });
    let moved = max_diff(&shifted.response, &calm.response);
    println!("a 5% right CG moves the roll response by up to {moved:?}");
    assert!(moved < Rate::new(0.05), "the integral had the offset trimmed");
    assert!(moved > Rate::new(0.0));
}

#[test]
fn turbulence_disturbs_repeatably() {
    let gains = load();
    let turbulence = Turbulence {
        rms: [AngAccel::new(2.0); 3],
        bandwidth: Freq::new(10.0),
        seed: 1,
    };
    let gusty = Conditions { turbulence: Some(turbulence), ..Conditions::default() };
    let calm = roll(&gains, &Conditions::default());
    let (a, b) = (roll(&gains, &gusty), roll(&gains, &gusty));
    assert_eq!(a.response, b.response);
    let moved = max_diff(&a.response, &calm.response);
    println!("2 rot/s² turbulence moves the response by up to {moved:?}");
    assert!(moved > Rate::new(1e-3));
    let reseeded = roll(&gains, &Conditions { turbulence: Some(Turbulence { seed: 2, ..turbulence }), ..gusty });
    assert_ne!(reseeded.response, a.response);
    let yaw_only = Turbulence { rms: [AngAccel::new(0.0), AngAccel::new(0.0), AngAccel::new(2.0)], ..turbulence };
    let off_axis = max_diff(&roll(&gains, &Conditions { turbulence: Some(yaw_only), ..gusty }).response, &calm.response);
    assert!(off_axis < 0.1 * moved, "yaw turbulence barely reaches roll: {off_axis:?} vs {moved:?}");
}

#[test]
fn noise_rms_is_the_recordings_before_scaling() {
    use whippyflight_sim::noise::Noise;
    // Two frames: gyro ±3, ±4, 0 (1e-5 rot/s); motors ±1, ±1, ±1, ±1 (1e-3 rot/s).
    let mut bytes = [1000f32.to_le_bytes(), 2u32.to_le_bytes()].concat();
    for sign in [1i16, -1] {
        for x in [3i16, 4, 0, 1, 1, 1, 1] {
            bytes.extend((sign * x).to_le_bytes());
        }
    }
    let noise = Noise::from_bytes(&bytes, Time::new(0.0), 2.0, 0.5).unwrap();
    let (gyro, motor) = noise.rms();
    assert!((gyro - Rate::new(1e-5 * (25.0f64 / 3.0).sqrt())).abs() < Rate::new(1e-9));
    assert!((motor - Rate::new(1e-3)).abs() < Rate::new(1e-9));
}

#[test]
fn motor_noise_follows_each_motors_speed() {
    use whippyflight_control::types::RotPerSec;
    use whippyflight_sim::noise::{Noise, SpeedLaw};
    let mut bytes = [1000f32.to_le_bytes(), 1u32.to_le_bytes()].concat();
    for x in [3i16, 4, 0, 1000, 1000, 1000, 1000] {
        bytes.extend(x.to_le_bytes());
    }
    let flat = Noise::from_bytes(&bytes, Time::new(0.0), 1.0, 2.0).unwrap();
    let law = SpeedLaw { reference: Rate::new(100.0), exponent: 0.5 };
    let lawful = flat.scaled(1.0, 2.0).with_speed_law(Some(law)).unwrap();
    let speeds = [100.0, 400.0, 25.0, 0.0].map(RotPerSec::new);
    let (a, b) = (flat.at(Time::new(0.0), speeds), lawful.at(Time::new(0.0), speeds));
    assert_eq!(a.gyro, b.gyro);
    for (m, factor) in [1.0f32, 2.0, 0.5, 0.0].into_iter().enumerate() {
        assert!((b.motor[m] - factor * a.motor[m]).abs() < RotPerSec::new(1e-6), "motor {m}");
    }
    assert_eq!(lawful.scaled(1.0, 1.0).speed_law(), Some(law));
    assert!(flat.scaled(1.0, 1.0).with_speed_law(Some(SpeedLaw { exponent: 3.0, ..law })).is_err());
    assert!(flat.scaled(1.0, 1.0).with_speed_law(Some(SpeedLaw { reference: Rate::new(0.0), ..law })).is_err());
}
