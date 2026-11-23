//! RX dropout ramp.

mod common;

use common::micros;
use whippyflight_control::dropout::DropoutRamp;
use whippyflight_control::types::{RotPerSec, Seconds};

const PERIOD: Seconds = Seconds::new(0.004);
const ZERO: RotPerSec = RotPerSec::new(0.0);

fn roll(rate: f32) -> [RotPerSec; 3] {
    [RotPerSec::new(rate), ZERO, ZERO]
}

/// Feeds `rate(n)` at packet times n·T, skipping `missed`; returns the
/// roll rate put out for each packet fed, by packet index.
fn feed(ramp: &mut DropoutRamp, rate: impl Fn(u32) -> f32, packets: u32, missed: &[u32]) -> Vec<(u32, f32)> {
    (0..packets)
        .filter(|n| !missed.contains(n))
        .map(|n| (n, f32::from(ramp.packet(roll(rate(n)), micros(n as f32 * PERIOD), PERIOD)[0] / RotPerSec::new(1.0))))
        .collect()
}

#[test]
fn passes_packets_on_schedule_through() {
    let mut ramp = DropoutRamp::new();
    ramp.set_cap(Seconds::new(0.05));
    for (n, out) in feed(&mut ramp, |n| (n as f32 * 0.37).sin(), 50, &[]) {
        assert_eq!(out, (n as f32 * 0.37).sin());
    }
}

/// A step hidden in 3 missed packets is spread over the 3 periods after
/// them, in equal parts, then the input passes through again.
#[test]
fn ramps_over_the_missed_periods() {
    let mut ramp = DropoutRamp::new();
    ramp.set_cap(Seconds::new(0.05));
    let out = feed(&mut ramp, |n| if n >= 11 { 1.0 } else { 0.0 }, 20, &[10, 11, 12]);
    let at = |n: u32| out.iter().find(|(k, _)| *k == n).unwrap().1;
    assert_eq!(at(9), 0.0);
    for (k, n) in (13..=16).enumerate() {
        let want = k as f32 / 3.0;
        assert!((at(n) - want.min(1.0)).abs() < 1e-5, "packet {n}: {} vs {want}", at(n));
    }
    assert_eq!(at(19), 1.0);
}

/// A long dropout ramps over the cap, not the whole loss.
#[test]
fn caps_the_ramp() {
    let mut ramp = DropoutRamp::new();
    let cap = 2.0 * PERIOD;
    ramp.set_cap(cap);
    let out = feed(&mut ramp, |n| if n >= 50 { 1.0 } else { 0.0 }, 60, &(10..50).collect::<Vec<_>>());
    let at = |n: u32| out.iter().find(|(k, _)| *k == n).unwrap().1;
    assert_eq!(at(50), 0.0);
    assert!((at(51) - 0.5).abs() < 1e-5, "{}", at(51));
    assert_eq!(at(52), 1.0);
}

#[test]
fn zero_cap_steps() {
    let mut ramp = DropoutRamp::new();
    let out = feed(&mut ramp, |n| if n >= 11 { 1.0 } else { 0.0 }, 20, &[10, 11, 12]);
    assert_eq!(out.iter().find(|(k, _)| *k == 13).unwrap().1, 1.0);
}

/// A dropout during a ramp starts the next one from where the output was.
#[test]
fn a_dropout_mid_ramp_continues_from_the_output() {
    let mut ramp = DropoutRamp::new();
    ramp.set_cap(Seconds::new(0.05));
    let out = feed(&mut ramp, |n| if n >= 11 { 1.0 } else { 0.0 }, 30, &[10, 11, 12, 15, 16]);
    let at = |n: u32| out.iter().find(|(k, _)| *k == n).unwrap().1;
    let before = at(14);
    assert!(before > 0.0 && before < 1.0);
    assert!((at(17) - before).abs() < 1e-5, "{} vs {before}", at(17));
    assert_eq!(at(29), 1.0);
}
