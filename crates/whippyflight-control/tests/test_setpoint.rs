//! RX setpoint estimator (B-spline through the packets + slew/jerk-limited feedforward target).

mod common;

use common::micros;
use whippyflight_control::setpoint::SetpointEstimator;
use whippyflight_control::types::{Micros, RotPerSec, RotPerSecCu, RotPerSecSq, Seconds};
use whippyunits::value;

const PERIOD: Seconds = Seconds::new(0.002); // 500 Hz link
const DT: Seconds = Seconds::new(0.00025); // 4 kHz loop
const ZERO: RotPerSec = RotPerSec::new(0.0);

fn ticks_per_packet() -> u32 {
    f32::from(PERIOD / DT).round() as u32
}

fn roll(rate: RotPerSec) -> [RotPerSec; 3] {
    [rate, ZERO, ZERO]
}

/// No slew limit and the jerk limit off (zero): the target is the estimate.
const UNLIMITED: (RotPerSecSq, RotPerSecCu) = (RotPerSecSq::new(1e9), RotPerSecCu::new(0.0));

/// Feed `packets` of `stick(t)` on roll and tick through them; returns the
/// last (rate, target) and the largest per-tick target change.
fn run(
    est: &mut SetpointEstimator,
    stick: impl Fn(Seconds) -> RotPerSec,
    packets: u32,
) -> ([RotPerSec; 3], [RotPerSec; 3], RotPerSec) {
    let mut last = ([ZERO; 3], [ZERO; 3]);
    let mut max_step = ZERO;
    let mut prev_target: Option<RotPerSec> = None;
    for n in 0..packets {
        let t_packet = n as f32 * PERIOD;
        est.packet(roll(stick(t_packet)), micros(t_packet), PERIOD);
        for k in 0..ticks_per_packet() {
            let t = t_packet + k as f32 * DT;
            last = est.tick(micros(t), DT);
            if let Some(p) = prev_target {
                max_step = max_step.max((last.1[0] - p).abs());
            }
            prev_target = Some(last.1[0]);
        }
    }
    (last.0, last.1, max_step)
}

/// The spline reproduces a ramp 1.5 packet periods late.
#[test]
fn tracks_a_ramp_a_period_and_a_half_behind() {
    let mut est = SetpointEstimator::new();
    // The jerk limit trails a ramp by slope²/(2J); this checks the estimate.
    let (slew, jerk) = UNLIMITED;
    est.configure(slew, jerk);
    let slope = RotPerSecSq::new(10.0);
    let (rate, target, _) = run(&mut est, |t| slope * t, 200);
    let t_last = 199.0 * PERIOD + (f32::from(PERIOD / DT) - 1.0) * DT;
    let expected = slope * (t_last - 1.5 * PERIOD);
    assert!((rate[0] - expected).abs() <= RotPerSec::new(1e-4), "rate={:?} expected={expected:?}", rate[0]);
    assert!((target[0] - rate[0]).abs() <= RotPerSec::new(1e-5), "target={:?}", target[0]);
}

/// A flick that stops: neither the estimate nor the target passes where
/// the stick stopped.
#[test]
fn flick_does_not_overshoot() {
    let (slew, jerk) = (RotPerSecSq::new(127.0), RotPerSecCu::new(2315.0));
    for (amp, rise) in [(0.05f32, 0.010f32), (0.2, 0.030), (1.0, 0.010)] {
        let (amp, rise) = (RotPerSec::new(amp), Seconds::new(rise));
        let mut est = SetpointEstimator::new();
        est.configure(slew, jerk);
        let s_curve = |t: Seconds| {
            amp * 0.5 * (1.0 - (core::f32::consts::PI * f32::from(t / rise).clamp(0.0, 1.0)).cos())
        };
        let (mut peak, mut est_peak) = (ZERO, ZERO);
        for n in 0..200u32 {
            let t_packet = n as f32 * PERIOD;
            est.packet(roll(s_curve(t_packet)), micros(t_packet), PERIOD);
            for k in 0..ticks_per_packet() {
                let (rate, target) = est.tick(micros(t_packet + k as f32 * DT), DT);
                peak = peak.max(target[0]);
                est_peak = est_peak.max(rate[0]);
            }
        }
        assert!(est_peak <= amp * 1.00001, "{amp:?} over {rise:?}: estimate peak {}", f32::from(est_peak / amp));
        assert!(peak <= amp * 1.00001, "{amp:?} over {rise:?}: target peak {}", f32::from(peak / amp));
    }
}

/// A fast ramp into full stick that stops dead, every other packet a
/// repeat (a staircase, as CRSF delivers): the target stops on full stick,
/// with or without a brake margin.
#[test]
fn ramp_into_full_stick_does_not_overshoot() {
    let period = Seconds::new(1.0 / 250.0);
    let (rise, full) = (Seconds::new(0.06), RotPerSec::new(1000.0 / 360.0));
    for margin in [0.0, 0.2] {
        let mut est = SetpointEstimator::new();
        est.configure(RotPerSecSq::new(53.0), RotPerSecCu::new(2000.0));
        est.set_brake_margin(margin);
        let mut peak = ZERO;
        let ticks = f32::from(period / DT).round() as u32;
        for n in 0..100u32 {
            let t_packet = n as f32 * period;
            let held = (n - n % 2) as f32 * period;
            let s = f32::from(held / rise).clamp(0.0, 1.0);
            est.packet(roll(full * s * s), micros(t_packet), period);
            for k in 0..ticks {
                let (_, target) = est.tick(micros(t_packet + k as f32 * DT), DT);
                peak = peak.max(target[0]);
            }
        }
        assert!(peak <= full * 1.00001, "margin {margin}: target peak {}", f32::from(peak / full));
    }
}

#[test]
fn target_slew_is_limited() {
    let mut est = SetpointEstimator::new();
    let (slew, jerk) = (RotPerSecSq::new(100.0), RotPerSecCu::new(0.0));
    est.configure(slew, jerk);
    run(&mut est, |_| ZERO, 10);
    let (_, target, max_step) = run(&mut est, |_| RotPerSec::new(1.0), 1);
    assert!(max_step <= slew * DT + RotPerSec::new(1e-6), "max_step={max_step:?}");
    assert!(target[0] < RotPerSec::new(1.0));
}

/// A stick step: in transit the target's slope changes by at most J·dt per
/// tick, and it stops on the new value without passing it (landing snaps it
/// onto the goal, so the bound is checked away from it).
#[test]
fn target_jerk_is_limited_and_lands_without_overshoot() {
    let (slew, jerk) = (RotPerSecSq::new(100.0), RotPerSecCu::new(5000.0));
    let mut est = SetpointEstimator::new();
    est.configure(slew, jerk);
    run(&mut est, |_| ZERO, 10);
    let step = RotPerSec::new(1.0);
    let near = RotPerSec::new(1e-3);
    let (mut prev, mut prev_slope, mut max_dslope, mut peak) =
        (ZERO, RotPerSecSq::new(0.0), RotPerSecSq::new(0.0), ZERO);
    for n in 10..200u32 {
        let t_packet = n as f32 * PERIOD;
        est.packet(roll(step), micros(t_packet), PERIOD);
        for k in 0..ticks_per_packet() {
            let (rate, target) = est.tick(micros(t_packet + k as f32 * DT), DT);
            // Only while the goal (the estimate) has settled onto the step;
            // before that it overshoots by itself.
            if (rate[0] - step).abs() < near {
                peak = peak.max(target[0]);
            }
            let slope = (target[0] - prev) / DT;
            if (target[0] - rate[0]).abs() > near {
                max_dslope = max_dslope.max((slope - prev_slope).abs());
            }
            assert!(slope.abs() <= slew + RotPerSecSq::new(1e-2), "slope={slope:?}");
            (prev, prev_slope) = (target[0], slope);
        }
    }
    assert!(max_dslope <= jerk * DT * 1.01 + RotPerSecSq::new(1e-2), "max slope change {max_dslope:?}");
    assert!(peak <= step + RotPerSec::new(1e-4), "peak={peak:?}");
    assert!((prev - step).abs() < RotPerSec::new(1e-4), "settles on the step: {prev:?}");
}

/// A brake margin eases the stop at the reduced jerk, leaves at the full
/// one, and still lands on the step.
#[test]
fn brake_margin_eases_the_stop() {
    let (slew, jerk) = (RotPerSecSq::new(100.0), RotPerSecCu::new(5000.0));
    let margin = 0.3;
    let mut est = SetpointEstimator::new();
    est.configure(slew, jerk);
    est.set_brake_margin(margin);
    run(&mut est, |_| ZERO, 10);
    let step = RotPerSec::new(1.0);
    let near = RotPerSec::new(1e-3);
    let (mut prev, mut prev_slope, mut rise, mut ease, mut peak) =
        (ZERO, RotPerSecSq::new(0.0), RotPerSecSq::new(0.0), RotPerSecSq::new(0.0), ZERO);
    for n in 10..200u32 {
        let t_packet = n as f32 * PERIOD;
        est.packet(roll(step), micros(t_packet), PERIOD);
        for k in 0..ticks_per_packet() {
            let (rate, target) = est.tick(micros(t_packet + k as f32 * DT), DT);
            if (rate[0] - step).abs() < near {
                peak = peak.max(target[0]);
            }
            let slope = (target[0] - prev) / DT;
            if (target[0] - rate[0]).abs() > near {
                let dslope = slope - prev_slope;
                rise = rise.max(dslope);
                ease = ease.max(-dslope);
            }
            (prev, prev_slope) = (target[0], slope);
        }
    }
    assert!(rise >= jerk * DT * 0.99, "leaves at the full jerk: {rise:?}");
    assert!(ease <= (1.0 - margin) * jerk * DT * 1.02 + RotPerSecSq::new(1e-2), "eases at {ease:?}");
    assert!(peak <= step + RotPerSec::new(1e-4), "peak={peak:?}");
    assert!((prev - step).abs() < RotPerSec::new(1e-4), "settles on the step: {prev:?}");
}

#[test]
fn holds_after_link_loss() {
    let mut est = SetpointEstimator::new();
    let (slew, jerk) = UNLIMITED;
    est.configure(slew, jerk);
    let ramp = RotPerSecSq::new(10.0);
    run(&mut est, |t| ramp * t, 50);
    let last_packet = micros(49.0 * PERIOD);
    let (rate_a, target_a) = est.tick(last_packet + Micros::new(10_000), DT);
    let (rate_b, target_b) = est.tick(last_packet + Micros::new(50_000), DT);
    assert_eq!(rate_a, rate_b);
    assert_eq!(target_a, rate_a);
    assert_eq!(target_b, rate_b);
}

/// A dropped packet mid-ramp: its slot holds the last value, so the
/// estimate neither stalls nor jumps to catch up, and never runs backwards.
#[test]
fn keeps_its_time_base_through_a_dropped_packet() {
    let mut est = SetpointEstimator::new();
    let (slew, jerk) = UNLIMITED;
    est.configure(slew, jerk);
    let ramp = RotPerSecSq::new(10.0);
    let normal = ramp * DT;
    let mut prev: Option<RotPerSec> = None;
    let mut largest = ZERO;
    for n in 0..60u32 {
        let t_packet = n as f32 * PERIOD;
        if n != 40 {
            est.packet(roll(ramp * t_packet), micros(t_packet), PERIOD);
        }
        for k in 0..ticks_per_packet() {
            let (rate, _) = est.tick(micros(t_packet + k as f32 * DT), DT);
            if let Some(p) = prev {
                assert!(rate[0] >= p, "runs backwards at packet {n}");
                largest = largest.max(rate[0] - p);
            }
            prev = Some(rate[0]);
        }
    }
    assert!(largest <= 2.0 * normal, "largest step {largest:?}, ramp step {normal:?}");
}

#[test]
fn reseeds_after_a_long_gap() {
    let mut est = SetpointEstimator::new();
    let ramp = RotPerSecSq::new(10.0);
    run(&mut est, |t| ramp * t, 50);
    // Next packet 100 ms later at a very different value: no velocity kick.
    est.packet(roll(RotPerSec::new(-0.5)), Micros::new(200_000), PERIOD);
    let (rate, _) = est.tick(Micros::new(200_000), DT);
    assert_eq!(rate[0], RotPerSec::new(-0.5));
}

/// A slow ramp of 1 µs-quantized stick (SITL's 1500 → 1800 over 0.8 s,
/// 216 deg/s at full), at the derived airframes' target limits: the target
/// stays within 1 deg/s of the estimate on average.  The estimate follows
/// the 1 µs staircase (±0.6 deg/s per packet), which the jerk limit can't
/// follow, and the nonlinear limiter loses some of the mean on it.  Not
/// jerk-limiting small goal jumps turned every RX frame of a flick into a
/// slope step the feedforward inverts into a voltage spike (saturation,
/// overshoot).
#[test]
fn quantized_ramp_is_followed() {
    let per_us = RotPerSec::new(216.0 / 300.0 / 360.0); // per µs of stick
    let ramp_time = Seconds::new(0.8);
    for (slew, jerk) in [(127.0, 2315.0), (154.0, 3853.0), (100.0, 5000.0)] {
        let (slew, jerk) = (RotPerSecSq::new(slew), RotPerSecCu::new(jerk));
        let mut est = SetpointEstimator::new();
        est.configure(slew, jerk);
        let (mut sum, mut n) = (ZERO, 0);
        for k in 0..f32::from(ramp_time / DT) as u32 {
            let t = k as f32 * DT;
            if k % ticks_per_packet() == 0 {
                est.packet(roll(f32::from(300.0 * t / ramp_time).round() * per_us), micros(t), PERIOD);
            }
            let (rate, target) = est.tick(micros(t), DT);
            if t > Seconds::new(0.1) {
                sum += target[0] - rate[0];
                n += 1;
            }
        }
        let shortfall_dps = value!(-sum / n as f32, deg / s, f32);
        assert!(shortfall_dps.abs() <= 1.0, "slew {slew:?} jerk {jerk:?}: target trails by {shortfall_dps} deg/s");
    }
}

/// Conditioning pulls a rising target back to a reachable value below it,
/// but not up to one above it; `None` changes nothing.
#[test]
fn condition_holds_the_target_at_the_reachable() {
    let rising = || {
        let mut est = SetpointEstimator::new();
        est.configure(RotPerSecSq::new(100.0), RotPerSecCu::new(2000.0));
        let step = |t: Seconds| if t >= PERIOD { RotPerSec::new(1.0) } else { ZERO };
        let (_, target, _) = run(&mut est, step, 10);
        (est, target[0])
    };
    let next = |est: &mut SetpointEstimator| est.tick(micros(10.0 * PERIOD), DT).1[0];
    let (mut plain, at) = rising();
    let free = next(&mut plain);
    assert!(free > at, "still rising: {at:?} → {free:?}");

    let (mut none, _) = rising();
    none.condition(None);
    assert_eq!(next(&mut none), free);

    let (mut behind, _) = rising();
    behind.condition(Some(roll(at - RotPerSec::new(0.1))));
    assert!(next(&mut behind) < at, "pulled back");

    let (mut ahead, _) = rising();
    ahead.condition(Some(roll(at + RotPerSec::new(0.1))));
    assert_eq!(next(&mut ahead), free, "not pushed forward");
}
