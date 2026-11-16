#!/usr/bin/env python3
"""
Extract sensor noise from a blackbox log (.bbl/.bfl, LOG#n for one flight
of a multi-flight file, or a blackbox_decode CSV) into the fixture the
whippyflight-sim evals replay ($NOISE_PATH).

  python -m whippyflight_betaflight.extract_noise LOG OUT.bin [--seconds 3.0] [--json INFO.json]
      [--fit COEFFICIENTS.json --diagnostics DIAGNOSTICS.json]

Noise is what the controller would see on top of the true motion:
  - gyro: BF's filtered gyro (gyroADC, after its LPFs/RPM/dynamic notches);
  - motor speed: eRPM telemetry in shaft rev/s (from the log's motor_poles, or
    --motor-poles; the glue converts the same way).
With --fit, the motion its model explains is taken out first: the body rates
its angular accelerations integrate to, and the motor speeds its motor model
reaches from the logged voltages.  What remains is zero-phase high-passed at
each channel's noise split from --diagnostics (fit_diagnostics.noise_split:
above it the fit's residual is the sensors', below it the airframe's), or
without them at --gyro-hp and --erpm-hp Hz.  Content below the cut-offs is
not represented.  With --fit it also measures how the motor-speed noise
grows with the motor's speed over the whole flight (`speed_exponent`), for
the sim to scale each motor's replay by (speed / the window's)^exponent;
--json writes it.

The window is the stretch of flight (sticks within --max-stick rot/s, all
motors above idle) whose mean motor speed is closest to --hover-rps
(default: the median of those samples).

Also prints BF's own motor-voltage roughness over that window, computed like
the evals' `roughness_v` (one-pole high-pass at ROUGH_HP_HZ), as a budget
reference: that much ripple is known to fly on this airframe.  --json also
writes it, with the window, for the Configurator.

Output format (little-endian; whippyflight-sim's noise.rs):
  f32 sample_rate_hz, u32 n, then n frames of i16[7]:
  gyro roll/pitch/yaw (FRD: BF's FLU gyro with pitch and yaw negated,
  GYRO_LSB rot/s), motor 0..3 by BF motor index (ERPM_LSB shaft rev/s).
"""

import argparse
import csv
import json
import struct

import numpy as np
from scipy import signal

GYRO_LSB = 1e-5
ERPM_LSB = 1e-3
ROUGH_HP_HZ = 50.0
FRD_FROM_FLU = np.array([1.0, -1.0, -1.0])


COLS = ["time", "vbatLatest"] + [f"gyroADC[{i}]" for i in range(3)] + \
       [f"motor[{i}]" for i in range(4)] + [f"eRPM[{i}]" for i in range(4)] + \
       [f"setpoint[{i}]" for i in range(3)]


def load(spec, motor_poles=None):
    """A blackbox_decode CSV, or a .bbl/.bfl (first usable flight, or LOG#n), and its Flight (None for a CSV)."""
    from whippyflight_betaflight.fit_model import load_flights, shaft_rps
    if spec.lower().endswith(".csv"):
        if not motor_poles:
            raise SystemExit("a CSV doesn't record motor_poles; pass --motor-poles")
        with open(spec) as f:
            r = csv.reader(f)
            hdr = next(r)
            idx = [hdr.index(c) for c in COLS]
            rows = []
            for row in r:
                try:
                    rows.append([float(row[i]) for i in idx])
                except (ValueError, IndexError):
                    pass
        d = np.array(rows)
    else:
        flights = load_flights(spec, motor_poles=motor_poles)
        if not flights:
            raise SystemExit(f"no usable flight in {spec}")
        d = np.column_stack([flights[0].cols[c] for c in COLS])
        motor_poles = flights[0].motor_poles
    t = (d[:, 0] - d[0, 0]) * 1e-6
    return dict(t=t, fs=1.0 / np.median(np.diff(t)), vbat=d[:, 1] / 100.0,
                gyro=d[:, 2:5] / 360.0, motor=d[:, 5:9] / 2047.0, erpm=shaft_rps(d[:, 9:13], motor_poles),
                setpoint=d[:, 13:16] / 360.0), (None if spec.lower().endswith(".csv") else flights[0])


AXES_FLU = ("roll", "pitch", "yaw")


def explained(flight, coeffs, lo, hi):
    """
    The motion `coeffs`' model explains over samples [lo, hi): motor speeds
    (shaft rev/s, by BF motor index) its motor model reaches from the logged
    voltages, as the telemetry reports them, and body rates (rot/s, BF's FLU)
    integrated from the angular accelerations they drive as the airframe feels
    them (not the telemetry's, whose noise the yaw model would differentiate
    into them).  None where a row is unusable.
    """
    from whippyflight_betaflight import fit_model as fm
    coeffs = fm.regressor_frame(coeffs)
    arms, drag = fm.stored_arms(coeffs), fm.drag_axes(coeffs)
    volts_map = coeffs.get("fit", {}).get("volts", "glue")
    clean = fm.with_modeled_speeds(flight, coeffs, volts_map)
    t = flight.t[lo:hi]
    rates = np.zeros((hi - lo, 3))
    for a, axis in enumerate(AXES_FLU):
        X, _, names, idx = fm.design(clean, axis, arms, volts_map, drag, gated=False)
        alpha = np.full(len(flight.t), np.nan)
        alpha[idx] = X @ np.array([coeffs[axis][n] for n in names])
        alpha = alpha[lo:hi]
        if np.isnan(alpha).any():
            return None
        rates[:, a] = np.concatenate([[0.0], np.cumsum(0.5 * (alpha[1:] + alpha[:-1]) * np.diff(t))])
    reported = fm.modeled_speeds(flight, coeffs["motors"], volts_map, fm.voltage_delay(coeffs))
    return rates, np.column_stack(reported)[lo:hi]


LAW_BLOCK_S = 0.1
LAW_TRIM_S = 0.2
MAX_SPEED_EXPONENT = 2.0  # whippyflight-sim noise::MAX_SPEED_EXPONENT


def speed_exponent(flight, coeffs, erpm, fs, hp_hz, reference):
    """
    How the motor-speed noise grows with the motor's speed over the whole
    flight: the slope of log RMS against log mean speed, per motor, over
    LAW_BLOCK_S blocks of every usable stretch, the noise taken as in the
    window (the motor model's speeds out, high-passed at `hp_hz`).  Blocks
    below a tenth of `reference` are left out; None without enough spread.
    """
    from whippyflight_betaflight import fit_model as fm
    coeffs = fm.regressor_frame(coeffs)
    volts_map = coeffs.get("fit", {}).get("volts", "glue")
    model = np.column_stack(fm.modeled_speeds(flight, coeffs["motors"], volts_map, fm.voltage_delay(coeffs)))
    sos = signal.butter(4, hp_hz, "highpass", fs=fs, output="sos")
    block, trim = int(LAW_BLOCK_S * fs), int(LAW_TRIM_S * fs)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flight.good.astype(int), [0]])))
    speed, rms = [], []
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < 2 * trim + block + 30:
            continue
        noise = signal.sosfiltfilt(sos, erpm[s:e] - model[s:e], axis=0)
        for b in range(trim, e - s - trim - block + 1, block):
            speed.append(erpm[s + b:s + b + block].mean(0))
            rms.append(noise[b:b + block].std(0))
    speed, rms = np.ravel(speed), np.ravel(rms)
    keep = (speed > 0.1 * reference) & (rms > 0)
    if keep.sum() < 20 or np.percentile(speed[keep], 90) < 1.5 * np.percentile(speed[keep], 10):
        return None
    k = np.polyfit(np.log(speed[keep]), np.log(rms[keep]), 1)[0]
    return float(np.clip(k, 0.0, MAX_SPEED_EXPONENT))


def one_pole_hp(x, fc, fs):
    k = 2 * np.pi * fc / fs
    k = k / (k + 1)
    return x - signal.lfilter([k], [1, k - 1], x, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--hover-rps", type=float, default=None,
                    help="target mean motor speed, shaft rev/s (default: the median over "
                         "qualifying samples)")
    ap.add_argument("--max-stick", type=float, default=0.55,
                    help="the largest stick rate in the window, rot/s")
    ap.add_argument("--gyro-hp", type=float, default=80.0)
    ap.add_argument("--erpm-hp", type=float, default=40.0)
    ap.add_argument("--motor-poles", type=int, default=None,
                    help="the motors' pole count (default: the log's motor_poles)")
    ap.add_argument("--json", help="also write the window and BF's roughness_v here")
    ap.add_argument("--fit", help="the fit's coefficients.json: take out the motion its model explains")
    ap.add_argument("--diagnostics", help="the fit's diagnostics.json: high-pass at its noise splits")
    args = ap.parse_args()

    d, flight = load(args.log, args.motor_poles)
    fs, n = d["fs"], int(args.seconds * d["fs"])
    spinning = d["erpm"].mean(1)
    idle = 0.4 * np.median(spinning[spinning > 0])
    ok = (d["erpm"].min(1) > idle) & (np.abs(d["setpoint"]).max(1) < args.max_stick)
    run = np.convolve(ok, np.ones(n), "valid") >= n
    starts = np.flatnonzero(run)
    if len(starts) == 0:
        raise SystemExit(f"no {args.seconds:.1f} s window meets the criteria")
    mean_w = np.convolve(d["erpm"].mean(1), np.ones(n) / n, "valid")
    hover = args.hover_rps if args.hover_rps is not None else float(np.median(d["erpm"][ok].mean(1)))
    a = starts[np.argmin(np.abs(mean_w[starts] - hover))]
    pad = int(0.2 * fs)
    lo, hi = max(a - pad, 0), min(a + n + pad, len(d["t"]))

    split = {}
    if args.diagnostics:
        with open(args.diagnostics) as f:
            split = json.load(f).get("noise_split_hz") or {}
    gyro_hp = [split.get(axis) or args.gyro_hp for axis in AXES_FLU]
    erpm_hp = split.get("motors") or args.erpm_hp
    gyro, erpm = d["gyro"][lo:hi], d["erpm"][lo:hi]
    exponent = None
    if args.fit:
        if flight is None:
            raise SystemExit("--fit needs a blackbox log, not a CSV")
        with open(args.fit) as f:
            coeffs = json.load(f)
        exponent = speed_exponent(flight, coeffs, d["erpm"], fs, erpm_hp, mean_w[a])
        motion = explained(flight, coeffs, lo, hi)
        if motion is None:
            raise SystemExit("the window has rows the fit can't use; pick another flight")
        gyro, erpm = gyro - motion[0], erpm - motion[1]

    def hp(x, fc):
        sos = signal.butter(4, fc, "highpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, x, axis=0)[a - lo:a - lo + n]

    gyro = np.column_stack([hp(gyro[:, i], gyro_hp[i]) for i in range(3)]) * FRD_FROM_FLU
    erpm = hp(erpm, erpm_hp)
    frames = np.hstack([gyro / GYRO_LSB, erpm / ERPM_LSB])
    if np.abs(frames).max() > 32767:
        raise SystemExit("noise exceeds the i16 range; raise GYRO_LSB/ERPM_LSB")
    with open(args.out, "wb") as f:
        f.write(struct.pack("<fI", fs, n))
        f.write(np.round(frames).astype("<i2").tobytes())

    volts = d["motor"][lo:hi] * d["vbat"][lo:hi, None]
    rough = one_pole_hp(volts, ROUGH_HP_HZ, fs)[a - lo:a - lo + n]
    print("high-passed at gyro " + "/".join(f"{x:g}" for x in gyro_hp) + f" Hz, motor speed {erpm_hp:g} Hz"
          + (", the fit's motion taken out" if args.fit else ""))
    print(f"window {d['t'][a]:.2f}–{d['t'][a + n - 1]:.2f} s at {fs:.0f} Hz, "
          f"mean motor speed {mean_w[a]:.0f} rev/s, vbat {d['vbat'][a:a + n].mean():.2f} V")
    print("gyro noise rms (rot/s) roll/pitch/yaw: " + " ".join(f"{x:.5f}" for x in gyro.std(0)))
    print("motor speed noise rms (rev/s) motors 0-3: " + " ".join(f"{x:.3f}" for x in erpm.std(0)))
    if exponent is not None:
        print(f"motor speed noise grows as speed^{exponent:.2f} about {mean_w[a]:.0f} rev/s")
    print(f"BF roughness_v (rms of {ROUGH_HP_HZ:.0f} Hz high-passed motor volts, mean over motors): "
          f"{np.sqrt((rough ** 2).mean()):.4f} V")
    print(f"wrote {args.out} ({8 + frames.size * 2} bytes)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"log": args.log, "window_s": [float(d["t"][a]), float(d["t"][a + n - 1])],
                       "bf_rough_v": float(np.sqrt((rough ** 2).mean())),
                       "motor_noise_speed": None if exponent is None else
                       {"reference_rps": float(mean_w[a]), "exponent": exponent}}, f)


if __name__ == "__main__":
    main()
