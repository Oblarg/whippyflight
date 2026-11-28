#!/usr/bin/env python3
"""
Break a motor's voltage roughness down by its source, from a log flown with
debug_mode = WHIPPYFLIGHT (one motor, whippyflight_debug_motor).

  python -m whippyflight_betaflight.effort LOG [--from S] [--to S] [--hp-hz 50]

LOG is a .bbl/.bfl; LOG#n picks one flight of a multi-flight file (default:
every flight logged in that mode).  Flights before the effort trace logged
other values in that mode, so an older log reads as nonsense here.  Every term is high-passed like the evals'
`roughness_v` (one-pole at --hp-hz) and reported as its rms and its share of
the applied voltage's high-passed variance, cov(term, applied) / var(applied).
The shares of the command's terms and the allocator's (the applied voltage
less the command: the collective and any clipping) sum to 1.  The schedule
jitter is the part of the feedback (P + I + D) due to the gains moving with
the motor speed faster than the controller's SCHEDULE_TRACE_LPF; its share
is of the same variance, and is already counted in P, I and D.
"""

import argparse

import numpy as np
from scipy import signal

from whippyflight_betaflight.bbl import read_logs

ROUGH_HP_HZ = 50.0

# build/debug.h: DEBUG_WHIPPYFLIGHT.  Logs record the mode by index.
DEBUG_WHIPPYFLIGHT = 114

TERMS = ("P", "I", "D", "feedforward", "affine")
SLOT_JITTER, SLOT_APPLIED, SLOT_SPEED = 5, 6, 7
MV, DECI_RPS = 1e-3, 0.1


def one_pole_hp(x, fc, fs):
    """x less its one-pole low-pass at `fc`, from rest at zero (the evals' roughness filter)."""
    k = 2 * np.pi * fc / fs
    k = k / (k + 1)
    return x - signal.lfilter([k], [1, k - 1], x, axis=0)


def effort_logs(spec):
    """(flight number, log) for LOG#n, or every flight in LOG logged in WHIPPYFLIGHT."""
    path, _, which = spec.partition("#")
    logs = list(enumerate(read_logs(path), 1))
    if which:
        logs = [logs[int(which) - 1]]
    picked = [(n, log) for n, log in logs if int(log.header.get("debug_mode", -1)) == DEBUG_WHIPPYFLIGHT]
    if which and not picked:
        raise SystemExit(f"{spec}: debug_mode is {logs[0][1].header.get('debug_mode')}, "
                         f"not WHIPPYFLIGHT ({DEBUG_WHIPPYFLIGHT})")
    return picked


def breakdown(log, start_s, end_s, hp_hz):
    t = (log["time"] - log["time"][0]) * 1e-6
    keep = (t >= start_s) & (t <= (end_s if end_s is not None else t[-1]))
    fs = log.log_hz
    slot = lambda i: log[f"debug[{i}]"][keep].astype(float)
    terms = {name: slot(i) * MV for i, name in enumerate(TERMS)}
    applied = slot(SLOT_APPLIED) * MV
    terms["allocator"] = applied - sum(terms[name] for name in TERMS)
    jitter = slot(SLOT_JITTER) * MV
    speed = slot(SLOT_SPEED) * DECI_RPS

    # From the first sample, so a term's offset doesn't ring the filter.
    hp = lambda x: one_pole_hp(x - x[0], hp_hz, fs)
    applied_hp = hp(applied)
    var = np.mean(applied_hp ** 2)
    rows = []
    for name, x in [*terms.items(), ("schedule jitter", jitter)]:
        x_hp = hp(x)
        rows.append((name, np.sqrt(np.mean(x_hp ** 2)), np.mean(x_hp * applied_hp) / var))
    return dict(seconds=keep.sum() / fs, fs=fs, rough=np.sqrt(var), speed=np.median(speed),
                jitter_traced=bool(np.any(jitter)), rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--from", dest="start", type=float, default=0.0, help="seconds into the flight")
    ap.add_argument("--to", dest="end", type=float, default=None, help="seconds into the flight")
    ap.add_argument("--hp-hz", type=float, default=ROUGH_HP_HZ)
    args = ap.parse_args()

    logs = effort_logs(args.log)
    if not logs:
        raise SystemExit(f"{args.log}: no flight logged with debug_mode = WHIPPYFLIGHT")
    for n, log in logs:
        b = breakdown(log, args.start, args.end, args.hp_hz)
        motor = log.header.get("whippyflight_debug_motor", "?")
        print(f"flight {n}, motor {motor}: {b['seconds']:.1f} s at {b['fs']:.0f} Hz, "
              f"median filtered speed {b['speed']:.0f} rev/s")
        print(f"  applied roughness (rms above {args.hp_hz:.0f} Hz): {b['rough'] * 1e3:.1f} mV")
        print(f"  {'term':<16}{'rms mV':>9}{'share':>9}")
        for name, rms, share in b["rows"]:
            print(f"  {name:<16}{rms * 1e3:>9.1f}{share:>9.1%}")
        if not b["jitter_traced"]:
            print("  (schedule jitter reads zero throughout: untraced)")


if __name__ == "__main__":
    main()
