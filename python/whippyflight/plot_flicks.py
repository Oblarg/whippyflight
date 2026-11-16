#!/usr/bin/env python3
"""
Plot the stick flicks the Configurator's WhippyFlight tab flies computed
gains through (crates/whippyflight-sim/src/flick.rs, written as
gains/flicks.json): one full-stick flick each way per axis, the fastest
raised cosine within the setpoint target's slew and jerk limits, or a
slower one if asked for.

    python3 plot_flicks.py flicks.json OUT_DIR

Writes OUT_DIR/flicks.png: per flick, the whole flick and its onset and
release up close (stick, feedforward target and response in rot/s, motor
saturation shaded), and under it the four motor voltages and shaft speeds
(rev/s) over the same spans.  Needs matplotlib.
"""

import json
import os
import sys


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def times(report, f):
    """Trace times, ms from the stick's start."""
    return [1e3 * i * report["trace_dt"] for i in range(len(f["response"]))]


def shade_saturation(ax, t, saturated):
    start = None
    for ti, s in zip(t + [None], saturated + [0.0]):
        if s > 0 and start is None:
            start = ti
        elif s == 0 and start is not None:
            ax.axvspan(start, ti if ti is not None else t[-1], color="red", alpha=0.12, lw=0)
            start = None


DIRECTIONS = {"roll": ("right", "left"), "pitch": ("up", "down"), "yaw": ("right", "left")}


def describe(m):
    """The controller's part: how far the gyro trails the target, and how it lands."""
    behind = m["gyro_lag_ms"] - m["target_lag_ms"]
    rough = "" if m.get("rough_v") is None else f", roughness {1e3 * m['rough_v']:.0f} mV"
    return (f"gyro {behind:.1f} ms behind target, overshoot {100 * m['overshoot']:+.1f}%, "
            f"sat {100 * m['sat_frac']:.0f}%{rough}")


def windows(report, f):
    """(title, ms span) of the whole flick, its onset and its release."""
    rise = f["rise_ms"]
    release = rise + 1e3 * report["hold_s"]
    near = 1.6 * rise + 30
    return [("", None), ("onset", (-5, near)), ("release", (release - 5, release + near))]


def plot(report, out_dir):
    """Write the PNG; returns its paths."""
    plt = pyplot()
    os.makedirs(out_dir, exist_ok=True)
    flicks = report["flicks"]
    fig, axes = plt.subplots(3 * len(flicks), 3, figsize=(15, 9.6 * len(flicks)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1, 1]})
    for row, motor_row, speed_row, f in zip(axes[::3], axes[1::3], axes[2::3], flicks):
        t = times(report, f)
        name = f"{f['axis']} {DIRECTIONS[f['axis']][f['amp'] < 0]} {abs(f['amp']):.2f} rot/s"
        for ax, (part, span) in zip(row, windows(report, f)):
            shade_saturation(ax, t, f["saturated"])
            ax.plot(t, f["stick"], color="0.5", ls="--", lw=1, label="stick")
            ax.plot(t, f["target"], color="tab:orange", lw=1, label="target")
            ax.plot(t, f["response"], color="tab:blue", lw=1.2, label="response")
            if span:
                ax.set_xlim(*span)
                ax.set_title(f"{name} {part}", fontsize=9)
            else:
                ax.set_title(f"{name}, {f['rise_ms']:.0f} ms stick: {describe(f['metrics'])}", fontsize=9)
            ax.set_xlabel("ms from stick start")
            ax.set_ylabel("rot/s")
            ax.grid(alpha=0.3)
        motors = [(motor_row, "motor voltage", "V", f["volts"]),
                  (speed_row, "motor speed", "rev/s", f["motor_speeds"])]
        for axes_row, what, unit, traces in motors:
            for ax, (part, span) in zip(axes_row, windows(report, f)):
                shade_saturation(ax, t, f["saturated"])
                for m, values in enumerate(traces):
                    ax.plot(t, values, lw=1, label=f"motor {m + 1}")
                if span:
                    ax.set_xlim(*span)
                ax.set_title(f"{name} {what} {part}".rstrip(), fontsize=9)
                ax.set_xlabel("ms from stick start")
                ax.set_ylabel(unit)
                ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    rates = {"fc": "the FC's max rates", "log": "the log's max rates", "manual": "hand-set rates"}.get(
        report["rates_from"], "Betaflight's default max rates")
    slowed = "" if report.get("min_rise_ms") is None else f", stick slowed to {report['min_rise_ms']:.0f} ms"
    zeroed = [t.upper() for t in "pid" if not report.get("feedback", {}).get(t, True)]
    if len(zeroed) == 3:
        slowed += ", feedforward only"
    elif zeroed:
        slowed += f", {'/'.join(zeroed)} feedback zeroed"
    cond = report.get("conditions") or {}
    if cond.get("gyro_filter") and not report.get("gyro_filtered", True):
        slowed += ", unfiltered gyro"
    elif not cond.get("gyro_filter", True):
        slowed += ", no gyro filters"
    lpf = report.get("motor_lpf_hz")
    if lpf is not None:
        source = {"fc": " (FC)", "log": " (log)"}.get(report.get("motor_lpf_from"), "")
        slowed += f", motor-speed low-pass{source} " + (f"{lpf:.0f} Hz" if lpf > 0 else "off")
    latency = report.get("latency")
    if latency and (latency["actuator_ms"] or latency["telemetry_ms"]):
        source = " (set)" if cond.get("actuator_delay_ms") is not None or cond.get("telemetry_delay_ms") is not None \
            else ""
        slowed += (f", delays{source} motor command {latency['actuator_ms']:.2f} ms, "
                   f"speed telemetry {latency['telemetry_ms']:.2f} ms")
    noise = report.get("noise_rms")
    if cond.get("noisy_flights") and noise and (noise["gyro"] or noise["motor"]):
        slowed += f", noise {noise['gyro']:.2g} rot/s {noise['motor']:.2g} rev/s RMS"
    cg = cond.get("cg_shift_pct") or [0, 0]
    if any(cg):
        slowed += f", CG {cg[0]:+g}% right {cg[1]:+g}% forward"
    turbulence = report.get("turbulence")
    if turbulence:
        rms = "/".join(f"{x:.2f}" for x in turbulence["rms"])
        slowed += f", turbulence {rms} rot/s² at {turbulence['hz']:.1f} Hz"
    hover, bf = report.get("hover_rough_v"), report.get("bf_rough_v")
    bf = "" if hover is None else (
        f"; noise roughness at hover {1e3 * hover:.1f} mV"
        + ("" if bf is None else f" (BF's total ripple at hover on the log {1e3 * bf:.0f} mV)"))
    robust = []
    margin = report.get("delay_margin_ms")
    if margin is not None:
        robust.append("delay margin none" if margin < 0 else f"delay margin {margin:.1f} ms")
    if report.get("trim_deg") is not None:
        robust.append(f"5% CG shift {report['trim_deg']:.1f}°")
    robust = "" if not robust else "; " + ", ".join(robust)
    fig.suptitle(f"Full-stick flicks at {rates}, on the {report['fit']} plant{robust}\n"
                 + (slowed[2:] + "\n" if slowed else "")
                 + f"target limits {report['slew']:.0f} rot/s², {report['jerk']:.0f} rot/s³; red: motors saturated{bf}",
                 fontsize=10, y=1 - 0.15 / fig.get_figheight(), va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    path = os.path.join(out_dir, "flicks.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        report = json.load(f)
    for path in plot(report, sys.argv[2]):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
