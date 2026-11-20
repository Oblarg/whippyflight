#!/usr/bin/env python3
"""
Where a log's flight sits in a .wfg blob's motor-speed space, and how far the
gains the blob schedules there stray from the same gains frozen at hover (a
fixed PID behind a mixer).

    python -m whippyflight_betaflight.plot_occupancy GAINS.wfg OUT_DIR LOG [LOG ...] [--motor-poles N]

Each LOG is a .bbl/.bfl (its every usable flight, or LOG#n for its n-th) or a
blackbox_decode CSV, as fit_model takes them; their flights are pooled, and
only their airborne, impact-free samples count (fit_model's gating).
Writes OUT_DIR/occupancy.png: per motor, the flight time over its own speed
and the other three's mean (the axes of plot_gains' lattices), and over the
spread of the four speeds; and OUT_DIR/deviation.png: per axis, a line per
gain, the fraction of flight time the scheduled gains differ from hover's by
more than a given amount, as ‖K(ω) − K(hover)‖ / ‖K(hover)‖ over the four motors;
and OUT_DIR/deviation_ff.png: the same for the feedforward's horizon inverse:
its rate feedforward G_h⁺, the voltages per rate change wanted over the horizon,
and its acceleration feedforward G_h⁺·F_h,ω, each axis's share of cancelling
the motor speeds' free response.  Inverting motors that lag behind an
integrator takes a term on the setpoint's rate of change and one on its
acceleration; Betaflight's feedforward and feedforward boost are those terms at
fixed gains, which these correspond to.  And OUT_DIR/deviation_affine.png:
the fraction of flight time the affine linearizer's voltage u_aff = G·ω|ω|,
which Betaflight has no counterpart of, differs from hover's by more than a
given amount, over the four motors.
Speeds off the grid clamp to it, as the firmware's do.  Needs matplotlib.
"""

import argparse
import os
import sys

import numpy as np

from whippyflight import plot_gains as pg
from whippyflight_betaflight.fit_model import load_flights, shaft_rps

# Relative gain deviations the summaries count flight time beyond.
THRESHOLDS = (0.1, 0.25, 0.5)
BINS = 80
# (style, width) per deviation line, so lines that coincide stay visible.
LINES = [("-", 3.0), ("--", 1.8), (":", 1.6), ("-.", 1.4)]


def flight_speeds(specs, motor_poles=None):
    """(labels, motor speeds [sample, motor] rev/s by BF motor index over the flights'
    airborne samples, flight time per sample s)."""
    flights = [f for spec in specs for f in load_flights(spec, motor_poles=motor_poles)]
    if not flights:
        raise SystemExit(f"no usable flight in {', '.join(specs)}")
    rps, dt = [], []
    for flight in flights:
        speeds = np.column_stack([shaft_rps(flight.cols[f"eRPM[{i}]"], flight.motor_poles) for i in range(4)])
        rps.append(speeds[flight.good])
        dt.append(np.full(int(flight.good.sum()), 1.0 / flight.log_hz))
    return [f.label for f in flights], np.concatenate(rps), np.concatenate(dt)


def beyond(x, dt, threshold):
    """The fraction of the flight time `dt` per sample that `x` exceeds `threshold`."""
    return float(dt[x > threshold].sum() / dt.sum())


def quantile(x, dt, q):
    """The `q` quantile of `x` over flight time `dt` per sample."""
    order = np.argsort(x)
    return float(np.interp(q, np.cumsum(dt[order]) / dt.sum(), x[order]))


def deviations(blob, rps):
    """[sample, gain, axis] relative distance of the scheduled gains from hover's over the
    four motors, gains in the blob's order; and each sample's speeds clamped to the grid."""
    clamped = np.clip(rps, blob["min_rps"], blob["max_rps"])
    hover = np.full((1, 4), blob["hover_rps"])
    out = np.empty((len(rps), len(blob["tables"]), 3))
    for g, table in enumerate(blob["tables"]):
        gain = pg.interpolator(blob, table)
        at_hover = gain(hover)[0]
        out[:, g, :] = np.linalg.norm(gain(clamped) - at_hover, axis=1) / np.linalg.norm(at_hover, axis=0)
    return out, clamped


def plot_occupancy(blob, rps, dt, out_dir, title):
    """Write occupancy.png; returns its path."""
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    plt = pg.pyplot()
    lo, hi, hover = blob["min_rps"], blob["max_rps"], blob["hover_rps"]
    edges = np.linspace(min(lo, rps.min()), max(hi, rps.max()), BINS + 1)
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.4), squeeze=False)
    order = sorted(range(4), key=lambda m: pg.SHORT[blob["positions"][m]])
    for ax, motor in zip(axes[0], order):
        short = pg.SHORT[blob["positions"][motor]]
        others = np.delete(rps, motor, axis=1).mean(axis=1)
        *_, mesh = ax.hist2d(rps[:, motor], others, bins=[edges, edges], weights=dt,
                             norm=LogNorm(), cmap="viridis")
        ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec="0.4", ls="--", lw=1))
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="0.6", lw=0.8)
        ax.plot([hover], [hover], marker="+", color="red", markersize=12, mew=2)
        ax.set_aspect("equal")
        ax.set_title(f"{short}", fontsize=10)
        ax.set_xlabel(f"{short} speed (rev/s)", fontsize=8)
        ax.set_ylabel("the other motors' mean speed (rev/s)", fontsize=8)
        fig.colorbar(mesh, ax=ax, fraction=0.046).set_label("flight time per bin (s)", fontsize=8)
    ax = axes[0, 4]
    spread = rps.max(axis=1) - rps.min(axis=1)
    ax.hist(spread, bins=BINS, weights=dt, color="tab:blue")
    for q in (50, 90, 99):
        at = quantile(spread, dt, q / 100)
        ax.axvline(at, color="0.3", ls=":", lw=1)
        ax.annotate(f"{q}%", (at, 1.0), xycoords=("data", "axes fraction"), fontsize=7,
                    ha="left", va="top", xytext=(2, -2), textcoords="offset points")
    ax.set_title("spread of the four speeds", fontsize=10)
    ax.set_xlabel("fastest − slowest motor (rev/s)", fontsize=8)
    ax.set_ylabel("flight time per bin (s)", fontsize=8)
    fig.suptitle(f"{title}: {dt.sum():.0f} s airborne; dashed: the gain grid "
                 f"{lo:.0f}–{hi:.0f} rev/s, +: hover {hover:.0f} rev/s, grey: every motor at one speed",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "occupancy.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def ff_deviations(blob, clamped):
    """[sample, gain, axis] relative distance of the feedforward's gains (plot_gains.FF_GAINS)
    from hover's."""
    hover = pg.ff_gains(blob, np.full((1, 4), blob["hover_rps"]))
    out = np.empty((len(clamped), len(pg.FF_GAINS), 3))
    for k, (gain, at_hover) in enumerate(zip(pg.ff_gains(blob, clamped), hover)):
        gain, at_hover = gain.reshape(len(gain), 3, -1), at_hover.reshape(1, 3, -1)
        out[:, k, :] = np.linalg.norm(gain - at_hover, axis=2) / np.linalg.norm(at_hover, axis=2)
    return out


def survival(ax, d, dt, style, label):
    """Draw the fraction of flight time `dt` per sample that `d` exceeds, over `d` (%), with
    the time beyond THRESHOLDS in the label."""
    order = np.argsort(d)
    past = 1.0 - np.cumsum(dt[order]) / dt.sum()
    beyonds = " / ".join(f"{100 * beyond(d, dt, t):.0f}" for t in THRESHOLDS)
    ax.plot(100 * d[order], 100 * past, lw=style[1], ls=style[0], label=f"{label}: {beyonds}%")


def threshold_legend(ax):
    ax.legend(title="time beyond " + " / ".join(f"{100 * t:.0f}" for t in THRESHOLDS) + "%",
              fontsize=8, title_fontsize=8)


def plot_affine(blob, clamped, dt, path, what):
    """Write the affine linearizer's voltage over the flight to `path`, titled `what`:
    its deviation from hover's over the four motors; returns `path`."""
    plt = pg.pyplot()
    volts = pg.affine_voltage(blob, clamped)
    at_hover = pg.affine_voltage(blob, np.full((1, 4), blob["hover_rps"]))[0]
    dev = np.linalg.norm(volts - at_hover, axis=1) / np.linalg.norm(at_hover)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    survival(ax, dev, dt, LINES[0], "u_aff")
    ax.set_xlim(0, 100 * max(quantile(dev, dt, 0.999), THRESHOLDS[-1]) * 1.05)
    ax.set_ylim(0, 100)
    ax.set_xlabel("‖u_aff(ω) − u_aff(hover)‖ / ‖u_aff(hover)‖ over the four motors (%)", fontsize=8)
    ax.set_ylabel("flight time beyond it (%)", fontsize=8)
    threshold_legend(ax)
    fig.suptitle(what, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_deviation(dev, names, dt, path, what):
    """Write the deviations `dev` [sample, gain, axis] of the gains `names` to `path`, titled
    `what`; returns `path`."""
    plt = pg.pyplot()
    top = max(max(quantile(dev[:, g, a], dt, 0.999) for g in range(dev.shape[1]) for a in range(3)),
              THRESHOLDS[-1])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True, squeeze=False)
    for col, axis in enumerate(pg.AXES):
        ax = axes[0, col]
        for row, name in enumerate(names):
            survival(ax, dev[:, row, col], dt, LINES[row], name)
        ax.set_xlim(0, 100 * top * 1.05)
        ax.set_ylim(0, 100)
        ax.set_title(axis, fontsize=10)
        ax.set_xlabel("deviation from the hover gains (%)", fontsize=8)
        threshold_legend(ax)
    axes[0, 0].set_ylabel("flight time beyond it (%)", fontsize=8)
    fig.suptitle(what, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gains", help=".wfg blob")
    parser.add_argument("out_dir")
    parser.add_argument("logs", nargs="+", metavar="LOG",
                        help=".bbl/.bfl (LOG#n for its n-th flight) or blackbox_decode CSV")
    parser.add_argument("--motor-poles", type=int, default=None,
                        help="motor poles, for logs that don't record them")
    args = parser.parse_args()
    blob = pg.read_wfg(args.gains)
    labels, rps, dt = flight_speeds(args.logs, args.motor_poles)
    dev, clamped = deviations(blob, rps)
    ff_dev = ff_deviations(blob, clamped)
    off_grid = beyond(np.any(clamped != rps, axis=1).astype(float), dt, 0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    title = f"{', '.join(labels)} on {os.path.basename(args.gains)}"
    feedback = [name for name, _ in pg.GAINS]
    feedforward = [name for name, _ in pg.FF_GAINS]
    for path in (
            plot_occupancy(blob, rps, dt, args.out_dir, title),
            plot_deviation(dev, feedback, dt, os.path.join(args.out_dir, "deviation.png"),
                           f"{title}: how far the scheduled feedback gains stray from the same "
                           f"gains frozen at hover, ‖K(ω) − K(hover)‖ / ‖K(hover)‖ over the four motors"),
            plot_deviation(ff_dev, feedforward, dt, os.path.join(args.out_dir, "deviation_ff.png"),
                           f"{title}: how far the feedforward's gains stray from the same inverse "
                           f"frozen at hover (horizon {blob['ff_horizon']} ticks)\nrate feedforward "
                           f"G_h⁺ (Betaflight's feedforward), acceleration feedforward G_h⁺·F_h,ω "
                           f"(Betaflight's feedforward boost)"),
            plot_affine(blob, clamped, dt, os.path.join(args.out_dir, "deviation_affine.png"),
                        f"{title}: how far the affine linearizer's voltage u_aff = G·ω|ω|\n"
                        f"strays from hover's (no Betaflight counterpart)")):
        print(f"wrote {path}", file=sys.stderr)
    print(f"{dt.sum():.1f} s airborne, {100 * off_grid:.1f}% of it off the grid", file=sys.stderr)
    print("flight time the scheduled gains stray from hover's by more than "
          + ", ".join(f"{100 * t:.0f}%" for t in THRESHOLDS) + ":", file=sys.stderr)
    for names, d in ((feedback, dev), (feedforward, ff_dev)):
        for row, name in enumerate(names):
            for col, axis in enumerate(pg.AXES):
                print(f"  {name} on {axis:5}: "
                      + ", ".join(f"{100 * beyond(d[:, row, col], dt, t):4.0f}%" for t in THRESHOLDS),
                      file=sys.stderr)


if __name__ == "__main__":
    main()
