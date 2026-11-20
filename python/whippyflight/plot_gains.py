#!/usr/bin/env python3
"""
Plot a .wfg blob's control terms over motor-speed space, as the firmware
evaluates them (whippyflight-control interpolation.rs, controller.rs).

    python -m whippyflight.plot_gains GAINS.wfg OUT_DIR

Writes OUT_DIR/gains_<motor>.png per motor (FL, FR, RL, RR): a lattice
with a row per feedback gain (I, D, P) and a column per axis (roll, pitch,
yaw), each panel the magnitude of that motor's gain on that axis over its
own speed and the other three's, shared.  Dots mark the grid points the
gains are solved at, the cross hover.  OUT_DIR/gains_diagonal.png: the same
gains with every motor at one speed, a line per motor.
OUT_DIR/feedforward_<motor>.png and feedforward_diagonal.png: the same for
the feedforward's horizon inverse, its rate feedforward G_h⁺ (voltage per
rate change over the horizon) and its acceleration feedforward G_h⁺·F_h,ω
(voltage per motor-speed deviation, the norm over the four motors' speeds).
OUT_DIR/affine.png: the affine linearizer's voltage u_aff = G·ω|ω|, the
motors' mean with every motor at one speed.  Needs matplotlib.
"""

import os
import struct
import sys

import numpy as np
import scipy as sc

from whippyflight import coefficients
from whippyflight import gen_wfg as g

AXES = ["roll", "pitch", "yaw"]
# (name, unit) of the blob's gain tables, in their order.
GAINS = [("I", "V/rot"), ("D", "V·s²/rot"), ("P", "V·s/rot")]
# (name, unit) of the feedforward's gains (ff_gains), and the reference model's per-motor
# slots (gains.rs REF_MOTOR_SLOTS) of F_h's motor-speed columns and of G_h.
FF_GAINS = [("rate feedforward", "V·s/rot"), ("acceleration feedforward", "V·s/rev")]
REF_FH_SPEED, REF_GH = slice(8, 11), slice(11, 14)
WIDTH = 12
REF_WIDTH = 4 * len(g.REF_MOTOR_SLOTS[0])
SHORT = {"front_left": "FL", "rear_left": "RL", "front_right": "FR", "rear_right": "RR"}
POINTS = 121


def read_wfg(path):
    """The blob at `path`: its grid, hover speed, motor positions, gain tables
    [N, N, N, N, motor, axis] (motors by firmware index, axes FRD [roll, pitch, yaw]),
    feedforward horizon (ticks) and reference model: its speed-free values and its
    per-motor table [N, motor, slot] (gains.rs REF_CONST_FLOATS, REF_MOTOR_SLOTS); and
    the affine linearizer G [motor, motor] (V per (rev/s)²)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, n, ff_horizon = struct.unpack_from("<IHBB", data)
    if magic != g.MAGIC:
        sys.exit(f"{path} is not a .wfg blob")
    if version != g.VERSION:
        sys.exit(f"{path} is .wfg version {version}; this reads version {g.VERSION}")
    min_rps, max_rps = struct.unpack_from("<2f", data, 0x08)
    (size,) = struct.unpack_from("<I", data, 0x1C)
    if size != len(data) or g.crc32(data[:-4]) != struct.unpack_from("<I", data, len(data) - 4)[0]:
        sys.exit(f"{path} is truncated or corrupt")
    at = g.HEADER_SIZE
    scales = []
    for width in (WIDTH, WIDTH, WIDTH):
        scales.append((np.frombuffer(data, "<f4", width, at), np.frombuffer(data, "<f4", width, at + 4 * width)))
        at += 8 * width
    ref_const = np.frombuffer(data, "<f4", len(g.REF_CONST_SLOTS), at).astype(np.float64)
    at += 4 * len(g.REF_CONST_SLOTS)
    ref_lo, ref_step = (np.frombuffer(data, "<f4", REF_WIDTH, at + 4 * REF_WIDTH * k) for k in (0, 1))
    at += 8 * REF_WIDTH
    affine = np.frombuffer(data, "<f4", 16, at).astype(np.float64).reshape(4, 4)
    at += 4 * (16 + 2 + 4 + 5 + 2)
    (hover_rps,) = struct.unpack_from("<f", data, at)
    at += 4 * (2 + 8)
    positions = list(data[at:at + 4])
    at += 8
    tables = []
    for lo, step in scales:
        q = np.frombuffer(data, "<u2", n ** 4 * WIDTH, at)
        at += 2 * q.size
        table = lo.astype(np.float64) + step.astype(np.float64) * q.reshape(-1, WIDTH)
        tables.append(table.reshape([n] * 4 + [4, 3]))
    q = np.frombuffer(data, "<u2", n * REF_WIDTH, at)
    ref_motor = ref_lo.astype(np.float64) + ref_step.astype(np.float64) * q.reshape(n, REF_WIDTH)
    return {"resolution": n, "min_rps": min_rps, "max_rps": max_rps, "hover_rps": hover_rps,
            "positions": [coefficients.POSITIONS[p] for p in positions], "tables": tables,
            "ff_horizon": ff_horizon, "ref_const": ref_const,
            "ref_motor": ref_motor.reshape(n, 4, len(g.REF_MOTOR_SLOTS[0])), "affine": affine}


def interpolator(blob, table):
    """The table at motor speeds [..., 4] (rev/s) on the grid, multilinear."""
    speeds = np.linspace(blob["min_rps"], blob["max_rps"], blob["resolution"])
    return sc.interpolate.RegularGridInterpolator([speeds] * 4, table)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def ff_gains(blob, speeds):
    """The feedforward's gains at motor speeds `speeds` [sample, motor] (rev/s, clamped to
    the grid), per axis: the rate feedforward G_h⁺[:, a] [sample, axis, motor] and the
    acceleration feedforward, the axis's share G_h⁺[:, a]·F_h,ω[a, :] of cancelling the
    motor speeds' free response [sample, axis, motor, motor speed].  G_h⁺ = G_hᵀ(G_h·G_hᵀ)⁻¹
    from the interpolated G_h, as the firmware's."""
    clamped = np.clip(speeds, blob["min_rps"], blob["max_rps"])
    nodes = np.linspace(blob["min_rps"], blob["max_rps"], blob["resolution"])
    table = blob["ref_motor"]
    slots = np.stack([np.stack([np.interp(clamped[:, m], nodes, table[:, m, s])
                                for s in range(table.shape[2])], axis=-1)
                      for m in range(4)], axis=1)
    g = slots[..., REF_GH].transpose(0, 2, 1)
    f = slots[..., REF_FH_SPEED].transpose(0, 2, 1)
    gt = g.transpose(0, 2, 1)
    inverse = (gt @ np.linalg.inv(g @ gt)).transpose(0, 2, 1)
    return inverse, inverse[..., :, None] * f[:, :, None, :]


def affine_voltage(blob, speeds):
    """The affine linearizer's voltage [sample, motor] at motor speeds `speeds` [sample, motor]
    (rev/s, clamped to the grid), u_aff = G·ω̂|ω̂|."""
    clamped = np.clip(speeds, blob["min_rps"], blob["max_rps"])
    return (clamped * np.abs(clamped)) @ blob["affine"].T


def feedback_terms(blob):
    """(name, unit, magnitude at speeds [sample, motor] → [sample, motor, axis]) per feedback gain."""
    return [(name, unit, lambda s, gain=interpolator(blob, t): np.abs(gain(s)))
            for (name, unit), t in zip(GAINS, blob["tables"])]


def feedforward_terms(blob):
    """(name, unit, magnitude at speeds [sample, motor] → [sample, motor, axis]) per feedforward
    gain, the acceleration feedforward's the norm over the motor speeds it acts on."""
    rate = lambda s: np.abs(ff_gains(blob, s)[0]).transpose(0, 2, 1)
    acceleration = lambda s: np.linalg.norm(ff_gains(blob, s)[1], axis=3).transpose(0, 2, 1)
    return [(*FF_GAINS[0], rate), (*FF_GAINS[1], acceleration)]


def motor_order(blob):
    return sorted(range(4), key=lambda m: SHORT[blob["positions"][m]])


def lattice_speeds(blob, motor):
    """(own, others) meshgrids over the grid, and the speeds [own, others, motor] they make."""
    lo, hi = blob["min_rps"], blob["max_rps"]
    own, others = np.meshgrid(np.linspace(lo, hi, POINTS), np.linspace(lo, hi, POINTS))
    speeds = np.repeat(others[..., None], 4, axis=-1)
    speeds[..., motor] = own
    return own, others, speeds


def surface(fig, ax, blob, own, others, z, unit):
    """Draw `z` over (own, others) with the solved grid points and hover."""
    lo, hi, hover = blob["min_rps"], blob["max_rps"], blob["hover_rps"]
    nodes = np.linspace(lo, hi, blob["resolution"])
    mesh = ax.pcolormesh(own, others, z, cmap="plasma", shading="auto")
    if z.max() > z.min():
        ax.contour(own, others, z, levels=10, colors="white", linewidths=0.6, alpha=0.7)
    ax.plot(*np.array([(a, b) for a in nodes for b in nodes]).T, ls="", marker=".", color="white",
            markersize=4)
    ax.plot([hover], [hover], marker="+", color="cyan", markersize=12, mew=2)
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax).set_label(unit, fontsize=8)


def grid_lines(ax, blob):
    for node in np.linspace(blob["min_rps"], blob["max_rps"], blob["resolution"]):
        ax.axvline(node, color="0.85", lw=0.8)
    ax.axvline(blob["hover_rps"], color="tab:cyan", lw=1.2, ls="--", label="hover")


def save(fig, out_dir, name):
    plt = pyplot()
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot(blob, out_dir, title):
    """Write the PNGs; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    horizon = f"feedforward over a {blob['ff_horizon']}-tick horizon"
    paths = []
    for terms, stem, what in ((feedback_terms(blob), "gains", "feedback gains"),
                              (feedforward_terms(blob), "feedforward", horizon)):
        paths += plot_lattices(blob, terms, out_dir, stem, f"{title}, {what}")
        paths.append(plot_diagonal(blob, terms, out_dir, stem, f"{title}, {what}"))
    paths.append(plot_affine(blob, out_dir, title))
    return paths


def plot_lattices(blob, terms, out_dir, stem, title):
    """Write OUT_DIR/<stem>_<motor>.png per motor, a row per term of `terms`; returns their paths."""
    plt = pyplot()
    lo, hi, hover = blob["min_rps"], blob["max_rps"], blob["hover_rps"]
    paths = []
    for motor in motor_order(blob):
        short = SHORT[blob["positions"][motor]]
        own, others, speeds = lattice_speeds(blob, motor)
        fig, axes = plt.subplots(len(terms), 3, figsize=(15, 4 * len(terms)), squeeze=False)
        for row, (name, unit, magnitude) in enumerate(terms):
            values = magnitude(speeds.reshape(-1, 4))[:, motor, :]
            for col, axis in enumerate(AXES):
                ax = axes[row, col]
                surface(fig, ax, blob, own, others, values[:, col].reshape(own.shape), unit)
                ax.set_title(f"{short} {name} on {axis}", fontsize=10)
                ax.set_xlabel(f"{short} speed (rev/s)", fontsize=8)
                ax.set_ylabel("the other motors' speed (rev/s)", fontsize=8)
        fig.suptitle(f"{title}, motor {motor + 1} ({blob['positions'][motor].replace('_', ' ')}): "
                     f"grid {lo:.0f}–{hi:.0f} rev/s at resolution {blob['resolution']}, "
                     f"hover {hover:.0f} rev/s (+)", fontsize=11)
        paths.append(save(fig, out_dir, f"{stem}_{short}.png"))
    return paths


def plot_diagonal(blob, terms, out_dir, stem, title):
    """Write OUT_DIR/<stem>_diagonal.png, `terms` with every motor at one speed; returns its path."""
    plt = pyplot()
    speed = np.linspace(blob["min_rps"], blob["max_rps"], POINTS)
    speeds = np.repeat(speed[:, None], 4, axis=1)
    fig, axes = plt.subplots(len(terms), 3, figsize=(15, 3.4 * len(terms)), squeeze=False)
    for row, (name, unit, magnitude) in enumerate(terms):
        values = magnitude(speeds)
        for col, axis in enumerate(AXES):
            ax = axes[row, col]
            grid_lines(ax, blob)
            for motor in motor_order(blob):
                ax.plot(speed, values[:, motor, col], lw=1.3, label=SHORT[blob["positions"][motor]])
            ax.set_title(f"{name} on {axis}", fontsize=10)
            ax.set_xlabel("every motor's speed (rev/s)", fontsize=8)
            ax.set_ylabel(unit, fontsize=8)
            ax.set_ylim(0, 1.08 * values[..., col].max())
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{title}, every motor at one speed: grey lines mark the grid points the "
                 f"gains are solved at", fontsize=11)
    return save(fig, out_dir, f"{stem}_diagonal.png")


def plot_affine(blob, out_dir, title):
    """Write OUT_DIR/affine.png: the motors' mean u_aff with all at one speed; returns its path."""
    plt = pyplot()
    speed = np.linspace(blob["min_rps"], blob["max_rps"], POINTS)
    mean = affine_voltage(blob, np.repeat(speed[:, None], 4, axis=1)).mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    grid_lines(ax, blob)
    ax.plot(speed, mean, lw=1.6, color="tab:red", label="motors' mean")
    ax.set_xlabel("every motor's speed (rev/s)", fontsize=8)
    ax.set_ylabel("V", fontsize=8)
    ax.set_ylim(min(0, 1.08 * mean.min()), max(0, 1.08 * mean.max()))
    ax.legend(fontsize=7)
    diag = np.abs(np.diag(blob["affine"]))
    cross = (np.abs(blob["affine"]).sum(axis=1) - diag) / diag
    ax.set_title(f"{title}, affine linearizer u_aff = G·ω|ω|, every motor at one speed:\n"
                 f"the motors' mean; the other motors' terms of G are at most "
                 f"{cross.max():.1e} of a motor's own", fontsize=10)
    return save(fig, out_dir, "affine.png")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    blob = read_wfg(sys.argv[1])
    for path in plot(blob, sys.argv[2], os.path.basename(sys.argv[1])):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
