#!/usr/bin/env python3
"""
Plot a .wfg blob's feedback gains over motor-speed space, as the firmware
interpolates them (whippyflight-control interpolation.rs).

    python -m whippyflight.plot_gains GAINS.wfg OUT_DIR

Writes OUT_DIR/gains_<motor>.png per motor (FL, FR, RL, RR): a lattice
with a row per gain (I, D, P) and a column per axis (roll, pitch, yaw),
each panel the magnitude of that motor's gain on that axis over its own
speed and the other three's, shared.  Dots mark the grid points the gains
are solved at, the cross hover.  And OUT_DIR/gains_diagonal.png: the same
gains with every motor at one speed, a line per motor.  Needs matplotlib.
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
WIDTH = 12
REF_WIDTH = 4 * len(g.REF_MOTOR_SLOTS[0])
SHORT = {"front_left": "FL", "rear_left": "RL", "front_right": "FR", "rear_right": "RR"}
POINTS = 121


def read_wfg(path):
    """The blob at `path`: its grid, hover speed, motor positions and gain tables
    [N, N, N, N, motor, axis] (motors by firmware index, axes FRD [roll, pitch, yaw])."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, n = struct.unpack_from("<IHB", data)
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
    at += 4 * len(g.REF_CONST_SLOTS) + 8 * REF_WIDTH + 4 * (16 + 2 + 4 + 5 + 2)
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
    return {"resolution": n, "min_rps": min_rps, "max_rps": max_rps, "hover_rps": hover_rps,
            "positions": [coefficients.POSITIONS[p] for p in positions], "tables": tables}


def interpolator(blob, table):
    """The table at motor speeds [..., 4] (rev/s) on the grid, multilinear."""
    speeds = np.linspace(blob["min_rps"], blob["max_rps"], blob["resolution"])
    return sc.interpolate.RegularGridInterpolator([speeds] * 4, table)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot(blob, out_dir, title):
    """Write the PNGs; returns their paths."""
    plt = pyplot()
    os.makedirs(out_dir, exist_ok=True)
    lo, hi, hover = blob["min_rps"], blob["max_rps"], blob["hover_rps"]
    own, others = np.meshgrid(np.linspace(lo, hi, POINTS), np.linspace(lo, hi, POINTS))
    gains = [interpolator(blob, t) for t in blob["tables"]]
    nodes = np.linspace(lo, hi, blob["resolution"])
    solved = np.array([(a, b) for a in nodes for b in nodes]).T
    paths = []
    for motor in sorted(range(4), key=lambda m: SHORT[blob["positions"][m]]):
        short = SHORT[blob["positions"][motor]]
        speeds = np.repeat(others[..., None], 4, axis=-1)
        speeds[..., motor] = own
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
        for row, ((name, unit), gain) in enumerate(zip(GAINS, gains)):
            values = gain(speeds.reshape(-1, 4))[:, motor, :]
            for col, axis in enumerate(AXES):
                ax = axes[row, col]
                z = np.abs(values[:, col]).reshape(own.shape)
                mesh = ax.pcolormesh(own, others, z, cmap="plasma", shading="auto")
                if z.max() > z.min():
                    ax.contour(own, others, z, levels=10, colors="white", linewidths=0.6, alpha=0.7)
                ax.plot(*solved, ls="", marker=".", color="white", markersize=4)
                ax.plot([hover], [hover], marker="+", color="cyan", markersize=12, mew=2)
                ax.set_aspect("equal")
                ax.set_title(f"{short} {name} on {axis}", fontsize=10)
                ax.set_xlabel(f"{short} speed (rev/s)", fontsize=8)
                ax.set_ylabel("the other motors' speed (rev/s)", fontsize=8)
                fig.colorbar(mesh, ax=ax).set_label(unit, fontsize=8)
        fig.suptitle(f"{title}, motor {motor + 1} ({blob['positions'][motor].replace('_', ' ')}): "
                     f"grid {lo:.0f}–{hi:.0f} rev/s at resolution {blob['resolution']}, "
                     f"hover {hover:.0f} rev/s (+)", fontsize=11)
        fig.tight_layout()
        path = os.path.join(out_dir, f"gains_{short}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    paths.append(plot_diagonal(blob, gains, out_dir, title))
    return paths


def plot_diagonal(blob, gains, out_dir, title):
    """Write the gains with every motor at one speed; returns its path."""
    plt = pyplot()
    lo, hi, hover = blob["min_rps"], blob["max_rps"], blob["hover_rps"]
    speed = np.linspace(lo, hi, POINTS)
    speeds = np.repeat(speed[:, None], 4, axis=1)
    order = sorted(range(4), key=lambda m: SHORT[blob["positions"][m]])
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), squeeze=False)
    for row, ((name, unit), gain) in enumerate(zip(GAINS, gains)):
        values = np.abs(gain(speeds))
        for col, axis in enumerate(AXES):
            ax = axes[row, col]
            for node in np.linspace(lo, hi, blob["resolution"]):
                ax.axvline(node, color="0.85", lw=0.8)
            ax.axvline(hover, color="tab:cyan", lw=1.2, ls="--", label="hover")
            for motor in order:
                ax.plot(speed, values[:, motor, col], lw=1.3, label=SHORT[blob["positions"][motor]])
            ax.set_title(f"{name} on {axis}", fontsize=10)
            ax.set_xlabel("every motor's speed (rev/s)", fontsize=8)
            ax.set_ylabel(unit, fontsize=8)
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{title}, every motor at one speed: grey lines mark the grid points the "
                 f"gains are solved at", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "gains_diagonal.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    blob = read_wfg(sys.argv[1])
    for path in plot(blob, sys.argv[2], os.path.basename(sys.argv[1])):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
