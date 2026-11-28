"""
Plant coefficient JSONs (what a firmware's fit tool writes and gen_wfg
reads), in the canonical frame.

The model is a quad-X with its motors in slot order [front_left, rear_left,
front_right, rear_right] and body rates in FRD [roll, pitch, yaw] (x
forward, y right, z down).  The pitch arms dFront/dBack and roll arms
dLeft/dRight are physical, and drag coefficients map FRD body specific
force to FRD angular acceleration.  The yaw torque pattern is + on
front_left and rear_right; the fitted yaw coefficients carry the sign of
the airframe's spin.

The `frame` block says how the firmware numbers the motors:

    "frame": {
      "axes": "FRD",
      "motors": [{"position": "rear_right", "spin": "cw"}, ...]
    }

one entry per firmware motor index, spin viewed from above.

Fits without a `frame` block are Betaflight fits from before it existed:
in Betaflight's motor order, solved in the physical frame rotated 180°
about z, with drag fitted in Betaflight's FLU axes.  `normalize` converts
them.

Usage:
    python -m whippyflight.coefficients <fit.json> [--write OUT]
prints (or writes) the fit in the canonical frame.
"""

import argparse
import copy
import json
import sys

# Slot order: the gain model's motor order and the blob's position codes.
POSITIONS = ["front_left", "rear_left", "front_right", "rear_right"]
SPINS = ["cw", "ccw"]
DIAGONAL = {"front_left": "rear_right", "rear_right": "front_left",
            "front_right": "rear_left", "rear_left": "front_right"}
BESIDE = {"front_left": "front_right", "front_right": "front_left",
          "rear_left": "rear_right", "rear_right": "rear_left"}

# Betaflight's QUADX mixer (mixerQuadX), by motor index, and its default
# (props-in) spins; yaw_motors_reversed flips every spin.
BETAFLIGHT_QUADX = ["rear_right", "front_right", "rear_left", "front_left"]
BETAFLIGHT_PROPS_IN = ["cw", "ccw", "ccw", "cw"]

# Arms the lever-arm weights fall back to without fitted ones: symmetric.
DEFAULT_ARMS = {"dFront": 1.0, "dBack": 1.0, "dLeft": 1.0, "dRight": 1.0}
# What gen_wfg assumed for a legacy fit without arms (legacy frame).
LEGACY_DEFAULT_ARMS = {"dFront": 1.0, "dBack": 1.7, "dLeft": 1.0, "dRight": 1.0}
ARM_AXIS = {"dFront": "pitch", "dBack": "pitch", "dLeft": "roll", "dRight": "roll"}


class FrameError(ValueError):
    pass


def check_frame(frame):
    """Validate a `frame` block; raises FrameError."""
    if frame.get("axes") != "FRD":
        raise FrameError(f"frame.axes is {frame.get('axes')!r}, not 'FRD'")
    motors = frame.get("motors")
    if not isinstance(motors, list) or len(motors) != 4:
        raise FrameError("frame.motors must list 4 motors (quad-X)")
    for m in motors:
        if m.get("position") not in POSITIONS:
            raise FrameError(f"motor position {m.get('position')!r} is not one of {POSITIONS}")
        if m.get("spin") not in SPINS:
            raise FrameError(f"motor spin {m.get('spin')!r} is not one of {SPINS}")
    positions = [m["position"] for m in motors]
    if sorted(positions) != sorted(POSITIONS):
        raise FrameError(f"frame.motors positions {positions} don't cover every corner once")
    spin = {m["position"]: m["spin"] for m in motors}
    for p in POSITIONS:
        if spin[DIAGONAL[p]] != spin[p] or spin[BESIDE[p]] == spin[p]:
            raise FrameError(f"spins {spin}: diagonal motors must share a direction, "
                             "the two pairs opposite")


def is_legacy(coeffs):
    return "frame" not in coeffs


def betaflight_frame(airframe):
    """The `frame` block of a Betaflight QUADX airframe (the fit's
    `airframe` block)."""
    mixer = airframe.get("mixer", "QUADX")
    if mixer != "QUADX":
        raise FrameError(f"mixer {mixer}: only QUADX is supported")
    reversed_ = bool(airframe.get("yaw_motors_reversed", False))
    flip = {"cw": "ccw", "ccw": "cw"}
    return {"axes": "FRD", "motors": [
        {"position": p, "spin": flip[s] if reversed_ else s}
        for p, s in zip(BETAFLIGHT_QUADX, BETAFLIGHT_PROPS_IN)]}


def normalize(coeffs):
    """`coeffs` in the canonical frame (a copy).  Canonical fits are checked
    and returned as they are; legacy Betaflight fits are converted."""
    out = copy.deepcopy(coeffs)
    if not is_legacy(out):
        check_frame(out["frame"])
        return out
    # The legacy model frame is the physical one turned 180° about z: its
    # front is the physical rear and its left the physical right, so the
    # arms swap; the plant coefficients are the same in either.
    arms = {k: out.get(ARM_AXIS[k], {}).get(k, v) for k, v in LEGACY_DEFAULT_ARMS.items()}
    out["pitch"]["dFront"], out["pitch"]["dBack"] = arms["dBack"], arms["dFront"]
    out["roll"]["dLeft"], out["roll"]["dRight"] = arms["dRight"], arms["dLeft"]
    # Drag was fitted Betaflight gyro on Betaflight accelerometer, both FLU:
    # FRD negates y and z, so roll←y and pitch←x change sign.
    for axis, term in [("roll", "accY"), ("pitch", "accX")]:
        if term in out.get(axis, {}):
            out[axis][term] = -out[axis][term]
    out["frame"] = betaflight_frame(out.get("airframe", {}))
    return out


def load(path):
    """Read and normalize a coefficient JSON."""
    with open(path) as f:
        return normalize(json.load(f))


def arms(coeffs, overrides=None):
    """The physical arms {dFront, dBack, dLeft, dRight} of a canonical fit:
    the fitted ones, else `overrides` (a dict, entries may be None), else
    symmetric."""
    overrides = overrides or {}
    return {k: coeffs.get(ARM_AXIS[k], {}).get(
                k, overrides.get(k) if overrides.get(k) is not None else d)
            for k, d in DEFAULT_ARMS.items()}


def layout(coeffs):
    """(slot per firmware motor, spin code per firmware motor) of a
    canonical fit: the blob's layout block."""
    motors = coeffs["frame"]["motors"]
    return ([POSITIONS.index(m["position"]) for m in motors],
            [SPINS.index(m["spin"]) for m in motors])


def main():
    ap = argparse.ArgumentParser(description="Print a coefficient JSON in the canonical frame.")
    ap.add_argument("fit")
    ap.add_argument("--write", metavar="OUT", help="write it here instead")
    args = ap.parse_args()
    text = json.dumps(load(args.fit), indent=2) + "\n"
    if args.write:
        with open(args.write, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
