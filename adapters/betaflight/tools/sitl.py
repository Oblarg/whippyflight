#!/usr/bin/env python3
"""
Rebuild the Betaflight SITL firmware with WhippyFlight and fly an airframe
of this checkout's fixtures through it (sitl_test.py, and optionally
sitl_rc_phase.py): flick plots with and without feedback in
target/sitl/<airframe>/.

    python3 adapters/betaflight/tools/sitl.py [--airframe NAME] [--rc-phase]

Run with the interpreter the whippyflight packages are installed in.  The
Betaflight checkout is $BETAFLIGHT_DIR, else one next to this checkout.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
from pipeline import GENERATED, airframes, committed, generate  # noqa: E402
from sitl_test import BETAFLIGHT_DIR, WHIPPYFLIGHT_DIR  # noqa: E402

OUT = os.path.join(WHIPPYFLIGHT_DIR, "target")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--airframe", help="fly this airframe's blob (its loop rate must be one SITL "
                                       "supports); default: the committed blob")
    ap.add_argument("--rc-phase", action="store_true", help="also run sitl_rc_phase.py")
    args = ap.parse_args()
    frames = airframes()
    a = frames[args.airframe] if args.airframe else committed(frames)
    os.makedirs(GENERATED, exist_ok=True)
    if a["blob"].startswith(GENERATED):
        generate(a, a["blob"])
    no_feedback = os.path.join(GENERATED, f"{a['name']}_no_feedback.wfg")
    generate(a, no_feedback, ["--feedback-scale", "0"])
    subprocess.run(["make", "TARGET=SITL", "OPTIONS=USE_WHIPPYFLIGHT", f"-j{os.cpu_count()}"],
                   cwd=BETAFLIGHT_DIR, check=True, stdout=subprocess.DEVNULL)
    env = {**os.environ, "WFG_PATH": a["blob"], "PLANT_JSON": a["fit"],
           "WFG_NO_FEEDBACK_PATH": no_feedback, "SITL_OUT": os.path.join(OUT, "sitl", a["name"])}
    subprocess.run([sys.executable, os.path.join(HERE, "sitl_test.py")], env=env, check=True)
    if args.rc_phase:
        subprocess.run([sys.executable, os.path.join(HERE, "sitl_rc_phase.py")], env=env, check=True)


if __name__ == "__main__":
    main()
