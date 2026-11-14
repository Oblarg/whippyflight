#!/usr/bin/env python3
"""
The test pipeline over the airframes in fixtures/airframes/airframes.json.

Each airframe is a fit (coefficient JSON), a sensor-noise fixture from the
same airframe's logs, and the gen_wfg arguments its blob is generated with.
One of them names a committed blob (fixtures/test_gains.wfg, which the Rust
tests embed and the firmware SITL tests fly); the others' blobs are
generated into target/airframes/.

    python3 tools/pipeline.py gen            regenerate every airframe's blob
    python3 tools/pipeline.py check          committed blob up to date with its fit
    python3 tools/pipeline.py test           cargo test, the whole workspace
    python3 tools/pipeline.py eval [--save]  SUMMARY evals per airframe, against
                                             fixtures/airframes/eval_baseline.json
    python3 tools/pipeline.py all            check, test, eval

Run with the interpreter the whippyflight package is installed in (see the
README); gen_wfg runs under it too.
"""

import argparse
import filecmp
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from sweep_qr import GEN_WFG, build_evals

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AIRFRAMES = os.path.join(REPO, "fixtures", "airframes")
MANIFEST = os.path.join(AIRFRAMES, "airframes.json")
BASELINE = os.path.join(AIRFRAMES, "eval_baseline.json")
GENERATED = os.path.join(REPO, "target", "airframes")


def airframes():
    with open(MANIFEST) as f:
        frames = json.load(f)
    for name, a in frames.items():
        a["name"] = name
        a["fit"] = os.path.join(AIRFRAMES, a["fit"])
        a["noise"] = os.path.join(AIRFRAMES, a["noise"])
        a["blob"] = os.path.normpath(os.path.join(AIRFRAMES, a["blob"])) if "blob" in a \
            else os.path.join(GENERATED, f"{name}.wfg")
    return frames


def committed(frames):
    (a,) = [a for a in frames.values() if not a["blob"].startswith(GENERATED)]
    return a


def generate(a, out, extra_args=()):
    r = subprocess.run(GEN_WFG + [a["fit"], out] + a["gen_args"] + list(extra_args),
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"gen_wfg failed for {a['name']}:\n{r.stderr}")


def cmd_gen(frames, _args):
    os.makedirs(GENERATED, exist_ok=True)
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda a: generate(a, a["blob"]), frames.values()))
    for a in frames.values():
        print(f"  {a['name']:12s} → {os.path.relpath(a['blob'], REPO)}")


def cmd_check(frames, _args):
    a = committed(frames)
    with tempfile.TemporaryDirectory() as tmp:
        fresh = os.path.join(tmp, "fresh.wfg")
        generate(a, fresh)
        if not filecmp.cmp(fresh, a["blob"], shallow=False):
            sys.exit(f"{os.path.relpath(a['blob'], REPO)} is stale: regenerate it from "
                     f"{os.path.relpath(a['fit'], REPO)} (pipeline.py gen)")
    print(f"  {os.path.relpath(a['blob'], REPO)} matches {os.path.relpath(a['fit'], REPO)}")


def cmd_test(_frames, _args):
    subprocess.run(["cargo", "test", "--no-fail-fast", "--release"], cwd=REPO, check=True)


def summaries(a, evals):
    out = {}
    for exe, test in evals:
        stdout = subprocess.run([exe, test, "--exact", "--ignored", "--nocapture"], check=True,
                                capture_output=True, text=True,
                                env={**os.environ, "WFG_PATH": a["blob"], "PLANT_JSON": a["fit"],
                                     "NOISE_PATH": a["noise"]}).stdout
        for line in stdout.splitlines():
            if line.startswith("SUMMARY"):
                out.update((k, float(v)) for k, v in (kv.split("=") for kv in line.split()[1:]))
    return out


def cmd_eval(frames, args):
    cmd_gen(frames, args)
    evals = build_evals()
    with ThreadPoolExecutor() as pool:
        results = dict(zip(frames, pool.map(lambda a: summaries(a, evals), frames.values())))
    baseline = {}
    if os.path.exists(BASELINE) and not args.save:
        with open(BASELINE) as f:
            baseline = json.load(f)
    names = list(results)
    metrics = list(dict.fromkeys(k for r in results.values() for k in r))
    print(f"\n{'':24s}" + "".join(f"{n:>24s}" for n in names))
    changed = 0
    for m in metrics:
        cells = []
        for n in names:
            v, b = results[n].get(m), baseline.get(n, {}).get(m)
            if v is None:
                cells.append(f"{'—':>24s}")
            elif b is None or b == v:
                cells.append(f"{v:24.5g}")
            else:
                changed += 1
                rel = f"{100 * (v - b) / abs(b):+.1f}%" if b else "new"
                cells.append(f"{f'{v:.5g} ({rel})':>24s}")
        print(f"{m:24s}" + "".join(cells))
    if args.save:
        with open(BASELINE, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"\nsaved {os.path.relpath(BASELINE, REPO)}")
    elif baseline:
        print(f"\n{changed} value(s) differ from {os.path.relpath(BASELINE, REPO)} (relative change in parentheses)")


def cmd_all(frames, args):
    for step in (cmd_check, cmd_test, cmd_eval):
        print(f"\n=== {step.__name__[4:]} ===", flush=True)
        step(frames, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "check", "test"):
        sub.add_parser(name)
    for name in ("eval", "all"):
        p = sub.add_parser(name)
        p.add_argument("--save", action="store_true", help="write the results as the new baseline")
    args = ap.parse_args()
    frames = airframes()
    {"gen": cmd_gen, "check": cmd_check, "test": cmd_test, "eval": cmd_eval,
     "all": cmd_all}[args.cmd](frames, args)


if __name__ == "__main__":
    main()
