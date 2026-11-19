#!/usr/bin/env python3
"""
Generate gain blobs over a grid of the two LQR weights that set the
controller's tradeoffs, for the Configurator to score and draw as a Pareto
front (see sweep_qr.py for what they trade).  σ_deriv trades disturbance
rejection against motor roughness and delay margin; σ_int trades attitude
hold against delay margin.  The integral's time constant τ_int is held at
the gen_wfg.py arguments'.

Everything is found on the loop at hover linearized as the sim flies it
(hover.py), which measures how fast a roll knock dies out, the extra gyro
delay the loop stays stable with, the attitude a CG shift costs and the
motor roughness the fit's recorded sensor noise causes (when it has one).

First σ_rate, unless --derived-sigma-rate gives it: the one whose loop
alone (no derivative or integral weight) recovers a knock twice as slowly
as the stiffest such loop, which the plant's own lags limit.  The front is
at --sigma-rate if given, else at the derived one.

Then the derivative filter's time constant τ_deriv, unless
--derived-tau-deriv gives it: the one whose front does best, by the
hypervolume its measures dominate (log scale, from the worst of each over
every τ_deriv tried).  It is tried on a log grid from one loop tick to
τ_int, then refined by pattern search.  The grid is at --tau-deriv if given,
else at the derived one.

Then the grid with that time constant.  σ_int runs from where a roll knock
dies out fastest without derivative weight (looser trades hold for margin
the loop has to spare) to where the loop goes unstable.  At each σ_int
level, σ_deriv runs from where it speeds a knock's recovery by a tenth to
where it is fastest: stiffer is worse on every measure the front draws.
Levels are midpoints of equal steps in ln σ.

  python -m whippyflight.front_wfg FIT.json OUT_DIR [--sigma-rate R] [--tau-deriv S]
                       [--deriv-levels N] [--int-levels N] [--jobs N]
                       [gen_wfg.py arguments, e.g. --dt 1.25e-4]

Writes OUT_DIR/<n>.wfg and OUT_DIR/front.json:
  {"dt": s, "sigma_rate": rot/s, "derived_sigma_rate": rot/s (at this front's τ_int, dt and
   grid), "tau_deriv": s,
   "derived_tau_deriv": s (at this front's σ_rate, τ_int, dt and grid), "tau_int": s,
   "resolution": N,
   "idle_rps": rev/s (0: none),
   "points": [{"sigma_deriv", "sigma_int", "grid": [deriv level, int level],
               "wfg": "<n>.wfg"}, ...]}
"""

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from whippyflight import hover

FRONT = "front.json"
SEARCHED = ("--sigma-deriv", "--sigma-int")
# Relative width the grid's ends are found to, and τ_deriv's search's.
TOLERANCE, SEARCH_TOLERANCE = 1e-3, 1e-2
# How much σ_deriv's loose end speeds a knock's recovery over no derivative weight.
DERIV_ONSET = 0.1
# How much slower σ_rate's loop alone recovers a knock than the stiffest rate loop can.
RATE_SLOWDOWN = 2.0
# τ_deriv's search: grid points, levels per weight of each candidate's front, and the step
# (ln τ) it refines down to.
TAU_GRID, TAU_FRONT_LEVELS, TAU_STEP = 5, 4, np.log(1.1)


def solve(argv):
    """gen_wfg on `argv`, quietly: (blob, its arguments with the defaults filled in)."""
    from whippyflight import gen_wfg
    args = gen_wfg.parse_args(argv)
    with contextlib.redirect_stderr(io.StringIO()):
        blob, _ = gen_wfg.generate(args)
    return blob, args


def inv(w):
    return 1.0 / w if w > 0 else np.inf


def best(cost, top, tolerance):
    """The weight in (0, `top`) where `cost` is least (golden section)."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = 0.0, top
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    ca, cb = cost(a), cost(b)
    while hi - lo > tolerance * hi:
        if ca <= cb:
            hi, b, cb = b, a, ca
            a = hi - ratio * (hi - lo)
            ca = cost(a)
        else:
            lo, a, ca = a, b, cb
            b = lo + ratio * (hi - lo)
            cb = cost(b)
    return 0.5 * (lo + hi)


def boundary(stable, tolerance):
    """The weight 1/σ where `stable(1/σ)` turns false, from stable at 0."""
    if not stable(0.0):
        return 0.0
    lo, hi = 0.0, 1.0
    while stable(hi):
        lo, hi = hi, 2.0 * hi
        if hi > 1e12:
            raise SystemExit("the loop never goes unstable: no range to derive")
    while hi - lo > tolerance * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(mid) else (lo, mid)
    return lo


def golden(cost, lo, hi, width):
    """The x in [`lo`, `hi`] where `cost` is least, to `width` (golden section)."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    ca, cb = cost(a), cost(b)
    while hi - lo > width:
        if ca <= cb:
            hi, b, cb = b, a, ca
            a = hi - ratio * (hi - lo)
            ca = cost(a)
        else:
            lo, a, ca = a, b, cb
            b = lo + ratio * (hi - lo)
            cb = cost(b)
    return 0.5 * (lo + hi)


def levels(loose, stiff, n):
    """Midpoints of n equal steps in ln σ from σ `loose` to `stiff`."""
    lo, hi = np.log(loose), np.log(stiff)
    return [float(np.exp(lo + (hi - lo) * (i + 0.5) / n)) for i in range(n)]


_loop = None


def _init(coefficients_path, gen_argv):
    global _loop
    _loop = hover.HoverLoop(coefficients_path, gen_argv)


LOOSEN = np.log(2.0)
LOOSEST = np.log(1e12)


def _int_range(taus, tolerance=TOLERANCE):
    """(loose, stiff) σ_int without derivative weight: where a knock dies out fastest, and
    where the loop goes unstable; None if it is unstable without integral weight too."""
    edge = boundary(lambda v: _loop.stable((np.inf, inv(v), *taus)), tolerance)
    if edge <= 0.0:
        return None
    start = np.log(1.0 / edge)

    def kick(x):
        return _loop.kick_iae((np.inf, np.exp(x), *taus))

    x, at = start, kick(start)
    while (looser := kick(x + LOOSEN)) < at:
        x, at = x + LOOSEN, looser
        if x - start > LOOSEST:
            raise SystemExit("integral weight never stops speeding a knock's recovery")
    return float(np.exp(golden(kick, max(start, x - LOOSEN), x + LOOSEN, tolerance))), 1.0 / edge


def _deriv_range(taus, sigma_int, tolerance=TOLERANCE):
    """(loose, stiff) σ_deriv at `sigma_int`: where it speeds a knock's recovery by
    DERIV_ONSET over none, and where knocks die out fastest (stiffer is worse on every
    measure the front draws); None if it never helps."""
    edge = boundary(lambda u: _loop.stable((inv(u), sigma_int, *taus)), tolerance)
    if edge <= 0.0:
        return None
    stiff = 1.0 / best(lambda u: _loop.kick_iae((inv(u), sigma_int, *taus)), edge, tolerance)
    onset = (1.0 - DERIV_ONSET) * _loop.kick_iae((np.inf, sigma_int, *taus))

    def kick(x):
        return _loop.kick_iae((np.exp(x), sigma_int, *taus))

    lo = np.log(stiff)
    if not kick(lo) < onset:
        return None
    hi = lo + LOOSEN
    while kick(hi) < onset:
        lo, hi = hi, hi + LOOSEN
        if hi - np.log(stiff) > LOOSEST:
            raise SystemExit("derivative weight never stops speeding a knock's recovery")
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if kick(mid) < onset else (lo, mid)
    return float(np.exp(hi)), stiff


def sigma_rate(coefficients_path, gen_argv, tolerance=TOLERANCE):
    """σ_rate whose loop alone (no derivative or integral weight) recovers a knock
    RATE_SLOWDOWN times slower than the stiffest such loop."""
    def kick(x):
        loop = hover.HoverLoop(coefficients_path, [*gen_argv, "--sigma-rate", repr(float(np.exp(x)))])
        # Without derivative weight τ_deriv is moot; one tick makes the filter degenerate.
        return loop.kick_iae((np.inf, np.inf, loop.args.tau_int, loop.args.tau_int))

    x, at = 0.0, kick(0.0)
    while (stiffer := kick(x - LOOSEN)) < (1.0 - tolerance) * at:
        x, at = x - LOOSEN, stiffer
        if -x > LOOSEST:
            raise SystemExit("rate weight never stops speeding a knock's recovery")
    floor = kick(golden(kick, x - LOOSEN, x + LOOSEN, tolerance))
    target = RATE_SLOWDOWN * floor
    lo, hi = x, x + LOOSEN
    while kick(hi) < target:
        lo, hi = hi, hi + LOOSEN
        if hi - x > LOOSEST:
            raise SystemExit("rate weight never slows a knock's recovery")
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if kick(mid) < target else (lo, mid)
    return float(np.exp(hi))


def grid(ints, deriv_ranges, deriv_levels):
    """[(deriv level, int level, σ_deriv, σ_int)] over σ_ints `ints` and each one's
    σ_deriv range."""
    return [(i, j, sd, si)
            for j, (si, span) in enumerate(zip(ints, deriv_ranges)) if span is not None
            for i, sd in enumerate(levels(*span, deriv_levels))]


def taus_at(log_tau_deriv):
    """(τ_deriv, τ_int) for ln τ_deriv/dt, τ_int the gen_wfg arguments'."""
    return float(_loop.dt * np.exp(log_tau_deriv)), float(_loop.args.tau_int)


def _coarse_ints(log_tau_deriv):
    int_range = _int_range(taus_at(log_tau_deriv), SEARCH_TOLERANCE)
    return levels(*int_range, TAU_FRONT_LEVELS) if int_range else []


def _coarse_span(log_tau_deriv, sigma_int):
    return _deriv_range(taus_at(log_tau_deriv), sigma_int, SEARCH_TOLERANCE)


def _coarse_measures(log_tau_deriv, sigma_deriv, sigma_int):
    """A coarse front point's measures, log scale, each to minimize; None if it has none."""
    m = _loop.measures((sigma_deriv, sigma_int, *taus_at(log_tau_deriv)), SEARCH_TOLERANCE)
    if m is None or not m[1] > 0:
        return None
    kick, margin, trim, rough = m
    return [np.log(kick), -np.log(margin), np.log(trim)] + ([] if rough is None else [np.log(rough)])


def fronts(pool, candidates):
    """{ln τ_deriv/dt: its coarse front's measures} for `candidates`, each stage of every
    candidate's front at once on `pool`."""
    ints = dict(zip(candidates, pool.map(_coarse_ints, candidates)))
    at = [(c, si) for c in candidates for si in ints[c]]
    spans = dict(zip(at, pool.map(_coarse_span, *zip(*at)))) if at else {}
    points = [(c, sd, si) for c in candidates
              for _, _, sd, si in grid(ints[c], [spans[c, si] for si in ints[c]], TAU_FRONT_LEVELS)]
    measured = list(pool.map(_coarse_measures, *zip(*points))) if points else []
    return {c: np.array([m for (p, *_), m in zip(points, measured) if p == c and m is not None])
            for c in candidates}


def hypervolume(points, ref):
    """Volume `points` dominate (minimizing) below `ref`."""
    points = points[np.all(points < ref, axis=1)]
    if not len(points):
        return 0.0
    if points.shape[1] == 1:
        return float(ref[0] - points[:, 0].min())
    points = points[np.argsort(points[:, -1])]
    total = 0.0
    for k in range(len(points)):
        upper = points[k + 1, -1] if k + 1 < len(points) else ref[-1]
        if upper > points[k, -1]:
            total += (upper - points[k, -1]) * hypervolume(points[:k + 1, :-1], ref[:-1])
    return total


def tau_deriv(pool, top):
    """The ln τ_deriv/dt in [0, `top`] whose front's hypervolume is largest."""
    tried = {}

    def score(candidates):
        tried.update(fronts(pool, [c for c in candidates if c not in tried]))
        scored = [f for f in tried.values() if len(f)]
        if not scored:
            sys.exit("the loop is unstable at hover at every τ_deriv tried; loosen σ_rate")
        ref = np.vstack(scored).max(axis=0)
        return max(tried, key=lambda c: hypervolume(tried[c], ref) if len(tried[c]) else -1.0)

    at = score([round(top * (i + 0.5) / TAU_GRID, 9) for i in range(TAU_GRID)])
    step = top / TAU_GRID
    while step > TAU_STEP:
        step /= 2.0
        while True:
            moved = score([round(float(np.clip(at + s, 0.0, top)), 9) for s in (-step, step)])
            print(f"  τ_deriv {_loop.dt * np.exp(moved) * 1e3:.3g} ms ({len(tried)} tried)",
                  flush=True)
            if moved == at:
                break
            at = moved
    return at


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coefficients", help="the fit's coefficients.json")
    ap.add_argument("out", help="directory for the blobs and front.json")
    ap.add_argument("--sigma-rate", type=float, default=None,
                    help="rate weight, rot/s (default: derived)")
    ap.add_argument("--derived-sigma-rate", type=float, default=None,
                    help="with --sigma-rate: this front's σ_rate derived earlier, rot/s, to record "
                         "instead of deriving it again")
    ap.add_argument("--tau-deriv", type=float, default=None,
                    help="derivative filter time constant, s (default: derived)")
    ap.add_argument("--derived-tau-deriv", type=float, default=None,
                    help="with --tau-deriv: this front's τ_deriv derived earlier, s, to record "
                         "instead of deriving it again")
    ap.add_argument("--deriv-levels", type=int, default=6)
    ap.add_argument("--int-levels", type=int, default=7)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args, gen_args = ap.parse_known_args()
    if any(a in gen_args for a in SEARCHED):
        ap.error("the searched parameters can't be passed through to gen_wfg.py")
    if args.deriv_levels < 1 or args.int_levels < 1:
        ap.error("each weight needs at least one level")
    for given in (args.tau_deriv, args.derived_tau_deriv):
        if given is not None and not given > 0:
            ap.error("τ_deriv must be positive")
    if args.derived_tau_deriv is not None and args.tau_deriv is None:
        ap.error("--derived-tau-deriv needs --tau-deriv")
    for given in (args.sigma_rate, args.derived_sigma_rate):
        if given is not None and not given > 0:
            ap.error("σ_rate must be positive")
    if args.derived_sigma_rate is not None and args.sigma_rate is None:
        ap.error("--derived-sigma-rate needs --sigma-rate")

    os.makedirs(args.out, exist_ok=True)
    derived_rate = args.derived_sigma_rate
    if derived_rate is None:
        print("Finding the rate weight...", flush=True)
        derived_rate = sigma_rate(args.coefficients, gen_args)
        print(f"  σ_rate {derived_rate:.3g} rot/s", flush=True)
    rate = derived_rate if args.sigma_rate is None else args.sigma_rate
    gen_args = [*gen_args, "--sigma-rate", repr(rate)]
    _init(args.coefficients, gen_args)
    dt = _loop.dt
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(args.coefficients, gen_args)) as pool:
        derived = args.derived_tau_deriv
        if derived is None:
            print(f"Finding the derivative time constant (τ_int {_loop.args.tau_int:.3g} s)...",
                  flush=True)
            if _loop.args.tau_int <= dt:
                sys.exit("τ_int must be longer than a loop tick")
            derived = taus_at(tau_deriv(pool, np.log(_loop.args.tau_int / dt)))[0]
            print(f"  τ_deriv {derived * 1e3:.3g} ms", flush=True)
        taus = (derived if args.tau_deriv is None else args.tau_deriv, float(_loop.args.tau_int))
        print("Finding the weights' ranges at hover...", flush=True)
        int_range = _int_range(taus)
        if int_range is None:
            sys.exit("the loop is unstable at hover even without integral or derivative weight; "
                     "loosen σ_rate")
        ints = levels(*int_range, args.int_levels)
        spans = list(pool.map(_deriv_range, [taus] * len(ints), ints))
    print(f"  σ_int from {int_range[0]:.4g} to {int_range[1]:.4g}; σ_deriv from "
          + ", ".join(f"{s[0]:.4g} to {s[1]:.4g}" if s else "none" for s in spans), flush=True)
    points_grid = grid(ints, spans, args.deriv_levels)

    print(f"Solving gains at {len(points_grid)} weightings on {args.jobs} processes...", flush=True)
    tau_args = ["--tau-deriv", repr(taus[0])]
    points, gen = [None] * len(points_grid), None
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(solve, [args.coefficients, os.devnull, *gen_args, *tau_args,
                                "--sigma-deriv", repr(deriv), "--sigma-int", repr(integ)]): n
            for n, (_, _, deriv, integ) in enumerate(points_grid)
        }
        for done, future in enumerate(as_completed(futures), 1):
            n = futures[future]
            i, j, deriv, integ = points_grid[n]
            try:
                blob, gen = future.result()
            except (Exception, SystemExit) as e:
                print(f"  [{done}/{len(points_grid)}] σ_deriv {deriv:.4g}, σ_int {integ:.4g}: "
                      f"failed: {e}", file=sys.stderr, flush=True)
                continue
            name = f"{n}.wfg"
            with open(os.path.join(args.out, name), "wb") as f:
                f.write(blob)
            points[n] = {"sigma_deriv": deriv, "sigma_int": integ, "grid": [i, j], "wfg": name}
            print(f"  [{done}/{len(points_grid)}] σ_deriv {deriv:.4g}, σ_int {integ:.4g}", flush=True)
    points = [p for p in points if p is not None]
    if not points:
        sys.exit("no gains could be computed")
    with open(os.path.join(args.out, FRONT), "w") as f:
        json.dump({"dt": gen.dt, "sigma_rate": gen.sigma_rate, "derived_sigma_rate": derived_rate,
                   "tau_deriv": taus[0],
                   "derived_tau_deriv": derived, "tau_int": taus[1],
                   "resolution": gen.resolution, "idle_rps": gen.idle_rps,
                   "points": points}, f, indent=1)
    print(f"Wrote {len(points)} blobs and {FRONT} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
