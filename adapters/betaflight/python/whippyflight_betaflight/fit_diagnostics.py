"""
Fit-quality diagnostics for fit_model.py, for judging whether test flights
excited the airframe enough and whether the model explains them.

`diagnostics()` returns a JSON-ready dict (what the Configurator's
WhippyFlight tab draws); `plot()` renders the same data as matplotlib PNGs.

Per axis (yaw, pitch, roll: angular acceleration in rot/s²; motors: volts):
  * R² of the pooled fit on its own rows, and held out: leave one flight
    out with two or more flights, else each quarter of the lone flight
    scored by a fit to the other three;
  * coefficients with relative standard errors;
  * measured vs predicted on a random sample of rows, and the residual
    histogram;
  * a time-series excerpt: the most heavily excited window of the longest
    flight, measured and predicted.
Per channel (rate axes and motors), the noise split: where the spectrum of
what the model leaves unexplained over every usable row (not only the excited
ones), on the shortest finite difference, stops falling (unmodeled dynamics)
and starts rising (sensor noise, differentiated): its smoothed minimum, Hz.
The rate axes' residuals are of the motor model's speeds rather than the
telemetry, so what rises is the gyro's noise and not the motor speeds' (the
yaw model differentiates them); the motors' is of the telemetry, whose noise
it measures.  Below the split the residual is the airframe's, above it the
sensor's; noise extraction and the turbulence band divide there.
Per rate axis, turbulence: that residual (of the whole plant the sims fly,
motors and body, driven by the logged voltages) from TURBULENCE_LOW_HZ to the
axis's split, as the one-pole spectrum with the same power and the same split about
the band's middle: its RMS (rot/s²) and corner (Hz), as the flick sim's
turbulence takes them.
Per flight: kept time, gating, the pooled fit's R² on that flight alone, and
the motor-pairing check.  `warnings` lists what looks wrong, as stable codes
with parameters, so a UI can phrase them.

    python -m whippyflight_betaflight.fit_diagnostics diagnostics.json OUT_DIR

re-plots a saved fit's diagnostics without refitting (needs matplotlib).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.signal import welch

from whippyflight_betaflight import fit_model as fm

UNITS = {"yaw": "rot/s²", "pitch": "rot/s²", "roll": "rot/s²", "motors": "V"}
MAX_POINTS = 1500
HIST_BINS = 41
EXCERPT_S = 6.0
HELDOUT_FOLDS = 4

# Below these the fit is suspect enough to flag.
MIN_R2 = {"yaw": 0.5, "pitch": 0.5, "roll": 0.5, "motors": 0.9}
MAX_HELDOUT_DROP = 0.1
MIN_TOTAL_KEPT_S = 30.0
MAX_REL_SE = 0.05

# Slower than this is a bias over one flick (a 0.3 s hold between its rise and
# release), the CG offset's and the pilot's to stand for, not turbulence.
TURBULENCE_LOW_HZ = 2.0
# The noise split is searched from here up to SPLIT_MAX of the log rate, on
# third-octave bands smoothed over SPLIT_SMOOTH of them.
SPLIT_MIN_HZ = 5.0
SPLIT_MAX = 0.25
SPLIT_SMOOTH = 3
# The residual's finite difference: two samples, its sinc null at the log rate.
SPLIT_WINDOW = 2
# Welch segments: as long as a run allows up to the longer, none from runs shorter than the shorter.
SEGMENT_S = (2.0, 4.0)


def _rows(flights, axis, arms, volts_map, drag):
    parts = [fm.design(f, axis, arms, volts_map, drag) for f in flights]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _sse_sst(y, pred):
    resid = y - pred
    return float(resid @ resid), float((y - y.mean()) @ (y - y.mean())) if len(y) else 0.0


def _residual_spectra(flights, axis, coeffs, arms, volts_map, drag):
    """
    The model's residual over every usable row on the SPLIT_WINDOW difference,
    per contiguous run (each motor's apart): [(freqs, sinc-corrected PSD, rows)].
    """
    out = []
    for f in flights:
        X, y, names, idx = fm.design(f, axis, arms, volts_map, drag, gated=False, window=SPLIT_WINDOW)
        resid = y - X @ np.array([coeffs[axis][n] for n in names])
        shortest, longest = (int(s * f.log_hz) for s in SEGMENT_S)
        for run in np.split(np.arange(len(idx)), np.flatnonzero(np.diff(idx) != 1) + 1):
            if len(run) < shortest:
                continue
            fr, psd = welch(resid[run], fs=f.log_hz, nperseg=min(longest, len(run)))
            out.append((fr, psd / np.sinc(fr * SPLIT_WINDOW / f.log_hz) ** 2, len(run)))
    return out


def _band_power(spectra, lo, hi):
    """Mean over rows of the power in [lo, hi)."""
    rows = sum(n for *_, n in spectra)
    return sum(psd[(fr >= lo) & (fr < hi)].sum() * (fr[1] - fr[0]) * n for fr, psd, n in spectra) / rows


def _split(spectra, log_hz):
    """The smoothed third-octave minimum of the residual's spectrum, Hz; None without one."""
    edges = SPLIT_MIN_HZ * 2.0 ** (np.arange(int(3 * np.log2(SPLIT_MAX * log_hz / SPLIT_MIN_HZ)) + 1) / 3)
    density = np.array([_band_power(spectra, a, b) / (b - a) for a, b in zip(edges, edges[1:])])
    if len(density) < SPLIT_SMOOTH or not (density > 0).all():
        return None
    smooth = np.exp(np.convolve(np.log(density), np.ones(SPLIT_SMOOTH) / SPLIT_SMOOTH, "valid"))
    k = int(np.argmin(smooth)) + SPLIT_SMOOTH // 2
    return float(np.sqrt(edges[k] * edges[k + 1]))


def modeled(flights, coeffs, volts_map):
    """`flights` with the motor model's speeds for the telemetry's."""
    return [fm.with_modeled_speeds(f, coeffs, volts_map) for f in flights]


def noise_split(flights, clean, coeffs, arms, volts_map, drag):
    """Per channel, the noise split (Hz), or None; `clean` is `modeled(flights, …)`."""
    log_hz = min(f.log_hz for f in flights)
    out = {}
    for axis in fm.AXES:
        spectra = _residual_spectra(flights if axis == "motors" else clean, axis, coeffs, arms, volts_map, drag)
        hz = _split(spectra, log_hz) if spectra else None
        out[axis] = round(hz, 1) if hz else None
    return out


def _one_pole_power(lo, hi, hz):
    """∫ 1/(1 + (f/hz)²) df over [lo, hi]."""
    return hz * (np.arctan(hi / hz) - np.arctan(lo / hz))


def turbulence(clean, coeffs, arms, volts_map, drag, split):
    """
    Per rate axis, {rms, hz} of the one-pole spectrum with the residual's power
    from TURBULENCE_LOW_HZ to the axis's `split` and its division about the
    band's geometric middle; None if too little data, no split, or a spectrum
    no one-pole has.
    """
    out = {}
    for axis in ("yaw", "pitch", "roll"):
        lo, hi = TURBULENCE_LOW_HZ, split.get(axis)
        spectra = _residual_spectra(clean, axis, coeffs, arms, volts_map, drag)
        if not spectra or not hi or hi <= 2 * lo:
            out[axis] = None
            continue
        mid = np.sqrt(lo * hi)
        low, high = _band_power(spectra, lo, mid), _band_power(spectra, mid, hi)
        # A one-pole spectrum's division runs from the flat (mid − lo)/(hi − mid) at
        # a high corner to the f⁻² (1/lo − 1/mid)/(1/mid − 1/hi) at a low one.
        division = lambda hz: _one_pole_power(lo, mid, hz) / _one_pole_power(mid, hi, hz) - low / high
        if not (low > 0 and high > 0) or division(1e-3) * division(1e4) >= 0:
            out[axis] = None
            continue
        hz = brentq(division, 1e-3, 1e4)
        s0 = (low + high) / _one_pole_power(lo, hi, hz)
        out[axis] = {"rms": round(float(np.sqrt(s0 * hz * np.pi / 2)), 5), "hz": round(float(hz), 3)}
    return out


def heldout_r2(flights, arms, volts_map, drag):
    """
    Per-axis held-out R², pooled over folds as 1 − ΣSSE/ΣSST, and the method.
    Folds partition the pooled fit's own rows (by flight, or a lone flight's
    rows into contiguous blocks of equal count) rather than re-gating pieces
    of a flight, so every fold is scored on rows the deployed fit was made
    from.  Blocks are equal in rows, not duration: excitation comes in bursts,
    and equal-duration folds can leave a fold trained on a handful of rows.
    """
    method = "leave one flight out" if len(flights) >= 2 else f"{HELDOUT_FOLDS} blocks of one flight"
    out = {}
    for axis in fm.AXES:
        X, y, fold = [], [], []
        for k, f in enumerate(flights):
            Xf, yf, _, idx = fm.design(f, axis, arms, volts_map, drag)
            X.append(Xf)
            y.append(yf)
            if len(flights) >= 2:
                fold.append(np.full(len(yf), k))
            else:
                fold.append(np.arange(len(yf)) * HELDOUT_FOLDS // max(len(yf), 1))
        X, y, fold = np.vstack(X), np.concatenate(y), np.concatenate(fold)
        sse = sst = 0.0
        for k in np.unique(fold):
            held = fold == k
            if held.all():
                continue
            a, b = _sse_sst(y[held], X[held] @ fm.ols(X[~held], y[~held])[0])
            sse, sst = sse + a, sst + b
        out[axis] = round(1.0 - sse / sst, 4) if sst else None
    return out, method


def _sample(n, k, rng):
    return np.sort(rng.choice(n, size=k, replace=False)) if n > k else np.arange(n)


def _excerpt(flight, axis, beta, arms, volts_map, drag):
    """Measured and predicted over the EXCERPT_S window with the most rows."""
    X, y, _, idx = fm.design(flight, axis, arms, volts_map, drag)
    if axis == "motors" and len(idx) > 1:
        # Rows are the four motors' in turn; the first motor's run ends where idx wraps.
        wraps = np.flatnonzero(np.diff(idx) < 0)
        end = wraps[0] + 1 if len(wraps) else len(idx)
        X, y, idx = X[:end], y[:end], idx[:end]
    if not len(idx):
        return None
    t = flight.t[idx]
    counts = np.searchsorted(t, t + EXCERPT_S) - np.arange(len(t))
    start = int(np.argmax(counts))
    stop = start + int(counts[start])
    pick = start + _sample(stop - start, MAX_POINTS, np.random.default_rng(0))
    return {"flight": flight.label,
            "window_s": EXCERPT_S,
            "t": np.round(t[pick] - t[start], 4).tolist(),
            "measured": np.round(y[pick], 5).tolist(),
            "predicted": np.round(X[pick] @ beta, 5).tolist()}


def diagnostics(flights, coeffs, volts_map):
    arms = fm.stored_arms(coeffs)
    drag = fm.drag_axes(coeffs)
    clean = modeled(flights, coeffs, volts_map)
    # The motor rows' voltages at the fit's delay (only the motor rows read them).
    lagged = [fm.with_voltage_delay(f, fm.voltage_delay(coeffs)) for f in flights]
    heldout, method = heldout_r2(lagged, arms, volts_map, drag)
    rng = np.random.default_rng(0)
    longest = max(lagged, key=lambda f: f.kept_s())
    axes = {}
    warnings = []
    split = noise_split(lagged, clean, coeffs, arms, volts_map, drag)
    for axis in fm.AXES:
        names = fm.design(flights[0], axis, arms, volts_map, drag)[2]
        beta = np.array([coeffs[axis][n] for n in names])
        X, y = _rows(lagged, axis, arms, volts_map, drag)
        pred = X @ beta
        resid = y - pred
        sse, sst = _sse_sst(y, pred)
        r2 = 1.0 - sse / sst if sst else 0.0
        pick = _sample(len(y), MAX_POINTS, rng)
        lim = float(np.percentile(np.abs(resid), 99.5)) if len(resid) else 1.0
        counts, edges = np.histogram(resid, bins=HIST_BINS, range=(-lim, lim))
        rel_se = coeffs.get("fit", {}).get(axis, {}).get("rel_se", {})
        axes[axis] = {
            "unit": UNITS[axis],
            "rows": int(len(y)),
            "r2": round(r2, 4),
            "heldout_r2": heldout[axis],
            "measured_std": round(float(y.std()), 5),
            "residual_std": round(float(resid.std()), 5),
            "coefficients": [{"name": n, "value": float(b), "rel_se": rel_se.get(n)}
                             for n, b in zip(names, beta)],
            "scatter": {"measured": np.round(y[pick], 5).tolist(),
                        "predicted": np.round(pred[pick], 5).tolist()},
            "residual_hist": {"edges": np.round(edges, 5).tolist(), "counts": counts.tolist()},
            "excerpt": _excerpt(longest, axis, beta, arms, volts_map, drag),
        }
        if r2 < MIN_R2[axis]:
            warnings.append({"code": "low_r2", "axis": axis, "value": round(r2, 3), "limit": MIN_R2[axis]})
        if heldout[axis] is not None and r2 - heldout[axis] > MAX_HELDOUT_DROP:
            warnings.append({"code": "overfit", "axis": axis, "r2": round(r2, 3),
                             "heldout_r2": heldout[axis]})
        for c in axes[axis]["coefficients"]:
            if c["rel_se"] is not None and c["rel_se"] > MAX_REL_SE:
                warnings.append({"code": "uncertain_coefficient", "axis": axis, "name": c["name"],
                                 "rel_se": c["rel_se"], "limit": MAX_REL_SE})

    per_flight = []
    for f in flights:
        pairing = fm.pairing_check(f)
        mismatched = [a for a, s in pairing.items() if max(s, key=s.get) != a]
        per_flight.append({"label": f.label,
                           "duration_s": round(float(f.t[-1] - f.t[0]), 1),
                           "kept_s": round(f.kept_s(), 1),
                           "log_hz": round(f.log_hz, 1),
                           "gating": f.gating,
                           "r2": {a: round(v, 4) for a, v in fm.evaluate(coeffs, f, arms, volts_map).items()},
                           "pairing": {a: {k: round(v, 4) for k, v in s.items()} for a, s in pairing.items()},
                           "pairing_mismatch": mismatched})
        if mismatched:
            warnings.append({"code": "pairing_mismatch", "flight": f.label, "axes": mismatched})
    kept = sum(f.kept_s() for f in flights)
    if kept < MIN_TOTAL_KEPT_S:
        warnings.append({"code": "little_data", "kept_s": round(kept, 1), "limit": MIN_TOTAL_KEPT_S})

    return {"version": 1,
            "heldout_method": method,
            "kept_s": round(kept, 1),
            "drag_axes": list(drag),
            "axes": axes,
            "noise_split_hz": split,
            "turbulence": turbulence(clean, coeffs, arms, volts_map, drag, split),
            "flights": per_flight,
            "warnings": warnings}


def plot(diag, out_dir):
    """One PNG per axis (excerpt, measured vs predicted, residuals) and flights.png."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for axis, a in diag["axes"].items():
        fig, (ax_t, ax_s, ax_h) = plt.subplots(1, 3, figsize=(15, 4.2),
                                               gridspec_kw={"width_ratios": [2.2, 1, 1]})
        title = f"{axis}: R² {a['r2']:.3f}"
        if a["heldout_r2"] is not None:
            title += f", held out {a['heldout_r2']:.3f}"
        fig.suptitle(f"{title} ({a['rows']} rows)")
        ex = a["excerpt"]
        if ex:
            ax_t.plot(ex["t"], ex["measured"], ".", ms=2, label="measured")
            ax_t.plot(ex["t"], ex["predicted"], ".", ms=2, label="predicted")
            ax_t.set_title(ex["flight"], fontsize=9)
            ax_t.set_xlim(0, ex.get("window_s", EXCERPT_S))
            ax_t.set_xlabel("s (gaps: samples left out of the fit)")
            ax_t.set_ylabel(a["unit"])
            ax_t.legend(markerscale=4)
        s = a["scatter"]
        ax_s.plot(s["measured"], s["predicted"], ".", ms=2, alpha=0.5)
        if s["measured"]:
            lo, hi = min(s["measured"]), max(s["measured"])
            ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax_s.set_xlabel(f"measured ({a['unit']})")
        ax_s.set_ylabel("predicted")
        h = a["residual_hist"]
        ax_h.stairs(h["counts"], h["edges"], fill=True)
        ax_h.set_xlabel(f"residual ({a['unit']})")
        ax_h.set_title(f"σ {a['residual_std']:.3g} of {a['measured_std']:.3g}", fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{axis}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    flights = diag["flights"]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(flights) + 3), 4))
    width = 0.8 / len(fm.AXES)
    for k, axis in enumerate(fm.AXES):
        ax.bar([i + k * width for i in range(len(flights))], [f["r2"][axis] for f in flights],
               width, label=axis)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(flights))])
    ax.set_xticklabels([f["label"] for f in flights], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² of the pooled fit")
    ax.set_ylim(min(0.0, *(f["r2"][a] for f in flights for a in fm.AXES)), 1.0)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "flights.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)
    return written


def main():
    import json
    import sys

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        diag = json.load(f)
    for path in plot(diag, sys.argv[2]):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
