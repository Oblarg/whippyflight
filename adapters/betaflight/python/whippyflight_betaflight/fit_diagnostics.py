"""
Plots of a fit's diagnostics (diagnostics.json, which whippyflight-fit writes
next to the coefficients): per axis (yaw, pitch, roll: angular acceleration in
rot/s²; motors: volts) a time-series excerpt of measured and predicted, the
measured-against-predicted scatter and the residual histogram; and per flight, the
pooled fit's R² on it.

    python -m whippyflight_betaflight.fit_diagnostics diagnostics.json OUT_DIR

Needs matplotlib.
"""

AXES = ("yaw", "pitch", "roll", "motors")
EXCERPT_S = 6.0


def plot(diag, out_dir):
    """One PNG per axis (excerpt, measured against predicted, residuals) and flights.png."""
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
        ax_s.plot(s["predicted"], s["measured"], ".", ms=2, alpha=0.5)
        if s["predicted"]:
            lo, hi = min(s["predicted"]), max(s["predicted"])
            ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax_s.set_xlabel(f"predicted ({a['unit']})")
        ax_s.set_ylabel("measured")
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
    width = 0.8 / len(AXES)
    for k, axis in enumerate(AXES):
        ax.bar([i + k * width for i in range(len(flights))], [f["r2"][axis] for f in flights],
               width, label=axis)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(flights))])
    ax.set_xticklabels([f["label"] for f in flights], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² of the pooled fit")
    ax.set_ylim(min(0.0, *(f["r2"][a] for f in flights for a in AXES)), 1.0)
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
