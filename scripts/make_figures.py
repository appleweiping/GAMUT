"""Paper figures from results/*/h*_a*.json (static PDFs for the ICML template).

Palette: the validated reference categorical order (slots 1-3: blue, orange, aqua; adjacent CVD dE >= 9.2);
aqua is below 3:1 contrast on white, so every series also carries a distinct marker and a legend entry.
fig 1 (observation.pdf): FLARE's per-class 68% coverage against forecast origin (small multiples) and coverage of
correctly vs incorrectly classified objects.
fig 2 (ztf_classes.pdf): per-class coverage against origin for FLARE, two-step CP and GAMUT at 68% and 95%, plus
mean band length.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "font.family": "serif", "legend.frameon": False, "pdf.fonttype": 42})
CLS = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
LABEL = {"SN_Ia": "SN Ia", "SN_CC": "CC SN", "SLSN": "SLSN", "AGN": "AGN", "TDE": "TDE", "CV": "CV"}


def load(bench):
    R = {}
    for f in (ROOT / "results" / bench).glob("h*_a*.json"):
        h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
        R[(h, a)] = json.loads(f.read_text())
    return R


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def observation():
    R = load("ztf")
    hs = sorted(h for (h, a) in R if a == 0.32 and "FLARE" in R[(h, a)])
    if not hs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    # dot plot: classes on y, coverage on x, one marker per origin (sequential blue ramp: early = dark)
    ramp = ["#0d3a73", "#1f5faa", "#2a78d6", "#6fa3e6", "#a9c8f0"]
    for j, h in enumerate(hs):
        pc = R[(h, 0.32)]["FLARE"]["per_class"]
        ax.scatter([pc[c] for c in CLS], np.arange(len(CLS)) + (j - 2) * 0.1, s=9, color=ramp[j % 5], zorder=3,
                   label=f"day {h:g}", edgecolor="white", linewidth=0.3)
    ax.axvline(0.68, color=INK2, ls="--", lw=0.7)
    ax.set_yticks(np.arange(len(CLS))); ax.set_yticklabels([LABEL[c] for c in CLS]); ax.invert_yaxis()
    ax.set_xlabel("class coverage (68% band)"); style(ax); ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(fontsize=5, loc="lower left", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    ax = axes[1]
    mis = [R[(h, 0.32)]["FLARE"]["misclassified_coverage"] for h in hs]
    f = [R[(h, 0.32)]["FLARE"]["misclassified_fraction"] for h in hs]
    marg = [R[(h, 0.32)]["FLARE"]["marginal"] for h in hs]
    cor = [(m - ff * mm) / (1 - ff) for m, ff, mm in zip(marg, f, mis)]
    ax.plot(hs, cor, color=BLUE, marker="o", ms=3, lw=1.2, label="correct class")
    ax.plot(hs, mis, color=ORANGE, marker="s", ms=3, lw=1.2, label="misclassified")
    ax.axhline(0.68, color=INK2, ls="--", lw=0.7)
    ax.set_xlabel("forecast origin (days)"); ax.set_ylabel("coverage"); style(ax)
    ax.legend(fontsize=5, loc="center left", bbox_to_anchor=(0.02, 0.66))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "observation.pdf"); plt.close(fig)


def ztf_classes():
    R = load("ztf")
    methods = [("FLARE", "FLARE", ORANGE, "s"), ("TwoStepCP", "Two-step CP", AQUA, "^"), ("GAMUT", "GAMUT", BLUE, "o")]
    fig, axes = plt.subplots(2, 7, figsize=(6.9, 2.3), sharex=True)
    for r, a in enumerate((0.32, 0.05)):
        hs = sorted(h for (h, aa) in R if aa == a and all(m in R[(h, aa)] for m, *_ in methods))
        if not hs:
            continue
        for j, c in enumerate(CLS):
            ax = axes[r, j]
            for m, lab, col, mk in methods:
                ax.plot(hs, [R[(h, a)][m]["per_class"][c] for h in hs], color=col, marker=mk, ms=2.5, lw=1.0, label=lab)
            ax.axhline(1 - a, color=INK2, ls="--", lw=0.7); style(ax)
            ax.set_ylim(*((0.45, 0.9) if a > 0.1 else (0.8, 1.0)))
            if j > 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(LABEL[c], fontsize=7, color=INK)
            if j == 0:
                ax.set_ylabel(f"coverage ({int(round(100 * (1 - a)))}%)")
        ax = axes[r, 6]
        for m, lab, col, mk in methods:
            ax.plot(hs, [R[(h, a)][m]["width"] for h in hs], color=col, marker=mk, ms=2.5, lw=1.0, label=lab)
        style(ax)
        if r == 0:
            ax.set_title("length (mag)", fontsize=7, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("origin (d)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "ztf_classes.pdf"); plt.close(fig)


def day50():
    """In the format of the FLARE forecast paper's Fig. 2: per class, the fraction of day-50 measurements (future
    detections with 45 <= t <= 55 d) inside the 68% and 95% bands, by forecast origin, for FLARE and GAMUT, with 95%
    object-level bootstrap intervals (500 resamples)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from gamut import data
    hs = [5.0, 10.0, 15.0, 20.0, 30.0]; rng = np.random.default_rng(50)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.3), sharey=False)
    stats = {}
    for ax, a in zip(axes, (0.32, 0.05)):
        for j, (meth, col, mk) in enumerate((("FLARE", ORANGE, "s"), ("GAMUT", BLUE, "o"))):
            for i, c in enumerate(CLS):
                xs, ys, lo, hi = [], [], [], []
                for k, h in enumerate(hs):
                    t = data.load("ztf", h); P = np.load(ROOT / "results" / "ztf" / f"points_h{h:g}_a{a:g}.npz")
                    m = (t.y[t.obj] == CLS.index(c)) & (t.t >= 45) & (t.t <= 55)
                    if m.sum() < 5:
                        continue
                    cov = P[f"{meth}__cov"][m].astype(float); o = t.obj[m]
                    uo, inv = np.unique(o, return_inverse=True); s_ = np.bincount(inv, cov); n_ = np.bincount(inv)
                    d = rng.integers(0, len(uo), (500, len(uo)))
                    bs = s_[d].sum(1) / n_[d].sum(1)
                    xs.append(i + (k - 2) * 0.09 + (j - 0.5) * 0.04); ys.append(cov.mean())
                    lo.append(cov.mean() - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - cov.mean())
                    stats[(meth, a, c, h)] = float(cov.mean())
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, ms=2.4, lw=0.7, color=col, capsize=0,
                            label=("FLARE (deployed)" if meth == "FLARE" else "GAMUT") if i == 0 else None)
        ax.axhline(1 - a, color=INK2, ls="--", lw=0.7)
        ax.set_xticks(range(len(CLS))); ax.set_xticklabels([LABEL[c] for c in CLS])
        ax.set_title(f"nominal {int(round(100 * (1 - a)))}% band", fontsize=7, color=INK)
        ax.set_ylim((0.2, 1.0) if a > 0.1 else (0.6, 1.0)); style(ax)
    axes[0].set_ylabel("fraction of day-50 measurements\ninside the band")
    axes[1].legend(loc="lower left", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "day50.pdf"); plt.close(fig)
    (ROOT / "results" / "day50.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in stats.items()}, indent=1))


if __name__ == "__main__":
    observation(); ztf_classes(); day50()
