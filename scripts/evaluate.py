"""Leave-one-fold-out evaluation of every method on one benchmark and forecast origin.

For each outer fold k, a method is calibrated on the out-of-fold outputs of the other folds' objects and evaluated on
fold k's future points (the cross-conformal protocol of FLARE's forecast bands, same folds). Predictions of all folds
are pooled and scored (gamut.metrics.evaluate); per-point outcomes are saved for paired bootstraps and figures.
Outputs: results/<bench>/h<h>_a<alpha>.json and results/<bench>/points_h<h>_a<alpha>.npz.
Run: python scripts/evaluate.py --bench ztf --h 5 [--alpha 0.32 0.05] [--methods ...] [--ablations]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamut import data, latent_baselines, methods, metrics  # noqa: E402

from gamut import variants  # noqa: E402
ALL = {**methods.METHODS, **latent_baselines.LATENT_BASELINES, **variants.VARIANTS}


def run(task, name, fn, alpha, kw):
    n = len(task.yt)
    cov = np.zeros(n, bool); lo = np.full(n, np.nan); hi = np.full(n, np.nan); ln = np.full(n, np.nan)
    pt_fold = task.fold[task.obj]
    diags = []
    for k in np.unique(task.fold):
        test = np.flatnonzero(pt_fold == k); cal = np.flatnonzero(pt_fold != k)
        c, l, h, L = fn(task, cal, test, alpha, seed=1000 + int(k), **kw)
        cov[test], lo[test], hi[test], ln[test] = c, l, h, L
        if fn is methods.gamut:
            diags.append(json.loads(json.dumps(methods.DIAG)))
    rows = np.arange(n)
    # an empty band (no grid value inside) is a miss of zero length; its hull degenerates to the pooled forecast
    empty = ~(np.isfinite(lo) & np.isfinite(hi))
    cov = cov & ~empty
    lo = np.where(empty, task.mu, lo); hi = np.where(empty, task.mu, hi); ln = np.where(empty, 0.0, ln)
    res = metrics.evaluate(task, rows, cov, lo, hi, ln, alpha)
    res["empty_fraction"] = float(empty.mean())
    if diags:
        res["diag"] = diags
    return res, dict(cov=cov, lo=lo, hi=hi, length=ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.32, 0.05])
    ap.add_argument("--methods", nargs="*", default=None); ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--names", nargs="*", default=None, help="exact method or ablation names to (re)compute")
    a = ap.parse_args()
    task = data.load(a.bench, a.h)
    todo = {k: (v, {}) for k, v in ALL.items() if a.methods is None or k in a.methods}
    if a.ablations:
        todo.update({k: (methods.gamut, kw) for k, kw in methods.ABLATIONS.items()})
    if a.names is not None:
        pool = {**{k: (v, {}) for k, v in ALL.items()}, **{k: (methods.gamut, kw) for k, kw in methods.ABLATIONS.items()}}
        todo = {k: pool[k] for k in a.names}
    out_dir = ROOT / "results" / a.bench; out_dir.mkdir(parents=True, exist_ok=True)
    for alpha in a.alpha:
        path = out_dir / f"h{a.h:g}_a{alpha:g}.json"
        prev = json.loads(path.read_text()) if path.exists() else {}
        pts_path = out_dir / f"points_h{a.h:g}_a{alpha:g}.npz"
        pts = dict(np.load(pts_path)) if pts_path.exists() else {}
        pts.pop("obj", None); pts.pop("y", None)
        for name, (fn, kw) in todo.items():
            t0 = time.time()
            res, arr = run(task, name, fn, alpha, kw)
            res["seconds"] = round(time.time() - t0, 1)
            if fn in (methods.gamut,) or fn in variants.VARIANTS.values():
                res["version"] = methods.GAMUT_VERSION
            elif name in ("KandinskyCP", "MDCP"):
                res["version"] = methods.GRID_VERSION
            prev[name] = res
            key = name.replace(" ", "_").replace("/", "_")
            for f, v in arr.items():
                pts[f"{key}__{f}"] = v
            print(f"{a.bench} h={a.h:g} alpha={alpha}: {name:34s} marg {res['marginal']:.3f} worst {res['worst_class']:.3f} "
                  f"ccg {res['ccg']:.3f} width {res['width']:.3f} IS {res['interval_score']:.3f} "
                  f"cls-IS {res['class_avg_interval_score']:.3f} miscls {res['misclassified_coverage']:.3f} ({res['seconds']}s)", flush=True)
            print("      per class " + " ".join(f"{c}:{v:.2f}" for c, v in res["per_class"].items()), flush=True)
        prev["_meta"] = dict(bench=a.bench, h=a.h, alpha=alpha, classes=task.classes, n_points=int(len(task.yt)),
                             n_objects=int(len(np.unique(task.obj))))
        path.write_text(json.dumps(prev, indent=1))
        np.savez_compressed(pts_path, obj=task.obj, y=task.y[task.obj], **pts)


if __name__ == "__main__":
    main()
