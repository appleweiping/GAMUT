"""Hyperparameter analysis on validation data only (nested cross-conformal).

For every outer fold k, the four remaining folds form the development pool; within it, each inner fold j is
calibrated on the other three and evaluated as validation data. The outer test fold k is never used. Validation
predictions of all (k, j) pairs are pooled and scored with gamut.metrics.evaluate.
Output: results/<bench>/validation_h<h>_a<alpha>.json (one entry per variant).
Run: python scripts/validate.py --bench ztf --h 5 --alpha 0.32 --variants '{"default": {}, "min_bin=50": {"min_bin": 50}}'
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
from gamut import data, methods, metrics  # noqa: E402

GRID = {                                        # variations around the final configuration (GAMUT v8)
    "default": {},
    "min_bin=50": {"min_bin": 50}, "min_bin=500": {"min_bin": 500}, "min_bin=inf (one law per class)": {"min_bin": 10 ** 9},
    "bandwidth x1 (Silverman)": {"bw_mult": 1.0}, "bandwidth x4": {"bw_mult": 4.0},
    "prune=1e-4": {"prune": 1e-4}, "prune=1e-2": {"prune": 1e-2},
    "dual iterations=5": {"n_iter": 5}, "dual iterations=20": {"n_iter": 20},
    "no finite-sample correction": {"finite": False}, "point-level unit": {"unit": "point"},
    "raw posterior (no temperature)": {"posterior": "classifier"}, "fixed lead bins (no merging)": {"cells": "bins"},
}


def nested(task, fn, alpha, kw):
    pt_fold = task.fold[task.obj]
    rows, cov, lo, hi, ln = [], [], [], [], []
    for k in np.unique(task.fold):
        for j in np.unique(task.fold):
            if j == k:
                continue
            test = np.flatnonzero(pt_fold == j)
            cal = np.flatnonzero((pt_fold != j) & (pt_fold != k))
            c, l, h, L = fn(task, cal, test, alpha, seed=100 * int(k) + int(j), **kw)
            rows.append(test); cov.append(c); lo.append(l); hi.append(h); ln.append(L)
    rows, cov, lo, hi, ln = map(np.concatenate, (rows, cov, lo, hi, ln))
    ok = np.isfinite(lo) & np.isfinite(hi)
    return metrics.evaluate(task, rows[ok], cov[ok], lo[ok], hi[ok], ln[ok], alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.32, 0.05])
    ap.add_argument("--variants", default=None, help="JSON dict name -> GAMUT kwargs (default: GRID)")
    a = ap.parse_args()
    task = data.load(a.bench, a.h)
    variants = json.loads(a.variants) if a.variants else GRID
    for alpha in a.alpha:
        path = ROOT / "results" / a.bench / f"validation_h{a.h:g}_a{alpha:g}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = json.loads(path.read_text()) if path.exists() else {}
        for name, kw in variants.items():
            t0 = time.time()
            res = nested(task, methods.gamut, alpha, kw)
            res["seconds"] = round(time.time() - t0, 1); res["kwargs"] = kw
            out[name] = res
            path.write_text(json.dumps(out, indent=1))
            print(f"{a.bench} h={a.h:g} a={alpha}: {name:34s} worst {res['worst_class']:.3f} ccg {res['ccg']:.3f} "
                  f"width {res['width']:.3f} cls-IS {res['class_avg_interval_score']:.3f} ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
