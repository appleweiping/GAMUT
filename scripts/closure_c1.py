"""C1 (hypothesis-conditional forecaster) trained on FLARE's `train` split only, for the closure-probability study.

The cross-fitted C1 of the main experiments sees test-split objects in training, which would let test information
reach the Venn-Abers calibrator fitted on `val` and would give C1 more training data than FLARE's forecaster. Here the
three LightGBM models (median, q16, q84) with the one-hot class appended are fitted on `train` objects (85/15 split
for early stopping, the recipe of scripts/crossfit.py) and queried under every class hypothesis on the future points
of `val` and `test` objects.
Output: data/ztf/closure_c1_h{h}.npz with F-row indices and (n, K) predictions.
Run: python scripts/closure_c1.py --h 5 [--threads 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from benchmarks import registry  # noqa: E402
from crossfit import FORECAST_PARAMS, QUANTILES, cap_rows, fit_lgb, long_frame  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--h", type=float, required=True); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--pooled", action="store_true", help="fit the pooled forecaster (no class input) instead, as a control")
    a = ap.parse_args()
    B = registry.load("ztf", a.h)
    meta = pd.read_parquet(ROOT / "data" / "ztf" / "meta.parquet")
    y = meta.y.to_numpy(); split = meta.split.to_numpy(); K = len(B.classes)
    F = B.F; Fobj = F.obj.to_numpy()
    tr = np.flatnonzero(split == "train")
    fit_obj, es_obj = train_test_split(tr, test_size=.15, stratify=y[tr], random_state=0)
    r_fit = cap_rows(np.flatnonzero(np.isin(Fobj, fit_obj)), Fobj, a.cap, 11)
    r_es = cap_rows(np.flatnonzero(np.isin(Fobj, es_obj)), Fobj, a.cap, 12)
    r_pr = np.flatnonzero(np.isin(Fobj, np.flatnonzero(np.isin(split, ["val", "test"]))))
    Xfit, Xes, Xpr = (long_frame(B.X, F, rr, a.h) for rr in (r_fit, r_es, r_pr))

    def with_class(D, cls):
        D = D.copy()
        for c in range(K):
            D[f"hyp_{c}"] = (cls == c).astype(float)
        return D
    params = dict(FORECAST_PARAMS, learning_rate=a.lr)
    models = {"mu": params, "q16": dict(params, objective="quantile", alpha=QUANTILES["q16"]),
              "q84": dict(params, objective="quantile", alpha=QUANTILES["q84"])}
    yfit, yes = F.dmag.to_numpy()[r_fit], F.dmag.to_numpy()[r_es]
    out = dict(rows=r_pr)
    if a.pooled:
        for name, p in models.items():
            m = fit_lgb(p, Xfit, yfit, Xes, yes, a.threads)
            out[f"pool_{name}"] = m.predict(Xpr, num_threads=a.threads)
            print(f"h={a.h:g} pooled {name}: {m.best_iteration} iterations", flush=True)
        np.savez_compressed(ROOT / "data" / "ztf" / f"closure_pool_h{a.h:g}.npz", **out)
        return
    Hfit, Hes = with_class(Xfit, y[Fobj[r_fit]]), with_class(Xes, y[Fobj[r_es]])
    for name, p in models.items():
        m = fit_lgb(p, Hfit, yfit, Hes, yes, a.threads)
        out[f"hcf_{name}"] = np.stack([m.predict(with_class(Xpr, np.full(len(Xpr), c)), num_threads=a.threads) for c in range(K)], 1)
        print(f"h={a.h:g} {name}: {m.best_iteration} iterations", flush=True)
    np.savez_compressed(ROOT / "data" / "ztf" / f"closure_c1_h{a.h:g}.npz", **out)
    print(f"h={a.h:g}: train objects {len(tr)}, predicted rows {len(r_pr)}", flush=True)


if __name__ == "__main__":
    main()
