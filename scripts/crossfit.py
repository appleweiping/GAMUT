"""Cross-fitted classifier and forecaster outputs for one benchmark and forecast origin.

For each outer fold k (the benchmark's folds), models are fitted on the other folds (an 85/15 object split for
early stopping) and predict the held-out fold, so every object receives out-of-fold (OOF) outputs:
  * classifier: class probabilities P (n, K) and the hierarchy/argmax label;
  * base forecaster (FLARE recipe): L1 point forecast mu and quantile forecasts q16, q84, q025, q975 of the
    target at each future point (long format: prefix features + time + band);
  * hypothesis-conditional forecaster (GAMUT C1): the same models with the one-hot TRUE class appended in
    training; at prediction every class hypothesis c is queried, giving mu_c, q16_c, q84_c for all c.
Outputs: data/<bench>/oof_h{h}.npz. Resumable per fold (data/<bench>/parts/).
Training uses learning rate 0.05 (FLARE: 0.02) and at most 24 future points per object (test rows are all kept).
Run: python scripts/crossfit.py --bench ztf --h 5 [--threads 4] [--lr 0.05] [--cap 24]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks import registry  # noqa: E402

FORECAST_PARAMS = dict(objective="regression_l1", learning_rate=0.02, num_leaves=127, min_child_samples=100,
                       feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.0, verbosity=-1, seed=42)
QUANTILES = {"q16": 0.16, "q84": 0.84, "q025": 0.025, "q975": 0.975}


def long_frame(X, F, idx_rows, horizon):
    """Rows of the long-format design for future points idx_rows of table F."""
    f = F.iloc[idx_rows]
    D = X.iloc[f.obj.to_numpy()].reset_index(drop=True)
    D["fc_day"] = f.t.to_numpy(); D["fc_band"] = f.band.to_numpy().astype(float); D["fc_lead"] = f.lead.to_numpy()
    return D


def cap_rows(rows, Fobj, cap, seed):
    """At most `cap` training points per object (seeded), so that densely observed objects do not dominate."""
    if cap <= 0:
        return rows
    rng = np.random.default_rng(seed)
    keep = []
    obj = Fobj[rows]
    order = np.argsort(obj, kind="stable"); rows, obj = rows[order], obj[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obj)) + 1, len(obj)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        r = rows[s0:s1]
        keep.append(r if len(r) <= cap else np.sort(rng.choice(r, cap, replace=False)))
    return np.concatenate(keep)


def fit_lgb(params, Xtr, ytr, Xes, yes, threads, rounds=4000, patience=150):
    p = dict(params, num_threads=threads)
    d = lgb.Dataset(Xtr, ytr, free_raw_data=True); e = lgb.Dataset(Xes, yes, reference=d)
    return lgb.train(p, d, rounds, valid_sets=[e], callbacks=[lgb.early_stopping(patience, verbose=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--no-hcf", action="store_true")
    a = ap.parse_args()
    B = registry.load(a.bench, a.h)                      # meta, X (features), F (future points), classes, classifier fn
    out_dir = ROOT / "data" / a.bench / "parts"; out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(FORECAST_PARAMS, learning_rate=a.lr)
    K = len(B.classes); y = B.meta.y.to_numpy(); fold = B.meta.fold.to_numpy()
    F = B.F; Fobj = F.obj.to_numpy()
    tag = f"h{a.h:g}"
    folds = sorted(set(fold)) if a.folds is None else a.folds
    for k in folds:
        part = out_dir / f"{tag}_fold{k}.npz"
        if part.exists():
            print(f"{tag} fold {k}: exists", flush=True); continue
        t0 = time.time()
        tr_obj = np.flatnonzero(fold != k); te_obj = np.flatnonzero(fold == k)
        fit_obj, es_obj = train_test_split(tr_obj, test_size=.15, stratify=y[tr_obj], random_state=k)
        # classifier
        P, L = B.classifier(B.X, y, fit_obj, es_obj, te_obj, a.threads)
        # forecaster rows
        r_fit = cap_rows(np.flatnonzero(np.isin(Fobj, fit_obj)), Fobj, a.cap, 1000 + k)
        r_es = cap_rows(np.flatnonzero(np.isin(Fobj, es_obj)), Fobj, a.cap, 2000 + k)
        r_te = np.flatnonzero(np.isin(Fobj, te_obj))
        Xfit, Xes, Xte = (long_frame(B.X, F, r, a.h) for r in (r_fit, r_es, r_te))
        yfit, yes = F.dmag.to_numpy()[r_fit], F.dmag.to_numpy()[r_es]
        out = dict(te_obj=te_obj, P=P, L=L, rows=r_te)
        models = {"mu": params, **{q: dict(params, objective="quantile", alpha=v) for q, v in QUANTILES.items()}}
        for name, pr in models.items():
            m = fit_lgb(pr, Xfit, yfit, Xes, yes, a.threads)
            out[name] = m.predict(Xte, num_threads=a.threads)
            out[f"{name}_iters"] = m.best_iteration
        if not a.no_hcf:
            def with_class(D, cls_idx):
                D = D.copy()
                for c in range(K):
                    D[f"hyp_{c}"] = (cls_idx == c).astype(float)
                return D
            zfit, zes = y[Fobj[r_fit]], y[Fobj[r_es]]
            Hfit, Hes = with_class(Xfit, zfit), with_class(Xes, zes)
            for name in ("mu", "q16", "q84"):
                pr = models[name]
                m = fit_lgb(pr, Hfit, yfit, Hes, yes, a.threads)
                out[f"hcf_{name}"] = np.stack([m.predict(with_class(Xte, np.full(len(Xte), c)), num_threads=a.threads)
                                               for c in range(K)], axis=1)          # (rows, K)
                out[f"hcf_{name}_iters"] = m.best_iteration
        np.savez_compressed(part, **out)
        print(f"{tag} fold {k}: done in {time.time()-t0:.0f}s, rows fit {len(r_fit)} test {len(r_te)}, "
              f"iters mu {out['mu_iters']}", flush=True)
    parts = [out_dir / f"{tag}_fold{k}.npz" for k in sorted(set(fold))]
    if all(p.exists() for p in parts):
        n = len(y); nr = len(F)
        P = np.full((n, K), np.nan); L = np.full(n, -1)
        cols = {}
        for p in parts:
            z = np.load(p)
            P[z["te_obj"]] = z["P"]; L[z["te_obj"]] = z["L"]
            for key in z.files:
                if key in ("te_obj", "P", "L", "rows") or key.endswith("_iters"):
                    continue
                arr = z[key]
                if key not in cols:
                    cols[key] = np.full((nr,) + arr.shape[1:], np.nan)
                cols[key][z["rows"]] = arr
        np.savez_compressed(ROOT / "data" / a.bench / f"oof_{tag}.npz", P=P, L=L, y=y, fold=fold, **cols)
        print("assembled", ROOT / "data" / a.bench / f"oof_{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
