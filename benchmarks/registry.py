"""Uniform access to the forecasting benchmarks: meta (y, fold), prefix features X, future points F, classes,
and the benchmark's classifier recipe (FLARE's hierarchy for ZTF, a class-weighted LightGBM elsewhere)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class Bench:
    name: str
    h: float
    meta: pd.DataFrame
    X: pd.DataFrame
    F: pd.DataFrame
    classes: list
    classifier: Callable


def _lgb_classifier(X, y, fit, es, te, threads, K):
    import lightgbm as lgb
    cnt = np.bincount(y[fit], minlength=K)
    w = (len(fit) / (K * np.maximum(cnt, 1))) ** .8
    p = dict(objective="multiclass", num_class=K, learning_rate=.05, num_leaves=31, min_child_samples=20,
             feature_fraction=.7, bagging_fraction=.85, bagging_freq=1, lambda_l2=1., verbosity=-1, seed=42,
             num_threads=threads)
    d = lgb.Dataset(X.iloc[fit], y[fit], weight=w[y[fit]]); e = lgb.Dataset(X.iloc[es], y[es], reference=d)
    m = lgb.train(p, d, 3000, valid_sets=[e], callbacks=[lgb.early_stopping(100, verbose=False)])
    P = m.predict(X.iloc[te], num_threads=threads)
    return P, P.argmax(1)


def _flare_classifier(X, y, fit, es, te, threads):
    import os
    import sys
    sys.path.insert(0, str(Path(os.environ.get("FLARE_REPO", ROOT.parent / "flare")) / "scripts"))   # FLARE checkout
    from train_early_cc import hierarchy_nocal
    from flare.config import BTS6
    feats = [c for c in X.columns if c not in ("has_prefix", "peak_mag")]
    m = hierarchy_nocal(X[feats], y, fit, es, SimpleNamespace(threads=threads, rounds=2000))
    Xt = X[feats].iloc[te][m.feature_names]
    return m.predict_proba(Xt), np.array([BTS6.classes.index(c) for c in m.predict(Xt)])


def load(name: str, h: float) -> Bench:
    d = DATA / name
    meta = pd.read_parquet(d / "meta.parquet")
    tag = f"{h:g}"
    X = pd.read_parquet(d / f"X_h{tag}.parquet")
    F = pd.read_parquet(d / f"F_h{tag}.parquet")
    classes = list(pd.read_json(d / "classes.json", typ="series")) if (d / "classes.json").exists() else None
    if name == "ztf":
        from .ztf import CLASSES
        clf = _flare_classifier
        classes = CLASSES
    else:
        K = len(classes)
        clf = lambda X_, y_, f_, e_, t_, th: _lgb_classifier(X_.drop(columns=[c for c in ("has_prefix", "peak_mag") if c in X_]), y_, f_, e_, t_, th, K)
    return Bench(name, h, meta, X, F, classes, clf)
