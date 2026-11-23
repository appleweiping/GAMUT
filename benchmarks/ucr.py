"""UCR archive datasets (Dau et al. 2019) as prefix-to-future forecasting benchmarks with a latent class.

Each series x[0:L] (TRAIN and TEST files pooled; labels are the latent classes) is observed up to a prefix fraction
h in {20, 30, 40, 50, 60} percent of its length. Inputs: the prefix resampled to 32 points, its mean, std, min, max,
last value, slope and curvature over the last 10% and the 25/75% quantiles. Targets: up to 24 equally spaced future
positions j > prefix end, as x[j] - x[prefix end - 1] ("change from the last observation", the analogue of the
magnitude relative to a reference), with t = 100 j / L in percent of the series and lead = t - h.
Folds: StratifiedKFold(5, shuffle, seed 42).
Run: python -m benchmarks.ucr --dataset StarLightCurves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ucr"
HORIZONS = (20, 30, 40, 50, 60)
N_FUT = 24


def load(ds):
    parts = [np.loadtxt(RAW / ds / f"{ds}_{s}.txt") for s in ("TRAIN", "TEST")]
    A = np.vstack(parts)
    return A[:, 0], A[:, 1:]


def features(P):
    n, p = P.shape
    grid = np.linspace(0, p - 1, 32)
    R = np.stack([np.interp(grid, np.arange(p), r) for r in P])
    k = max(3, int(0.1 * p)); tail = P[:, -k:]; tt = np.arange(k) - (k - 1) / 2
    slope = (tail * tt).sum(1) / (tt ** 2).sum()
    curv = np.polyfit(tt, tail.T, 2)[0]
    F = pd.DataFrame(R, columns=[f"r{i}" for i in range(32)])
    F["mean"], F["std"], F["min"], F["max"] = P.mean(1), P.std(1), P.min(1), P.max(1)
    F["last"], F["slope"], F["curv"] = P[:, -1], slope, curv
    F["q25"], F["q75"] = np.quantile(P, .25, 1), np.quantile(P, .75, 1)
    F["n_obs_total"] = float(p)
    return F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args()
    lab, S = load(a.dataset)
    S = S[:, ~np.isnan(S).all(0)]
    labels = sorted(np.unique(lab)); y = np.array([labels.index(v) for v in lab]); n, L = S.shape
    name = f"ucr_{a.dataset.lower()}"; out = ROOT / "data" / name; out.mkdir(parents=True, exist_ok=True)
    fold = np.full(n, -1)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y)):
        fold[te] = k
    classes = [f"class_{int(v)}" for v in labels]
    pd.DataFrame(dict(obj_id=np.arange(n), cls=[classes[v] for v in y], y=y, fold=fold)).to_parquet(out / "meta.parquet")
    (out / "classes.json").write_text(json.dumps(classes))
    for h in HORIZONS:
        p = int(round(L * h / 100))
        X = features(S[:, :p]); X.insert(0, "peak_mag", S[:, p - 1]); X.insert(0, "has_prefix", True)
        X.to_parquet(out / f"X_h{h}.parquet")
        js = np.unique(np.linspace(p, L - 1, N_FUT).round().astype(int))
        obj = np.repeat(np.arange(n), len(js)); jj = np.tile(js, n)
        Fd = pd.DataFrame(dict(obj=obj, t=100 * jj / L, band=0, dmag=S[obj, jj] - S[obj, p - 1]))
        Fd["lead"] = Fd.t - h
        Fd.to_parquet(out / f"F_h{h}.parquet")
    print(name, "n", n, "L", L, "classes", dict(zip(classes, np.bincount(y))), flush=True)


if __name__ == "__main__":
    main()
