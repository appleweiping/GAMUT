"""ZTF Bright Transient Survey benchmark (the public FLARE benchmark) as a forecasting task.

For every spectroscopically classified object of the benchmark (10,497 objects, six classes) and every
forecast origin h in {5, 10, 15, 20, 30} days after the first detection:
  * inputs: FLARE's prefix-causal feature vector (flare.analysis.feature_frame on the detections up to h,
    with the catalogue context of the benchmark), exactly the inference path of the FLARE forecaster;
  * targets: every real later g- or r-band detection with h < t <= 90 d, as AB magnitude relative to the
    brightest detection within the prefix (the FLARE forecast target);
  * the latent class (SN Ia, SN CC, SLSN, AGN, TDE, CV) and the evaluation folds: StratifiedKFold(5,
    shuffle, seed 42) over the pool in the order train, val, test (the order of FLARE's forecast caches).
Output (data/ztf/): meta.parquet, X_h{h}.parquet, F_h{h}.parquet (long table of future points).
Run: python -m benchmarks.ztf --paper-data <FLARE_PAPER_DATA> [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ztf"
HORIZONS = (5, 10, 15, 20, 30)
T_MAX = 90.0
CLASSES = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
_G = {}


def _init(paper_data, ctx):
    from flare.analysis import feature_frame, observed_events
    from flare.data import load_events, COL
    _G.update(ff=feature_frame, oe=observed_events, le=load_events, COL=COL, root=Path(paper_data), ctx=ctx)


def _object(job):
    i, fp, oid = job
    arr = _G["le"](str(_G["root"] / "data" / fp)); COL = _G["COL"]
    t, band, lf = arr[:, COL["dt"]], arr[:, COL["band_id"]], arr[:, COL["logflux"]]
    mag = 23.9 - 2.5 * lf
    ctx = {k: (None if pd.isna(v) else float(v)) for k, v in _G["ctx"].loc[oid].items()}
    out = {}
    for h in HORIZONS:
        try:
            pre = _G["oe"](arr, float(h))
        except ValueError:
            out[h] = None; continue
        f, peak = _G["ff"](pre, ctx)
        fut = (t > h) & (t <= T_MAX) & np.isin(band, (0, 1)) & np.isfinite(mag)
        out[h] = (f.iloc[0].to_dict(), float(peak),
                  np.column_stack([np.full(fut.sum(), i), t[fut], band[fut], mag[fut] - peak]))
    return i, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--paper-data", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.paper_data).resolve().parents[0]))
    root = Path(a.paper_data)
    from evaluate_unified import context_table  # FLARE scripts/ on sys.path via PYTHONPATH
    pool = pd.concat([pd.read_csv(root / "data/splits" / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    pool = pool[pool.cls.isin(CLASSES)].reset_index(drop=True)
    y = np.array([CLASSES.index(c) for c in pool.cls])
    fold = np.full(len(pool), -1)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        fold[te] = k
    ctx = context_table(root, pool.obj_id.values)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(i, r.filepath, r.obj_id) for i, r in enumerate(pool.itertuples())]
    with Pool(a.workers, initializer=_init, initargs=(a.paper_data, ctx)) as p:
        res = dict(p.imap_unordered(_object, jobs, chunksize=32))
    meta = pd.DataFrame(dict(obj_id=pool.obj_id, cls=pool.cls, y=y, fold=fold, split=pool.split, bts_type=pool.bts_type))
    meta.to_parquet(OUT / "meta.parquet")
    for h in HORIZONS:
        rows, peaks, F, keep = [], [], [], np.zeros(len(pool), bool)
        for i in range(len(pool)):
            r = res[i][h]
            if r is None:
                rows.append({}); peaks.append(np.nan); continue
            rows.append(r[0]); peaks.append(r[1]); F.append(r[2]); keep[i] = True
        X = pd.DataFrame(rows); X.insert(0, "peak_mag", peaks); X.insert(0, "has_prefix", keep)
        X.to_parquet(OUT / f"X_h{h}.parquet")
        F = pd.DataFrame(np.concatenate(F), columns=["obj", "t", "band", "dmag"]).astype(dict(obj=int, band=int))
        F["lead"] = F.t - h
        F.to_parquet(OUT / f"F_h{h}.parquet")
        print(f"h={h}: objects with prefix {keep.sum()}, future points {len(F)}, "
              f"objects with >=1 future point {F.obj.nunique()}", flush=True)


if __name__ == "__main__":
    main()
