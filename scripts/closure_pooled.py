"""Control for the closure-probability study: the same simulation with a POOLED next-point law.

S3 = P(singleton | next magnitude ~ pooled forecaster + pooled empirical standardized residuals of next points on
`val`, 8 quantile draws), with the pooled LightGBM median/q16/q84 trained on FLARE's `train` split
(scripts/closure_c1.py --pooled). It differs from S2 only in the next-point law: no class input, no posterior. Rows,
outcomes and the deployed set are those of results/closure/h{h}.parquet; the column `s3` is added in place.
Run: python scripts/closure_pooled.py --h 5 [--workers 4]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from closure import FLARE, PAPER, WINDOW, N_DRAW_CLASS, _set_and_post  # noqa: E402
_G = {}


def _init(h, Q, ctx):
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(FLARE / "scripts"))
    from flare import load_analysis
    _G.update(A=load_analysis(), h=h, Q=Q, ctx=ctx)


def _rows(job):
    from flare.analysis import append_detection, observed_events
    from flare.data import load_events
    oid, fp, items = job
    A, h = _G["A"], _G["h"]; clf = A._classifier_at(h)
    ctx = {k: (np.nan if pd.isna(v) else float(v)) for k, v in _G["ctx"].loc[oid].items()}
    pre = observed_events(load_events(str(PAPER / "data" / fp)), float(h))
    _, _, peak = _set_and_post(pre, ctx, clf)
    out = []
    for (b, t, mu, sig) in items:
        draws = peak + mu + sig * _G["Q"]
        out.append((oid, b, float(np.mean([len(_set_and_post(append_detection(pre, t, b, m, 0.08), ctx, clf)[0]) == 1 for m in draws]))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--h", type=int, required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); h = a.h
    sys.path.insert(0, str(FLARE / "scripts"))
    from evaluate_unified import context_table
    from gamut import data
    df = pd.read_parquet(ROOT / "results" / "closure" / f"h{h}.parquet")
    meta = pd.read_parquet(ROOT / "data" / "ztf" / "meta.parquet").reset_index().rename(columns={"index": "i"})
    files = pd.concat([pd.read_csv(PAPER / "data" / "splits" / f"{s}.csv") for s in ("val", "test")])[["obj_id", "filepath"]]
    meta = meta.merge(files, on="obj_id").set_index("obj_id")
    F = pd.read_parquet(ROOT / "data" / "ztf" / f"F_h{h}.parquet")
    z = np.load(ROOT / "data" / "ztf" / f"closure_pool_h{h}.npz")
    pos = {int(r): j for j, r in enumerate(z["rows"])}
    MU = z["pool_mu"]; SG = np.maximum(np.abs(z["pool_q84"] - z["pool_q16"]) / 2, data.SCALE_FLOOR)
    Fo, Fb, Ft, Fy = F.obj.to_numpy(), F.band.to_numpy(), F.t.to_numpy(), F.dmag.to_numpy()
    jobs, resid = {}, []
    for r in df.itertuples():
        i = int(meta.loc[r.obj_id, "i"])
        fr = np.flatnonzero((Fo == i) & (Fb == r.band) & np.isclose(Ft, r.t_next))
        if not len(fr) or int(fr[0]) not in pos:
            continue
        j = pos[int(fr[0])]
        jobs.setdefault(r.obj_id, (meta.loc[r.obj_id, "filepath"], []))[1].append((int(r.band), float(r.t_next), float(MU[j]), float(SG[j])))
        if r.split == "val":
            resid.append((Fy[fr[0]] - MU[j]) / SG[j])
    Q = np.quantile(np.array(resid), (np.arange(N_DRAW_CLASS) + 0.5) / N_DRAW_CLASS)
    ctx = context_table(PAPER, np.array(list(jobs)))
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(h, Q, ctx)) as p:
        res = [x for rows in p.imap_unordered(_rows, [(o, fp, it) for o, (fp, it) in jobs.items()], chunksize=4) for x in rows]
    s3 = pd.DataFrame(res, columns=["obj_id", "band", "s3"])
    df = df.drop(columns=[c for c in ("s3",) if c in df]).merge(s3, on=["obj_id", "band"], how="left")
    df.to_parquet(ROOT / "results" / "closure" / f"h{h}.parquet")
    print(f"h={h}: s3 computed for {s3.shape[0]} rows of {len(df)}; val residuals {len(resid)}", flush=True)


if __name__ == "__main__":
    main()
