"""Closure probability: P(the deployed 90% prediction set becomes a singleton after one more observation).

Pre-registered in refine-logs/EXPERIMENT_PLAN.md ("Closure probability"). FLARE's deployed day-5/10 classifiers were
trained on FLARE's `train` split; objects of `val` (calibration of our scores) and `test` (evaluation) are used here.

For each object and band b in {g, r} with a real next detection in band b within (h, h + 2] days:
  outcome  y   = 1{ set(prefix + real next detection) is a singleton }
  S0 (deployed, as shipped)  FLARE's next_observation_value p_singleton (16 Gaussian draws at h + 1 day)
  S1 (deployed, real time)   the same construction with the draws at the real next time t*
  S2 (ours, latent-class mixture; C1 trained on FLARE's `train` split, scripts/closure_c1.py)
        p_close = sum_c p_hat_c(x) * P(singleton | next magnitude ~ class-c law at t*),
        class-c law = GAMUT's hypothesis-conditional forecaster (out-of-fold) + class-c empirical standardized
        residuals of next points on `val` (8 quantile draws), p_hat = deployed posterior temperature-scaled on `val`.
Output: results/closure/h{h}.parquet (one row per object and band).
Run: python scripts/closure.py --h 5 [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = Path(os.environ.get("FLARE_PAPER_DATA", ROOT.parent / "paper_data"))   # FLARE paper data (contains data/splits, data/<events>)
FLARE = Path(os.environ.get("FLARE_REPO", ROOT.parent / "flare"))               # FLARE checkout (deployed models in models/)
WINDOW = 2.0          # a "next observation" is the first detection in the band within this many days after h
N_DRAW_CLASS = 8
P_MIN = 0.01
_G = {}


def _init(h, T, Q, ctx):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(FLARE / "scripts"))
    from flare import load_analysis
    _G.update(A=load_analysis(), h=h, T=T, Q=Q, ctx=ctx)


def _ctx(oid):
    row = _G["ctx"].loc[oid]
    return {k: (np.nan if pd.isna(v) else float(v)) for k, v in row.items()}


def _set_and_post(a, ctx, clf):
    import flare.data as D
    from flare.analysis import feature_frame
    X, peak = feature_frame(D.recompute_colors(a), ctx)
    Xc = X.reindex(columns=clf.feature_names)
    return clf.conformal_report(Xc)[0]["set"], clf.predict_proba(Xc)[0], peak


def _temper(p, T):
    s = np.log(np.maximum(p, 1e-8)) / T
    e = np.exp(s - s.max())
    return e / e.sum()


def _object(job):
    """job: (obj_id, filepath, split, y, nexts) with nexts = [(band, t, mag, err, hmu(K), hsig(K))]."""
    from flare.analysis import NEXT_OBS_SCATTER, append_detection, observed_events
    from flare.data import load_events
    oid, fp, split, y, nexts = job
    A, h = _G["A"], _G["h"]
    clf = A._classifier_at(h)
    ctx = _ctx(oid)
    full = load_events(str(PAPER / "data" / fp))
    try:
        pre = observed_events(full, float(h))
    except ValueError:
        return []
    set0, post, peak = _set_and_post(pre, ctx, clf)
    phat = _temper(post, _G["T"])
    nxt = A.next_observation_value(full, h, ctx)                       # S0, exactly as deployed
    fc = A.forecast(full, h, ctx)
    rows = []
    for (b, t, mag, err, hmu, hsig) in nexts:
        name = "gr"[b]
        dep = nxt["bands"].get(name, {})
        s0 = dep.get("p_singleton", np.nan) if dep.get("status") == "ok" else np.nan
        set1, _, _ = _set_and_post(append_detection(pre, t, b, mag, err), ctx, clf)
        # S1: deployed construction at the real next time
        curve = fc["bands"].get(name, {})
        s1 = np.nan
        if curve.get("status") == "ok":
            (d0, d1), (m0, m1) = curve["days"][:2], curve["magnitude"][:2]
            mean = m0 + (m1 - m0) * (t - d0) / (d1 - d0)
            rng = np.random.default_rng(0)
            draws = rng.normal(mean, NEXT_OBS_SCATTER[h][b], 16)
            s1 = float(np.mean([len(_set_and_post(append_detection(pre, t, b, m, 0.08), ctx, clf)[0]) == 1 for m in draws]))
        # S2: latent-class mixture
        keep = np.flatnonzero(phat >= P_MIN)
        w = phat[keep] / phat[keep].sum()
        pc = np.full(len(phat), np.nan)
        for c in keep:
            draws = peak + hmu[c] + hsig[c] * _G["Q"][c]
            pc[c] = np.mean([len(_set_and_post(append_detection(pre, t, b, m, 0.08), ctx, clf)[0]) == 1 for m in draws])
        s2 = float(np.sum(w * pc[keep]))
        rows.append(dict(obj_id=oid, split=split, y=y, band=b, t_next=t, dt_next=t - h, set_before=len(set0),
                         set_before_has_true=None, outcome=int(len(set1) == 1), set_after=len(set1),
                         s0=s0, s1=s1, s2=s2, **{f"phat{k}": float(v) for k, v in enumerate(phat)},
                         **{f"pclose{k}": float(v) for k, v in enumerate(pc)}))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--h", type=int, required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--c1", choices=["train", "crossfit"], default="train",
                    help="train: C1 fitted on FLARE's train split only (scripts/closure_c1.py); crossfit: the main experiments' C1")
    a = ap.parse_args(); h = a.h
    sys.path.insert(0, str(FLARE / "scripts"))
    from evaluate_unified import context_table
    from flare import load_analysis
    from flare.analysis import observed_events
    from flare.data import COL, load_events
    from gamut import data
    from gamut.methods import _temperature
    task = data.load("ztf", float(h))
    meta = pd.read_parquet(ROOT / "data" / "ztf" / "meta.parquet")
    files = pd.concat([pd.read_csv(PAPER / "data" / "splits" / f"{s}.csv") for s in ("val", "test")])[["obj_id", "filepath"]]
    meta = meta.reset_index().rename(columns={"index": "i"}).merge(files, on="obj_id")
    meta = meta[meta.split.isin(["val", "test"])]
    if a.limit:
        meta = meta.groupby("split").head(a.limit)
    ctx = context_table(PAPER, meta.obj_id.values)
    K = task.K
    Fdf = pd.read_parquet(ROOT / "data" / "ztf" / f"F_h{h}.parquet")
    if a.c1 == "train":
        z = np.load(ROOT / "data" / "ztf" / f"closure_c1_h{h}.npz")
        pos = {int(r): j for j, r in enumerate(z["rows"])}
        HMU = z["hcf_mu"]; HSG = np.maximum(np.abs(z["hcf_q84"] - z["hcf_q16"]) / 2, data.SCALE_FLOOR)
    # next real detection per object and band, with the out-of-fold C1 forecasts at that point
    jobs, resid = [], {c: [] for c in range(K)}
    A = load_analysis(); clf = A._classifier_at(h)
    Pval, Yval = [], []
    for r in meta.itertuples():
        full = load_events(str(PAPER / "data" / r.filepath))
        try:
            observed_events(full, float(h))
        except ValueError:
            continue
        nexts = []
        for b in (0, 1):
            m = (full[:, COL["dt"]] > h) & (full[:, COL["dt"]] <= h + WINDOW) & (full[:, COL["band_id"]] == b)
            if not m.any():
                continue
            j = np.flatnonzero(m)[0]
            t = float(full[j, COL["dt"]]); mag = 23.9 - 2.5 * float(full[j, COL["logflux"]]); err = 2.5 * float(full[j, COL["logflux_err"]])
            if a.c1 == "train":
                fr = np.flatnonzero((Fdf.obj.to_numpy() == r.i) & (Fdf.band.to_numpy() == b) & np.isclose(Fdf.t.to_numpy(), t))
                if not len(fr) or int(fr[0]) not in pos:
                    continue
                j = pos[int(fr[0])]; hmu, hsig, yt = HMU[j].copy(), HSG[j].copy(), float(Fdf.dmag.to_numpy()[fr[0]])
            else:
                rows = np.flatnonzero((task.obj == r.i) & (task.band == b) & np.isclose(task.t, t))
                if not len(rows):
                    continue
                k = rows[0]; hmu, hsig, yt = task.hmu[k].copy(), task.hsig[k].copy(), float(task.yt[k])
            if not (np.all(np.isfinite(hmu)) and np.all(np.isfinite(hsig))):
                continue
            nexts.append((b, t, mag, err, hmu, hsig))
            if r.split == "val":
                resid[int(r.y)].append((yt - hmu[int(r.y)]) / hsig[int(r.y)])
        if nexts:
            jobs.append((r.obj_id, r.filepath, r.split, int(r.y), nexts))
        if r.split == "val":
            ctxd = {k_: (np.nan if pd.isna(v) else float(v)) for k_, v in ctx.loc[r.obj_id].items()}
            from flare.analysis import feature_frame
            import flare.data as D
            X, _ = feature_frame(D.recompute_colors(observed_events(full, float(h))), ctxd)
            Pval.append(clf.predict_proba(X.reindex(columns=clf.feature_names))[0]); Yval.append(int(r.y))
    T = _temperature(np.array(Pval), np.array(Yval), K)
    pooled = np.concatenate([np.array(v) for v in resid.values() if len(v)])
    qs = (np.arange(N_DRAW_CLASS) + 0.5) / N_DRAW_CLASS
    Q = {c: np.quantile(np.array(resid[c]) if len(resid[c]) >= 30 else pooled, qs) for c in range(K)}
    print(f"h={h}: jobs {len(jobs)} (val {sum(j[2] == 'val' for j in jobs)}, test {sum(j[2] == 'test' for j in jobs)}), "
          f"T={T:.3f}, residual n per class {[len(resid[c]) for c in range(K)]}", flush=True)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(h, T, Q, ctx)) as p:   # spawn: LightGBM/OpenMP is not fork-safe
        out = [row for rows in p.imap_unordered(_object, jobs, chunksize=4) for row in rows]
    df = pd.DataFrame(out); df["h"] = h; df["T"] = T
    (ROOT / "results" / "closure").mkdir(parents=True, exist_ok=True)
    suffix = ("_smoke" if a.limit else "") + ("" if a.c1 == "train" else "_crossfitC1")
    df.drop(columns=["set_before_has_true"]).to_parquet(ROOT / "results" / "closure" / f"h{h}{suffix}.parquet")
    print(f"h={h}: rows {len(df)}, outcome rate {df.outcome.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
