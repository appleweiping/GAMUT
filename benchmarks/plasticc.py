"""PLAsTiCC (Kessler et al. 2019; unblinded release, Zenodo 2539456) as light-curve forecasting benchmarks.

Two benchmarks with different cadence and selection:
  * plasticc_train: the 7,848-object training set (spectroscopically biased, bright, WFD + DDF);
  * plasticc_ddf:   the deep-drilling-field objects of test chunk 01 with their unblinded classes (deep, unbiased);
  * plasticc_wfd03: the same for test chunk 03 (second held-out set);
  * plasticc_wfd:   the wide-fast-deep objects of test chunk 02 (held-out set, pre-registered;
                    never used for design). Rows are pre-filtered to the detections that to_events keeps, and objects to
                    those that can pass the quality cut, before feature extraction (equivalent result, for speed).
Each object's g, r, i detections (detected_bool = 1, passbands 1-3) are converted to the FLARE event layout (time since
first detection, log flux in microJansky: FLUXCAL zero point 27.5 vs AB 23.9), the FLARE quality cut is applied (at least
8 detections with at least two in g and in r within 100 days), and the same prefix-causal FLARE features and forecast
targets as the ZTF benchmark are built at origins h in {5, 10, 15, 20, 30} d (future g/r detections with h < t <= 90 d,
magnitude relative to the brightest prefix detection). No catalogue context. Classes with fewer than 30 objects are dropped.
Run: python -m benchmarks.plasticc --which train|ddf|wfd [--workers 4]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "plasticc"
HORIZONS = (5, 10, 15, 20, 30)
T_MAX = 90.0
NAMES = {6: "uLens", 15: "TDE", 16: "EB", 42: "SNII", 52: "SNIax", 53: "Mira", 62: "SNIbc", 64: "KN", 65: "Mdwarf",
         67: "SNIa91bg", 88: "AGN", 90: "SNIa", 92: "RRL", 95: "SLSN", 991: "uLensBinary", 992: "ILOT", 993: "CaRT",
         994: "PISN", 995: "uLensString"}
_G = {}


def _init():
    from flare.analysis import feature_frame, observed_events
    from flare.data import COL, recompute_colors
    _G.update(ff=feature_frame, oe=observed_events, COL=COL, rc=recompute_colors)


def to_events(df):
    """FLARE 15-column event array from one object's PLAsTiCC rows (detections in g, r, i only)."""
    COL = _G["COL"]
    d = df[(df.detected_bool == 1) & df.passband.isin([1, 2, 3]) & (df.flux > 0)].sort_values("mjd")
    if len(d) == 0:
        return None
    t = d.mjd.to_numpy() - d.mjd.to_numpy()[0]
    keep = t <= 100.0
    d, t = d[keep], t[keep]
    a = np.full((len(d), len(COL)), np.nan)
    a[:, COL["dt"]] = t
    a[:, COL["dt_prev"]] = np.r_[0.0, np.diff(t)]
    b = d.passband.to_numpy() - 1
    a[:, COL["band_id"]] = b
    a[:, COL["logflux"]] = np.log10(d.flux.to_numpy()) - 1.44          # FLUXCAL (ZP 27.5) -> microJy (ZP 23.9)
    a[:, COL["logflux_err"]] = d.flux_err.to_numpy() / (d.flux.to_numpy() * np.log(10))
    for k, name in enumerate(("band_ztfg", "band_ztfr", "band_ztfi")):
        a[:, COL[name]] = (b == k).astype(float)
    a[:, COL["label"]] = 0.0
    return _G["rc"](a)


def _object(job):
    i, df = job
    arr = to_events(df)
    if arr is None:
        return i, None
    COL = _G["COL"]; band = arr[:, COL["band_id"]]
    if len(arr) < 8 or (band == 0).sum() < 2 or (band == 1).sum() < 2:
        return i, None
    t, mag = arr[:, COL["dt"]], 23.9 - 2.5 * arr[:, COL["logflux"]]
    out = {}
    for h in HORIZONS:
        try:
            pre = _G["oe"](arr, float(h))
        except ValueError:
            out[h] = None; continue
        f, peak = _G["ff"](pre, {})
        fut = (t > h) & (t <= T_MAX) & np.isin(band, (0, 1)) & np.isfinite(mag)
        out[h] = (f.iloc[0].to_dict(), float(peak), np.column_stack([np.full(fut.sum(), i), t[fut], band[fut], mag[fut] - peak]))
    return i, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", choices=["train", "ddf", "wfd", "wfd03"], required=True)
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    if a.which == "train":
        lc = pd.read_csv(RAW / "plasticc_train_lightcurves.csv.gz"); meta = pd.read_csv(RAW / "plasticc_train_metadata.csv.gz")
        meta["label"] = meta.target
    elif a.which in ("wfd", "wfd03"):
        parts = []
        chunk = "02" if a.which == "wfd" else "03"
        for ch in pd.read_csv(RAW / f"plasticc_test_lightcurves_{chunk}.csv.gz", chunksize=5_000_000):
            parts.append(ch[(ch.detected_bool == 1) & ch.passband.isin([1, 2, 3]) & (ch.flux > 0)])
        lc = pd.concat(parts, ignore_index=True); del parts
        cnt = lc.groupby("object_id").passband.agg(n="size", g=lambda b: (b == 1).sum(), r=lambda b: (b == 2).sum())
        cand = cnt.index[(cnt.n >= 8) & (cnt.g >= 2) & (cnt.r >= 2)]
        meta = pd.read_csv(RAW / "plasticc_test_metadata.csv.gz", usecols=["object_id", "ddf_bool", "true_target"])
        meta = meta[meta.object_id.isin(cand) & (meta.ddf_bool == 0)].copy()
        meta["label"] = meta.true_target
        lc = lc[lc.object_id.isin(meta.object_id)]
        print("wfd candidates after pre-filter:", len(meta), flush=True)
    else:
        lc = pd.read_csv(RAW / "plasticc_test_lightcurves_01.csv.gz")
        meta = pd.read_csv(RAW / "plasticc_test_metadata.csv.gz", usecols=["object_id", "ddf_bool", "true_target"])
        meta = meta[meta.object_id.isin(lc.object_id.unique()) & (meta.ddf_bool == 1)].copy()
        meta["label"] = meta.true_target
        lc = lc[lc.object_id.isin(meta.object_id)]
    name = f"plasticc_{a.which}"
    out = ROOT / "data" / name; out.mkdir(parents=True, exist_ok=True)
    groups = {oid: g for oid, g in lc.groupby("object_id", sort=False)}
    meta = meta[meta.object_id.isin(groups)].reset_index(drop=True)
    jobs = [(i, groups[oid]) for i, oid in enumerate(meta.object_id)]
    with Pool(a.workers, initializer=_init) as p:
        res = dict(p.imap_unordered(_object, jobs, chunksize=64))
    ok = np.array([res[i] is not None for i in range(len(meta))])
    meta = meta[ok].reset_index(drop=True); res = [res[i] for i in np.flatnonzero(ok)]
    counts = meta.label.value_counts()
    keep_labels = sorted(counts[counts >= 30].index)
    sel = meta.label.isin(keep_labels).to_numpy()
    meta = meta[sel].reset_index(drop=True); res = [r for r, s in zip(res, sel) if s]
    classes = [NAMES.get(int(l), str(l)) for l in keep_labels]
    y = np.array([keep_labels.index(l) for l in meta.label])
    fold = np.full(len(y), -1)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        fold[te] = k
    pd.DataFrame(dict(obj_id=meta.object_id, label=meta.label, cls=[classes[v] for v in y], y=y, fold=fold)).to_parquet(out / "meta.parquet")
    (out / "classes.json").write_text(json.dumps(classes))
    for h in HORIZONS:
        rows, peaks, F, has = [], [], [], []
        for i, r in enumerate(res):
            rr = r[h]
            if rr is None:
                rows.append({}); peaks.append(np.nan); has.append(False); continue
            f, pk, pts = rr
            pts = pts.copy(); pts[:, 0] = i
            rows.append(f); peaks.append(pk); F.append(pts); has.append(True)
        X = pd.DataFrame(rows); X.insert(0, "peak_mag", peaks); X.insert(0, "has_prefix", has)
        X.to_parquet(out / f"X_h{h}.parquet")
        Fd = pd.DataFrame(np.concatenate(F), columns=["obj", "t", "band", "dmag"]).astype(dict(obj=int, band=int)); Fd["lead"] = Fd.t - h
        Fd.to_parquet(out / f"F_h{h}.parquet")
        print(f"{name} h={h}: objects {len(y)}, future points {len(Fd)}", flush=True)
    print(name, "classes", dict(zip(classes, np.bincount(y))), flush=True)


if __name__ == "__main__":
    main()
