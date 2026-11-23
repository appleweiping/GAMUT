"""Object-level bootstrap (objects are the exchangeable units) for every setting with saved per-point outcomes.

For every (benchmark, origin, level) and method: per-class point-pooled coverage with a 95% percentile interval
(500 resamples of objects within each class), and whether the class is significantly under-covered (upper bound below
nominal). For GAMUT against every latent-class-valid baseline: the ratio of mean band lengths with its interval
(paired: the same resampled objects for both methods).
Output: results/bootstrap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
B = 500
METHODS = ["FLARE", "SplitCP", "CQR", "DCP", "LocalizedCP", "MixtureCP", "GCC-prob", "PartitionCP", "CPTC", "TwoStepCP",
           "KandinskyCP", "MDCP", "MaxClassCP", "MDCP + C1", "TwoStepCP + C1", "MaxClassCP + C1", "GAMUT", "Oracle-true-class"]
sys.path.insert(0, str(ROOT))
from gamut import methods as _m  # noqa: E402
METHODS += list(_m.ABLATIONS)
VALIDB = ["TwoStepCP", "KandinskyCP", "MDCP", "MaxClassCP", "MDCP + C1", "TwoStepCP + C1", "MaxClassCP + C1"]


def key(m):
    return m.replace(" ", "_").replace("/", "_")


def main():
    out = {}
    import zlib
    seed = lambda *parts: zlib.crc32("|".join(map(str, parts)).encode())   # stable per setting (and method): results do not depend on the method list
    for f in sorted((ROOT / "results").glob("*/points_h*_a*.npz")):
        bench = f.parent.name
        if bench.startswith("_"):
            continue
        h = float(f.stem.split("_h")[1].split("_a")[0]); a = float(f.stem.split("_a")[1])
        try:
            P = dict(np.load(f))
        except Exception:
            print("skip (being written)", f, flush=True); continue
        obj = P["obj"]; yobj = P["y"]
        uo, inv = np.unique(obj, return_inverse=True)
        zo = np.zeros(len(uo), int); zo[inv] = yobj
        npts = np.bincount(inv)
        res = {}
        idx = {c: np.flatnonzero(zo == c) for c in np.unique(zo)}
        try:
            from gamut import data
            t = data.load(bench, h); lb = t.lead_bin if len(t.lead_bin) == len(obj) else None
        except Exception:
            lb = None
        rng = np.random.default_rng(seed(bench, f"{h:g}", f"{a:g}"))
        draws = {c: rng.integers(0, len(ix), (B, len(ix))) for c, ix in idx.items()}
        allix = np.arange(len(uo)); dall = rng.integers(0, len(uo), (B, len(uo)))
        for m in METHODS:
            if f"{key(m)}__cov" not in P:
                continue
            cv = np.bincount(inv, P[f"{key(m)}__cov"].astype(float))
            ln = np.bincount(inv, np.nan_to_num(P[f"{key(m)}__length"]))
            per = {}
            for c, ix in idx.items():
                bc = cv[ix][draws[c]].sum(1) / npts[ix][draws[c]].sum(1)
                lo, hi = np.percentile(bc, [2.5, 97.5])
                per[int(c)] = dict(cov=float(cv[ix].sum() / npts[ix].sum()), lo=float(lo), hi=float(hi), under=bool(hi < 1 - a))
            cells = {}; crng = np.random.default_rng(seed(bench, f"{h:g}", f"{a:g}", m))
            if lb is not None:
                cov_pt = P[f"{key(m)}__cov"].astype(float)
                for c in np.unique(zo):
                    for b in np.unique(lb):
                        sel = (yobj == c) & (lb == b)
                        if not sel.any():
                            continue
                        uo2, inv2 = np.unique(obj[sel], return_inverse=True)
                        if len(uo2) < max(1 / a - 1, 3):
                            continue
                        co = np.bincount(inv2, cov_pt[sel]) / np.bincount(inv2)
                        bs = co[crng.integers(0, len(co), (B, len(co)))].mean(1)
                        cells[f"{int(c)}|{int(b)}"] = dict(n=int(len(uo2)), cov=float(co.mean()), hi=float(np.percentile(bs, 97.5)),
                                                          under=bool(np.percentile(bs, 97.5) < 1 - a))
            res[m] = dict(per_class=per, n_under=int(sum(v["under"] for v in per.values())), cells=cells,
                          n_cells_under=int(sum(v["under"] for v in cells.values())), _len=ln, _cv=cv)
        if "GAMUT" in res:
            g = res["GAMUT"]["_len"]
            for m in VALIDB:
                if m in res:
                    r = res[m]["_len"]
                    ratio = g[dall].sum(1) / r[dall].sum(1)
                    res["GAMUT"][f"width_ratio_vs_{m}"] = dict(ratio=float(g.sum() / r.sum()), lo=float(np.percentile(ratio, 2.5)),
                                                             hi=float(np.percentile(ratio, 97.5)))
        for m in res:
            res[m].pop("_len", None); res[m].pop("_cv", None)
        out[f"{bench}|{h:g}|{a:g}"] = res
        print(bench, h, a, {m: (r["n_under"], r["n_cells_under"]) for m, r in res.items()}, flush=True)
    (ROOT / "results" / "bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
