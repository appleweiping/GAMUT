"""Validity on the horizon cells GAMUT actually calibrates (its merged cells), for GAMUT and, on the same cells, for
FLARE and the always-valid baselines. The (class, FLARE lead-time bin) check in the paper is stricter than GAMUT's
target when bins are merged; this script reports the target itself.

For every main (benchmark, origin, level): the merged cell of a point is GAMUT's cell map of the fold in which the
point was tested (results JSON, GAMUT diag); per (class, merged cell) with >= max(1/alpha - 1, 3) objects, the
object-level coverage and its 95% bootstrap upper bound (cells are pooled over folds when all folds share one
cell map, and are fold-specific otherwise, which lowers the test's power there) (500 resamples, seeded per setting); a cell is under-covered
when the upper bound is below nominal.
Output: results/merged_cells.json
Run: python scripts/merged_cells.py
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamut import data  # noqa: E402

MAIN = ["ztf", "plasticc_train", "plasticc_ddf", "ucr_starlightcurves", "ucr_ecg5000", "ucr_electricdevices"]
METHODS = ["GAMUT", "GAMUT-cert (split certificate)", "FLARE", "TwoStepCP", "MDCP", "MaxClassCP"]
NB = 500


def key(m):
    return m.replace(" ", "_").replace("/", "_")


def main():
    out = {}
    for b in MAIN:
        for f in sorted((ROOT / "results" / b).glob("h*_a*.json")):
            if "validation" in f.name:
                continue
            h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
            R = json.loads(f.read_text())
            diag = R.get("GAMUT", {}).get("diag")
            if not diag:
                continue
            P = np.load(ROOT / "results" / b / f"points_h{h:g}_a{a:g}.npz")
            t = data.load(b, h)
            folds = np.unique(t.fold)
            pf = t.fold[t.obj]
            cell = np.full(len(t.obj), -1)
            same = all(d["cells"] == diag[0]["cells"] for d in diag)
            for i, k in enumerate(folds):
                m = pf == k                                       # pooled over folds when every fold has the same cell map
                cell[m] = np.asarray(diag[i]["cells"])[t.lead_bin[m]] + (0 if same else 1000 * i)
            z = t.y[t.obj]
            rng = np.random.default_rng(zlib.crc32(f"{b}|{h:g}|{a:g}|merged".encode()))
            res = {}
            for meth in METHODS:
                kk = f"{key(meth)}__cov"
                if kk not in P:
                    continue
                cov = P[kk].astype(float); n_under = 0; n_cells = 0; deficits = []
                for c in np.unique(z):
                    for g in np.unique(cell[z == c]):
                        sel = (z == c) & (cell == g)
                        uo, inv = np.unique(t.obj[sel], return_inverse=True)
                        if len(uo) < max(1 / a - 1, 3):
                            continue
                        co = np.bincount(inv, cov[sel]) / np.bincount(inv)
                        hi = np.percentile(co[rng.integers(0, len(co), (NB, len(co)))].mean(1), 97.5)
                        n_cells += 1; n_under += hi < 1 - a
                        if hi < 1 - a:
                            deficits.append(float(1 - a - co.mean()))
                res[meth] = dict(n_under=int(n_under), n_cells=int(n_cells), deficits=deficits)
            out[f"{b}|{h:g}|{a:g}"] = res
            print(b, h, a, {m: (r["n_under"], r["n_cells"]) for m, r in res.items()}, flush=True)
    (ROOT / "results" / "merged_cells.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
