"""Registered block B1: paired object-level bootstrap of the class-averaged interval score, GAMUT vs the best valid
baseline of each setting (the preregistered experiment plan).

For every main (benchmark, origin, level): the baselines valid under the bootstrap rule (results/bootstrap.json); the
one with the lowest class-averaged interval score is the comparator. Objects are resampled within each class (1,000
resamples, seeded per setting); the statistic is the ratio of class-averaged interval scores GAMUT / comparator.
Output: results/is_bootstrap.json
Run: python scripts/is_bootstrap.py [--method GAMUT-Q] [--benches plasticc_wfd03] [--out results/is_bootstrap_q.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamut import data  # noqa: E402
from gamut.metrics import interval_score  # noqa: E402

MAIN = ["ztf", "plasticc_train", "plasticc_ddf", "ucr_starlightcurves", "ucr_ecg5000", "ucr_electricdevices"]
BASE = ["FLARE", "SplitCP", "CQR", "DCP", "LocalizedCP", "MixtureCP", "GCC-prob", "PartitionCP", "CPTC", "TwoStepCP",
        "KandinskyCP", "MDCP", "MaxClassCP", "TwoStepCP + C1", "MDCP + C1", "MaxClassCP + C1"]
NB = 1000


def key(m):
    return m.replace(" ", "_").replace("/", "_")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="GAMUT"); ap.add_argument("--benches", nargs="*", default=MAIN)
    ap.add_argument("--out", default=str(ROOT / "results" / "is_bootstrap.json"))
    ap.add_argument("--comparator", default=None, help="fixed comparator (e.g. TwoStepCP) instead of the best valid baseline")
    a_ = ap.parse_args(); M = a_.method; MK = key(M)
    boot = json.loads((ROOT / "results" / "bootstrap.json").read_text())
    out = {}
    for b in a_.benches:
        for f in sorted((ROOT / "results" / b).glob("h*_a*.json")):
            if "validation" in f.name:
                continue
            h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
            R = json.loads(f.read_text()); bk = boot.get(f"{b}|{h:g}|{a:g}", {})
            valid = [m for m in BASE if m in R and m in bk and bk[m]["n_under"] == 0]
            if a_.comparator is not None:
                valid = [a_.comparator] if a_.comparator in R else []
            if not valid or M not in R:
                continue
            comp = min(valid, key=lambda m: R[m]["class_avg_interval_score"])
            P = np.load(ROOT / "results" / b / f"points_h{h:g}_a{a:g}.npz")
            task = data.load(b, h)
            assert len(task.yt) == len(P["obj"])
            isg = interval_score(task.yt, P[f"{MK}__lo"], P[f"{MK}__hi"], a)
            isc = interval_score(task.yt, P[f"{key(comp)}__lo"], P[f"{key(comp)}__hi"], a)
            uo, inv = np.unique(P["obj"], return_inverse=True)
            zo = task.y[uo]; npts = np.bincount(inv)
            sg = np.bincount(inv, isg); sc = np.bincount(inv, isc)
            rng = np.random.default_rng(zlib.crc32(f"{b}|{h:g}|{a:g}".encode()))
            ratios = []
            cls = [np.flatnonzero(zo == c) for c in np.unique(zo)]
            for _ in range(NB):
                g = c_ = 0.0
                for ix in cls:
                    d = ix[rng.integers(0, len(ix), len(ix))]
                    g += sg[d].sum() / npts[d].sum(); c_ += sc[d].sum() / npts[d].sum()
                ratios.append(g / c_)
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            gv = bk.get(M, {}).get("n_under", 1) == 0
            point = R[M]["class_avg_interval_score"] / R[comp]["class_avg_interval_score"]
            out[f"{b}|{h:g}|{a:g}"] = dict(comparator=comp, ratio=float(point), lo=float(lo), hi=float(hi), gamut_valid=bool(gv),
                                          better=bool(hi < 1), worse=bool(lo > 1))
            print(f"{M} {b} h={h:g} a={a:g}: vs {comp:16s} ratio {point:.3f} [{lo:.3f}, {hi:.3f}] gamut_valid={gv}", flush=True)
    Path(a_.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
