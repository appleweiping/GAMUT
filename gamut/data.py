"""Point-level arrays for one benchmark and forecast origin, assembled from the cross-fit outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCALE_FLOOR = 0.05            # FLARE: scale = max(|q84 - q16| / 2, 0.05)
ASTRO = {"ztf", "plasticc_train", "plasticc_ddf", "plasticc_wfd", "plasticc_wfd03"}
FLARE_GRID = np.array([10, 15, 20, 25, 30, 40, 50, 60, 75, 90], float)


@dataclass
class Task:
    bench: str
    h: float
    classes: list
    # object level
    y: np.ndarray          # (n,) latent class
    fold: np.ndarray       # (n,)
    P: np.ndarray          # (n, K) out-of-fold class probabilities
    L: np.ndarray          # (n,) classifier decision
    # point level
    obj: np.ndarray        # (m,) object index of each future point
    t: np.ndarray
    band: np.ndarray
    yt: np.ndarray         # target
    lead_bin: np.ndarray
    mu: np.ndarray
    sig: np.ndarray
    q: dict                # name -> (m,) quantile forecasts
    hmu: np.ndarray        # (m, K) hypothesis-conditional point forecasts
    hsig: np.ndarray       # (m, K) hypothesis-conditional scales
    hq: dict

    @property
    def K(self):
        return len(self.classes)


def lead_edges(bench, h):
    if bench in ASTRO:                      # bins centred on FLARE's grid days (absolute time since first detection)
        mids = (FLARE_GRID[1:] + FLARE_GRID[:-1]) / 2
        return mids
    return np.linspace(h, 100, 7)[1:-1]     # UCR: five equal bins of the remaining series


def load(bench: str, h: float) -> Task:
    d = ROOT / "data" / bench
    tag = f"h{h:g}"
    z = np.load(d / f"oof_{tag}.npz")
    F = pd.read_parquet(d / f"F_{tag}.parquet")
    meta = pd.read_parquet(d / "meta.parquet")
    if bench == "ztf":
        from benchmarks.ztf import CLASSES as classes
    else:
        classes = list(pd.read_json(d / "classes.json", typ="series"))
    ok = np.isfinite(z["mu"]) & np.isfinite(F.dmag.to_numpy())
    sig = np.maximum(np.abs(z["q84"] - z["q16"]) / 2, SCALE_FLOOR)
    hsig = np.maximum(np.abs(z["hcf_q84"] - z["hcf_q16"]) / 2, SCALE_FLOOR)
    t = F.t.to_numpy()
    return Task(bench=bench, h=h, classes=classes, y=meta.y.to_numpy(), fold=meta.fold.to_numpy(), P=z["P"], L=z["L"],
                obj=F.obj.to_numpy()[ok], t=t[ok], band=F.band.to_numpy()[ok], yt=F.dmag.to_numpy()[ok],
                lead_bin=np.searchsorted(lead_edges(bench, h), t[ok], side="right"),
                mu=z["mu"][ok], sig=sig[ok], q={k: z[k][ok] for k in ("q16", "q84", "q025", "q975")},
                hmu=z["hcf_mu"][ok], hsig=hsig[ok], hq={k: z[f"hcf_{k}"][ok] for k in ("q16", "q84")})
