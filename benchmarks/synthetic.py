"""Synthetic latent-class forecasting benchmark with a known posterior and known class-conditional laws.

K = 3 classes with priors (0.70, 0.25, 0.05): the rare class has the widest residuals. Prefix features
x | Z = c ~ N(m_c, I_2); the true posterior p(c | x) is available in closed form. Each object has 10 future
points at lead times t = 10, 20, ..., 100: y = a_c x_1 t / 100 + s_c (0.3 + t / 100) eps. The classifier
posterior handed to the methods is the true posterior sharpened or blurred by an inverse temperature kappa
(kappa = 1: Bayes-optimal); the forecasters are the true class means and scales (hypothesis-conditional) and the
posterior-mixture mean and quantiles (pooled), so that differences come from the calibration step alone.
Writes data/synthetic_k<kappa>/ in the format of benchmarks/registry.py and scripts/crossfit.py.
Run: python -m benchmarks.synthetic --kappa 1.0 [--n 6000 --seed 0]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PRIOR = np.array([0.70, 0.25, 0.05])
MEANS = np.array([[0.0, 0.0], [1.2, 0.6], [0.6, 1.4]])
SLOPE = np.array([1.0, -0.5, 2.0])
SCALE = np.array([0.4, 0.8, 1.6])
T = np.arange(10, 101, 10, dtype=float)


def posterior(x):
    ll = -0.5 * ((x[:, None, :] - MEANS[None]) ** 2).sum(-1) + np.log(PRIOR)[None]
    ll -= ll.max(1, keepdims=True); p = np.exp(ll)
    return p / p.sum(1, keepdims=True)


def class_law(x, t):
    """(n, m, K) means and scales of y given x, lead t, class."""
    mu = SLOPE[None, None, :] * x[:, 0][:, None, None] * (t[None, :, None] / 100)
    sd = np.broadcast_to(SCALE[None, None, :] * (0.3 + t[None, :, None] / 100), mu.shape)
    return mu, sd


def mixture_quantiles(P, mu, sd, levels):
    """Quantiles of sum_c P_c N(mu_c, sd_c) by bisection; P (n, K), mu/sd (n, m, K) -> (len(levels), n, m)."""
    lo = (mu - 10 * sd).min(-1); hi = (mu + 10 * sd).max(-1)
    out = []
    for q in levels:
        a, b = lo.copy(), hi.copy()
        for _ in range(60):
            mid = (a + b) / 2
            F = (P[:, None, :] * norm.cdf((mid[..., None] - mu) / sd)).sum(-1)
            a = np.where(F < q, mid, a); b = np.where(F < q, b, mid)
        out.append((a + b) / 2)
    return np.stack(out)


def make(kappa=1.0, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.choice(3, n, p=PRIOR)
    x = MEANS[z] + rng.normal(size=(n, 2))
    p_true = posterior(x)
    S = np.log(np.maximum(p_true, 1e-12)) * kappa; S -= S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)
    mu, sd = class_law(x, T)
    y = mu[np.arange(n), :, z] + sd[np.arange(n), :, z] * rng.normal(size=(n, len(T)))
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(n, int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, z)):
        fold[te] = k
    q = mixture_quantiles(P, mu, sd, [0.025, 0.16, 0.5, 0.84, 0.975])
    d = ROOT / "data" / f"synthetic_k{kappa:g}"; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(obj_id=np.arange(n), cls=z, y=z, fold=fold)).to_parquet(d / "meta.parquet")
    (d / "classes.json").write_text(json.dumps(["common", "medium", "rare"]))
    obj = np.repeat(np.arange(n), len(T)); tt = np.tile(T, n)
    pd.DataFrame(dict(obj=obj, t=tt, band=np.zeros(len(obj), int), dmag=y.ravel(), lead=tt)).to_parquet(d / "F_h0.parquet")
    flat = lambda A: A.reshape(n * len(T), -1) if A.ndim == 3 else A.ravel()
    np.savez_compressed(d / "oof_h0.npz", P=P, L=P.argmax(1), y=z, fold=fold, p_true=p_true,
                        mu=flat(q[2]), q025=flat(q[0]), q16=flat(q[1]), q84=flat(q[3]), q975=flat(q[4]),
                        hcf_mu=flat(mu), hcf_q16=flat(mu - sd), hcf_q84=flat(mu + sd))
    print(f"synthetic kappa={kappa}: n={n}, class counts {np.bincount(z)}, classifier accuracy {np.mean(P.argmax(1) == z):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.0]); ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in a.kappa:
        make(k, a.n, a.seed)
