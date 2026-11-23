"""Oracle band of Prop. 1 on the synthetic benchmark (true posterior, true Gaussian class laws) and the Prop. 2 check.

1) Oracle: dual weights v* found by the fixed point of Eq. (dual) on a fresh Monte Carlo sample of 200,000 objects
   (population thresholds, no finite-sample correction), then the oracle band length on the benchmark's objects.
2) Prop. 2: for every single-interval method in results/synthetic_k*/points_h0_a*.npz, the class coverage predicted
   from the analytic decomposition  P(M | Z=c) = E[p_c M]/pi_c + E[Cov(1{Z=c}, M | X)]/pi_c  (true p, true laws)
   against the empirical class coverage.
Output: results/synthetic_oracle.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.synthetic import MEANS, PRIOR, T, class_law, posterior  # noqa: E402


def dual(Tm, z, alpha, iters=400):
    """Tm: (n, K) sbar p_c f_c at the true value; v equalizing the per-class alpha-quantiles (population sample)."""
    K = Tm.shape[1]
    v = 1 / PRIOR.copy()
    for _ in range(iters):
        tau = np.array([np.quantile((Tm @ v)[z == c], alpha) for c in range(K)])
        ref = np.exp(np.mean(np.log(tau)))
        v *= np.clip((ref / tau) ** 0.5, 0.5, 2.0); v /= np.exp(np.mean(np.log(v)))
    tau = np.array([np.quantile((Tm @ v)[z == c], alpha) for c in range(K)])
    return v, tau.min(), tau


def band_length(v, t, p, mu, sd, G=1200):
    sbar = (p * sd).sum(1)
    lo = (mu - 8 * sd).min(1); hi = (mu + 8 * sd).max(1)
    Y = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, G)[None]
    S = np.zeros_like(Y)
    for c in range(len(v)):
        S += v[c] * p[:, c][:, None] * norm.pdf(Y, mu[:, c][:, None], sd[:, c][:, None])
    return (S * sbar[:, None] >= t).sum(1) * (hi - lo) / (G - 1)


def main():
    out = {}
    rng = np.random.default_rng(123)
    n = 200_000
    z = rng.choice(3, n, p=PRIOR); x = MEANS[z] + rng.normal(size=(n, 2))
    t_lead = rng.choice(T, n)
    p = posterior(x)
    from benchmarks.synthetic import SLOPE, SCALE
    mu = SLOPE[None, :] * x[:, 0:1] * (t_lead[:, None] / 100)
    sd = SCALE[None, :] * (0.3 + t_lead[:, None] / 100) * np.ones((n, 1))
    y = mu[np.arange(n), z] + sd[np.arange(n), z] * rng.normal(size=n)
    edges = np.linspace(0, 100, 7)[1:-1]                      # the benchmark's lead-time bins (h = 0)
    bins = np.searchsorted(edges, t_lead, side="right")
    sbar = (p * sd).sum(1)
    for alpha in (0.32, 0.05):
        Tm = p * norm.pdf(y[:, None], mu, sd) * sbar[:, None]
        L = np.full(n, np.nan); cov = {c: [] for c in range(3)}; V = {}
        evalpts = np.random.default_rng(7).choice(n, 24000, replace=False)
        for b in np.unique(bins):
            m = bins == b
            v, t, tau = dual(Tm[m], z[m], alpha)
            V[int(b)] = v.tolist()
            sub = evalpts[m[evalpts]]
            L[sub] = band_length(v, t, p[sub], mu[sub], sd[sub])
            s_ = Tm[m] @ v
            for c in range(3):
                cov[c].append(float(np.mean(s_[z[m] == c] >= t)))
        used = evalpts
        out[f"oracle_a{alpha:g}"] = dict(v=V, mean_length=float(L[used].mean()),
                                         class_coverage={str(c): float(np.mean(cov[c])) for c in range(3)})
        print(f"alpha={alpha}: oracle (per-bin, scale-normalized) length {L[used].mean():.3f}, class coverage", {c: round(np.mean(cov[c]), 3) for c in range(3)})
    # Prop. 2 check on the benchmark's own points
    for kappa in ("1", "0.5", "2"):
        d = ROOT / "data" / f"synthetic_k{kappa}"
        zz = np.load(d / "oof_h0.npz")
        ptrue = zz["p_true"]; zobj = zz["y"]
        import pandas as pd
        F = pd.read_parquet(d / "F_h0.parquet")
        obj = F.obj.to_numpy(); yt = F.dmag.to_numpy()
        hm = zz["hcf_mu"]; hs = (zz["hcf_q84"] - zz["hcf_q16"]) / 2
        for alpha in (0.32, 0.05):
            pf = ROOT / "results" / f"synthetic_k{kappa}" / f"points_h0_a{alpha:g}.npz"
            if not pf.exists():
                continue
            P = np.load(pf)
            res = {}
            for key in ("FLARE", "SplitCP", "GCC-prob", "MixtureCP", "LocalizedCP", "CQR"):
                if f"{key}__lo" not in P:
                    continue
                lo, hi = P[f"{key}__lo"], P[f"{key}__hi"]
                # m_c(x) = P(miss | x, Z=c) under the true class law at each point
                mc = 1 - (norm.cdf((hi[:, None] - hm) / hs) - norm.cdf((lo[:, None] - hm) / hs))      # (points, K)
                pc = ptrue[obj]
                mbar = (pc * mc).sum(1)
                emp = {}; pred = {}; term1 = {}
                for c in range(3):
                    sel = zobj[obj] == c
                    emp[c] = float(1 - np.mean(~P[f"{key}__cov"][sel]))
                    pi_c = pc[:, c].mean()
                    term1[c] = float(np.mean(pc[:, c] * mbar) / pi_c)
                    gap = float(np.mean(pc[:, c] * (mc[:, c] - mbar)) / pi_c)
                    pred[c] = float(1 - term1[c] - gap)
                res[key] = dict(empirical=emp, predicted=pred, weighted_miss=term1)
            out[f"prop2_k{kappa}_a{alpha:g}"] = res
    (ROOT / "results" / "synthetic_oracle.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
