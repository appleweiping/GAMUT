"""central-interval ablation, the interval-score variant of GAMUT (registered in the preregistered experiment plan before implementation).
Kept in its own module so that gamut/methods.py stays byte-identical to the code frozen for the held-out runs."""
from __future__ import annotations

import numpy as np

from .methods import (DIAG, EGRID, _apply_temperature, _class_thresholds, _horizon_cells, _residual_density_parts,
                      _split_ab, _temperature, adaptive_grid)


def gamut_q(task, cal, test, alpha, crc=False, n_iter=100, min_bin=150, bw_mult=2.0, unit="object", finite=True,
            prune=1e-3, seed=0, **_):
    """central-interval ablation: the interval-score variant (registered in the preregistered experiment plan before implementation).

    Same C1 (hypothesis-conditional forecaster with per-(class, lead-bin) kernel residual laws), C2 (temperature-scaled
    posterior) and C3 (dual fixed point equalising per-class object-level CRC thresholds, adaptive horizon cells) as
    GAMUT; only the score differs: s_v(x, y) = min(F_v(y|x), 1 - F_v(y|x)) with F_v the CDF of the dual-weighted
    posterior mixture sum_c v_c p_c(x) F_c(y|x) / sum_c v_c p_c(x). The band is the central interval
    [F_v^{-1}(t), F_v^{-1}(1 - t)]: a single interval whose misses split between the tails, which is what the interval
    score rewards. crc=True: weights and laws on half A, thresholds on half B (the certificate of the strict-split guarantee applies, since
    the score is again monotone in t)."""
    K = task.K; nb = int(task.lead_bin.max()) + 1
    cal_objs = np.unique(task.obj[cal])
    if crc:
        A, B = _split_ab(task, cal_objs, seed)
        inA = np.isin(task.obj[cal], A); calA, calB = cal[inA], cal[~inA]
    else:
        A, calA, calB = cal_objs, cal, cal
    HMU, HSG = task.hmu, task.hsig
    pi = np.bincount(task.y[A], minlength=K) / len(A)
    T = _temperature(task.P[A], task.y[A], K)
    post = lambda objs: _apply_temperature(task.P[objs], T)
    zA = task.y[task.obj[calA]]; bA_ = task.lead_bin[calA]
    step = EGRID[1] - EGRID[0]
    eA = (task.yt[calA][:, None] - HMU[calA]) / HSG[calA]
    cdf = np.zeros((K, nb, len(EGRID)))
    for c in range(K):
        mc = zA == c
        d_all = _residual_density_parts(eA[mc, c], bw_mult=bw_mult)[0]
        for b in range(nb):
            mb = mc & (bA_ == b)
            d = _residual_density_parts(eA[mb, c], bw_mult=bw_mult)[0] if mb.sum() >= min_bin else d_all
            cc = np.cumsum(d) * step
            cdf[c, b] = cc / cc[-1]

    def Fval(rows, Y):                                     # Y: (n, G) -> (n, G, K) class CDFs
        e = (Y[:, :, None] - HMU[rows][:, None, :]) / HSG[rows][:, None, :]
        idx = np.clip(((e - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1)
        out = cdf[np.arange(K)[None, None, :], task.lead_bin[rows][:, None, None], idx]
        out = np.where(e < EGRID[0], 0.0, np.where(e > EGRID[-1], 1.0, out))
        return out

    def score(F, Pr, v):                                   # F: (n, K) at the true y; central-interval score
        w = Pr * v[None, :]; Fv = (F * w).sum(1) / w.sum(1)
        return np.minimum(Fv, 1 - Fv)

    FA = Fval(calA, task.yt[calA][:, None])[:, 0, :]; PA = post(task.obj[calA])
    FB, PB = (FA, PA) if calB is calA else (Fval(calB, task.yt[calB][:, None])[:, 0, :], post(task.obj[calB]))
    gid = _horizon_cells(task.lead_bin[calA], task.obj[calA], zA, K, nb, int(np.ceil(2 / alpha)))
    bA, bB, lead_cell = gid[task.lead_bin[calA]], gid[task.lead_bin[calB]], gid[task.lead_bin]
    nmin = int(np.ceil(1 / alpha - 1 - 1e-9))
    V = np.ones((nb, K)); Tb = np.full(nb, np.nan)
    DIAG.clear(); DIAG.update(iterations=[], excluded=[], fallback_bins=[], temperature=float(T), cells=gid.tolist())
    for b in range(int(gid.max()) + 1):
        mA, mB = bA == b, bB == b
        if not mB.any():
            continue
        zb = task.y[task.obj[calA[mA]]]
        keep = np.array([len(np.unique(task.obj[calA[mA]][zb == c])) for c in range(K)]) >= nmin
        v = 1.0 / np.maximum(pi, 1e-6)
        for it in range(n_iter):
            tau = _class_thresholds(score(FA[mA], PA[mA], v), zb, task.obj[calA[mA]], K, alpha, unit, finite)
            ok = keep & np.isfinite(tau) & (tau > 0)
            if not ok.any():
                break
            ref = np.exp(np.mean(np.log(tau[ok])))
            v = v * np.clip(np.where(ok, (ref / np.where(ok, tau, ref)) ** 0.5, 1.0), 0.5, 2.0); v = v / np.exp(np.mean(np.log(v)))
            if np.max(np.abs(np.log(tau[ok] / ref))) < 0.005:
                break
        DIAG["iterations"].append(it + 1)
        zbB = task.y[task.obj[calB[mB]]]
        tB = _class_thresholds(score(FB[mB], PB[mB], v), zbB, task.obj[calB[mB]], K, alpha, unit, finite)
        nB = np.array([len(np.unique(task.obj[calB[mB]][zbB == c])) for c in range(K)])
        okB = (nB >= nmin) & np.isfinite(tB)
        DIAG["excluded"] += [(int(b), int(c), int(nB[c])) for c in range(K) if 0 < nB[c] < nmin]
        V[gid == b] = v; Tb[gid == b] = np.min(tB[okB]) if okB.any() else np.nanmin(tB)
    if np.isnan(Tb).any():
        good = np.flatnonzero(~np.isnan(Tb))
        for b in np.flatnonzero(np.isnan(Tb)):
            j = good[np.argmin(np.abs(good - b))]; V[b] = V[j]; Tb[b] = Tb[j]
    bt = lead_cell[test]; Pt = post(task.obj[test])
    Ft = Fval(test, task.yt[test][:, None])[:, 0, :]
    w = Pt * V[bt]; Fv_true = (Ft * w).sum(1) / w.sum(1)
    t_hat = Tb[bt]
    cov = np.minimum(Fv_true, 1 - Fv_true) >= t_hat
    lo = np.full(len(test), np.nan); hi = np.full(len(test), np.nan)
    for s0 in range(0, len(test), 1000):
        sl = slice(s0, s0 + 1000); rows = test[sl]; Pr = Pt[sl]
        act = Pr >= prune
        lo_g = np.where(act, HMU[rows] - 8 * HSG[rows], np.inf).min(1); hi_g = np.where(act, HMU[rows] + 8 * HSG[rows], -np.inf).max(1)
        Y, _ = adaptive_grid(lo_g, hi_g, HMU[rows], HSG[rows])
        ww = Pr * V[bt[sl]]
        Fv = (Fval(rows, Y) * ww[:, None, :]).sum(2) / ww.sum(1)[:, None]
        Fv = np.maximum.accumulate(Fv, axis=1)                                  # numerical monotonicity
        t = t_hat[sl]
        for i in range(len(rows)):
            if not np.isfinite(t[i]) or t[i] <= 0:
                lo[s0 + i], hi[s0 + i] = Y[i, 0], Y[i, -1]; continue
            if t[i] > 0.5:
                continue                                                         # empty band
            lo[s0 + i] = np.interp(t[i], Fv[i], Y[i]); hi[s0 + i] = np.interp(1 - t[i], Fv[i], Y[i])
    length = np.where(np.isfinite(lo) & np.isfinite(hi), hi - lo, 0.0)
    return cov, lo, hi, length


VARIANTS = {"central-interval ablation": gamut_q}