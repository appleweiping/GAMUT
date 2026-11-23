"""Interval-forecast calibration methods: external baselines and GAMUT.

Every method is calibrated on the out-of-fold outputs of the calibration objects (the other folds) and produces, for the
evaluated future points, the hull [lo, hi] of its band, the total length of the band (a union of intervals for GAMUT) and
whether each true value is covered. Nonconformity of a pooled forecast: r = |y - mu| / sigma with FLARE's scale
sigma = max(|q84 - q16| / 2, 0.05)."""
from __future__ import annotations

import numpy as np

MIN_N = 20                                      # FLARE: Mondrian cells with fewer points fall back
GAMUT_VERSION = "v8"                            # v8: residual-law support +-60 scales (was +-12); adaptive evaluation grid
GRID_VERSION = "g2"                             # grid-based baselines (Kandinsky CP, MDCP) with the same adaptive grid
DIAG = {}                                       # diagnostics of the last gamut() call (iterations, excluded cells)


def conf_quantile(scores, alpha, weights=None):
    """Finite-sample (1 - alpha) conformal quantile; +inf when unattainable. With weights (one test unit of weight 1)."""
    s = np.asarray(scores, float)
    if len(s) == 0:
        return np.inf
    if weights is None:
        v = np.sort(s); k = int(np.ceil((len(v) + 1) * (1 - alpha)))
        return float(v[k - 1]) if k <= len(v) else np.inf
    o = np.argsort(s); s, w = s[o], np.asarray(weights, float)[o]
    c = np.cumsum(w) / (w.sum() + 1.0)
    k = int(np.searchsorted(c, 1 - alpha, side="left"))
    return float(s[k]) if k < len(s) else np.inf


def _cells(*keys):
    """Integer cell id for combinations of integer keys."""
    key = np.zeros(len(keys[0]), np.int64)
    for k in keys:
        key = key * 1000 + np.asarray(k, np.int64)
    return key


def mondrian_quantile(scores, cell, alpha, fallback_cell=None, min_n=MIN_N):
    """dict cell -> conformal quantile, with a fallback dict for cells under min_n."""
    out = {}
    for c in np.unique(cell):
        m = cell == c
        out[c] = conf_quantile(scores[m], alpha) if m.sum() >= min_n else None
    fb = {}
    if fallback_cell is not None:
        for c in np.unique(fallback_cell):
            fb[c] = conf_quantile(scores[fallback_cell == c], alpha)
    return out, fb


def _lookup(qmap, fbmap, cell, fb_cell, glob):
    v = np.array([qmap.get(c) if qmap.get(c) is not None else np.nan for c in cell], float)
    if fb_cell is not None:
        fbv = np.array([fbmap.get(c, glob) for c in fb_cell], float)
        v = np.where(np.isfinite(v), v, fbv)
    return np.where(np.isfinite(v), v, glob)


def _band(task, rows, center, half):
    lo, hi = center - half, center + half
    cov = (task.yt[rows] >= lo) & (task.yt[rows] <= hi)
    return cov, lo, hi, hi - lo


LOCAL = np.linspace(-3.0, 3.0, 49)              # standardized offsets added around every hypothesis centre


def adaptive_grid(lo_g, hi_g, centers, scales, G=400):
    """Per-row irregular grid: G uniform points on [lo_g, hi_g] plus 49 points within +-3 scales of every hypothesis
    centre, so that bands much narrower than the global range (e.g. around a point mass) are always resolved.
    Returns sorted values Y (n, G') and midpoint-rule cell widths W (n, G')."""
    U = lo_g[:, None] + (hi_g - lo_g)[:, None] * np.linspace(0, 1, G)[None, :]
    Lc = (centers[:, :, None] + scales[:, :, None] * LOCAL[None, None, :]).reshape(len(lo_g), -1)
    Y = np.sort(np.concatenate([U, Lc], 1), 1)
    gaps = np.diff(Y, axis=1)
    W = np.zeros_like(Y); W[:, 1:] += gaps / 2; W[:, :-1] += gaps / 2
    return Y, W


def band_from_inside(Y, W, inside):
    """Hull (midpoints to the neighbouring grid values) and total length of the grid set."""
    any_in = inside.any(1); r = np.arange(len(Y)); G = Y.shape[1]
    first = inside.argmax(1); last = G - 1 - inside[:, ::-1].argmax(1)
    lo = np.where(any_in, Y[r, first] - (Y[r, first] - Y[r, np.maximum(first - 1, 0)]) / 2, np.nan)
    hi = np.where(any_in, Y[r, last] + (Y[r, np.minimum(last + 1, G - 1)] - Y[r, last]) / 2, np.nan)
    return lo, hi, (inside * W).sum(1)


def r_pooled(task, rows):
    return np.abs(task.yt[rows] - task.mu[rows]) / task.sig[rows]


# ---------------------------------------------------------------- external baselines
def split_cp(task, cal, test, alpha, **_):
    """Split conformal with normalized residuals (Lei et al. 2018; Papadopoulos et al. 2002): one global multiplier."""
    m = conf_quantile(r_pooled(task, cal), alpha)
    return _band(task, test, task.mu[test], m * task.sig[test])


def flare_mondrian(task, cal, test, alpha, **_):
    """The deployed FLARE bands: multiplier per (PREDICTED class = argmax probability, band, lead bin), fallback to
    (band, lead bin) for cells under 20 points (cross-conformal Mondrian by predicted class)."""
    pc = task.P.argmax(1)
    r = r_pooled(task, cal)
    cc = _cells(pc[task.obj[cal]], task.band[cal], task.lead_bin[cal]); fc = _cells(task.band[cal], task.lead_bin[cal])
    q, fb = mondrian_quantile(r, cc, alpha, fc)
    glob = conf_quantile(r, alpha)
    m = _lookup(q, fb, _cells(pc[task.obj[test]], task.band[test], task.lead_bin[test]), _cells(task.band[test], task.lead_bin[test]), glob)
    return _band(task, test, task.mu[test], m * task.sig[test])


def oracle_true_class(task, cal, test, alpha, **_):
    """Mondrian by the TRUE class (unattainable: needs the test label); reference for class-conditional validity."""
    r = r_pooled(task, cal)
    cc = _cells(task.y[task.obj[cal]], task.band[cal], task.lead_bin[cal]); fc = _cells(task.band[cal], task.lead_bin[cal])
    q, fb = mondrian_quantile(r, cc, alpha, fc)
    glob = conf_quantile(r, alpha)
    m = _lookup(q, fb, _cells(task.y[task.obj[test]], task.band[test], task.lead_bin[test]), _cells(task.band[test], task.lead_bin[test]), glob)
    return _band(task, test, task.mu[test], m * task.sig[test])


def _qpair(task, alpha):
    return ("q16", "q84") if alpha > 0.1 else ("q025", "q975")


def cqr(task, cal, test, alpha, **_):
    """Conformalized quantile regression (Romano, Patterson & Candes 2019)."""
    a, b = _qpair(task, alpha)
    lo_c, hi_c = np.minimum(task.q[a][cal], task.q[b][cal]), np.maximum(task.q[a][cal], task.q[b][cal])
    E = np.maximum(lo_c - task.yt[cal], task.yt[cal] - hi_c)
    Q = conf_quantile(E, alpha)
    lo, hi = np.minimum(task.q[a][test], task.q[b][test]) - Q, np.maximum(task.q[a][test], task.q[b][test]) + Q
    cov = (task.yt[test] >= lo) & (task.yt[test] <= hi)
    return cov, lo, hi, hi - lo


def _cdf_knots(task, rows):
    """Monotone quantile knots (levels, values) from the five forecast quantiles (median = L1 point forecast)."""
    lv = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    V = np.column_stack([task.q["q025"][rows], task.q["q16"][rows], task.mu[rows], task.q["q84"][rows], task.q["q975"][rows]])
    V = np.maximum.accumulate(np.sort(V, axis=1), axis=1)
    return lv, V


def _cdf(lv, V, y):
    """Piecewise-linear CDF through the knots, linear tails to levels 0 and 1 at +-2x the outer spacing."""
    lo_ext = V[:, 0] - 2 * (V[:, 1] - V[:, 0] + 1e-6); hi_ext = V[:, -1] + 2 * (V[:, -1] - V[:, -2] + 1e-6)
    VV = np.column_stack([lo_ext, V, hi_ext]); LV = np.r_[0.0, lv, 1.0]
    out = np.empty(len(y))
    for i in range(len(y)):
        out[i] = np.interp(y[i], VV[i], LV)
    return out, VV, LV


def dcp(task, cal, test, alpha, **_):
    """Distributional conformal prediction (Chernozhukov, Wuthrich & Zhu 2021) on the forecaster's quantile CDF."""
    lv, V = _cdf_knots(task, cal)
    Fc, _, _ = _cdf(lv, V, task.yt[cal])
    tau = conf_quantile(np.abs(Fc - 0.5), alpha)
    tau = min(tau, 0.5)
    lvt, Vt = _cdf_knots(task, test)
    Ft, VV, LV = _cdf(lvt, Vt, task.yt[test])
    cov = np.abs(Ft - 0.5) <= tau
    lo = np.array([np.interp(0.5 - tau, LV, VV[i]) for i in range(len(test))])
    hi = np.array([np.interp(0.5 + tau, LV, VV[i]) for i in range(len(test))])
    return cov, lo, hi, hi - lo


def localized_cp(task, cal, test, alpha, **_):
    """Localized conformal prediction (Guan 2023) with a Gaussian kernel in the classifier-probability space, per lead bin,
    bandwidth = median pairwise distance of calibration objects (fixed rule)."""
    r = r_pooled(task, cal)
    cal_obj = np.unique(task.obj[cal]); Pc = task.P[cal_obj]
    rng = np.random.default_rng(0)
    sub = Pc[rng.choice(len(Pc), min(500, len(Pc)), replace=False)]
    d = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)); bw = np.median(d[d > 0]) if (d > 0).any() else 1.0
    pos = {o: i for i, o in enumerate(cal_obj)}
    cal_oi = np.array([pos[o] for o in task.obj[cal]])
    m = np.full(len(test), np.nan)
    test_obj = task.obj[test]
    for g in np.unique(task.lead_bin[test]):
        cm = task.lead_bin[cal] == g
        if cm.sum() < MIN_N:
            cm = np.ones(len(cal), bool)
        rs = r[cm]; order = np.argsort(rs); rs = rs[order]; oi = cal_oi[cm][order]
        tm = np.flatnonzero(task.lead_bin[test] == g)
        uo = np.unique(test_obj[tm])
        for s in range(0, len(uo), 64):
            chunk = uo[s:s + 64]
            W = np.exp(-((task.P[chunk][:, None, :] - Pc[None, :, :]) ** 2).sum(-1) / (2 * bw ** 2))   # (c, n_cal_obj)
            Wp = W[:, oi]                                                                              # (c, points)
            cs = np.cumsum(Wp, axis=1) / (Wp.sum(1, keepdims=True) + W.max(1, keepdims=True))       # test mass = max kernel
            k = (cs < 1 - alpha).sum(1)
            q = np.where(k < len(rs), rs[np.minimum(k, len(rs) - 1)], np.inf)
            for ci, o in enumerate(chunk):
                m[tm[test_obj[tm] == o]] = q[ci]
    return _band(task, test, task.mu[test], m * task.sig[test])


def _class_cdf_grid(task, cal, grid):
    """Per (class, lead bin) empirical CDF of r on a grid, with the (n+1) finite-sample correction."""
    r = r_pooled(task, cal); z = task.y[task.obj[cal]]; g = task.lead_bin[cal]
    nb = int(task.lead_bin.max()) + 1
    Fz = np.zeros((task.K, nb, len(grid)))
    for c in range(task.K):
        for b in range(nb):
            m = (z == c) & (g == b)
            if m.sum() < MIN_N:
                m = z == c
            v = np.sort(r[m]); Fz[c, b] = np.searchsorted(v, grid, side="right") / (len(v) + 1)
    return Fz


def mixture_cp(task, cal, test, alpha, **_):
    """Probability-weighted mixture of the class-wise conformal score distributions: conformal prediction under label shift
    (Podkopaev & Ramdas 2021) with the test object's posterior as target label distribution; the professor's 'band built
    as a mixture of the class-wise bands weighted by the classifier probabilities'."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(60), 600))
    Fz = _class_cdf_grid(task, cal, grid)
    Pt = task.P[task.obj[test]]                                         # (m, K)
    Fm = np.einsum("mk,mkg->mg", Pt, Fz[:, task.lead_bin[test], :].transpose(1, 0, 2))
    k = (Fm < 1 - alpha).sum(1)
    m = np.where(k < len(grid), grid[np.minimum(k, len(grid) - 1)], np.inf)
    return _band(task, test, task.mu[test], m * task.sig[test])


def _qr(Phi, s, level, max_n=15000, seed=0):
    from sklearn.linear_model import QuantileRegressor
    if len(s) > max_n:
        idx = np.random.default_rng(seed).choice(len(s), max_n, replace=False); Phi, s = Phi[idx], s[idx]
    for solver, pen in (("highs", 0.0), ("highs-ipm", 0.0), ("highs", 1e-6)):     # fall back only if HiGHS fails
        try:
            m = QuantileRegressor(quantile=level, alpha=pen, fit_intercept=False, solver=solver).fit(Phi, s)
            return m.coef_
        except (TypeError, ValueError):
            continue
    raise RuntimeError("quantile regression failed with every solver")


def _design_p(task, rows, nb):
    P = task.P[task.obj[rows]]
    B = np.eye(nb)[task.lead_bin[rows]][:, 1:]
    return np.column_stack([np.ones(len(rows)), P[:, :-1], B, task.band[rows]])


def gcc_prob(task, cal, test, alpha, **_):
    """Conformal prediction with conditional guarantees (Gibbs, Cherian & Candes 2025), unaugmented quantile-regression
    form, with the function class spanned by the classifier probabilities, lead-bin indicators and band."""
    nb = int(task.lead_bin.max()) + 1
    beta = _qr(_design_p(task, cal, nb), r_pooled(task, cal), 1 - alpha)
    m = np.maximum(_design_p(task, test, nb) @ beta, 1e-3)
    return _band(task, test, task.mu[test], m * task.sig[test])


def partition_cp(task, cal, test, alpha, n_groups=8, **_):
    """Mondrian conformal over a learned partition of the classifier-probability space (k-means on calibration objects'
    probabilities), in the spirit of conditional coverage via learned partitions (Kiyani, Pappas & Hassani 2024)."""
    from sklearn.cluster import KMeans
    cal_obj = np.unique(task.obj[cal])
    km = KMeans(n_groups, n_init=4, random_state=0).fit(np.sqrt(task.P[cal_obj]))
    grp = km.predict(np.sqrt(task.P))
    r = r_pooled(task, cal)
    cc = _cells(grp[task.obj[cal]], task.lead_bin[cal]); fc = _cells(task.lead_bin[cal])
    q, fb = mondrian_quantile(r, cc, alpha, fc)
    glob = conf_quantile(r, alpha)
    m = _lookup(q, fb, _cells(grp[task.obj[test]], task.lead_bin[test]), _cells(task.lead_bin[test]), glob)
    return _band(task, test, task.mu[test], m * task.sig[test])


# ---------------------------------------------------------------- early design (union of class bands), kept for the record
def _split_ab(task, cal_objs, seed):
    from sklearn.model_selection import train_test_split
    y = task.y[cal_objs]
    strat = y if np.bincount(y, minlength=task.K).min() >= 2 else None
    A, B = train_test_split(cal_objs, test_size=0.5, stratify=strat, random_state=seed)
    return np.sort(A), np.sort(B)


def _union(lo, hi, member):
    """Hull, total length and point-wise membership of the union of the selected intervals (per row)."""
    lo_m = np.where(member, lo, np.inf); hi_m = np.where(member, hi, -np.inf)
    hull_lo, hull_hi = lo_m.min(1), hi_m.max(1)
    order = np.argsort(lo_m, axis=1)
    L = np.take_along_axis(lo_m, order, 1); H = np.take_along_axis(hi_m, order, 1)
    length = np.zeros(len(lo)); cur_lo = L[:, 0].copy(); cur_hi = H[:, 0].copy()
    for j in range(1, L.shape[1]):
        valid = np.isfinite(L[:, j])
        new = valid & (L[:, j] > cur_hi)
        length += np.where(new, cur_hi - cur_lo, 0.0)
        cur_lo = np.where(new, L[:, j], cur_lo)
        cur_hi = np.where(valid & ~new, np.maximum(cur_hi, H[:, j]), np.where(new, H[:, j], cur_hi))
    length += cur_hi - cur_lo
    return hull_lo, hull_hi, length


def gamut_union(task, cal, test, alpha, hcf=True, screen="conformal", crc=True, adapt=True, beta=None, joint=True, seed=0, **_):
    """Early design (v0, superseded by gamut): hypothesis-conditional forecasts (C1), conformal screening of plausible classes (C2) and screening-aware
    class-conditional calibration by per-class conformal risk control (C3).

    For each class c: residual under hypothesis c, r_c = |y - mu_c| / sigma_c. On calibration half A: the class set
    Gamma_beta(x) = {c : 1 - p_c(x) <= q_c} with q_c the class-conditional (1 - beta) conformal quantile of 1 - p_c over
    class-c objects, and a base multiplier m_c(x) for class c (conditional quantile regression of r_c on p_c(x) and the
    lead bin over class-c points when adapt=True, else a per-lead-bin conformal quantile). On half B, per class, conformal
    risk control (Angelopoulos et al. 2024) of the object-level loss l_i(lam) = fraction of object i's future points not
    covered by [mu_c +- lam m_c sigma_c] or with c outside Gamma(x_i), which depends on class c's own lam only, so that
    E[l(lam_c) | Z = c] <= alpha for every class. The band is the union over c in Gamma(x) of the class intervals."""
    beta = alpha / 2 if beta is None else beta
    K = task.K; nb = int(task.lead_bin.max()) + 1
    cal_objs = np.unique(task.obj[cal]); A, B = _split_ab(task, cal_objs, seed)
    inA = np.isin(task.obj[cal], A); calA, calB = cal[inA], cal[~inA]
    HMU = task.hmu if hcf else np.repeat(task.mu[:, None], K, 1)
    HSG = task.hsig if hcf else np.repeat(task.sig[:, None], K, 1)
    # C2: class-conditional conformal class sets (thresholds on A)
    qcls = np.ones(K)
    if screen == "conformal":
        for c in range(K):
            oc = A[task.y[A] == c]
            qcls[c] = min(conf_quantile(1 - task.P[oc, c], beta), 1.0) if len(oc) else 1.0

    def gamma(objs, qc):
        if screen == "all":
            return np.ones((len(objs), K), bool)
        if screen == "argmax":
            G = np.zeros((len(objs), K), bool); G[np.arange(len(objs)), task.P[objs].argmax(1)] = True; return G
        G = (1 - task.P[objs]) <= qc[None, :]
        G[np.arange(len(objs)), task.P[objs].argmax(1)] = True             # never empty
        return G

    # C3a: base multipliers on A (true-class hypothesis residuals)
    def design(rows, c):
        Bm = np.eye(nb)[task.lead_bin[rows]][:, 1:]
        return np.column_stack([np.ones(len(rows)), task.P[task.obj[rows], c], Bm])

    coefs, binq = {}, {}
    level_a = 1 - (alpha - beta) if not crc else 1 - alpha
    for c in range(K):
        rc = calA[task.y[task.obj[calA]] == c]
        r = np.abs(task.yt[rc] - HMU[rc, c]) / HSG[rc, c]
        if adapt and len(rc) >= 5 * (nb + 2):
            coefs[c] = _qr(design(rc, c), r, level_a, max_n=15000, seed=seed + c)
        cells = task.lead_bin[rc]
        binq[c] = {b: (conf_quantile(r[cells == b], 1 - level_a) if (cells == b).sum() >= MIN_N else None) for b in range(nb)}
        binq[c]["all"] = conf_quantile(r, 1 - level_a) if len(r) else np.inf

    def base_mult(rows, c):
        if c in coefs:
            return np.maximum(design(rows, c) @ coefs[c], 0.05)
        v = np.array([binq[c].get(b) if binq[c].get(b) is not None else binq[c]["all"] for b in task.lead_bin[rows]], float)
        return v

    # C3b: per-class conformal risk control on B
    lam = np.ones(K)
    grid = np.exp(np.linspace(np.log(0.3), np.log(8.0), 160))
    if crc and joint:
        # composite loss: the whole union band with one global scale lam on every class component; the band is
        # monotone in lam, so lam* = max over classes of the class-wise CRC solutions is valid for every class
        rowsB = calB
        GB = gamma(task.obj[rowsB], qcls)
        baseB = np.column_stack([base_mult(rowsB, c) for c in range(K)]) * HSG[rowsB]      # (points, K) half-width at lam=1
        dist = np.abs(task.yt[rowsB][:, None] - HMU[rowsB])                                 # (points, K)
        need = np.where(GB, dist / baseB, np.inf).min(1)                                    # smallest lam covering the point
        zb = task.y[task.obj[rowsB]]
        lam_c = np.full(K, grid[0])
        for c in range(K):
            m = zb == c
            if not m.any():
                continue
            uo, inv = np.unique(task.obj[rowsB][m], return_inverse=True); npts = np.bincount(inv)
            miss = need[m][:, None] > grid[None, :]
            Lobj = np.zeros((len(uo), len(grid))); np.add.at(Lobj, inv, miss.astype(float)); Lobj /= npts[:, None]
            risk = (Lobj.sum(0) + 1.0) / (len(uo) + 1.0)
            ok = np.flatnonzero(risk <= alpha)
            lam_c[c] = grid[ok[0]] if len(ok) else grid[-1]
        lam[:] = lam_c.max()
    elif crc:
        for c in range(K):
            ob = B[task.y[B] == c]
            if len(ob) == 0:
                lam[c] = grid[-1]; continue
            rows = calB[task.y[task.obj[calB]] == c]
            inG = gamma(task.obj[rows], qcls)[:, c]
            r = np.abs(task.yt[rows] - HMU[rows, c]) / HSG[rows, c]
            ratio = r / base_mult(rows, c)
            uo, inv = np.unique(task.obj[rows], return_inverse=True); npts = np.bincount(inv)
            feasible = None
            for attempt in range(2):
                miss = (~inG)[:, None] | (ratio[:, None] > grid[None, :])                 # (points, grid)
                Lobj = np.zeros((len(uo), len(grid)))
                np.add.at(Lobj, inv, miss.astype(float)); Lobj /= npts[:, None]
                n = len(uo)
                risk = (Lobj.sum(0) + 1.0) / (n + 1.0)
                ok = np.flatnonzero(risk <= alpha)
                if len(ok):
                    feasible = grid[ok[0]]; break
                qcls[c] = 1.0; inG = np.ones(len(rows), bool)                            # screening infeasible: always keep c
            lam[c] = feasible if feasible is not None else grid[-1]
    # prediction
    G = gamma(task.obj[test], qcls)
    lo = np.empty((len(test), K)); hi = np.empty((len(test), K))
    for c in range(K):
        half = lam[c] * base_mult(test, c) * HSG[test, c]
        lo[:, c] = HMU[test, c] - half; hi[:, c] = HMU[test, c] + half
    yv = task.yt[test][:, None]
    cov = (G & (yv >= lo) & (yv <= hi)).any(1)
    hull_lo, hull_hi, length = _union(lo, hi, G)
    return cov, hull_lo, hull_hi, length


METHODS = {
    "SplitCP": split_cp, "FLARE": flare_mondrian, "CQR": cqr, "DCP": dcp, "LocalizedCP": localized_cp,
    "MixtureCP": mixture_cp, "GCC-prob": gcc_prob, "PartitionCP": partition_cp,
    "Oracle-true-class": oracle_true_class,
}


# ---------------------------------------------------------------- GAMUT (coverage-constrained, dual-weighted)
EGRID = np.linspace(-60.0, 60.0, 4801)                  # standardized residual grid (step 0.025); wide support for heavy tails


def _residual_density(e, bw=None, bw_mult=1.0):
    """Smoothed density of standardized residuals on EGRID (histogram + Gaussian kernel, Silverman bandwidth)."""
    e = np.clip(e[np.isfinite(e)], EGRID[0], EGRID[-1])
    if len(e) < 5:
        e = np.r_[e, np.random.default_rng(0).normal(0, 1, 50)]
    step = EGRID[1] - EGRID[0]
    hist = np.bincount(np.clip(((e - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1), minlength=len(EGRID)).astype(float)
    s = min(np.std(e), (np.quantile(e, .75) - np.quantile(e, .25)) / 1.349) or np.std(e) or 1.0
    bw = bw or max(bw_mult * 1.06 * s * len(e) ** -0.2, 2 * step)
    k = np.arange(-int(4 * bw / step), int(4 * bw / step) + 1) * step
    ker = np.exp(-0.5 * (k / bw) ** 2); ker /= ker.sum()
    d = np.convolve(hist, ker, mode="same")
    d = d / (d.sum() * step)
    return np.maximum(d, 1e-9)


def _residual_density_parts(e, bw_mult=1.0):
    """As _residual_density, also returning the kernel (on EGRID steps) and the normalizer, for leave-one-object-out
    corrections: density = conv / norm with conv the kernel-smoothed histogram."""
    e = np.clip(e[np.isfinite(e)], EGRID[0], EGRID[-1])
    step = EGRID[1] - EGRID[0]
    if len(e) < 5:
        return _residual_density(e, bw_mult=bw_mult), None, None, len(e)
    hist = np.bincount(np.clip(((e - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1), minlength=len(EGRID)).astype(float)
    sd = min(np.std(e), (np.quantile(e, .75) - np.quantile(e, .25)) / 1.349) or np.std(e) or 1.0
    bw = max(bw_mult * 1.06 * sd * len(e) ** -0.2, 2 * step)
    k = np.arange(-int(4 * bw / step), int(4 * bw / step) + 1) * step
    ker = np.exp(-0.5 * (k / bw) ** 2); ker /= ker.sum()
    conv = np.convolve(hist, ker, mode="same")
    norm = conv.sum() * step
    return np.maximum(conv / norm, 1e-9), ker, norm, len(e)


def _temperature(P, z, K):
    """C2 membership calibration: temperature T minimizing the NLL of the true classes of the calibration objects."""
    from scipy.optimize import minimize_scalar
    lp = np.log(np.maximum(P, 1e-8))

    def nll(logT):
        S = lp / np.exp(logT); S = S - S.max(1, keepdims=True)
        return float(-(S[np.arange(len(z)), z] - np.log(np.exp(S).sum(1))).mean())
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def _apply_temperature(P, T):
    S = np.log(np.maximum(P, 1e-8)) / T; S -= S.max(1, keepdims=True); E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def _class_thresholds(s, z, obj, K, alpha, unit, finite):
    """Per-class calibration thresholds of the score s (cover iff s >= t): for class c, the largest t whose
    class-c miscoverage risk is <= alpha. unit='point': each future point one unit; unit='object': each object one
    unit with loss = its fraction of missed points (conformal risk control, Angelopoulos et al. 2024). finite=True
    applies the (n_c + 1) finite-sample correction with n_c = number of class-c OBJECTS (the exchangeable units)."""
    t = np.full(K, np.nan)
    for c in range(K):
        m = z == c
        if not m.any():
            continue
        sc = s[m]; oc = obj[m]
        uo, inv, cnt = np.unique(oc, return_inverse=True, return_counts=True)
        n = len(uo)
        w = 1.0 / cnt[inv] if unit == "object" else np.full(len(sc), n / len(sc))   # weights sum to n (objects)
        budget = alpha * (n + 1) - 1 if finite else alpha * n
        o = np.argsort(sc); cw = np.cumsum(w[o])
        k = int(np.searchsorted(cw, budget, side="right"))       # points o[:k] may be missed
        t[c] = sc[o[k]] if k < len(sc) else np.inf
        if budget < 0:
            t[c] = -np.inf
    return t


def _horizon_cells(lead_bin_cal, obj_cal, z_cal, K, nb, n_target):
    """Contiguous groups of lead-time bins such that every class with >= n_target calibration objects overall has
    >= n_target objects with points in each group (greedy merging of the poorest group with its smaller neighbour).
    Uses calibration labels and lead times only, never outcomes. Returns group id per lead bin."""
    groups = [[b] for b in range(nb)]
    tot = np.array([len(np.unique(obj_cal[z_cal == c])) for c in range(K)])
    big = np.flatnonzero(tot >= n_target)

    def counts(g):
        m = np.isin(lead_bin_cal, g)
        return np.array([len(np.unique(obj_cal[m & (z_cal == c)])) for c in big])
    while len(groups) > 1:
        mins = [counts(g).min() if len(big) else n_target for g in groups]
        i = int(np.argmin(mins))
        if mins[i] >= n_target:
            break
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if mins[i - 1] <= mins[i + 1] else i + 1
        a, b = sorted((i, j))
        groups[a] = groups[a] + groups[b]; del groups[b]
    gid = np.zeros(nb, int)
    for k, g in enumerate(groups):
        gid[g] = k
    return gid


def gamut(task, cal, test, alpha, hcf=True, posterior="calibrated", weights="dual", crc=False, prune=1e-3,
          n_iter=100, min_bin=150, unit="object", finite=True, bw_mult=2.0, scale_norm=True, lead_mondrian=True, loo=True,
          cells="auto", incl="min", seed=0, **_):
    """GAMUT (Guarantees for All Members Under Type uncertainty).

    C1 hypothesis-conditional forecaster: mu_c(x), sigma_c(x) for every class hypothesis c (class as model input).
    C2 calibrated membership: classifier posterior p_c(x), temperature-scaled on the calibration objects.
    C3 dual-weighted level-set calibration: the band minimizing the scale-normalized length E[|C(X)| / sbar(X)] under
    one coverage constraint per class is the level set {y : s_v(x, y) = sbar(x) sum_c v_c p_c(x) f_c(y | x) >= 1}
    with global dual weights v (Prop. 1), sbar(x) = sum_c p_c(x) sigma_c(x) the posterior-mean forecast scale
    (scale_norm=False: absolute length, sbar = 1, which abandons high-variance objects); f_c is the
    hypothesis-c forecast with the class-c residual law (per lead bin). The dual weights are found by a multiplicative
    fixed point that equalizes the per-class calibration thresholds; the band uses the smallest class threshold, so
    for fixed v every class is covered at 1 - alpha (per-class conformal risk control). crc=True fits v on one half
    of the calibration objects and the thresholds on the other half (exact finite-sample certificate)."""
    K = task.K; nb = int(task.lead_bin.max()) + 1
    cal_objs = np.unique(task.obj[cal])
    if crc:
        A, B = _split_ab(task, cal_objs, seed)
        inA = np.isin(task.obj[cal], A); calA, calB = cal[inA], cal[~inA]
    else:
        A, calA, calB = cal_objs, cal, cal
    HMU = task.hmu if hcf else np.repeat(task.mu[:, None], K, 1)
    HSG = task.hsig if hcf else np.repeat(task.sig[:, None], K, 1)
    pi = np.bincount(task.y[A], minlength=K) / len(A)
    T = _temperature(task.P[A], task.y[A], K) if posterior == "calibrated" else 1.0

    def post(objs):
        if posterior == "uniform":
            return np.repeat(pi[None, :], len(objs), 0)
        if posterior == "onehot":
            Pm = np.zeros((len(objs), K)); Pm[np.arange(len(objs)), task.P[objs].argmax(1)] = 1.0; return Pm
        return _apply_temperature(task.P[objs], T) if posterior == "calibrated" else task.P[objs]

    # class-conditional residual laws on A (per lead bin when populated), with the bookkeeping needed to evaluate
    # every calibration point's own-class density leave-one-object-out (no in-sample optimism for small classes)
    zA = task.y[task.obj[calA]]; bA_ = task.lead_bin[calA]
    dens = np.zeros((K, nb, len(EGRID))); src = {}
    step = EGRID[1] - EGRID[0]
    eA = (task.yt[calA][:, None] - HMU[calA]) / HSG[calA]                      # (rows, K) residual under each hypothesis
    for c in range(K):
        mc = zA == c
        d_all, k_all, n_all, c_all = _residual_density_parts(eA[mc, c], bw_mult=bw_mult)
        for b in range(nb):
            mb = mc & (bA_ == b)
            if mb.sum() >= min_bin:
                d, kk, nn, cc = _residual_density_parts(eA[mb, c], bw_mult=bw_mult)
                dens[c, b] = d; src[(c, b)] = (mb, kk, nn, cc)
            else:
                dens[c, b] = d_all; src[(c, b)] = (mc, k_all, n_all, c_all)

    def loo_true_density():
        """(rows of calA,) own-class density at each calibration point with its own object's points removed."""
        idx = np.clip(((np.clip(eA[np.arange(len(calA)), zA], EGRID[0], EGRID[-1]) - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1)
        out = dens[zA, bA_, idx].copy()
        objA = task.obj[calA]
        for (c, b), (sample, ker, norm, n) in src.items():
            if ker is None or not loo:
                continue
            ev = np.flatnonzero((zA == c) & (bA_ == b))
            if not len(ev):
                continue
            half = len(ker) // 2
            smp = np.flatnonzero(sample)
            so = objA[smp]; order = np.argsort(so, kind="stable"); smp, so = smp[order], so[order]
            starts = np.searchsorted(so, objA[ev], "left"); ends = np.searchsorted(so, objA[ev], "right")
            for i, r in enumerate(ev):
                js = smp[starts[i]:ends[i]]
                dd = idx[r] - idx[js] + half
                own = ker[dd[(dd >= 0) & (dd < len(ker))]].sum()
                m = len(js)
                if n - m > 0:
                    out[r] = max((dens[c, b, idx[r]] * norm - own) / (norm * (n - m) / n), 1e-9)
        out[(eA[np.arange(len(calA)), zA] < EGRID[0]) | (eA[np.arange(len(calA)), zA] > EGRID[-1])] = 1e-12
        return out

    def fval(rows, yv):
        e = (yv[:, None] - HMU[rows]) / HSG[rows]
        idx = np.clip(((e - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1)
        out = dens[np.arange(K)[None, :], task.lead_bin[rows][:, None], idx] / HSG[rows]
        out[(e < EGRID[0]) | (e > EGRID[-1])] = 1e-12
        return out

    def sbar(rows, Pr):
        return (Pr * HSG[rows]).sum(1) if scale_norm else np.ones(len(rows))

    def terms(rows):
        Pr = post(task.obj[rows])
        return Pr * fval(rows, task.yt[rows]) * sbar(rows, Pr)[:, None]     # (points, K): sbar(x) p_c(x) f_c(y | x)

    def thresholds(s, rows):
        return _class_thresholds(s, task.y[task.obj[rows]], task.obj[rows], K, alpha, unit, finite)

    # C3: per lead-time bin (Mondrian over horizons, as FLARE's cells), dual weights by the multiplicative fixed point
    # that equalizes the per-class thresholds, then the smallest class threshold of the bin
    TA = terms(calA)
    PrA = post(task.obj[calA])
    TA[np.arange(len(calA)), zA] = PrA[np.arange(len(calA)), zA] * loo_true_density() / HSG[calA, zA] * sbar(calA, PrA)
    TB = TA if calB is calA else terms(calB)
    hier = cells == "hier"                                      # per-bin thresholds where a class is big, pooled elsewhere
    n_big = int(np.ceil(2 / alpha))
    if cells == "auto":                                         # data-adaptive horizon cells (calibration counts only)
        gid = _horizon_cells(task.lead_bin[calA], task.obj[calA], zA, K, nb, n_big)
    else:
        gid = np.arange(nb)
    bA = gid[task.lead_bin[calA]]; bB = gid[task.lead_bin[calB]]
    lead_cell = gid[task.lead_bin]
    if incl == "merge":                                         # a class joins a cell's threshold with >= ceil(2/alpha) objects
        nmin = int(np.ceil(2 / alpha))
    else:                                                       # CRC budget alpha(n+1)-1 >= 0  <=>  n >= 1/alpha - 1
        nmin = int(np.ceil(1 / alpha - 1 - 1e-9))
    DIAG.clear(); DIAG.update(iterations=[], excluded=[], fallback_bins=[], pooled=[], temperature=float(T))
    V = np.ones((nb, K)); Tb = np.full(nb, np.nan); Rb = np.full(nb, np.nan); SMALL = {}
    n_cut = n_big if hier else nmin
    DIAG["cells"] = gid.tolist()
    for b in (range(int(gid.max()) + 1) if lead_mondrian else [None]):
        mA = (bA == b) if b is not None else np.ones(len(calA), bool)
        mB = (bB == b) if b is not None else np.ones(len(calB), bool)
        if not mB.any():
            continue
        zb = task.y[task.obj[calA[mA]]]
        nobj = np.array([len(np.unique(task.obj[calA[mA]][zb == c])) for c in range(K)])
        keep = nobj >= n_cut                                       # classes with enough objects in this bin
        if weights == "mixture":
            v = np.ones(K)
        else:
            v = 1.0 / np.maximum(pi, 1e-6)
            if weights == "dual":
                for it in range(n_iter):
                    tau = thresholds(TA[mA] @ v, calA[mA])
                    ok = keep & np.isfinite(tau) & (tau > 0)
                    if not ok.any():
                        break
                    ref = np.exp(np.mean(np.log(tau[ok])))
                    upd = np.where(ok, (ref / np.where(ok, tau, ref)) ** 0.5, 1.0)
                    v = v * np.clip(upd, 0.5, 2.0); v = v / np.exp(np.mean(np.log(v)))
                    if np.max(np.abs(np.log(tau[ok] / ref))) < 0.005:
                        break
                DIAG["iterations"].append(it + 1)
        if weights == "mixture":                              # posterior mixture: one marginal threshold
            t = _class_thresholds(TB[mB] @ v, np.zeros(mB.sum(), int), task.obj[calB[mB]], 1, alpha, unit, finite)[0]
        else:
            tB = thresholds(TB[mB] @ v, calB[mB])
            zbB = task.y[task.obj[calB[mB]]]
            nobjB = np.array([len(np.unique(task.obj[calB[mB]][zbB == c])) for c in range(K)])
            okB = (nobjB >= n_cut) & np.isfinite(tB)
            if hier and b is not None:
                Rb[b] = np.quantile(TA[mA] @ v, alpha)                # reference scale of the bin, fixed given A
                SMALL[b] = [c for c in range(K) if 0 < nobjB[c] < n_big]
                t = np.min(tB[okB]) if okB.any() else np.nan
            else:
                DIAG["excluded"] += [(int(b) if b is not None else -1, int(c), int(nobjB[c])) for c in range(K) if 0 < nobjB[c] < nmin]
                t = np.min(tB[okB]) if okB.any() else np.nanmin(tB)
        if b is None:
            V[:] = v; Tb[:] = t
        else:
            V[b] = v; Tb[b] = t
    if hier and lead_mondrian and weights != "mixture" and SMALL:
        # classes that are small in some bins: one CRC threshold per class, pooled over those bins, on the score
        # normalised by the bins' reference scales; the bin threshold is lowered to cover it
        zB_ = task.y[task.obj[calB]]
        lam = {}
        for c in range(K):
            cells_c = [b for b in SMALL if c in SMALL[b]]
            if not cells_c:
                continue
            m = (zB_ == c) & np.isin(bB, cells_c)
            n_c = len(np.unique(task.obj[calB[m]]))
            if n_c < nmin:
                DIAG["excluded"].append((-2, int(c), int(n_c))); continue
            u = (TB[m] * V[bB[m]]).sum(1) / Rb[bB[m]]
            lam[c] = _class_thresholds(u, np.zeros(m.sum(), int), task.obj[calB[m]], 1, alpha, unit, finite)[0]
            DIAG["pooled"].append((int(c), int(n_c), [int(x) for x in cells_c]))
        for b, cl in SMALL.items():
            ls = [lam[c] for c in cl if c in lam]
            if ls and np.isfinite(Rb[b]):
                cand = min(ls) * Rb[b]
                Tb[b] = cand if not np.isfinite(Tb[b]) else min(Tb[b], cand)
    if lead_mondrian and np.isnan(Tb).any():                 # bins without calibration points: nearest bin
        good = np.flatnonzero(~np.isnan(Tb))
        for b in np.flatnonzero(np.isnan(Tb)):
            j = good[np.argmin(np.abs(good - b))]; V[b] = V[j]; Tb[b] = Tb[j]
            if (lead_cell[test] == b).any():
                DIAG["fallback_bins"].append(int(b))
    # prediction: coverage of the true value, and the band on a y-grid
    bt = lead_cell[test]
    s_true = (terms(test) * V[bt]).sum(1)
    cov = s_true >= Tb[bt]
    lo = np.empty(len(test)); hi = np.empty(len(test)); length = np.empty(len(test))
    Pt = post(task.obj[test])
    for s0 in range(0, len(test), 1500):
        rows = test[s0:s0 + 1500]; Pr = Pt[s0:s0 + 1500]
        act = Pr >= prune
        lo_g = np.where(act, HMU[rows] - 8 * HSG[rows], np.inf).min(1); hi_g = np.where(act, HMU[rows] + 8 * HSG[rows], -np.inf).max(1)
        Y, Wg = adaptive_grid(lo_g, hi_g, HMU[rows], HSG[rows])
        S = np.zeros_like(Y)
        for c in range(K):
            e = (Y - HMU[rows, c][:, None]) / HSG[rows, c][:, None]
            idx = np.clip(((e - EGRID[0]) / step).round().astype(int), 0, len(EGRID) - 1)
            fc = dens[c][task.lead_bin[rows][:, None], idx] / HSG[rows, c][:, None]
            fc[(e < EGRID[0]) | (e > EGRID[-1])] = 0.0
            S += V[lead_cell[rows], c][:, None] * Pr[:, c][:, None] * fc
        S *= sbar(rows, Pr)[:, None]
        inside = S >= Tb[lead_cell[rows]][:, None]
        lo[s0:s0 + 1500], hi[s0:s0 + 1500], length[s0:s0 + 1500] = band_from_inside(Y, Wg, inside)
    return cov, lo, hi, length


def _membership(task, A, posterior):
    """C2: posterior function objs -> (n, K) and the class priors on the calibration objects A."""
    K = task.K
    pi = np.bincount(task.y[A], minlength=K) / len(A)
    T = _temperature(task.P[A], task.y[A], K) if posterior == "calibrated" else 1.0

    def post(objs):
        if posterior == "uniform":
            return np.repeat(pi[None, :], len(objs), 0)
        if posterior == "onehot":
            Pm = np.zeros((len(objs), K)); Pm[np.arange(len(objs)), task.P[objs].argmax(1)] = 1.0; return Pm
        return _apply_temperature(task.P[objs], T) if posterior == "calibrated" else task.P[objs]
    return post, pi, T


def _residual_laws(task, rows, HMU, HSG, min_bin, bw_mult):
    """C1: per (class, lead bin) density of the standardized residual under the class hypothesis, on EGRID."""
    K = task.K; nb = int(task.lead_bin.max()) + 1
    z = task.y[task.obj[rows]]
    dens = np.zeros((K, nb, len(EGRID)))
    for c in range(K):
        mc = z == c
        e = (task.yt[rows] - HMU[rows, c]) / HSG[rows, c]
        d_all = _residual_density(e[mc], bw_mult=bw_mult)
        for b in range(nb):
            mb = mc & (task.lead_bin[rows] == b)
            dens[c, b] = _residual_density(e[mb], bw_mult=bw_mult) if mb.sum() >= min_bin else d_all
    return dens


def _class_dens(task, dens, rows, Y, HMU, HSG):
    """f_c(Y | x) for rows (n,) and values Y (n, G) -> (n, G, K)."""
    step = EGRID[1] - EGRID[0]
    e = (Y[:, :, None] - HMU[rows][:, None, :]) / HSG[rows][:, None, :]
    idx = np.clip(((e - EGRID[0]) / step).round().astype(np.int32), 0, len(EGRID) - 1)
    out = dens[np.arange(task.K)[None, None, :], task.lead_bin[rows][:, None, None], idx] / HSG[rows][:, None, :]
    out[(e < EGRID[0]) | (e > EGRID[-1])] = 0.0
    return out


def _grid(rows, HMU, HSG, Pr, G, width=8.0, prune=1e-4):
    act = Pr >= prune
    act[np.arange(len(rows)), Pr.argmax(1)] = True
    lo = np.where(act, HMU[rows] - width * HSG[rows], np.inf).min(1)
    hi = np.where(act, HMU[rows] + width * HSG[rows], -np.inf).max(1)
    return lo, hi, lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, G)[None, :]


METHODS["GAMUT-union (early design)"] = gamut_union
METHODS["GAMUT"] = gamut
ABLATIONS = {
    "GAMUT w/o C1 (pooled forecaster)": dict(hcf=False),
    "GAMUT w/o C2 (uniform posterior)": dict(posterior="uniform"),
    "GAMUT w/o C2 (argmax posterior)": dict(posterior="onehot"),
    "GAMUT w/o C2-calibration (raw posterior)": dict(posterior="classifier"),
    "GAMUT w/o C3-dual (posterior mixture)": dict(weights="mixture"),
    "GAMUT w/o C3-dual (prior-ratio weights)": dict(weights="balanced"),
    "GAMUT w/o lead-bin cells (one threshold)": dict(lead_mondrian=False),
    "GAMUT w/o scale normalization (absolute length)": dict(scale_norm=False),
    "GAMUT w/o leave-one-object-out laws (in-sample)": dict(loo=False),
    "GAMUT w/o adaptive cells (fixed lead bins)": dict(cells="bins"),
    "GAMUT-cert (split certificate)": dict(crc=True),
    "GAMUT (point-level unit)": dict(unit="point"),
}
