"""External baselines that use the latent class: two-step CP, max-class CP, CPTC, Kandinsky CP and MDCP.

Each follows the published algorithm; where the paper's components do not exist in this setting (per-source
networks, spline bases of a 100-dimensional prefix) the substitution is stated in the docstring and, whenever a
choice was needed, the stronger option was given to the baseline (our hypothesis-conditional forecaster, the
classifier posterior as basis)."""
from __future__ import annotations

import numpy as np

from .methods import MIN_N, _band, _qr, _split_ab, _union, conf_quantile, r_pooled


def _true_class_bin_quantiles(task, cal, r, alpha):
    """(K, nb) conformal (1 - alpha) quantiles of the residual r per (TRUE class, lead bin), falling back to the
    class-wide quantile for cells under MIN_N points and to the global quantile for empty classes."""
    nb = int(task.lead_bin.max()) + 1
    z = task.y[task.obj[cal]]; g = task.lead_bin[cal]
    glob = conf_quantile(r, alpha)
    Q = np.full((task.K, nb), glob)
    for c in range(task.K):
        mc = z == c
        qc = conf_quantile(r[mc], alpha) if mc.sum() >= MIN_N else glob
        for b in range(nb):
            m = mc & (g == b)
            Q[c, b] = conf_quantile(r[m], alpha) if m.sum() >= MIN_N else qc
    return Q


def _class_threshold_sets(task, cal_objs, test_objs, alpha_l):
    """ClassThr label sets (Sadinle et al. 2019, class-conditional; Timans et al. 2024 Sec. 5.1):
    Gamma(x) = {c : p_c(x) >= 1 - q_c}, q_c the (1 - alpha_l) conformal quantile of 1 - p_c over class-c objects."""
    K = task.K
    q = np.ones(K)
    for c in range(K):
        oc = cal_objs[task.y[cal_objs] == c]
        if len(oc):
            q[c] = min(conf_quantile(1 - task.P[oc, c], alpha_l), 1.0)
    G = task.P[test_objs] >= (1 - q)[None, :]
    G[np.arange(len(test_objs)), task.P[test_objs].argmax(1)] = True        # never empty
    return G


def two_step_cp(task, cal, test, alpha, alpha_l=0.01, **_):
    """Two-step conformal prediction (Timans, Straehle, Sakmann, Naesseth & Nalisnick, ECCV 2024): a class-conditional
    conformal label set at level alpha_l = 0.01 (their setting), then the class-conditional (true-class Mondrian)
    quantile of the box residual at level alpha, taking the MAXIMUM quantile over the classes in the label set.
    Guarantee (1 - alpha_l)(1 - alpha) for every class. Box residual = FLARE's normalized residual; cells per lead bin."""
    Q = _true_class_bin_quantiles(task, cal, r_pooled(task, cal), alpha)
    G = _class_threshold_sets(task, np.unique(task.obj[cal]), task.obj[test], alpha_l)
    m = np.where(G, Q[:, task.lead_bin[test]].T, -np.inf).max(1)
    return _band(task, test, task.mu[test], m * task.sig[test])


def max_class_cp(task, cal, test, alpha, **_):
    """Conservative max-class conformal prediction (Barber, Candes, Ramdas & Tibshirani 2021; the 'conservative CP'
    of Kandinsky CP): the largest true-class quantile over ALL classes, valid for every class whatever the posterior."""
    Q = _true_class_bin_quantiles(task, cal, r_pooled(task, cal), alpha)
    return _band(task, test, task.mu[test], Q.max(0)[task.lead_bin[test]] * task.sig[test])


def _class_intervals(task, cal, test, alpha):
    """Per-class intervals hmu_c +- Q_{c,b} hsig_c from the hypothesis-conditional forecaster (C1), with Q the
    true-class (class c, lead bin) conformal quantile of the class-c standardized residual."""
    K = task.K
    lo = np.empty((len(test), K)); hi = np.empty((len(test), K))
    for c in range(K):
        r = np.abs(task.yt[cal] - task.hmu[cal, c]) / task.hsig[cal, c]
        Q = _true_class_bin_quantiles(task, cal, r, alpha)[c]
        half = Q[task.lead_bin[test]] * task.hsig[test, c]
        lo[:, c] = task.hmu[test, c] - half; hi[:, c] = task.hmu[test, c] + half
    return lo, hi


def _union_band(task, test, lo, hi, G):
    yv = task.yt[test][:, None]
    cov = (G & (yv >= lo) & (yv <= hi)).any(1)
    hl, hh, length = _union(lo, hi, G)
    return cov, hl, hh, length


def two_step_cp_c1(task, cal, test, alpha, alpha_l=0.01, **_):
    """Two-step CP given GAMUT's C1: the same class-conditional label set at level alpha_l, then the UNION over the
    classes in the set of class-c intervals from the hypothesis-conditional forecaster. Same guarantee as two_step_cp."""
    lo, hi = _class_intervals(task, cal, test, alpha)
    G = _class_threshold_sets(task, np.unique(task.obj[cal]), task.obj[test], alpha_l)
    return _union_band(task, test, lo, hi, G)


def max_class_cp_c1(task, cal, test, alpha, **_):
    """Max-class CP given GAMUT's C1: the union over ALL classes of class-c intervals from the hypothesis-conditional
    forecaster; valid for every class whatever the posterior."""
    lo, hi = _class_intervals(task, cal, test, alpha)
    return _union_band(task, test, lo, hi, np.ones_like(lo, dtype=bool))


def mdcp_c1(task, cal, test, alpha, seed=0, **kw):
    """MDCP with GAMUT's C1 residual laws as working densities (see mdcp, laws='kde')."""
    return mdcp(task, cal, test, alpha, seed=seed, laws="kde", **kw)


def cptc(task, cal, test, alpha, **_):
    """CPTC (Sun & Yu, NeurIPS 2025), batch version: state-specific conformal sets from a state-conditioned forecaster
    f(x, z) (here the hypothesis-conditional forecaster), each calibrated on the calibration points of that state
    (true labels are known in calibration, so no state sampling is needed), aggregated by their implemented rule
    (Eq. 11): the union of the state sets of the most probable states whose posterior mass reaches 1 - alpha.
    The online ACI tracking of the paper has no counterpart in a batch evaluation and is omitted."""
    K = task.K
    lo = np.empty((len(test), K)); hi = np.empty((len(test), K))
    for c in range(K):
        r = np.abs(task.yt[cal] - task.hmu[cal, c]) / task.hsig[cal, c]
        Q = _true_class_bin_quantiles(task, cal, r, alpha)[c]
        half = Q[task.lead_bin[test]] * task.hsig[test, c]
        lo[:, c] = task.hmu[test, c] - half; hi[:, c] = task.hmu[test, c] + half
    Pt = task.P[task.obj[test]]
    order = np.argsort(-Pt, axis=1)
    cum = np.cumsum(np.take_along_axis(Pt, order, 1), axis=1)
    keep_sorted = np.c_[np.ones((len(test), 1), bool), cum[:, :-1] < 1 - alpha]
    G = np.zeros_like(keep_sorted); np.put_along_axis(G, order, keep_sorted, 1)
    yv = task.yt[test][:, None]
    cov = (G & (yv >= lo) & (yv <= hi)).any(1)
    hl, hh, length = _union(lo, hi, G)
    return cov, hl, hh, length


# ---------------------------------------------------------------- shared y-grid machinery
def _grid_band(task, test, member, centers, scales, G=400, width=8.0, chunk=1000):
    """Hull and length of {y : member(rows, Y)} on the adaptive grid of gamut.methods (uniform over centers +- width *
    scales, refined around every centre); the same measurement as for GAMUT."""
    from .methods import adaptive_grid, band_from_inside
    lo = np.full(len(test), np.nan); hi = np.full(len(test), np.nan); length = np.zeros(len(test))
    for s0 in range(0, len(test), chunk):
        sl = slice(s0, s0 + chunk); rows = test[sl]
        lo_g = (centers[sl] - width * scales[sl]).min(1); hi_g = (centers[sl] + width * scales[sl]).max(1)
        Y, W = adaptive_grid(lo_g, hi_g, centers[sl], scales[sl], G)
        lo[sl], hi[sl], length[sl] = band_from_inside(Y, W, member(rows, Y))
    return lo, hi, length


def _gauss(y, m, s):
    return np.exp(-0.5 * ((y - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)


# ---------------------------------------------------------------- Kandinsky CP
def _membership_features(task, rows, Y):
    """Features of (prefix, future value) for the membership posterior P(Z = c | x, y): log classifier posterior,
    standardized residual under each class hypothesis, its square, log class scale, lead-bin one-hot.
    rows: (n,), Y: (n, G) -> (n, G, d)."""
    nb = int(task.lead_bin.max()) + 1
    n, G = Y.shape
    lp = np.log(np.maximum(task.P[task.obj[rows]], 1e-6))                  # (n, K)
    e = (Y[:, :, None] - task.hmu[rows][:, None, :]) / task.hsig[rows][:, None, :]   # (n, G, K)
    ls = np.log(task.hsig[rows])
    B = np.eye(nb)[task.lead_bin[rows]][:, 1:]
    rep = lambda A: np.broadcast_to(A[:, None, :], (n, G, A.shape[1]))
    return np.concatenate([rep(lp), np.clip(e, -12, 12), np.minimum(e ** 2, 144), rep(ls), rep(B)], axis=2)


def _fit_softmax(Xf, z, K, l2=1e-3, iters=300):
    """Multinomial logistic regression by L-BFGS (scipy), standardized features."""
    from scipy.optimize import minimize
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    Xs = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]
    d = Xs.shape[1]
    Yz = np.eye(K)[z]

    def f(w):
        W = w.reshape(d, K); S = Xs @ W; S -= S.max(1, keepdims=True)
        lse = np.log(np.exp(S).sum(1)); P = np.exp(S - lse[:, None])
        nll = -(Yz * (S - lse[:, None])).sum() / len(Xs) + l2 * (W[:-1] ** 2).sum()
        g = Xs.T @ (P - Yz) / len(Xs); g[:-1] += 2 * l2 * W[:-1]
        return nll, g.ravel()
    w = minimize(f, np.zeros(d * K), jac=True, method="L-BFGS-B", options=dict(maxiter=iters)).x
    return mu, sd, w.reshape(d, K)


def _softmax_apply(model, Xf):
    mu, sd, W = model
    S = np.concatenate([(Xf - mu) / sd, np.ones(Xf.shape[:-1] + (1,))], axis=-1) @ W
    S -= S.max(-1, keepdims=True); E = np.exp(S)
    return E / E.sum(-1, keepdims=True)


def kandinsky_cp(task, cal, test, alpha, seed=0, max_fit=60000, **_):
    """Kandinsky conformal prediction (Bairaktari, Wu & Wu, ICML 2025), 'fractional group membership' application:
    weight functions Phi_c(x, y) = P(Z = c | x, y), here a multinomial logistic posterior on the classifier
    probabilities and the future value's residuals under each class hypothesis (fitted on calibration half A), plus
    lead-bin indicators; quantile regression of the normalized residual on this basis on half B (Algorithm 1);
    prediction set {y : S(x, y) <= q(x, y)} found on a y-grid (Algorithm 2). Guarantee: per-latent-class coverage
    within O(sqrt(K/n)) of nominal when Phi is the true posterior."""
    nb = int(task.lead_bin.max()) + 1
    A, B = _split_ab(task, np.unique(task.obj[cal]), seed)
    inA = np.isin(task.obj[cal], A); calA, calB = cal[inA], cal[~inA]
    rng = np.random.default_rng(seed)
    fitA = calA if len(calA) <= max_fit else rng.choice(calA, max_fit, replace=False)
    Xa = _membership_features(task, fitA, task.yt[fitA][:, None])[:, 0, :]
    model = _fit_softmax(Xa, task.y[task.obj[fitA]], task.K)

    def basis(rows, Y):
        Phi = _softmax_apply(model, _membership_features(task, rows, Y))          # (n, G, K)
        Bm = np.broadcast_to(np.eye(nb)[task.lead_bin[rows]][:, None, 1:], Phi.shape[:2] + (nb - 1,))
        return np.concatenate([Phi, Bm], axis=2)

    PhiB = basis(calB, task.yt[calB][:, None])[:, 0, :]
    beta = _qr(PhiB, r_pooled(task, calB), 1 - alpha, max_n=15000, seed=seed)

    def member(rows, Y):
        q = basis(rows, Y) @ beta
        return np.abs(Y - task.mu[rows][:, None]) / task.sig[rows][:, None] <= q

    cov = member(test, task.yt[test][:, None])[:, 0]
    C = np.c_[task.hmu[test], task.mu[test]]; S = np.c_[task.hsig[test], task.sig[test]]
    lo, hi, length = _grid_band(task, test, member, C, S, chunk=150)
    return cov, lo, hi, length


# ---------------------------------------------------------------- MDCP
def _softplus(t):
    return np.logaddexp(0.0, t)


def _sigmoid(t):
    return 0.5 * (1 + np.tanh(0.5 * t))


def mdcp(task, cal, test, alpha, seed=0, lr=1e-3, batch=256, max_epochs=10000, tol=1e-4, max_fit=60000, laws="gauss", **_):
    """Multi-distribution conformal prediction (Yang & Jin, ICML 2026), regression algorithm (Sec. 4.3, Alg. 1):
    Gaussian working models f_k(y | x) per source (here the hypothesis-conditional forecaster under class k, which
    shares strength across classes and is at least as accurate as per-source boosting for rare classes), a pooled
    Gaussian p_pool, Lagrange multipliers lambda_k(x) = softplus(Lambda(x)' theta_k) fitted by maximizing the empirical
    dual objective (Eq. 8/10) with Adam (lr 1e-3, minibatch 256, relative-change early stopping) on one half of the
    calibration objects, and max-p aggregation of per-source conformal p-values of the shared score -h on the other
    half; set computed on a y-grid (App. C.4). Basis Lambda(x): intercept, classifier posterior, lead-bin indicators
    (the paper uses splines of a low-dimensional x; the posterior is the informative low-dimensional summary here).
    laws="kde" (\"MDCP + C1 densities\"): the working models f_k are GAMUT's per-(class, lead bin) kernel residual laws
    under the class hypothesis (gamut.methods._residual_laws), fitted on half A, instead of Gaussians."""
    K = task.K; nb = int(task.lead_bin.max()) + 1
    A, B = _split_ab(task, np.unique(task.obj[cal]), seed)
    inA = np.isin(task.obj[cal], A); calA, calB = cal[inA], cal[~inA]
    rng = np.random.default_rng(seed)
    fitA = calA if len(calA) <= max_fit else rng.choice(calA, max_fit, replace=False)
    if laws == "kde":
        from .methods import _class_dens, _residual_laws
        dens = _residual_laws(task, calA, task.hmu, task.hsig, 150, 2.0)

    def fdens(rows, Y):                    # Y: (n, G) -> (n, G, K)
        if laws == "kde":
            return _class_dens(task, dens, rows, Y, task.hmu, task.hsig)
        return _gauss(Y[:, :, None], task.hmu[rows][:, None, :], task.hsig[rows][:, None, :])

    def Lam(rows):
        return np.c_[np.ones(len(rows)), task.P[task.obj[rows]], np.eye(nb)[task.lead_bin[rows]][:, 1:]]

    def fk(rows, yv):
        return fdens(rows, yv[:, None])[:, 0, :]                                         # (n, K)

    La = Lam(fitA); Fa = fk(fitA, task.yt[fitA])
    pp = np.maximum(_gauss(task.yt[fitA], task.mu[fitA], task.sig[fitA]), 1e-8)
    m = La.shape[1]
    theta = np.zeros((m, K))
    a_m = np.zeros_like(theta); a_v = np.zeros_like(theta); b1, b2, eps = 0.9, 0.999, 1e-8

    def objective(L, F, p, th):
        lam = _softplus(L @ th); h = (lam * F).sum(1)
        return float(np.mean(np.minimum(1 - h, 0) / p + (1 - alpha) * lam.sum(1)))
    prev = objective(La, Fa, pp, theta); step = 0
    for ep in range(max_epochs):
        perm = rng.permutation(len(fitA))
        for s0 in range(0, len(perm), batch):
            j = perm[s0:s0 + batch]
            Z = La[j] @ theta; lam = _softplus(Z); h = (lam * Fa[j]).sum(1)
            act = (h > 1).astype(float) / pp[j]                                         # d/dh of min(1-h,0)/p = -1/p
            gl = -act[:, None] * Fa[j] + (1 - alpha)                                     # dJ/dlam (ascent)
            g = La[j].T @ (gl * _sigmoid(Z)) / len(j)
            step += 1
            a_m = b1 * a_m + (1 - b1) * g; a_v = b2 * a_v + (1 - b2) * g * g
            theta += lr * (a_m / (1 - b1 ** step)) / (np.sqrt(a_v / (1 - b2 ** step)) + eps)
        cur = objective(La, Fa, pp, theta)
        if abs(cur - prev) / max(abs(prev), 1e-8) < tol:
            break
        prev = cur

    def hval(rows, Y):                     # Y: (n, G)
        lam = _softplus(Lam(rows) @ theta)                                              # (n, K)
        return (lam[:, None, :] * fdens(rows, Y)).sum(2)

    hB = hval(calB, task.yt[calB][:, None])[:, 0]; zB = task.y[task.obj[calB]]
    t_k = []
    for c in range(K):
        s = np.sort(hB[zB == c])
        if len(s) == 0:
            continue
        k = int(np.floor(alpha * (len(s) + 1)))                  # p-value >= alpha  <=>  h >= s_(k)
        t_k.append(s[k - 1] if k >= 1 else -np.inf)
    t = min(t_k)

    def member(rows, Y):
        return hval(rows, Y) >= t

    cov = member(test, task.yt[test][:, None])[:, 0]
    lo, hi, length = _grid_band(task, test, member, task.hmu[test], task.hsig[test])
    return cov, lo, hi, length


LATENT_BASELINES = {"TwoStepCP": two_step_cp, "MaxClassCP": max_class_cp, "CPTC": cptc, "KandinskyCP": kandinsky_cp,
                    "MDCP": mdcp, "MDCP + C1": mdcp_c1, "TwoStepCP + C1": two_step_cp_c1, "MaxClassCP + C1": max_class_cp_c1}
