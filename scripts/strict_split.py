"""Strict three-way split check of finite-sample validity (train / calibrate / test on disjoint objects).

The main protocol is cross-conformal (FLARE's): a calibration object's outputs come from models trained on folds
that include the test fold. Thm. 3 needs base models independent of both the calibration and the test objects.
Here, for rotation r = 0..4: test fold r, calibration fold r+1 (mod 5), and the classifier, base forecaster and
hypothesis-conditional forecaster are trained on the other three folds. Methods are calibrated on fold r+1 and
evaluated on fold r; predictions of the five rotations are pooled.
Output: results/<bench>_strict/h<h>_a<alpha>.json
Run: python scripts/strict_split.py --bench ztf --h 5 [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from benchmarks import registry  # noqa: E402
from crossfit import FORECAST_PARAMS, QUANTILES, cap_rows, fit_lgb, long_frame  # noqa: E402
from gamut import data, latent_baselines, methods, metrics  # noqa: E402

METHODS = {"FLARE": methods.flare_mondrian, "SplitCP": methods.split_cp, "TwoStepCP": latent_baselines.two_step_cp,
           "MaxClassCP": latent_baselines.max_class_cp, "Oracle-true-class": methods.oracle_true_class,
           "CPTC": latent_baselines.cptc, "KandinskyCP": latent_baselines.kandinsky_cp, "MDCP": latent_baselines.mdcp,
           "MDCP + C1": latent_baselines.mdcp_c1, "TwoStepCP + C1": latent_baselines.two_step_cp_c1,
           "MaxClassCP + C1": latent_baselines.max_class_cp_c1,
           "GAMUT": methods.gamut, "GAMUT-cert (split certificate)": (methods.gamut, dict(crc=True))}
NBOOT = 500


def object_bootstrap(pooled, covs, alpha, rng):
    """Object-level bootstrap as in scripts/bootstrap.py: per class, 500 resamples of the class's objects, point-pooled
    coverage with a 95% percentile interval; per (class, lead bin) cell with >= max(1/alpha - 1, 3) objects, the
    object-level coverage with its upper bound. Under-covered: upper bound below nominal."""
    obj = pooled.obj; z = pooled.y[obj]
    uo, inv = np.unique(obj, return_inverse=True); zo = np.zeros(len(uo), int); zo[inv] = z
    cv = np.bincount(inv, covs.astype(float)); npts = np.bincount(inv)
    per = {}
    for c in np.unique(zo):
        ix = np.flatnonzero(zo == c); d = rng.integers(0, len(ix), (NBOOT, len(ix)))
        bc = cv[ix][d].sum(1) / npts[ix][d].sum(1); lo, hi = np.percentile(bc, [2.5, 97.5])
        per[int(c)] = dict(n=int(len(ix)), cov=float(cv[ix].sum() / npts[ix].sum()), lo=float(lo), hi=float(hi),
                           under=bool(hi < 1 - alpha))
    cells = {}
    for c in np.unique(zo):
        for b in np.unique(pooled.lead_bin):
            sel = (z == c) & (pooled.lead_bin == b)
            if not sel.any():
                continue
            u2, i2 = np.unique(obj[sel], return_inverse=True)
            if len(u2) < max(1 / alpha - 1, 3):
                continue
            co = np.bincount(i2, covs[sel].astype(float)) / np.bincount(i2)
            bs = co[rng.integers(0, len(co), (NBOOT, len(co)))].mean(1)
            cells[f"{int(c)}|{int(b)}"] = dict(n=int(len(u2)), cov=float(co.mean()), hi=float(np.percentile(bs, 97.5)),
                                              under=bool(np.percentile(bs, 97.5) < 1 - alpha))
    return dict(per_class=per, n_under=int(sum(v["under"] for v in per.values())), cells=cells,
                n_cells_under=int(sum(v["under"] for v in cells.values())))


def rotation_outputs(B, h, r, threads, lr, cap, cache):
    path = cache / f"h{h:g}_rot{r}.npz"
    if path.exists():
        return dict(np.load(path))
    y = B.meta.y.to_numpy(); fold = B.meta.fold.to_numpy(); K = len(B.classes)
    F = B.F; Fobj = F.obj.to_numpy()
    te_f, ca_f = r, (r + 1) % 5
    tr_obj = np.flatnonzero((fold != te_f) & (fold != ca_f)); pr_obj = np.flatnonzero((fold == te_f) | (fold == ca_f))
    fit_obj, es_obj = train_test_split(tr_obj, test_size=.15, stratify=y[tr_obj], random_state=r)
    P, L = B.classifier(B.X, y, fit_obj, es_obj, pr_obj, threads)
    params = dict(FORECAST_PARAMS, learning_rate=lr)
    r_fit = cap_rows(np.flatnonzero(np.isin(Fobj, fit_obj)), Fobj, cap, 3000 + r)
    r_es = cap_rows(np.flatnonzero(np.isin(Fobj, es_obj)), Fobj, cap, 4000 + r)
    r_pr = np.flatnonzero(np.isin(Fobj, pr_obj))
    Xfit, Xes, Xpr = (long_frame(B.X, F, rr, h) for rr in (r_fit, r_es, r_pr))
    yfit, yes = F.dmag.to_numpy()[r_fit], F.dmag.to_numpy()[r_es]
    out = dict(pr_obj=pr_obj, P=P, L=L, rows=r_pr)
    models = {"mu": params, "q16": dict(params, objective="quantile", alpha=QUANTILES["q16"]),
              "q84": dict(params, objective="quantile", alpha=QUANTILES["q84"])}
    for name, p in models.items():
        out[name] = fit_lgb(p, Xfit, yfit, Xes, yes, threads).predict(Xpr, num_threads=threads)

    def with_class(D, cls):
        D = D.copy()
        for c in range(K):
            D[f"hyp_{c}"] = (cls == c).astype(float)
        return D
    Hfit, Hes = with_class(Xfit, y[Fobj[r_fit]]), with_class(Xes, y[Fobj[r_es]])
    for name, p in models.items():
        m = fit_lgb(p, Hfit, yfit, Hes, yes, threads)
        out[f"hcf_{name}"] = np.stack([m.predict(with_class(Xpr, np.full(len(Xpr), c)), num_threads=threads) for c in range(K)], 1)
    np.savez_compressed(path, **out)
    return out


def make_task(B, h, o):
    n = len(B.meta); K = len(B.classes)
    P = np.full((n, K), np.nan); L = np.full(n, -1); P[o["pr_obj"]] = o["P"]; L[o["pr_obj"]] = o["L"]
    F = B.F.iloc[o["rows"]]
    ok = np.isfinite(o["mu"]) & np.isfinite(F.dmag.to_numpy())
    t = F.t.to_numpy()[ok]
    sig = np.maximum(np.abs(o["q84"] - o["q16"]) / 2, data.SCALE_FLOOR)[ok]
    hsig = np.maximum(np.abs(o["hcf_q84"] - o["hcf_q16"]) / 2, data.SCALE_FLOOR)[ok]
    return data.Task(bench=B.name, h=h, classes=B.classes, y=B.meta.y.to_numpy(), fold=B.meta.fold.to_numpy(), P=P, L=L,
                     obj=F.obj.to_numpy()[ok], t=t, band=F.band.to_numpy()[ok], yt=F.dmag.to_numpy()[ok],
                     lead_bin=np.searchsorted(data.lead_edges(B.name, h), t, side="right"), mu=o["mu"][ok], sig=sig,
                     q={"q16": o["q16"][ok], "q84": o["q84"][ok]}, hmu=o["hcf_mu"][ok], hsig=hsig,
                     hq={"q16": o["hcf_q16"][ok], "q84": o["hcf_q84"][ok]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="ztf"); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=24)
    a = ap.parse_args()
    B = registry.load(a.bench, a.h)
    cache = ROOT / "data" / a.bench / "strict"; cache.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "results" / f"{a.bench}_strict"; out_dir.mkdir(parents=True, exist_ok=True)
    fold = B.meta.fold.to_numpy()
    tasks = []
    for r in range(5):
        t0 = time.time()
        o = rotation_outputs(B, a.h, r, a.threads, a.lr, a.cap, cache)
        tasks.append((r, make_task(B, a.h, o)))
        print(f"rotation {r}: outputs in {time.time() - t0:.0f}s", flush=True)
    for alpha in (0.32, 0.05):
        res = {}
        for name, spec in METHODS.items():
            fn, kw = spec if isinstance(spec, tuple) else (spec, {})
            parts = []; diags = []
            for r, task in tasks:
                pf = fold[task.obj]
                test = np.flatnonzero(pf == r); cal = np.flatnonzero(pf == (r + 1) % 5)
                c, lo, hi, ln = fn(task, cal, test, alpha, seed=5000 + r, **kw)
                if fn is methods.gamut:
                    diags.append(json.loads(json.dumps(methods.DIAG)))
                empty = ~(np.isfinite(lo) & np.isfinite(hi))
                c = c & ~empty; lo = np.where(empty, task.mu[test], lo); hi = np.where(empty, task.mu[test], hi)
                ln = np.where(empty, 0.0, ln)
                parts.append((task, test, c, lo, hi, ln))
            # pool: metrics on a concatenated pseudo-task
            covs = np.concatenate([p[2] for p in parts]); los = np.concatenate([p[3] for p in parts])
            his = np.concatenate([p[4] for p in parts]); lns = np.concatenate([p[5] for p in parts])
            T0 = parts[0][0]
            cat = lambda f: np.concatenate([getattr(p[0], f)[p[1]] for p in parts])
            pooled = data.Task(bench=T0.bench, h=T0.h, classes=T0.classes, y=T0.y, fold=T0.fold,
                               P=np.nanmean([p[0].P for p in parts], axis=0), L=T0.L, obj=cat("obj"), t=cat("t"),
                               band=cat("band"), yt=cat("yt"), lead_bin=cat("lead_bin"), mu=cat("mu"), sig=cat("sig"),
                               q={}, hmu=cat("hmu"), hsig=cat("hsig"), hq={})
            Lmerged = np.full(len(T0.y), -1)
            for p in parts:
                objs = np.unique(p[0].obj[p[1]]); Lmerged[objs] = p[0].L[objs]
            pooled.L = Lmerged
            r_ = metrics.evaluate(pooled, np.arange(len(covs)), covs, los, his, lns, alpha)
            # per (class, bin) object-level coverage
            z = pooled.y[pooled.obj]; cells = {}
            for c in range(pooled.K):
                for b in np.unique(pooled.lead_bin):
                    m = (z == c) & (pooled.lead_bin == b)
                    if m.any():
                        uo, inv = np.unique(pooled.obj[m], return_inverse=True)
                        co = np.bincount(inv, covs[m].astype(float)) / np.bincount(inv)
                        cells[f"{c}|{b}"] = [int(len(uo)), float(co.mean())]
            r_["cells_object"] = cells
            r_["bootstrap"] = object_bootstrap(pooled, covs, alpha, np.random.default_rng(0))
            if diags:
                r_["diag"] = diags
            res[name] = r_
            print(f"{a.bench} strict h={a.h:g} alpha={alpha}: {name:32s} marg {r_['marginal']:.3f} worst {r_['worst_class']:.3f} "
                  f"obj-worst {min(r_['per_class_object'].values()):.3f} width {r_['width']:.3f}", flush=True)
        (out_dir / f"h{a.h:g}_a{alpha:g}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
