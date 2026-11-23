"""Evaluation of the closure-probability study (pre-registered; see scripts/closure.py).

Calibrators are fitted on FLARE's `val` split and evaluated on `test`. Metrics: ECE (15 equal-mass bins), Brier, log
loss, AUC, calibration slope and intercept, per-true-class reliability; object-level bootstrap intervals (1,000
resamples) and paired differences against the deployed score.
Outputs: results/closure/eval.json, paper/tables/closure.tex, paper/closure_numbers.tex, paper/figures/closure.pdf
Run: python scripts/closure_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "closure"; PAPER = ROOT / "paper"
CLASSES = ["SN Ia", "SN CC", "SLSN", "AGN", "TDE", "CV"]
NB = 1000
EPS = 1e-4


def logit(p):
    p = np.clip(p, 0.005, 0.995); return np.log(p / (1 - p))


def fit_platt(s, y):
    m = LogisticRegression(C=1e6).fit(logit(s)[:, None], y)
    return lambda t: m.predict_proba(logit(t)[:, None])[:, 1]


def fit_iso(s, y):
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s, y)
    return lambda t: m.predict(t)


def fit_ivap(s, y):
    """Inductive Venn-Abers: p0, p1 from isotonic fits with the test point added as 0 and 1; p = p1 / (1 - p0 + p1)."""
    s = np.asarray(s, float); y = np.asarray(y, float)

    def pred(t):
        out = np.empty(len(t)); cache = {}
        for i, v in enumerate(np.asarray(t, float)):
            if v in cache:
                out[i] = cache[v]; continue
            ss = np.r_[s, v]
            p0 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ss, np.r_[y, 0.0]).predict([v])[0]
            p1 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ss, np.r_[y, 1.0]).predict([v])[0]
            cache[v] = out[i] = p1 / (1 - p0 + p1)
        return out
    return pred


def ece(p, y, bins=15):
    order = np.argsort(p, kind="stable"); parts = np.array_split(order, bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in parts if len(b)))


def metrics(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    out = dict(ece=ece(p, y), brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
               auc=float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"), mean_p=float(p.mean()), rate=float(y.mean()))
    lr = LogisticRegression(C=1e6).fit(logit(p)[:, None], y)
    out.update(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))
    return out


def boot(df, cols, y, rng):
    """Object-level bootstrap of ECE and Brier for every score column, and paired differences to the first column."""
    objs = df.obj_id.to_numpy(); uo, inv = np.unique(objs, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uo))]
    E = {c: [] for c in cols}; B = {c: [] for c in cols}
    for _ in range(NB):
        pick = rng.integers(0, len(uo), len(uo)); idx = np.concatenate([rows_of[k] for k in pick])
        for c in cols:
            p = np.clip(df[c].to_numpy()[idx], EPS, 1 - EPS); yy = y[idx]
            E[c].append(ece(p, yy)); B[c].append(float(np.mean((p - yy) ** 2)))
    ref = cols[0]; ref2 = "s0|ivap" if "s0|ivap" in cols else cols[0]
    out = {}
    for c in cols:
        e, b = np.array(E[c]), np.array(B[c])
        de, db = e - np.array(E[ref]), b - np.array(B[ref])
        db2 = b - np.array(B[ref2]); de2 = e - np.array(E[ref2])
        out[c] = dict(ece_ci=[float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5))],
                      brier_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                      d_ece_ci=[float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
                      d_brier_ci=[float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                      d_brier_va_ci=[float(np.percentile(db2, 2.5)), float(np.percentile(db2, 97.5))],
                      d_ece_va_ci=[float(np.percentile(de2, 2.5)), float(np.percentile(de2, 97.5))])
    return out


def variants(df):
    """Offline ablations of the mixture from the stored per-class closure estimates."""
    K = 6
    ph = df[[f"phat{k}" for k in range(K)]].to_numpy(); pc = df[[f"pclose{k}" for k in range(K)]].to_numpy()
    T = df["T"].to_numpy()[:, None]
    raw = ph ** T; raw /= raw.sum(1, keepdims=True)
    ok = np.isfinite(pc)
    mix = lambda W: np.nansum(np.where(ok, W, 0) * np.nan_to_num(pc), 1) / np.where(ok, W, 0).sum(1)
    am = np.nan_to_num(pc[np.arange(len(df)), ph.argmax(1)])
    return dict(s2_raw=mix(raw), s2_argmax=am)


def main():
    rng = np.random.default_rng(0)
    report = {}; tex_rows = []; macros = {}
    figdata = {}
    for h in (5, 10):
        f = RES / f"h{h}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        for k, v in variants(df).items():
            df[k] = v
        has3 = "s3" in df and df["s3"].notna().all()
        common = df[np.isfinite(df.s0) & np.isfinite(df.s1) & (np.isfinite(df.s3) if "s3" in df else True)].copy()
        val, te = common[common.split == "val"], common[common.split == "test"].copy()
        yv, yt = val.outcome.to_numpy().astype(float), te.outcome.to_numpy().astype(float)
        spec = [("s0", None, "Deployed $p_{\\rm singleton}$ (as shipped)"), ("s0", "platt", "\\quad + Platt"),
                ("s0", "iso", "\\quad + isotonic"), ("s0", "ivap", "\\quad + Venn--Abers"),
                ("s1", "ivap", "Deployed at the real time + Venn--Abers"),
                ("s3", "ivap", "Pooled forecaster, same residuals + Venn--Abers"),
                ("s2_argmax", "ivap", "Argmax class only + Venn--Abers"),
                ("s2_raw", "ivap", "Class mixture, raw posterior + Venn--Abers"),
                ("s2", None, "Class mixture (ours), uncalibrated"), ("s2", "ivap", "Class mixture (ours) + Venn--Abers")]
        cols = []
        res = {}
        for s, cal, disp in spec:
            if s not in te:
                continue
            name = f"{s}|{cal}"
            fit = dict(platt=fit_platt, iso=fit_iso, ivap=fit_ivap).get(cal)
            p = te[s].to_numpy() if fit is None else fit(val[s].to_numpy(), yv)(te[s].to_numpy())
            te[name] = p; cols.append(name)
            res[name] = dict(display=disp, **metrics(p, yt))
            # per class (test)
            pcl = {}
            for c in range(6):
                m = te.y.to_numpy() == c
                if m.sum() >= 5:
                    pcl[CLASSES[c]] = dict(n=int(m.sum()), mean_p=float(np.mean(p[m])), rate=float(yt[m].mean()),
                                           brier=float(np.mean((p[m] - yt[m]) ** 2)), ece=ece(np.clip(p[m], EPS, 1 - EPS), yt[m], bins=min(15, max(1, int(m.sum()) // 5))))
            res[name]["per_class"] = pcl
        bt = boot(te, cols, yt, rng)
        for name in cols:
            res[name].update(bt[name])
        # subset: sets with more than one class before the observation
        multi = te.set_before.to_numpy() > 1
        for name in cols:
            if multi.sum() > 30:
                res[name]["multi"] = metrics(te[name].to_numpy()[multi], yt[multi])
        if multi.sum() > 30:
            sub = te[multi].reset_index(drop=True)
            bsub = boot(sub, ["s0|None", "s0|ivap", "s2|ivap"], yt[multi], rng)
            res["s2|ivap"]["multi"]["d_brier_va_ci"] = bsub["s2|ivap"]["d_brier_va_ci"]
            res["s2|ivap"]["multi"]["d_brier_ci"] = bsub["s2|ivap"]["d_brier_ci"]
        # rows where the deployed score is undefined (no prefix detection in that band): only the mixture applies
        und = df[~np.isfinite(df.s0) & (df.split == "test")]
        report[h] = dict(n_test=int(len(te)), n_val=int(len(val)), n_test_objects=int(te.obj_id.nunique()),
                         rate_test=float(yt.mean()), n_multi=int(multi.sum()), n_deployed_undefined_test=int(len(und)),
                         T=float(df["T"].iloc[0]), methods=res)
        for name in cols:
            r = res[name]
            tex_rows.append(f"{h} & {r['display']} & {r['ece']:.3f} & [{r['ece_ci'][0]:.3f}, {r['ece_ci'][1]:.3f}] & {r['brier']:.3f} & "
                            f"{r['logloss']:.3f} & {r['auc']:.3f} & {r['slope']:.2f} \\\\")
        tex_rows.append("\\midrule")
        tag = {5: "Five", 10: "Ten"}[h]
        d, o = res["s0|None"], res["s2|ivap"]; dv = res["s0|ivap"]
        macros.update({f"Clo{tag}ECEDep": f"{d['ece']:.3f}", f"Clo{tag}ECEOurs": f"{o['ece']:.3f}",
                       f"Clo{tag}ECEDepVA": f"{dv['ece']:.3f}", f"Clo{tag}BrierDep": f"{d['brier']:.3f}",
                       f"Clo{tag}BrierOurs": f"{o['brier']:.3f}", f"Clo{tag}BrierDepVA": f"{dv['brier']:.3f}",
                       f"Clo{tag}AUCDep": f"{d['auc']:.3f}", f"Clo{tag}AUCOurs": f"{o['auc']:.3f}",
                       f"Clo{tag}SlopeDep": f"{d['slope']:.2f}", f"Clo{tag}N": f"{len(te):,}",
                       f"Clo{tag}NObj": f"{te.obj_id.nunique():,}", f"Clo{tag}Rate": f"{yt.mean():.2f}",
                       f"Clo{tag}Undef": f"{len(und):,}",
                       f"Clo{tag}DBrierHi": f"{o['d_brier_ci'][1]:.3f}", f"Clo{tag}DECEHi": f"{o['d_ece_ci'][1]:.3f}",
                       f"Clo{tag}DBrierVA": f"[{o['d_brier_va_ci'][0]:.3f}, {o['d_brier_va_ci'][1]:.3f}]",
                       f"Clo{tag}BrierGain": f"{100 * (1 - o['brier'] / d['brier']):.0f}\\%",
                       f"Clo{tag}BrierGainVA": f"{100 * (1 - o['brier'] / dv['brier']):.0f}\\%",
                       f"Clo{tag}LLDep": f"{d['logloss']:.3f}", f"Clo{tag}LLOurs": f"{o['logloss']:.3f}",
                       f"Clo{tag}BrierArgmax": f"{res['s2_argmax|ivap']['brier']:.3f}", f"Clo{tag}BrierRaw": f"{res['s2_raw|ivap']['brier']:.3f}",
                       f"Clo{tag}BrierRealTime": f"{res['s1|ivap']['brier']:.3f}", f"Clo{tag}ECEOursRaw": f"{res['s2|None']['ece']:.3f}"})
        figdata[h] = {n: (te[n].to_numpy(), yt) for n in ("s0|None", "s0|ivap", "s2|ivap")}
    # sensitivity: the first run, with the main experiments' cross-fitted C1 (trained on folds that include test objects)
    for h in (5, 10):
        f = RES / f"h{h}_crossfitC1.parquet"
        if f.exists() and h in report:
            dfc = pd.read_parquet(f); dfc = dfc[np.isfinite(dfc.s0) & np.isfinite(dfc.s1)]
            v, t = dfc[dfc.split == "val"], dfc[dfc.split == "test"]
            p = fit_ivap(v.s2.to_numpy(), v.outcome.to_numpy().astype(float))(t.s2.to_numpy())
            mm = metrics(p, t.outcome.to_numpy().astype(float))
            report[h]["crossfit_c1_s2_ivap"] = mm
            macros[f"Clo{ {5: 'Five', 10: 'Ten'}[h] }BrierCrossfit".replace(" ", "")] = f"{mm['brier']:.3f}"
    # registered comparisons (EXPERIMENT_PLAN.md): S2+IVAP vs S0 and vs S0+IVAP on ECE and Brier; AUC not lower
    reg = []
    for h, r in report.items():
        o = r["methods"]["s2|ivap"]; d = r["methods"]["s0|None"]; dv = r["methods"]["s0|ivap"]
        for ref, nm, dE, dB in (("S0", d, o["d_ece_ci"], o["d_brier_ci"]), ("S0 + VA", dv, o["d_ece_va_ci"], o["d_brier_va_ci"])):
            for met, diff, ci in (("ECE", o["ece"] - nm["ece"], dE), ("Brier", o["brier"] - nm["brier"], dB)):
                verdict = "met" if ci[1] < 0 else ("not met (no difference)" if ci[0] <= 0 else "not met (worse)")
                reg.append(f"{h} & ours vs {ref} & {met} & {diff:+.3f} & [{ci[0]:+.3f}, {ci[1]:+.3f}] & {verdict} \\\\")
            reg.append(f"{h} & ours vs {ref} & AUC & {o['auc'] - nm['auc']:+.3f} & -- & {'met' if o['auc'] >= nm['auc'] else 'not met'} \\\\")
        mm = r["methods"]["s2|ivap"].get("multi"); md = r["methods"]["s0|ivap"].get("multi"); m0 = r["methods"]["s0|None"].get("multi")
        if mm and md:
            ci = mm.get("d_brier_va_ci", [float("nan")] * 2)
            reg.append(f"{h} & sets with $>1$ class ($n={r['n_multi']}$) vs S0 + VA & Brier & {mm['brier'] - md['brier']:+.3f} & [{ci[0]:+.3f}, {ci[1]:+.3f}] & "
                       f"{'met' if ci[1] < 0 else 'not met'} \\\\")
            reg.append(f"{h} & sets with $>1$ class & ECE & ours {mm['ece']:.3f}, S0 {m0['ece']:.3f}, S0+VA {md['ece']:.3f} & -- & -- \\\\")
        reg.append("\\midrule")
    rtex = ("\\begin{table}[H]\n\\centering\n\\caption{Outcome of the pre-registered closure-probability comparisons on the test split "
            "(difference ours $-$ reference; paired object-level bootstrap interval). VA: Venn--Abers.}\n\\label{tab:closurereg}\n\\scriptsize\n"
            "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{rllccl}\n\\toprule\nDay & Comparison & Metric & Difference & 95\\% CI & Registered claim \\\\\n\\midrule\n"
            + "\n".join(reg[:-1]) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (PAPER / "tables" / "closure_reg.tex").write_text(rtex)
    for h, r in report.items():
        tag = {5: "Five", 10: "Ten"}[h]
        a_ = r["methods"]["s2_argmax|ivap"]["brier"]; o_ = r["methods"]["s2|ivap"]["brier"]
        macros[f"Clo{tag}MixGain"] = f"{a_ - o_:.3f}"
        if "s3|ivap" in r["methods"]:
            macros[f"Clo{tag}BrierPooled"] = f"{r['methods']['s3|ivap']['brier']:.3f}"
            macros[f"Clo{tag}AUCPooled"] = f"{r['methods']['s3|ivap']['auc']:.3f}"
        macros[f"Clo{tag}ECEVerdict"] = "lower" if r["methods"]["s2|ivap"]["d_ece_va_ci"][1] < 0 else "not significantly different"
    (PAPER / "closure_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "eval.json").write_text(json.dumps({str(k): v for k, v in report.items()}, indent=1))
    tex = ("\\begin{table}[H]\n\\centering\n\\caption{Closure probability on FLARE's test split (calibrators fitted on its validation split): "
           "probability that the deployed 90\\% set becomes a singleton after the object's real next detection. ECE with 15 equal-mass bins and its "
           "object-level 95\\% bootstrap interval; slope: calibration slope (1 is ideal).}\n\\label{tab:closure}\n\\scriptsize\n"
           "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{rlcccccc}\n\\toprule\nDay & Score & ECE & 95\\% CI & Brier & Log loss & AUC & Slope \\\\\n\\midrule\n"
           + "\n".join(tex_rows[:-1] if tex_rows and tex_rows[-1] == "\\midrule" else tex_rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    tex = tex.replace("\\\\\n\\midrule\n\\bottomrule", "\\\\\n\\bottomrule")
    (PAPER / "tables" / "closure.tex").write_text(tex)
    (PAPER / "closure_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    # per-class table
    pc_rows = []
    for h, r in report.items():
        for c in CLASSES:
            a = r["methods"]["s0|None"]["per_class"].get(c); b = r["methods"]["s0|ivap"]["per_class"].get(c)
            o = r["methods"]["s2|ivap"]["per_class"].get(c)
            if a and o:
                pc_rows.append(f"{h} & {c} & {a['n']} & {a['rate']:.2f} & {a['mean_p']:.2f} & {b['mean_p']:.2f} & {o['mean_p']:.2f} & "
                               f"{a['brier']:.3f} & {b['brier']:.3f} & {o['brier']:.3f} & {b['ece']:.3f} & {o['ece']:.3f} \\\\")
        pc_rows.append("\\midrule")
    ptex = ("\\begin{table}[H]\n\\centering\n\\caption{Closure probability per true class on the test split: observed closure rate, mean predicted "
            "probability and Brier score for the deployed score, the deployed score with Venn--Abers, and the class mixture with Venn--Abers (ours). "
            "Classes with at least five test rows.}\n\\label{tab:closureclass}\n\\scriptsize\n\\resizebox{\\columnwidth}{!}{%\n"
            "\\begin{tabular}{rlrccccccccc}\n\\toprule\n & & & & \\multicolumn{3}{c}{Mean predicted} & \\multicolumn{3}{c}{Brier} & \\multicolumn{2}{c}{ECE} \\\\\n"
            "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-12}\nDay & Class & $n$ & Rate & Dep. & Dep.+VA & Ours & Dep. & Dep.+VA & Ours & Dep.+VA & Ours \\\\\n\\midrule\n"
            + "\n".join(pc_rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n").replace("\\midrule\n\\bottomrule", "\\bottomrule")
    (PAPER / "tables" / "closure_class.tex").write_text(ptex)
    # reliability figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if figdata:
        fig, axes = plt.subplots(1, len(figdata), figsize=(3.3 * len(figdata), 3.0), squeeze=False)
        sty = {"s0|None": ("Deployed", "#9aa0a6", "o"), "s0|ivap": ("Deployed + Venn-Abers", "#e8710a", "s"),
               "s2|ivap": ("Class mixture + Venn-Abers (ours)", "#1a73e8", "D")}
        for ax, (h, dd) in zip(axes[0], sorted(figdata.items())):
            ax.plot([0, 1], [0, 1], ls="--", color="#5f6368", lw=1)
            for n, (p, y) in dd.items():
                order = np.argsort(p, kind="stable"); parts = [b for b in np.array_split(order, 10) if len(b)]
                ax.plot([p[b].mean() for b in parts], [y[b].mean() for b in parts], marker=sty[n][2], ms=4, lw=1.6,
                        color=sty[n][1], label=sty[n][0])
            ax.set_title(f"Day {h}", fontsize=9); ax.set_xlabel("predicted closure probability", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=7); ax.grid(alpha=.25, lw=.5)
        axes[0][0].set_ylabel("observed closure rate", fontsize=8)
        axes[0][0].legend(fontsize=6.5, frameon=False, loc="upper left")
        fig.tight_layout(); fig.savefig(PAPER / "figures" / "closure.pdf"); plt.close(fig)
    for h, r in report.items():
        print(f"day {h}: test rows {r['n_test']} (objects {r['n_test_objects']}), rate {r['rate_test']:.3f}, T {r['T']:.2f}, "
              f"deployed-undefined test rows {r['n_deployed_undefined_test']}")
        for n, m in r["methods"].items():
            print(f"   {m['display'][:48]:48s} ECE {m['ece']:.3f} [{m['ece_ci'][0]:.3f},{m['ece_ci'][1]:.3f}] Brier {m['brier']:.3f} "
                  f"LL {m['logloss']:.3f} AUC {m['auc']:.3f} slope {m['slope']:.2f} dB {np.round(m['d_brier_ci'],3)} dB_VA {np.round(m['d_brier_va_ci'],3)}")


if __name__ == "__main__":
    main()
