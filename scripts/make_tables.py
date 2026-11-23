"""Builds every number, table and figure of the paper from results/*/h*_a*.json (no number is typed by hand).

Outputs: paper/numbers.tex (macros), paper/tables/*.tex, paper/figures/*.pdf, results/summary.json.
Run: python scripts/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; PAPER = ROOT / "paper"
(PAPER / "tables").mkdir(parents=True, exist_ok=True); (PAPER / "figures").mkdir(parents=True, exist_ok=True)

MAIN = ["ztf", "plasticc_train", "plasticc_ddf", "ucr_starlightcurves", "ucr_ecg5000", "ucr_electricdevices"]
APPX = ["ucr_wafer"]
HELD = ["plasticc_wfd"]          # pre-registered held-out set (review round 4); never used for design
SYN = ["synthetic_k1", "synthetic_k0.5", "synthetic_k2"]
BNAME = {"ztf": "ZTF-BTS", "plasticc_train": "PLAsTiCC-train", "plasticc_ddf": "PLAsTiCC-DDF",
         "ucr_starlightcurves": "StarLight", "ucr_ecg5000": "ECG5000", "ucr_electricdevices": "ElecDevices",
         "ucr_wafer": "Wafer", "synthetic_k1": "Synth ($\\kappa{=}1$)", "synthetic_k0.5": "Synth ($\\kappa{=}0.5$)",
         "synthetic_k2": "Synth ($\\kappa{=}2$)", "plasticc_wfd": "PLAsTiCC-WFD (held out)"}
# method key -> (display name, family)
BASE = [("FLARE", "FLARE", "deployed"), ("SplitCP", "Split CP", "blind"), ("CQR", "CQR", "blind"), ("DCP", "DCP", "blind"),
        ("LocalizedCP", "Localized CP", "prob"), ("MixtureCP", "Label-shift mixture", "prob"), ("GCC-prob", "GCC ($p$)", "prob"),
        ("PartitionCP", "Learned partition", "prob"), ("CPTC", "CPTC", "prob"),
        ("TwoStepCP", "Two-step CP", "valid"), ("KandinskyCP", "Kandinsky CP", "valid"), ("MDCP", "MDCP", "valid"),
        ("MaxClassCP", "Max-class CP", "valid"),
        ("TwoStepCP + C1", "Two-step CP + C1", "validc1"), ("MDCP + C1", "MDCP + C1", "validc1"),
        ("MaxClassCP + C1", "Max-class CP + C1", "validc1")]
OURS = ("GAMUT", "\\gamut{} (ours)")
ORACLE = ("Oracle-true-class", "True-class Mondrian (reference)")
ABL = [("GAMUT", "\\gamut{} (full)"), ("GAMUT w/o C1 (pooled forecaster)", "w/o C1: pooled forecaster"),
       ("GAMUT w/o C2 (uniform posterior)", "w/o C2: uniform posterior"), ("GAMUT w/o C2 (argmax posterior)", "w/o C2: argmax posterior"),
       ("GAMUT w/o C2-calibration (raw posterior)", "w/o C2 calibration (raw $\\tilde p$)"),
       ("GAMUT w/o C3-dual (posterior mixture)", "w/o C3: posterior mixture"),
       ("GAMUT w/o C3-dual (prior-ratio weights)", "w/o C3: prior-ratio weights")]
EXTRA = [("GAMUT-cert (split certificate)", "\\gamut-cert (split certificate)"), ("GAMUT (point-level unit)", "\\gamut{} (point-level unit)"),
         ("GAMUT-union (early design)", "Union of class bands (early design)")]
TOL = {0.32: 0.02, 0.05: 0.015}
BASE_NAMES = [("GAMUT", "\\gamut"), ("GAMUT-cert (split certificate)", "\\gamut-cert"), ("Oracle-true-class", "Oracle (true class)"),
              ("TwoStepCP", "Two-step CP"), ("MaxClassCP", "Max-class CP"), ("SplitCP", "Split CP"), ("FLARE", "FLARE")]
LEVELS = (0.32, 0.05)


def load():
    R = {}
    for d in RES.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for f in d.glob("h*_a*.json"):
            h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
            R[(d.name, h, a)] = json.loads(f.read_text())
    return R


def settings(R, benches, alpha, need=("GAMUT",)):
    return sorted((b, h) for (b, h, a) in R if b in benches and a == alpha and all(n in R[(b, h, a)] for n in need))


BOOT = json.loads((RES / "bootstrap.json").read_text()) if (RES / "bootstrap.json").exists() else {}
_NAME = {}


def is_valid(res, alpha, setting=None, key=None):
    """Valid: no class significantly under-covered (object-level bootstrap, 95% upper bound below nominal); falls back to
    the pre-registered tolerance when no bootstrap is available for the setting."""
    if setting is not None and key is not None:
        bb = BOOT.get(f"{setting[0]}|{setting[1]:g}|{alpha:g}", {}).get(key)
        if bb is not None:
            return bb["n_under"] == 0
    return res["worst_class"] >= 1 - alpha - TOL[alpha]


def agg(R, benches, key, alpha, ref="FLARE"):
    """Aggregates over (bench, origin) settings with both key and ref present."""
    S = [(b, h) for (b, h) in settings(R, benches, alpha, need=(key, ref, "GAMUT"))]
    if not S:
        return None
    rs = [R[(b, h, alpha)][key] for b, h in S]; rf = [R[(b, h, alpha)][ref] for b, h in S]
    return dict(n=len(S), valid=np.mean([is_valid(r, alpha, (b, h), key) for r, (b, h) in zip(rs, S)]), worst=np.mean([r["worst_class"] for r in rs]),
                min_worst=np.min([r["worst_class"] for r in rs]), ccg=np.mean([r["ccg"] for r in rs]),
                rel_width=np.mean([r["width"] / f["width"] for r, f in zip(rs, rf)]),
                rel_cis=np.mean([r["class_avg_interval_score"] / f["class_avg_interval_score"] for r, f in zip(rs, rf)]),
                cells=np.mean([BOOT[f"{b}|{h:g}|{alpha:g}"][key]["n_cells_under"] for b, h in S
                               if key in BOOT.get(f"{b}|{h:g}|{alpha:g}", {})] or [np.nan]),
                marg=np.mean([r["marginal"] for r in rs]))


def f3(x):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f2(x):
    return "--" if x is None or not np.isfinite(x) else f"{x:.2f}"


def main_table(R):
    rows = []
    for key, disp, fam in BASE + [(OURS[0], OURS[1], "ours"), ("GAMUT-cert (split certificate)", "\\gamut-cert (split certificate)", "ours2"),
                                  (ORACLE[0], ORACLE[1], "oracle")]:
        cells = []
        for a in LEVELS:
            g = agg(R, MAIN, key, a)
            cells += ["--"] * 5 if g is None else [f"{100 * g['valid']:.0f}", f3(g["worst"]), f3(g["ccg"]),
                                                   "--" if not np.isfinite(g["cells"]) else f"{g['cells']:.1f}", f2(g["rel_width"])]
        rows.append((key, disp, fam, cells))
    # bold best among non-oracle for valid-rate (max), worst (closest to nominal from below handled as max), ccg (min)
    lines = []
    for key, disp, fam, cells in rows:
        pre = "\\rowcolor{gray!12}" if key == "GAMUT" else ""
        name = f"\\textcolor{{gray}}{{{disp}}}" if fam == "oracle" else disp
        lines.append(pre + name + " & " + " & ".join(cells) + " \\\\")
        if key in ("DCP", "CPTC", "MaxClassCP", "MaxClassCP + C1"):
            lines.append("\\midrule")
    n = agg(R, MAIN, "GAMUT", 0.32)
    body = "\n".join(lines)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{Summary over the six main benchmarks and all forecast origins ({n['n'] if n else 0} settings per level). \\emph{{Valid}}: percentage of settings in which no class is significantly under-covered (object-level bootstrap, 95\\% upper bound below nominal). \\emph{{Worst}}: mean over settings of the worst-class coverage. \\emph{{CCG}}: mean absolute class-coverage gap. \\emph{{Cells}}: mean number of (class, lead-time bin) cells per setting whose object-level coverage is significantly below nominal (bootstrap). \\emph{{Width}}: mean band length relative to FLARE's. Families from top: membership-blind, probability-conditioned, latent-class-valid, and the latter given our hypothesis-conditional forecaster (+\\,C1). \\gamut-cert is the split variant whose finite-sample guarantee (\\cref{{thm:valid}}) needs a strict split; here it is run under the same cross-fitted protocol as every method. The true-class Mondrian reference needs the test object's class.}}
\\label{{tab:main}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lccccc@{{\\hspace{{12pt}}}}ccccc}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{68\\% bands}} & \\multicolumn{{5}}{{c}}{{95\\% bands}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}}
Method & Valid (\\%) $\\uparrow$ & Worst $\\uparrow$ & CCG $\\downarrow$ & Cells $\\downarrow$ & Width $\\downarrow$ & Valid (\\%) $\\uparrow$ & Worst $\\uparrow$ & CCG $\\downarrow$ & Cells $\\downarrow$ & Width $\\downarrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (PAPER / "tables" / "main.tex").write_text(tex.replace("\\midrule\n\\rowcolor", "\\midrule\n\\rowcolor"))


def per_bench_table(R, benches, label, caption, methods):
    out = []
    for a in LEVELS:
        hdr = " & ".join(BNAME[b] for b in benches)
        lines = []
        for key, disp in methods:
            cells = []
            for b in benches:
                S = [h for (bb, h) in settings(R, [b], a, need=(key, "FLARE"))]
                if not S:
                    cells.append("--"); continue
                w = np.min([R[(b, h, a)][key]["worst_class"] for h in S])
                rw = np.mean([R[(b, h, a)][key]["width"] / R[(b, h, a)]["FLARE"]["width"] for h in S])
                cells.append(f"{w:.3f} / {rw:.2f}")
            lines.append(disp + " & " + " & ".join(cells) + " \\\\")
        out.append(f"\\multicolumn{{{len(benches) + 1}}}{{l}}{{\\textit{{{int(round(100 * (1 - a)))}\\% bands}}}} \\\\\n\\midrule\n" + "\n".join(lines))
    body = "\n\\midrule\n".join(out)
    tex = f"""\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'c' * len(benches)}}}
\\toprule
Method & {" & ".join(BNAME[b] for b in benches)} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table*}}
"""
    return tex


def per_origin_tables(R, bench):
    """Full metrics per origin for one benchmark (appendix)."""
    hs = sorted({h for (b, h, a) in R if b == bench})
    blocks = []
    for a in LEVELS:
        for h in hs:
            if (bench, h, a) not in R:
                continue
            res = R[(bench, h, a)]
            cls = res[next(iter(k for k in res if not k.startswith("_")))]["per_class"].keys()
            lines = []
            for key, disp in [(k, d) for k, d, _ in BASE] + [OURS, ORACLE] + [(k, d) for k, d in EXTRA]:
                if key not in res:
                    continue
                r = res[key]
                lines.append(disp + " & " + f3(r["marginal"]) + " & " + " & ".join(f2(r["per_class"].get(c, np.nan)) for c in cls)
                             + f" & {f3(r['worst_class'])} & {f3(r['ccg'])} & {f3(r['width'])} & {f3(r['interval_score'])} & {f3(r['class_avg_interval_score'])} \\\\")
            unit = "d" if bench in ("ztf", "plasticc_train", "plasticc_ddf", "plasticc_wfd") else "\\%"
            blocks.append(f"""\\begin{{table}}[H]
\\centering
\\caption{{{BNAME[bench]}, origin {h:g}{unit}, {int(round(100 * (1 - a)))}\\% bands. Per-class coverage (point-pooled), worst class, CCG, mean length, interval score (IS) and class-averaged IS.}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lc{'c' * len(cls)}ccccc}}
\\toprule
Method & Marg. & {" & ".join(c.replace('_', ' ') for c in cls)} & Worst & CCG & Width & IS & cls-IS \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
""")
    return "\n".join(blocks)


def ablation_table(R):
    lines = []
    STRUCT = [("GAMUT w/o adaptive cells (fixed lead bins)", "fixed lead bins (no merging)"),
              ("GAMUT w/o leave-one-object-out laws (in-sample)", "in-sample residual laws")]
    for key, disp in ABL + STRUCT[:1] + EXTRA[:1]:          # in-sample laws and point-level unit: tab:ablation_full
        cells = []
        for a in LEVELS:
            g = agg(R, MAIN, key, a, ref="GAMUT")
            cells += ["--"] * 5 if g is None else [f"{100 * g['valid']:.0f}", f3(g["worst"]), f3(g["ccg"]),
                                                   "--" if not np.isfinite(g["cells"]) else f"{g['cells']:.1f}", f2(g["rel_width"])]
        if key == "GAMUT":
            lines.append("\\rowcolor{gray!12}" + disp + " & " + " & ".join(cells) + " \\\\\n\\midrule")
        else:
            lines.append(disp + " & " + " & ".join(cells) + " \\\\")
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Ablations, aggregated over the six main benchmarks and all origins. Width is relative to full \\gamut. Columns as in \\cref{{tab:main}}.}}
\\label{{tab:ablation}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lccccc@{{\\hspace{{6pt}}}}ccccc}}
\\toprule
 & \\multicolumn{{5}}{{c}}{{68\\%}} & \\multicolumn{{5}}{{c}}{{95\\%}} \\\\
\\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}}
Variant & Val. & Worst & CCG & Cells & W. & Val. & Worst & CCG & Cells & W. \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (PAPER / "tables" / "ablation.tex").write_text(tex)


def numbers(R):
    M = {}
    z = R.get(("ztf", 5.0, 0.32), {})
    if "FLARE" in z:
        M["ZTFflareSLSNsixtyeight"] = f"{z['FLARE']['per_class']['SLSN']:.2f}"
        f = z["FLARE"]["misclassified_fraction"]; mc = z["FLARE"]["misclassified_coverage"]
        M["ZTFflareMisclsSixtyeight"] = f"{mc:.2f}"
        M["ZTFflareCorrectSixtyeight"] = f"{(z['FLARE']['marginal'] - f * mc) / (1 - f):.2f}"
    if "GAMUT" in z:
        M["ZTFgamutSLSNsixtyeight"] = f"{z['GAMUT']['per_class']['SLSN']:.2f}"
    # mean width reduction vs. the latent-class-valid baselines
    gains = []
    for a in LEVELS:
        for b, h in settings(R, MAIN, a, need=("GAMUT", "TwoStepCP", "MDCP", "MaxClassCP", "KandinskyCP")):
            for k in ("TwoStepCP", "MDCP", "MaxClassCP"):
                gains.append(1 - R[(b, h, a)]["GAMUT"]["width"] / R[(b, h, a)][k]["width"])
    M["WidthGainValidMean"] = f"{100 * np.mean(gains):.0f}\\%" if gains else "--"
    gains = []
    C1 = ("TwoStepCP + C1", "MDCP + C1", "MaxClassCP + C1")
    for a in LEVELS:
        for b, h in settings(R, MAIN, a, need=("GAMUT",) + C1):
            for k in C1:
                gains.append(1 - R[(b, h, a)]["GAMUT"]["width"] / R[(b, h, a)][k]["width"])
    if gains:
        M["WidthGainCOneMean"] = f"{100 * np.mean(gains):.0f}\\%"
    return M


MACROS = ["ZTFflareSLSNsixtyeight", "ZTFgamutSLSNsixtyeight", "ZTFflareMisclsSixtyeight", "ZTFflareCorrectSixtyeight",
          "WidthGainValidMean", "ZTFprobGainMax", "ZTFflareMargDev", "ZTFflareSLSNmin", "ZTFflareSLSNmax", "ZTFgamutMaxUnderSixtyeight", "ZTFgamutMaxUnderNinetyfive",
          "MainValidBestNonValidSixtyeight", "MainValidGamutSixtyeight", "MainCCGValidBaselinesSixtyeight",
          "MainWidthValidBaselinesSixtyeight", "MainCCGGamutSixtyeight", "MainCCGGamutNinetyfive", "MainWidthGamutSixtyeight",
          "MainWidthGamutNinetyfive", "AblWidthNoCone", "AblWidthUniform", "AblWidthArgmax", "AblWidthRaw", "AblWorstMixtureSixtyeight",
          "AblWidthBalanced", "SynRareBlindSixtyeight", "SynPropGapErr", "HyperMaxWorstChange", "HyperMaxWidthChange", "RuntimeGamutZTF",
          "DualItersMedian", "DualItersCapPct", "AblWorstNoLooZTF",
          "SynOracleGapSixtyeight", "SynOracleGapNinetyfive", "MainCellsGamutSixtyeight", "MainCellsFlareSixtyeight", "MainCellsKandinskySixtyeight",
          "MainCellsGamutNinetyfive", "MainCellsFlareNinetyfive", "MainValidTolGamutSixtyeight", "NinetyfiveWiderThanMDCP",
          "MainWidthCPTCNinetyfive", "MainValidCPTCSixtyeight", "MainWidthCertSixtyeight", "MainWidthCertNinetyfive",
          "ZTFWidthRatioSixtyeight", "ZTFWidthRatioNinetyfive", "ZTFfiveWidthRatioNinetyfive", "AblValidMixtureSixtyeight",
          "AblConeSynSixtyeight", "AblConeSynNinetyfive", "AblValidNoLooNinetyfive", "AblOneThreshWorstCell",
          "AblRawKtwoSixtyeight", "AblRawKtwoNinetyfive", "CertWorstSixtyeight", "CertWorstNinetyfive",
          "StrictWidthRatioNinetyfive", "DualItersCapSmall", "SynOracleGapValidSixtyeight", "SynOracleGapValidNinetyfive",
          "ZTFValidOverFlareSixtyeight", "ZTFValidOverFlareNinetyfive", "SynValidOverGamutSixtyeight", "SynValidOverGamutNinetyfive",
          "MainCellsTwoStepSixtyeight", "MainCellsMDCPSixtyeight", "MainCellsMaxClassSixtyeight", "MainCellsMDCPNinetyfive",
          "MainWidthCOneBaselinesSixtyeight", "MainWidthCOneBaselinesNinetyfive", "MainValidCOneBaselinesSixtyeight",
          "MainValidCOneBaselinesNinetyfive", "MainCellsCOneBaselinesSixtyeight", "MainCellsCOneBaselinesNinetyfive", "WidthGainCOneMean"]


def write_macros(M):
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())
    for k in MACROS:
        if k not in M:
            tex += f"\n\\newcommand{{\\{k}}}{{\\textbf{{??}}}}"
    (PAPER / "numbers.tex").write_text(tex + "\n")
    (RES / "macros.json").write_text(json.dumps(M, indent=1))


def bench_table():
    import pandas as pd
    import sys
    sys.path.insert(0, str(ROOT))
    from gamut import data
    rows = []
    origins = {"ztf": "5,10,15,20,30 d", "plasticc_train": "5,15,30 d", "plasticc_ddf": "5,15,30 d",
               "ucr_starlightcurves": "20,40,60\\%", "ucr_ecg5000": "20,40,60\\%", "ucr_electricdevices": "20,40,60\\%",
               "ucr_wafer": "20\\%", "plasticc_wfd": "5,15,30 d"}
    domain = {"ztf": "ZTF transients", "plasticc_train": "LSST sim.", "plasticc_ddf": "LSST sim. (DDF)",
              "ucr_starlightcurves": "stellar light curves", "ucr_ecg5000": "ECG", "ucr_electricdevices": "electricity",
              "ucr_wafer": "semiconductor", "plasticc_wfd": "LSST sim. (WFD)"}
    for b in MAIN + APPX + HELD:
        d = ROOT / "data" / b
        hs = sorted(float(x.stem.split("_h")[1]) for x in d.glob("oof_h*.npz"))
        if not hs:
            continue
        h = hs[0]
        t = data.load(b, h)
        objs = np.unique(t.obj); cnt = np.bincount(t.y[objs], minlength=t.K)
        rows.append(f"{BNAME[b]} & {domain[b]} & {len(objs):,} & {t.K} & {cnt.min()} & {len(t.yt):,} & {origins[b]} \\\\")
    tex = """\\begin{table}[t]
\\centering
\\caption{Benchmarks. Objects with at least one future point, and future points, at the earliest origin; \\emph{min} is the size of the smallest class.}
\\label{tab:bench}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{llrrrrl}
\\toprule
Benchmark & Domain & Objects & $K$ & min & Points & Origins \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (PAPER / "tables" / "bench.tex").write_text(tex)


# ---------------------------------------------------------------- appendix tables and remaining macros
PROB = ["LocalizedCP", "MixtureCP", "GCC-prob", "PartitionCP", "CPTC"]
VALIDB = ["TwoStepCP", "KandinskyCP", "MDCP", "MaxClassCP"]
_TASKS = {}


def _task(bench, h):
    import sys
    sys.path.insert(0, str(ROOT))
    from gamut import data
    if (bench, h) not in _TASKS:
        _TASKS[(bench, h)] = data.load(bench, h)
    return _TASKS[(bench, h)]


_PTS = {}
_CELLS = {}


def cell_stats(bench, h, a, keys, min_obj=40):
    """Mean |deviation| and min deviation of coverage from nominal over (class, lead bin) cells with >= min_obj objects."""
    f = RES / bench / f"points_h{h:g}_a{a:g}.npz"
    if not f.exists():
        return {}
    try:
        if (bench, h, a) not in _PTS:
            _PTS[(bench, h, a)] = dict(np.load(f))
        P = _PTS[(bench, h, a)]; t = _task(bench, h)
    except Exception:                       # file being written by a running job
        return {}
    out = {}
    if (bench, h) not in _CELLS:
        z = t.y[t.obj]; cl = []
        for c in range(t.K):
            for b in range(int(t.lead_bin.max()) + 1):
                m = (z == c) & (t.lead_bin == b)
                if m.any() and len(np.unique(t.obj[m])) >= min_obj:
                    cl.append(m)
        _CELLS[(bench, h)] = cl
    cells = _CELLS[(bench, h)]
    for k in keys:
        kk = k.replace(" ", "_").replace("/", "_") + "__cov"
        if kk not in P or not cells:
            continue
        dev = np.array([P[kk][m].mean() - (1 - a) for m in cells])
        out[k] = (float(np.abs(dev).mean()), float(dev.min()))
    return out


def observation_table(R, M):
    hs = sorted(h for (b, h, a) in R if b == "ztf" and a == 0.32 and "FLARE" in R[(b, h, a)])
    lines = []; gains = []; mdev = []
    for a in LEVELS:
        for h in hs:
            if ("ztf", h, a) not in R:
                continue
            r = R[("ztf", h, a)]; fl = r["FLARE"]
            f, mc = fl["misclassified_fraction"], fl["misclassified_coverage"]
            cor = (fl["marginal"] - f * mc) / (1 - f)
            bestp = max((r[k]["worst_class"], k) for k in PROB if k in r)
            gains.append(bestp[0] - fl["worst_class"]); mdev.append(abs(fl["marginal"] - (1 - a)))
            lines.append(f"{int(round(100 * (1 - a)))}\\% & {h:g} & {fl['marginal']:.3f} & " + " & ".join(f"{fl['per_class'][c]:.2f}" for c in fl["per_class"])
                         + f" & {cor:.2f} & {mc:.2f} & {100 * f:.0f}\\% & {bestp[0]:.3f} ({bestp[1]}) \\\\")
    cls = list(R[("ztf", hs[0], 0.32)]["FLARE"]["per_class"]) if hs else []
    tex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Observation study on ZTF-BTS: FLARE's bands by origin. Per-class coverage (point-pooled), coverage of correctly and incorrectly classified objects, the misclassified fraction, and the best worst-class coverage among the probability-conditioned methods.}}
\\label{{tab:observation}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{llc{'c' * len(cls)}cccl}}
\\toprule
Level & Origin (d) & Marg. & {' & '.join(c.replace('_', ' ') for c in cls)} & Correct & Miscl. & Miscl. frac. & Best prob.-conditioned worst \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    (PAPER / "tables" / "observation.tex").write_text(tex)
    if gains:
        M["ZTFprobGainMax"] = f"{max(gains):.3f}"; M["ZTFflareMargDev"] = f"{max(mdev):.3f}"
    sl = [R[("ztf", h, 0.32)]["FLARE"]["per_class"][c] for h in hs for c in ("SLSN", "TDE")]
    if sl:
        M["ZTFflareSLSNmin"] = f"{min(sl):.2f}"; M["ZTFflareSLSNmax"] = f"{max(sl):.2f}"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        gd = [(1 - a) - R[("ztf", h, a)]["GAMUT"]["per_class"][c] for h in hs if ("ztf", h, a) in R and "GAMUT" in R[("ztf", h, a)] for c in cls]
        if gd:
            M[f"ZTFgamutMaxUnder{tag}"] = f"{max(0.0, max(gd)):.3f}"


def ablation_full(R, M):
    keys = [k for k, _ in ABL] + ["GAMUT w/o lead-bin cells (one threshold)", "GAMUT w/o scale normalization (absolute length)",
                                  "GAMUT w/o leave-one-object-out laws (in-sample)", "GAMUT w/o adaptive cells (fixed lead bins)"] + [k for k, _ in EXTRA[:2]]
    disp = dict(ABL + EXTRA); disp.update({"GAMUT w/o lead-bin cells (one threshold)": "w/o C3 lead-time cells (one threshold)",
                                         "GAMUT w/o scale normalization (absolute length)": "w/o C3 scale normalisation",
                                         "GAMUT w/o leave-one-object-out laws (in-sample)": "w/o C1 leave-one-object-out laws",
                                         "GAMUT w/o adaptive cells (fixed lead bins)": "w/o C3 adaptive horizon cells"})
    blocks = []
    for a in LEVELS:
        lines = []
        for k in keys:
            cells = []
            for b in MAIN:
                S = [h for (bb, h) in settings(R, [b], a, need=(k, "GAMUT"))]
                if not S:
                    cells.append("--"); continue
                w = np.min([R[(b, h, a)][k]["worst_class"] for h in S])
                rw = np.mean([R[(b, h, a)][k]["width"] / R[(b, h, a)]["GAMUT"]["width"] for h in S])
                cd = [cell_stats(b, h, a, [k]).get(k, (np.nan, np.nan))[0] for h in S]
                cells.append(f"{w:.3f} / {rw:.2f} / {np.nanmean(cd):.3f}")
            lines.append(disp.get(k, k) + " & " + " & ".join(cells) + " \\\\")
        blocks.append(f"\\multicolumn{{{len(MAIN) + 1}}}{{l}}{{\\textit{{{int(round(100 * (1 - a)))}\\% bands}}}} \\\\\n\\midrule\n" + "\n".join(lines))
    tex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Ablations per benchmark: worst-class coverage (minimum over origins) / band length relative to \\gamut{{}} / mean absolute deviation from nominal over (class, lead-time bin) cells with at least 40 objects.}}
\\label{{tab:ablation_full}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{l{'c' * len(MAIN)}}}
\\toprule
Variant & {' & '.join(BNAME[b] for b in MAIN)} \\\\
\\midrule
{(chr(10) + '\\midrule' + chr(10)).join(blocks)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    (PAPER / "tables" / "ablation_full.tex").write_text(tex)
    for key, mac in [("GAMUT w/o C1 (pooled forecaster)", "AblWidthNoCone"), ("GAMUT w/o C2 (uniform posterior)", "AblWidthUniform"),
                     ("GAMUT w/o C2 (argmax posterior)", "AblWidthArgmax"), ("GAMUT w/o C2-calibration (raw posterior)", "AblWidthRaw"),
                     ("GAMUT w/o C3-dual (prior-ratio weights)", "AblWidthBalanced")]:
        g = [agg(R, MAIN, key, a, ref="GAMUT") for a in LEVELS]
        g = [x for x in g if x]
        if g:
            M[mac] = f"{100 * (np.mean([x['rel_width'] for x in g]) - 1):+.0f}\\%"
    z20 = R.get(("ztf", 20.0, 0.32), {}).get("GAMUT w/o leave-one-object-out laws (in-sample)")
    if z20:
        M["AblWorstNoLooZTF"] = f"{z20['per_class']['CV']:.3f}"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        for key, mac in (("GAMUT w/o adaptive cells (fixed lead bins)", "FixedBins"), ("GAMUT w/o C1 (pooled forecaster)", "NoCone"),
                         ("GAMUT w/o C2-calibration (raw posterior)", "Raw")):
            g = agg(R, MAIN, key, a, ref="GAMUT")
            if g:
                M[f"AblW{mac}{tag}"] = f"{100 * (g['rel_width'] - 1):+.0f}\\%"
                M[f"AblCells{mac}{tag}"] = f"{g['cells']:.1f}"; M[f"AblValid{mac}{tag}"] = f"{100 * g['valid']:.0f}\\%"
    g = agg(R, MAIN, "GAMUT w/o C3-dual (posterior mixture)", 0.32, ref="GAMUT")
    if g:
        M["AblWorstMixtureSixtyeight"] = f"{g['worst']:.3f}"


def hyper_table(M):
    lines = []; dw = []; dwid = []; dccg = []; dfix = []
    for f in sorted(RES.glob("*/validation_h*_a*.json")):
        b = f.parent.name; h = float(f.stem.split("_h")[1].split("_a")[0]); a = float(f.stem.split("_a")[1])
        V = json.loads(f.read_text())
        if "default" not in V:
            continue
        d = V["default"]
        for k, r in V.items():
            lines.append(f"{BNAME.get(b, b)} & {h:g} & {int(round(100 * (1 - a)))}\\% & {k.replace('_', ' ').replace('alpha', '$\\alpha$')} & {r['worst_class']:.3f} & {r['ccg']:.3f} & {r['width']:.3f} & {r['class_avg_interval_score']:.3f} \\\\")
            if k.startswith(("min_bin", "bandwidth", "prune", "dual iterations")):
                dw.append(abs(r["worst_class"] - d["worst_class"])); dwid.append(abs(r["width"] / d["width"] - 1))
            if k.startswith("fixed lead bins"):
                dfix.append(r["width"] / d["width"] - 1)
            if k == "dual iterations=5":
                dccg.append((r["ccg"] - d["ccg"], d["ccg"], r["ccg"], BNAME.get(b, b)))
    tex = f"""\\begin{{longtable}}{{lllp{{4.2cm}}cccc}}
\\caption{{Hyperparameter analysis on validation folds only (nested leave-one-fold-out inside the four calibration folds of every outer fold; the outer test fold is never used).}}
\\label{{tab:hyper}}\\\\
\\toprule
Benchmark & Origin & Level & Variant & Worst & CCG & Width & cls-IS \\\\
\\midrule
\\endfirsthead
\\toprule
Benchmark & Origin & Level & Variant & Worst & CCG & Width & cls-IS \\\\
\\midrule
\\endhead
{chr(10).join(lines) if lines else '\\multicolumn{8}{c}{(pending)} \\\\'}
\\bottomrule
\\end{{longtable}}
"""
    (PAPER / "tables" / "hyper.tex").write_text(tex)
    if dw:
        M["HyperMaxWorstChange"] = f"{max(dw):.3f}"; M["HyperMaxWidthChange"] = f"{100 * max(dwid):.0f}\\%"
    if dfix:
        M["HyperFixedBinsWidthMin"] = f"{100 * min(dfix):.0f}\\%"; M["HyperFixedBinsWidthMax"] = f"{100 * max(dfix):.0f}\\%"
    if dccg:
        w = max(dccg)
        M["HyperDualFiveCCGFrom"] = f"{w[1]:.3f}"; M["HyperDualFiveCCGTo"] = f"{w[2]:.3f}"; M["HyperDualFiveCCGBench"] = w[3]


def cert_table(R):
    lines = []
    for b in MAIN + APPX:
        for a in LEVELS:
            S = [h for (bb, h) in settings(R, [b], a, need=("GAMUT", "GAMUT-cert (split certificate)", "GAMUT (point-level unit)"))]
            if not S:
                continue
            row = [BNAME[b], f"{int(round(100 * (1 - a)))}\\%"]
            for k in ("GAMUT", "GAMUT-cert (split certificate)", "GAMUT (point-level unit)"):
                rr = [R[(b, h, a)][k] for h in S]
                row += [f"{np.mean([r['marginal'] for r in rr]):.3f}", f"{np.min([r['worst_class'] for r in rr]):.3f}",
                        f"{np.mean([r['width'] / R[(b, h, a)]['GAMUT']['width'] for r, h in zip(rr, S)]):.2f}"]
            lines.append(" & ".join(row) + " \\\\")
    tex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Cross-conformal protocol: \\gamut{{}} (in-sample dual weights, object-level unit), \\gamut-cert (split certificate; its finite-sample premise needs the strict split of \\cref{{tab:strict}}) and \\gamut{{}} with point-level units: mean marginal coverage, worst-class coverage (minimum over origins) and length relative to \\gamut.}}
\\label{{tab:cert}}
\\scriptsize
\\begin{{tabular}}{{llccc ccc ccc}}
\\toprule
 & & \\multicolumn{{3}}{{c}}{{\\gamut}} & \\multicolumn{{3}}{{c}}{{\\gamut-cert}} & \\multicolumn{{3}}{{c}}{{point-level unit}} \\\\
\\cmidrule(lr){{3-5}}\\cmidrule(lr){{6-8}}\\cmidrule(lr){{9-11}}
Benchmark & Level & Marg. & Worst & W. & Marg. & Worst & W. & Marg. & Worst & W. \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (PAPER / "tables" / "cert.tex").write_text(tex)


def synthetic_table(R, M):
    lines = []
    O = json.loads((RES / "synthetic_oracle.json").read_text()) if (RES / "synthetic_oracle.json").exists() else {}
    meth = [(k, d) for k, d, _ in BASE] + [OURS, ORACLE]
    blind = []
    for kappa in ("1", "0.5", "2"):
        b = f"synthetic_k{kappa}"
        for a in LEVELS:
            r = R.get((b, 0.0, a))
            if not r:
                continue
            for k, d in meth:
                if k not in r:
                    continue
                pc = r[k]["per_class"]
                lines.append(f"{kappa} & {int(round(100 * (1 - a)))}\\% & {d} & " + " & ".join(f"{pc[c]:.3f}" for c in ("common", "medium", "rare"))
                             + f" & {r[k]['width']:.3f} & {r[k].get('empty_fraction', 0):.3f} \\\\")
                if a == 0.32 and kappa == "1" and k in ["FLARE", "SplitCP", "CQR", "DCP"] + PROB:
                    blind.append(pc["rare"])
            o = O.get(f"oracle_a{a:g}")
            if o and kappa == "1":
                lines.append(f"-- & {int(round(100 * (1 - a)))}\\% & Oracle band (Prop.~\\ref{{prop:oracle}}) & " + " & ".join(f"{o['class_coverage'][str(c)]:.3f}" for c in range(3)) + f" & {o['mean_length']:.3f} & 0 \\\\")
            lines.append("\\midrule")
    if blind:
        M["SynRareBlindSixtyeight"] = f"{min(blind):.2f}--{max(blind):.2f}"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        r = R.get(("synthetic_k1", 0.0, a)); o = O.get(f"oracle_a{a:g}")
        if r and o and "GAMUT" in r:
            M[f"SynOracleGap{tag}"] = f"{100 * (r['GAMUT']['width'] / o['mean_length'] - 1):.1f}\\%"
            vb = [r[k]["width"] / o["mean_length"] - 1 for k in ("TwoStepCP", "MDCP", "MaxClassCP") if k in r]
            if vb:
                M[f"SynOracleGapValid{tag}"] = f"{100 * min(vb):.0f}--{100 * max(vb):.0f}\\%"
    errs = []; plines = []
    for key, val in O.items():
        if not key.startswith("prop2"):
            continue
        for m, d in val.items():
            for c in ("0", "1", "2"):
                errs.append(abs(d["empirical"][c] - d["predicted"][c]))
            plines.append(f"{key.replace('prop2_', '').replace('_', ' ')} & {m} & " + " & ".join(f"{d['empirical'][c]:.3f} / {d['predicted'][c]:.3f}" for c in ("0", "1", "2")) + " \\\\")
    if errs:
        M["SynPropGapErr"] = f"{max(errs):.3f}"
    tex = f"""\\begin{{longtable}}{{llp{{4cm}}ccccc}}
\\caption{{Synthetic family ($\\kappa$: posterior sharpness; 1 = Bayes). Per-class coverage, mean length and fraction of empty bands. The oracle band uses the true posterior and class laws.}}
\\label{{tab:synthetic}}\\\\
\\toprule
$\\kappa$ & Level & Method & common & medium & rare & Width & Empty \\\\
\\midrule
\\endfirsthead
\\toprule
$\\kappa$ & Level & Method & common & medium & rare & Width & Empty \\\\
\\midrule
\\endhead
{chr(10).join(lines)}
\\bottomrule
\\end{{longtable}}

\\begin{{table}}[H]
\\centering
\\caption{{\\cref{{prop:gap}} on synthetic data: empirical / predicted class coverage, the prediction computed from the true posterior and class laws for each method's band.}}
\\label{{tab:prop2}}
\\scriptsize
\\begin{{tabular}}{{llccc}}
\\toprule
Setting & Method & common & medium & rare \\\\
\\midrule
{chr(10).join(plines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (PAPER / "tables" / "synthetic.tex").write_text(tex)


def runtime_table(R, M):
    keys = [(k, d) for k, d, _ in BASE] + [OURS]
    lines = []
    for k, d in keys:
        secs = [R[(b, h, a)][k]["seconds"] / 5 for (b, h, a) in R if b == "ztf" and k in R[(b, h, a)] and "seconds" in R[(b, h, a)][k]]
        if secs:
            lines.append(f"{d} & {np.mean(secs):.1f} \\\\")
            if k == "GAMUT":
                M["RuntimeGamutZTF"] = f"{np.mean(secs):.0f}"
    tex = f"""\\begin{{table}}[H]
\\centering
\\caption{{Calibration plus prediction time per fold on ZTF-BTS (seconds, one CPU core, mean over origins and levels; about 190,000 calibration and 48,000 test points per fold at day 5).}}
\\label{{tab:runtime}}
\\scriptsize
\\begin{{tabular}}{{lc}}
\\toprule
Method & Seconds per fold \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (PAPER / "tables" / "runtime.tex").write_text(tex)


def unfavourable_table(R):
    lines = []; cnt = dict(notvalid=0, narrower=0, betteris=0, total=0)
    for a in LEVELS:
        for b, h in settings(R, MAIN + APPX, a):
            r = R[(b, h, a)]; g = r["GAMUT"]; cnt["total"] += 1
            tag = f"{BNAME[b]} & {h:g} & {int(round(100 * (1 - a)))}\\%"
            if g.get("empty_fraction", 0) > 0.01:
                lines.append(f"{tag} & \\gamut{{}} empty bands (counted as misses) & {100 * g['empty_fraction']:.1f}\\% of points \\\\")
            if not is_valid(g, a, (b, h), "GAMUT"):
                cnt["notvalid"] += 1
                lines.append(f"{tag} & \\gamut{{}} not valid & worst {g['worst_class']:.3f} \\\\")
            for k in VALIDB:
                if k in r and is_valid(r[k], a, (b, h), k) and r[k]["width"] < g["width"]:
                    cnt["narrower"] += 1
                    lines.append(f"{tag} & valid {k} narrower & {r[k]['width']:.3f} vs {g['width']:.3f} \\\\")
            best = min(((r[k]["class_avg_interval_score"], k) for k in VALIDB + ["FLARE"] + PROB if k in r and is_valid(r[k], a, (b, h), k)), default=None)
            if best and best[0] < g["class_avg_interval_score"]:
                cnt["betteris"] += 1
                lines.append(f"{tag} & valid {best[1]} lower cls-IS & {best[0]:.3f} vs {g['class_avg_interval_score']:.3f} \\\\")
    tex = f"""Every setting (benchmark, origin, level) of the six main benchmarks and Wafer in which \\gamut{{}} is not class-valid, a latent-class-valid baseline is narrower, or a valid method has a lower class-averaged interval score. Of {cnt['total']} settings: {cnt['notvalid']} not valid, {cnt['narrower']} narrower valid baseline (method-setting pairs), {cnt['betteris']} lower class-averaged IS by a valid method.

\\begin{{longtable}}{{lllll}}
\\toprule
Benchmark & Origin & Level & Event & Values \\\\
\\midrule
\\endhead
{chr(10).join(lines) if lines else '\\multicolumn{5}{c}{none} \\\\'}
\\bottomrule
\\end{{longtable}}
"""
    (PAPER / "tables" / "unfavourable.tex").write_text(tex)
    return cnt


def diag_macros(R, M):
    """Dual iterations, realised horizon cells, excluded class-cells and borrowed cells, for GAMUT and GAMUT-cert,
    cross-conformal (all benchmarks) and strict split (ZTF)."""
    it = []; rows = []
    sources = [((b, h, a), res) for (b, h, a), res in R.items() if not b.endswith("_strict")]   # strict runs are added below
    for f in sorted((RES / "ztf_strict").glob("h*_a*.json")) if (RES / "ztf_strict").exists() else []:
        h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
        sources.append((("ztf (strict split)", h, a), json.loads(f.read_text())))
    for (b, h, a), res in sorted(sources, key=lambda x: (str(x[0][0]), x[0][1], x[0][2])):
        for k, disp in (("GAMUT", "\\gamut"), ("GAMUT-cert (split certificate)", "cert")):
            if k not in res or not res[k].get("diag"):
                continue
            D = res[k]["diag"]
            if k == "GAMUT" and b in MAIN:
                for d in D:
                    it += d["iterations"]
            ncell = [max(d.get("cells", [0])) + 1 for d in D]
            excl = sum(len(d["excluded"]) for d in D); fb = sum(len(d["fallback_bins"]) for d in D)
            ex_cls = sorted({e[1] for d in D for e in d["excluded"]})
            rows.append(f"{BNAME.get(b, b.replace('_', ' '))} & {h:g} & {int(round(100 * (1 - a)))}\\% & {disp} & {min(ncell)}--{max(ncell)} & {excl} & {fb} \\\\")
    if it:
        it = np.array(it)
        M["DualItersMedian"] = f"{int(np.median(it))}"; M["DualItersCapPct"] = f"{100 * np.mean(it >= 100):.0f}\\%"
    tex = ("Realised horizon cells (out of FLARE's lead-time bins), class-cells excluded from the threshold because the class has fewer than $1/\\alpha-1$ "
           "(certification) objects in the cell, and cells that borrowed a neighbour's threshold, summed over folds or rotations.\n\n"
           "\\begin{longtable}{llllccc}\n\\toprule\nBenchmark & Origin & Level & Variant & Cells & Excluded & Borrowed \\\\\n\\midrule\n\\endhead\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{longtable}\n")
    (PAPER / "tables" / "excluded.tex").write_text(tex)


def strict_table(M):
    """Strict three-way split on ZTF at every origin: per-setting rows (appendix longtable) and a summary over origins."""
    lines = []; summ = {}
    files = sorted((RES / "ztf_strict").glob("h*_a*.json"), key=lambda f: (float(f.stem.split("_a")[0][1:]), -float(f.stem.split("_a")[1]))) \
        if (RES / "ztf_strict").exists() else []
    order = ["FLARE", "SplitCP", "CPTC", "TwoStepCP", "KandinskyCP", "MDCP", "MaxClassCP", "TwoStepCP + C1", "MDCP + C1",
             "MaxClassCP + C1", "GAMUT", "GAMUT-cert (split certificate)", "Oracle-true-class"]
    names = dict(BASE_NAMES); names.update({k: d for k, d, _ in BASE}); names["GAMUT"] = "\\gamut"
    for f in files:
        h = float(f.stem.split("_a")[0][1:]); a = float(f.stem.split("_a")[1])
        R = json.loads(f.read_text())
        if "FLARE" not in R:
            continue
        for k in [k for k in order if k in R]:
            r = R[k]; bt = r.get("bootstrap")
            rel = r["width"] / R["FLARE"]["width"]
            nu = "--" if bt is None else str(bt["n_under"]); nc = "--" if bt is None else str(bt["n_cells_under"])
            lines.append(f"{h:g} & {int(round(100 * (1 - a)))}\\% & {names.get(k, k)} & {r['marginal']:.3f} & "
                         + " & ".join(f"{v:.2f}" for v in r["per_class"].values())
                         + f" & {min(r['per_class_object'].values()):.3f} & {nu} & {nc} & {rel:.2f} \\\\")
            if bt is not None:
                summ.setdefault((k, a), []).append((bt["n_under"] == 0, bt["n_cells_under"], rel, h))
        lines.append("\\midrule")
    tex = ("{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}\n\\begin{longtable}{lllccccccccccc}\n\\caption{Strict three-way split on ZTF-BTS at every origin (training, calibration and test "
           "objects disjoint; five rotations pooled). Per-class coverage pooled over points, the worst object-averaged class coverage, the "
           "number of classes and of (class, lead-time bin) cells significantly under-covered (object-level bootstrap, as in the main "
           "text), and the length relative to FLARE's.}\n\\label{tab:strict}\\\\\n\\toprule\n"
           "Orig. & Lvl. & Method & Marg. & SN Ia & CC SN & SLSN & AGN & TDE & CV & Worst & Cls. & Cells & Width \\\\\n\\midrule\n\\endfirsthead\n"
           "\\toprule\nOrig. & Lvl. & Method & Marg. & SN Ia & CC SN & SLSN & AGN & TDE & CV & Worst & Cls. & Cells & Width \\\\\n\\midrule\n\\endhead\n"
           + ("\n".join(lines) if lines else "\\multicolumn{14}{c}{(pending)} \\\\") + "\n\\bottomrule\n\\end{longtable}}\n")
    tex = tex.replace("\\midrule\n\\bottomrule", "\\bottomrule")
    # summary over origins
    srows = []
    for k in order:
        cells = []
        for a in LEVELS:
            v = summ.get((k, a))
            cells += ["--"] * 3 if not v else [f"{100 * np.mean([x[0] for x in v]):.0f}", f"{np.mean([x[1] for x in v]):.1f}", f"{np.mean([x[2] for x in v]):.2f}"]
        if any(c != "--" for c in cells):
            srows.append(f"{names.get(k, k)} & " + " & ".join(cells) + " \\\\")
    norig = len({x[3] for v in summ.values() for x in v})
    stex = ("\\begin{table}[H]\n\\centering\n\\caption{Strict three-way split on ZTF-BTS, summary over the " + str(norig) + " origins run: "
            "percentage of origins with no class significantly under-covered, mean number of under-covered (class, lead-time bin) cells, "
            "and mean length relative to FLARE's. Only this protocol meets the premise of \\cref{thm:valid}.}\n\\label{tab:strictsum}\n\\scriptsize\n"
            "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccccc}\n\\toprule\n & \\multicolumn{3}{c}{68\\%} & \\multicolumn{3}{c}{95\\%} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMethod & Valid & Cells & Width & Valid & Cells & Width \\\\\n\\midrule\n"
            + "\n".join(srows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (PAPER / "tables" / "strict.tex").write_text(stex + "\n" + tex)
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        for k, mac in (("GAMUT", "Gamut"), ("GAMUT-cert (split certificate)", "Cert"), ("TwoStepCP", "TwoStep"), ("MDCP", "MDCP"),
                       ("MaxClassCP", "MaxClass"), ("FLARE", "Flare")):
            v = summ.get((k, a))
            if v:
                M[f"StrictValid{mac}{tag}"] = f"{100 * np.mean([x[0] for x in v]):.0f}\\%"
                M[f"StrictCells{mac}{tag}"] = f"{np.mean([x[1] for x in v]):.1f}"
                M[f"StrictWidth{mac}{tag}"] = f"{np.mean([x[2] for x in v]):.2f}"
    M["StrictNOrigins"] = str(norig)


def is_macros(R, M):
    """Pre-registered efficiency endpoint: class-averaged interval score among the valid methods of each setting."""
    cand = [k for k, _, _ in BASE] + ["GAMUT", "GAMUT-cert (split certificate)"]
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        wins = 0; ratios = []; n = 0
        for b, h in settings(R, MAIN, a):
            r = R[(b, h, a)]; n += 1
            valid = [k for k in cand if k in r and k != "GAMUT" and is_valid(r[k], a, (b, h), k)]
            g = r["GAMUT"]["class_avg_interval_score"]
            if not valid:
                wins += is_valid(r["GAMUT"], a, (b, h), "GAMUT"); continue
            best = min(r[k]["class_avg_interval_score"] for k in valid)
            wins += bool(is_valid(r["GAMUT"], a, (b, h), "GAMUT") and g <= best); ratios.append(g / best)
        if n:
            M[f"ISWinsGamut{tag}"] = f"{wins}/{n}"; M[f"ISMedRatio{tag}"] = f"{np.median(ratios):.2f}"
            M[f"ISMaxRatio{tag}"] = f"{max(ratios):.2f}"


def held_table(R, M):
    """Pre-registered held-out benchmark (PLAsTiCC-WFD, test chunk 02): the summary of tab:main for that benchmark alone."""
    rows = []
    for key, disp, fam in BASE + [(OURS[0], OURS[1], "ours"), ("GAMUT-cert (split certificate)", "\\gamut-cert (split certificate)", "ours2"),
                                  (ORACLE[0], ORACLE[1], "oracle")]:
        cells = []
        for a in LEVELS:
            g = agg(R, HELD, key, a)
            cells += ["--"] * 5 if g is None else [f"{100 * g['valid']:.0f}", f3(g["worst"]), f3(g["ccg"]),
                                                   "--" if not np.isfinite(g["cells"]) else f"{g['cells']:.1f}", f2(g["rel_width"])]
        if any(c != "--" for c in cells):
            rows.append(disp + " & " + " & ".join(cells) + " \\\\")
        if key in ("DCP", "CPTC", "MaxClassCP", "MaxClassCP + C1") and rows:
            rows.append("\\midrule")
    n = agg(R, HELD, "GAMUT", 0.32)
    tex = ("\\begin{table}[H]\n\\centering\n\\caption{Pre-registered held-out benchmark: wide-fast-deep objects of PLAsTiCC test chunk 02, "
           "never used during design, evaluated once with the frozen code (" + str(n["n"] if n else 0) + " origins per level). Columns as in \\cref{tab:main}.}\n"
           "\\label{tab:held}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lccccc@{\\hspace{8pt}}ccccc}\n\\toprule\n"
           " & \\multicolumn{5}{c}{68\\% bands} & \\multicolumn{5}{c}{95\\% bands} \\\\\n\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}\n"
           "Method & Valid & Worst & CCG & Cells & Width & Valid & Worst & CCG & Cells & Width \\\\\n\\midrule\n"
           + ("\n".join(rows) if rows else "\\multicolumn{11}{c}{(pending)} \\\\") + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (PAPER / "tables" / "held.tex").write_text(tex.replace("\\midrule\n\\bottomrule", "\\bottomrule"))
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        g = agg(R, HELD, "GAMUT", a)
        if g:
            M[f"HeldValidGamut{tag}"] = f"{100 * g['valid']:.0f}\\%"; M[f"HeldWidthGamut{tag}"] = f"{g['rel_width']:.2f}"
            M[f"HeldWorstGamut{tag}"] = f"{g['min_worst']:.3f}"; M[f"HeldCellsGamut{tag}"] = f"{g['cells']:.1f}"
        fl = agg(R, HELD, "FLARE", a)
        if fl:
            M[f"HeldValidFlare{tag}"] = f"{100 * fl['valid']:.0f}\\%"; M[f"HeldWorstFlare{tag}"] = f"{fl['min_worst']:.3f}"
        vb = [agg(R, HELD, k, a) for k in ("TwoStepCP", "MDCP", "MaxClassCP")]
        vb = [x for x in vb if x]
        if vb:
            M[f"HeldWidthValid{tag}"] = f"{min(x['rel_width'] for x in vb):.2f}--{max(x['rel_width'] for x in vb):.2f}"
            M[f"HeldValidValid{tag}"] = f"{100 * min(x['valid'] for x in vb):.0f}--{100 * max(x['valid'] for x in vb):.0f}\\%"
        ct = agg(R, HELD, "GAMUT-cert (split certificate)", a)
        if ct:
            M[f"HeldWidthCert{tag}"] = f"{ct['rel_width']:.2f}"; M[f"HeldValidCert{tag}"] = f"{100 * ct['valid']:.0f}\\%"


def extra_macros(R, M):
    """Macros added after review round 3 (guarantee-matched widths, tolerance validity, ablation details)."""
    tol = lambda key, a: np.mean([R[(b, h, a)][key]["worst_class"] >= 1 - a - TOL[a] for b, h in settings(R, MAIN, a, need=(key,))])
    M["MainValidTolGamutSixtyeight"] = f"{100 * tol('GAMUT', 0.32):.0f}\\%"
    S95 = settings(R, MAIN, 0.05, need=("GAMUT", "MDCP"))
    M["NinetyfiveWiderThanMDCP"] = f"{sum(R[(b, h, 0.05)]['GAMUT']['width'] > R[(b, h, 0.05)]['MDCP']['width'] for b, h in S95)}/{len(S95)}"
    for key, mac in (("CPTC", "CPTC"), ("GAMUT-cert (split certificate)", "Cert")):
        for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
            g = agg(R, MAIN, key, a)
            if g:
                M[f"MainWidth{mac}{tag}"] = f"{g['rel_width']:.2f}"; M[f"MainValid{mac}{tag}"] = f"{100 * g['valid']:.0f}\\%"
                if mac == "Cert":
                    M[f"CertWorst{tag}"] = f"{g['worst']:.3f}"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        Z = [(h, R[("ztf", h, a)]) for (b, h) in settings(R, ["ztf"], a, need=("GAMUT", "FLARE"))]
        if Z:
            M[f"ZTFWidthRatio{tag}"] = f"{np.mean([r['GAMUT']['width'] / r['FLARE']['width'] for _, r in Z]):.2f}"
        V = [np.mean([r[k]['width'] / r['FLARE']['width'] for _, r in Z if k in r]) for k in ("TwoStepCP", "MDCP", "MaxClassCP")] if Z else []
        if V:
            M[f"ZTFValidOverFlare{tag}"] = f"{100 * (min(V) - 1):.0f}--{100 * (max(V) - 1):.0f}\\%"
        sv = [R[(f"synthetic_k{k}", 0.0, a)][m]["width"] / R[(f"synthetic_k{k}", 0.0, a)]["GAMUT"]["width"] - 1
              for k in ("0.5", "1", "2") if (f"synthetic_k{k}", 0.0, a) in R for m in ("TwoStepCP", "MDCP", "MaxClassCP")
              if m in R[(f"synthetic_k{k}", 0.0, a)] and "GAMUT" in R[(f"synthetic_k{k}", 0.0, a)]]
        if sv:
            M[f"SynValidOverGamut{tag}"] = f"{100 * min(sv):.0f}--{100 * max(sv):.0f}\\%"
    if ("ztf", 5.0, 0.32) in R:
        r = R[("ztf", 5.0, 0.32)]; M["ZTFfiveWidthRatioSixtyeight"] = f"{r['GAMUT']['width'] / r['FLARE']['width']:.2f}"
    if ("ztf", 5.0, 0.05) in R:
        r = R[("ztf", 5.0, 0.05)]; M["ZTFfiveWidthRatioNinetyfive"] = f"{r['GAMUT']['width'] / r['FLARE']['width']:.2f}"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        r1 = R.get(("synthetic_k1", 0.0, a)); r2 = R.get(("synthetic_k2", 0.0, a))
        if r1 and "GAMUT w/o C1 (pooled forecaster)" in r1:
            M[f"AblConeSyn{tag}"] = f"{100 * (r1['GAMUT w/o C1 (pooled forecaster)']['width'] / r1['GAMUT']['width'] - 1):.0f}\\%"
        if r2 and "GAMUT w/o C2-calibration (raw posterior)" in r2:
            M[f"AblRawKtwo{tag}"] = f"{100 * (r2['GAMUT w/o C2-calibration (raw posterior)']['width'] / r2['GAMUT']['width'] - 1):.0f}\\%"
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        for key, mac in (("GAMUT w/o adaptive cells (fixed lead bins)", "FixedBins"), ("GAMUT w/o C1 (pooled forecaster)", "NoCone"),
                         ("GAMUT w/o C2-calibration (raw posterior)", "Raw")):
            g = agg(R, MAIN, key, a, ref="GAMUT")
            if g:
                M[f"AblW{mac}{tag}"] = f"{100 * (g['rel_width'] - 1):+.0f}\\%"
                M[f"AblCells{mac}{tag}"] = f"{g['cells']:.1f}"; M[f"AblValid{mac}{tag}"] = f"{100 * g['valid']:.0f}\\%"
    g = agg(R, MAIN, "GAMUT w/o C3-dual (posterior mixture)", 0.32, ref="GAMUT")
    if g:
        M["AblValidMixtureSixtyeight"] = f"{100 * g['valid']:.0f}\\%"
    g = agg(R, MAIN, "GAMUT w/o leave-one-object-out laws (in-sample)", 0.05, ref="GAMUT")
    if g:
        M["AblValidNoLooNinetyfive"] = f"{100 * g['valid']:.0f}\\%"
    cells = [c["cov"] for k, v in BOOT.items() if k.startswith("ztf|") and k.endswith("|0.32")
             for c in v.get("GAMUT w/o lead-bin cells (one threshold)", {}).get("cells", {}).values()]
    if cells:
        M["AblOneThreshWorstCell"] = f"{min(cells):.2f}"
    f = RES / "ztf_strict" / "h5_a0.05.json"
    if f.exists():
        r = json.loads(f.read_text()); M["StrictWidthRatioNinetyfive"] = f"{r['GAMUT']['width'] / r['FLARE']['width']:.2f}"
    it = []
    for b in ("plasticc_train", "ucr_ecg5000"):
        for (bb, h, a), res in R.items():
            if bb == b and a == 0.05 and "GAMUT" in res:
                for d in res["GAMUT"].get("diag", []):
                    it += d["iterations"]
    if it:
        M["DualItersCapSmall"] = f"{100 * np.mean(np.array(it) >= 100):.0f}\\%"
    # validity under both rules, all methods (appendix)
    lines = []
    for key, disp, _ in BASE + [(OURS[0], OURS[1], ""), ("GAMUT-cert (split certificate)", "\\gamut-cert", ""), (ORACLE[0], ORACLE[1], "")]:
        cells_ = []
        for a in LEVELS:
            g = agg(R, MAIN, key, a)
            cells_ += ["--", "--"] if g is None else [f"{100 * g['valid']:.0f}", f"{100 * tol(key, a):.0f}"]
        lines.append(disp + " & " + " & ".join(cells_) + " \\\\")
    tex = ("\\begin{table}[H]\n\\centering\n\\caption{Percentage of valid settings (six main benchmarks, all origins) under the bootstrap rule used in the main text and under the pre-registered fixed tolerance (worst class $\\ge$ nominal $-0.02$ / $-0.015$).}\n\\label{tab:validity}\n\\scriptsize\n"
           "\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{68\\%} & \\multicolumn{2}{c}{95\\%} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nMethod & bootstrap & tolerance & bootstrap & tolerance \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (PAPER / "tables" / "validity.tex").write_text(tex)


def main_macros(R, M):
    for a, tag in ((0.32, "Sixtyeight"), (0.05, "Ninetyfive")):
        g = agg(R, MAIN, "GAMUT", a)
        if not g:
            continue
        M[f"MainValidGamut{tag}"] = f"{100 * g['valid']:.0f}\\%"; M[f"MainCCGGamut{tag}"] = f"{g['ccg']:.3f}"
        M[f"MainWidthGamut{tag}"] = f"{g['rel_width']:.2f}"
        M[f"MainCellsGamut{tag}"] = f"{g['cells']:.1f}"
        fl = agg(R, MAIN, "FLARE", a)
        if fl:
            M[f"MainCellsFlare{tag}"] = f"{fl['cells']:.1f}"
        kd = agg(R, MAIN, "KandinskyCP", a)
        if kd:
            M[f"MainCellsKandinsky{tag}"] = f"{kd['cells']:.1f}"; M[f"MainValidKandinsky{tag}"] = f"{100 * kd['valid']:.0f}\\%"
        nv = [agg(R, MAIN, k, a) for k in ["FLARE", "SplitCP", "CQR", "DCP"] + PROB]
        nv = [x for x in nv if x]
        if nv:
            M[f"MainValidBestNonValid{tag}"] = f"{100 * max(x['valid'] for x in nv):.0f}\\%"
        for k, mac in (("TwoStepCP", "TwoStep"), ("MDCP", "MDCP"), ("MaxClassCP", "MaxClass")):
            x = agg(R, MAIN, k, a)
            if x:
                M[f"MainCells{mac}{tag}"] = f"{x['cells']:.1f}"
        c1 = [agg(R, MAIN, k, a) for k in ["TwoStepCP + C1", "MDCP + C1", "MaxClassCP + C1"]]
        c1 = [x for x in c1 if x]
        if len(c1) == 3:
            M[f"MainWidthCOneBaselines{tag}"] = f"{min(x['rel_width'] for x in c1):.2f}--{max(x['rel_width'] for x in c1):.2f}"
            M[f"MainValidCOneBaselines{tag}"] = f"{100 * min(x['valid'] for x in c1):.0f}--{100 * max(x['valid'] for x in c1):.0f}\\%"
            M[f"MainCellsCOneBaselines{tag}"] = f"{min(x['cells'] for x in c1):.1f}--{max(x['cells'] for x in c1):.1f}"
        vb = [agg(R, MAIN, k, a) for k in ["TwoStepCP", "MDCP", "MaxClassCP"]]
        vb = [x for x in vb if x]
        if vb:
            M[f"MainCCGValidBaselines{tag}"] = f"{min(x['ccg'] for x in vb):.3f}--{max(x['ccg'] for x in vb):.3f}"
            M[f"MainWidthValidBaselines{tag}"] = f"{min(x['rel_width'] for x in vb):.2f}--{max(x['rel_width'] for x in vb):.2f}"


def main():
    R = load()
    bench_table()
    main_table(R); ablation_table(R)
    meth = [(k, d) for k, d, _ in BASE] + [OURS, ORACLE]
    (PAPER / "tables" / "per_bench.tex").write_text(per_bench_table(
        R, MAIN + APPX + [b for b in HELD if any(k[0] == b for k in R)], "tab:perbench", "Per benchmark: worst-class coverage (minimum over origins) / mean band length relative to FLARE.", meth))
    with open(PAPER / "tables" / "per_origin.tex", "w") as fh:
        for b in MAIN + APPX + SYN + HELD:
            fh.write(per_origin_tables(R, b))
    M = numbers(R)
    observation_table(R, M); ablation_full(R, M); hyper_table(M); cert_table(R); synthetic_table(R, M); runtime_table(R, M)
    cnt = unfavourable_table(R); main_macros(R, M); diag_macros(R, M); strict_table(M); extra_macros(R, M); held_table(R, M); is_macros(R, M)
    print("unfavourable counts", cnt)
    for f in ("main.tex", "bench.tex", "ablation.tex"):  # these overflow the column; scale them to the line width
        fp = PAPER / "tables" / f; t = fp.read_text()
        if "\\resizebox" not in t:
            t = t.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
            fp.write_text(t)
    write_macros(M)
    summary = {f"{b}|{h:g}|{a:g}": {k: dict(worst=v["worst_class"], ccg=v["ccg"], width=v["width"], marg=v["marginal"])
                                       for k, v in res.items() if not k.startswith("_")} for (b, h, a), res in R.items()}
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print("settings with GAMUT:", {a: len(settings(R, MAIN, a)) for a in LEVELS}); print(M)


if __name__ == "__main__":
    main()
