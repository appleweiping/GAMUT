"""Evaluation metrics for interval forecasts under a latent class.

All coverage numbers are over future measurements (the convention of the FLARE paper), and also averaged per object.
lo, hi: hull of the band (for a union of intervals the hull is what a single band would report); length: total length
of the union (equal to hi - lo for a single interval)."""
from __future__ import annotations

import numpy as np


def interval_score(y, lo, hi, alpha):
    """Gneiting & Raftery (2007) interval score: width + (2/alpha) * distance outside the interval."""
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def evaluate(task, rows, covered, lo, hi, length, alpha, extra_masks=None):
    """rows: indices of the evaluated future points; covered/lo/hi/length aligned with rows."""
    y_obj = task.y[task.obj[rows]]
    nominal = 1 - alpha
    out = dict(n_points=int(len(rows)), marginal=float(covered.mean()))
    # object-averaged coverage
    o = task.obj[rows]
    uo, inv = np.unique(o, return_inverse=True)
    cov_obj = np.bincount(inv, covered.astype(float)) / np.bincount(inv)
    out["marginal_object"] = float(cov_obj.mean())
    per, per_w, per_is, per_obj = {}, {}, {}, {}
    isc = interval_score(task.yt[rows], lo, hi, alpha)
    ycls_obj = task.y[uo]
    for c, name in enumerate(task.classes):
        m = y_obj == c
        if m.any():
            per[name] = float(covered[m].mean()); per_w[name] = float(length[m].mean()); per_is[name] = float(isc[m].mean())
            per_obj[name] = float(cov_obj[ycls_obj == c].mean())
    out.update(per_class=per, per_class_object=per_obj, per_class_width=per_w, per_class_interval_score=per_is)
    vals = np.array(list(per.values()))
    out["worst_class"] = float(vals.min())
    out["ccg"] = float(np.mean(np.abs(vals - nominal)))                  # class-coverage gap
    out["under"] = float(np.max(np.maximum(nominal - vals, 0)))         # worst under-coverage
    out["width"] = float(length.mean()); out["hull_width"] = float((hi - lo).mean())
    out["interval_score"] = float(isc.mean())
    out["class_avg_interval_score"] = float(np.mean(list(per_is.values())))
    out["class_avg_width"] = float(np.mean(list(per_w.values())))
    # coverage of objects the classifier gets wrong (the population the professor's problem is about)
    wrong = task.L[task.obj[rows]] != y_obj
    out["misclassified_coverage"] = float(covered[wrong].mean()) if wrong.any() else float("nan")
    out["misclassified_fraction"] = float(wrong.mean())
    if extra_masks:
        for k, msk in extra_masks.items():
            out[k] = float(covered[msk].mean()) if msk.any() else float("nan")
    return out
