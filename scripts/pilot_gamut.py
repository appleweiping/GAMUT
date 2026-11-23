"""Pilot: GAMUT variants on one benchmark/origin/level (development use; prints one line per variant)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from gamut import data, methods, metrics
sys.path.insert(0, str(ROOT / "scripts")); from evaluate import run
bench, h, alpha = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
task = data.load(bench, h)
variants = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
for name, kw in variants.items():
    res, _ = run(task, name, methods.gamut, alpha, kw)
    print(f"{name:28s} marg {res['marginal']:.3f} worst {res['worst_class']:.3f} ccg {res['ccg']:.3f} width {res['width']:.3f} "
          f"hull {res['hull_width']:.3f} IS {res['interval_score']:.3f} cls-IS {res['class_avg_interval_score']:.3f} miscls {res['misclassified_coverage']:.3f} | "
          + " ".join(f"{c}:{v:.2f}" for c, v in res['per_class'].items()), flush=True)
