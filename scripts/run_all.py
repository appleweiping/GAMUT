"""Work-queue driver: evaluates every (benchmark, origin) whose cross-fit is assembled and whose results are missing.

Several workers can run at once; each claims a job by creating results/<bench>/.lock_h<h> atomically.
A job is complete when both levels' JSON files contain every method and ablation.
Run: python scripts/run_all.py [--loop] [--benches ztf plasticc_train ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORDER = ["ztf", "plasticc_train", "ucr_ecg5000", "ucr_starlightcurves", "plasticc_ddf", "ucr_electricdevices", "ucr_wafer", "plasticc_wfd",
         "synthetic_k1", "synthetic_k0.5", "synthetic_k2"]
H_ORDER = {"ztf": [5, 20, 10, 30, 15]}


def expected_names():
    from gamut import latent_baselines, methods
    return set(methods.METHODS) | set(latent_baselines.LATENT_BASELINES) | set(methods.ABLATIONS)


def missing(bench, h, names):
    """Names whose result is absent at either level, or a GAMUT entry from an older GAMUT version."""
    from gamut import methods
    out = set()
    for a in (0.32, 0.05):
        p = ROOT / "results" / bench / f"h{h:g}_a{a:g}.json"
        R = json.loads(p.read_text()) if p.exists() else {}
        for n in names:
            if "union" in n and bench != "ztf":          # the superseded early design is evaluated on ZTF only
                continue
            if n not in R or (n.startswith("GAMUT") and "union" not in n and R[n].get("version") != methods.GAMUT_VERSION) \
                    or (n in ("KandinskyCP", "MDCP") and R[n].get("version") != methods.GRID_VERSION):
                out.add(n)
    return sorted(out)


def jobs(benches):
    for b in benches:
        d = ROOT / "data" / b
        hs = sorted(float(f.stem.split("_h")[1]) for f in d.glob("oof_h*.npz")) if d.exists() else []
        pref = H_ORDER.get(b, [])
        hs = [h for h in pref if h in hs] + [h for h in hs if h not in pref]
        for h in hs:
            yield b, h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--loop", action="store_true"); ap.add_argument("--benches", nargs="*", default=ORDER)
    a = ap.parse_args()
    names = expected_names()
    while True:
        ran = False
        for b, h in jobs(a.benches):
            todo = missing(b, h, names)
            if not todo:
                continue
            (ROOT / "results" / b).mkdir(parents=True, exist_ok=True)
            lock = ROOT / "results" / b / f".lock_h{h:g}"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            except FileExistsError:
                continue
            log = ROOT / "results" / b / f"eval_h{h:g}.log"
            t0 = time.time()
            with open(log, "a") as fh:
                rc = subprocess.call([sys.executable, "-u", str(ROOT / "scripts" / "evaluate.py"), "--bench", b, "--h", f"{h:g}",
                                      "--names", *todo], stdout=fh, stderr=subprocess.STDOUT,
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            print(f"{b} h={h:g}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
            lock.unlink(missing_ok=True)
            ran = True
            break
        if not ran:
            if not a.loop:
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
