#!/bin/bash
# Pre-registered held-out evaluation (the preregistered experiment plan, round-4 response): PLAsTiCC test chunk 02 WFD.
# Waits for the download, builds the benchmark, cross-fits origins 5/15/30 with the PLAsTiCC recipe, evaluates every method.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
FLARE_REPO=${FLARE_REPO:-../flare}
while pgrep -f "plasticc_test_lightcurves_02" >/dev/null; do sleep 20; done
gzip -t data/raw/plasticc/plasticc_test_lightcurves_02.csv.gz || { echo "download corrupt"; exit 1; }
mkdir -p data/plasticc_wfd
[[ -f data/plasticc_wfd/F_h30.parquet ]] || env PYTHONPATH=$PWD:$FLARE_REPO $PY -u -m benchmarks.plasticc --which wfd --workers 3 > data/plasticc_wfd/build.log 2>&1 || { echo "build failed"; exit 1; }
for h in 5 15 30; do
  [[ -f data/plasticc_wfd/oof_h$h.npz ]] && continue
  env PYTHONPATH=$PWD $PY -u scripts/crossfit.py --bench plasticc_wfd --h $h --threads 2 --lr 0.1 --cap 16 > data/plasticc_wfd/crossfit_h$h.log 2>&1 || { echo "crossfit $h failed"; exit 1; }
done
$PY -u scripts/run_all.py --benches plasticc_wfd > results/worker_wfd.log 2>&1
echo HELDOUT_DONE
