#!/bin/bash
# waits for the ZTF chain, then cross-fits every other benchmark (sequential, 3 threads; lr 0.1, <= 16 training points per object, for compute).
# Origins: PLAsTiCC at 5, 15, 30 d (ZTF: all five, run separately); UCR at three prefix fractions (20/40/60%).
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
FLARE_REPO=${FLARE_REPO:-../flare}
while pgrep -f "crossfit.py --bench ztf" >/dev/null; do sleep 30; done
for b in plasticc_train ucr_ecg5000 ucr_starlightcurves plasticc_ddf ucr_electricdevices ucr_wafer; do
  if [[ $b == ucr_* ]]; then H="20 40 60"; else H="5 15 30"; fi
  for h in $H; do
    [[ -f data/$b/oof_h$h.npz ]] && continue
    env PYTHONPATH=$PWD $PY -u scripts/crossfit.py --bench $b --h $h --threads 3 --lr 0.1 --cap 16 > data/$b/crossfit_h$h.log 2>&1
  done
done
echo ALLDONE > data/queue_done.txt
