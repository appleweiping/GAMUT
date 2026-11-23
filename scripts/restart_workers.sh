#!/bin/bash
# waits for running evaluate.py children to finish, clears stale locks, starts two queue workers
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
FLARE_REPO=${FLARE_REPO:-../flare}
while pgrep -f "scripts/evaluate.py" >/dev/null; do sleep 20; done
rm -f results/*/.lock_h*
B="ztf ucr_electricdevices plasticc_train plasticc_ddf ucr_ecg5000 ucr_starlightcurves synthetic_k1 synthetic_k0.5 synthetic_k2 ucr_wafer"
nohup $PY -u scripts/run_all.py --loop --benches $B >> results/worker1.log 2>&1 &
sleep 5
nohup $PY -u scripts/run_all.py --loop --benches $B >> results/worker2.log 2>&1 &
sleep 5
nohup $PY -u scripts/run_all.py --loop --benches $B >> results/worker3.log 2>&1 &
