"""One-off: label the nested-validation entries computed before the bandwidth was selected (bandwidth x1 runs)."""
import glob, json
for f in glob.glob("results/ztf/validation_*.json"):
    V = json.load(open(f)); W = {}
    for k, v in V.items():
        if k == "default":
            W["initial configuration (bandwidth x1, Silverman)"] = v
        elif k == "bandwidth x2":
            W["bandwidth x2 (selected)"] = v
        elif k.startswith("bandwidth") or k.startswith("initial") or "(at bandwidth x1)" in k:
            W[k] = v
        else:
            W[k + " (at bandwidth x1)"] = v
    json.dump(W, open(f, "w"), indent=1)
