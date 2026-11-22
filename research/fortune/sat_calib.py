"""Iteration 18: steepness of goal satiation (never calibrated before)."""
import json
from dynasty_calib import evaluate
BASE = json.load(open("dynasty_calib_iter17.json"))["chosen"]
rows = []
for sk in (3.0, 5.0, 8.0, 12.0):
    e = evaluate({**BASE, "sat_k": sk})
    rows.append(dict(sat_k=sk, total=e["total"], fails=e["fails"]))
    print(f"sat_k={sk}: SF {e['sf']}/10 anchors {e['anchors']}/5 dyn {e['dyn']}/5 fails={e['fails']}", flush=True)
best = max(rows, key=lambda r: r["total"])
P = {**BASE, "sat_k": best["sat_k"]}
print("chosen sat_k", best["sat_k"])
oos = []
for seed in (101, 202, 303):
    e = evaluate(P, seed=seed, N_single=40000, N_dyn=40000)
    oos.append({k: e[k] for k in ("sf", "anchors", "dyn", "total", "fails", "dyn_vals", "mm", "m", "gens")} | dict(seed=seed))
    print(f"OOS {seed}: SF {e['sf']}/10 anchors {e['anchors']}/5 dyn {e['dyn']}/5 fails={e['fails']}", flush=True)
    print("   dyn:", {k: round(v, 3) for k, v in e["dyn_vals"].items()}, "gini:", [round(x["gini"], 3) for x in e["gens"]], flush=True)
json.dump(dict(grid=rows, chosen=P, oos=oos), open("final_calibration.json", "w"), indent=1, default=float)
