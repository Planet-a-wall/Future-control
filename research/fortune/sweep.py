"""Grid search over the four least-constrained parameters (calibration set, seed=1)."""
import itertools, json, sys
import numpy as np
import studies
from studies import score, verdict, TARGETS
FULL = dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True, merit=True)
grid = dict(habit_decay=[0.004, 0.008, 0.015], mu_max=[0.02, 0.025, 0.03],
            sigma=[0.08, 0.10], beta=[0.03, 0.04])
rows = []
for vals in itertools.product(*grid.values()):
    P = dict(zip(grid.keys(), vals))
    m = score("sweep", FULL, seed=1, P=P)
    v = verdict(m)
    rows.append(dict(P=P, n=sum(v.values()), fails=[k for k in v if not v[k]], m=m))
rows.sort(key=lambda r: -r["n"])
for r in rows[:8]:
    print(r["n"], "/", len(TARGETS), r["P"], "fails:", r["fails"],
          {k: round(r["m"][k], 3) for k in r["fails"]})
json.dump(rows, open("sweep_results.json", "w"), indent=1, default=float)
