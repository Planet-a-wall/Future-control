"""E6: does merit protect only if it buys honest feedback? (hypothesis test inside the model)"""
import json
import numpy as np
from model import simulate, traits
from validate import CAL, FULL
res = {}
for label, extra in [("merit = money + rescue only", {}), ("merit + honest feedback", dict(merit_feedback=True))]:
    fl = {**FULL, **extra}
    rows = []
    for om in (0.0, 0.05, 0.10, 0.20):
        n = 20000
        t = traits(n, np.random.default_rng(9), v=0.85, q_intr=0.9, omega=om)   # born poor, strong mind
        o = simulate(np.full(n, 0.1), t, fl, seed=10, P=CAL)
        g = o["ever_great"]
        f = g & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))
        rows.append(dict(omega=om, p_great=float(g.mean()), p_fall_given_great=float(f.sum() / max(g.sum(), 1)),
                         p_stay=float((g & ~f).mean()), p_jail=float(np.mean(o["jailed_t"] >= 0)),
                         median_final=float(np.median(o["K"]))))
        print(label, rows[-1])
    res[label] = rows
json.dump(res, open("merit_results.json", "w"), indent=1)
