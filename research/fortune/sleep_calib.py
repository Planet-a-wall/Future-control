"""Iteration 14: calibrate sleep module (short-sleep prevalence ~35%) while keeping all other anchors."""
import itertools, json
import numpy as np
from studies import verdict
from sleep_studies import score_full, pop_full, FINAL, ST
from mood_studies import MT, mood_metrics, check
rows = []
for sm, kw, rf in itertools.product((7.5, 7.6, 7.7), (1.0, 1.5), (0.08, 0.12)):
    P = {**FINAL, "sleep_mean": sm, "k_work": kw, "rho_fat": rf}
    m = score_full(P, seed=1, N_POP=20000, N_COH=3000); v = verdict(m)
    K0, tr, o = pop_full(1, P, N=20000)
    mm = mood_metrics(K0, tr, o); mm["short_sleep_share"] = float(np.mean(o["short_share"] > 0.5))
    mv = {**check(mm, MT), **{k: ST[k][0] <= mm[k] <= ST[k][1] for k in ST}}
    rows.append(dict(P=dict(sleep_mean=sm, k_work=kw, rho_fat=rf), n=sum(v.values()) + sum(mv.values()),
                     fails=[(k, round(m[k], 3)) for k in v if not v[k]] + [(k, round(mm[k], 3)) for k in mv if not mv[k]]))
    print(rows[-1], flush=True)
rows.sort(key=lambda r: -r["n"])
best = {**FINAL, **rows[0]["P"]}
print("chosen", rows[0])
oos = []
for seed in (101, 202, 303):
    m = score_full(best, seed); v = verdict(m)
    K0, tr, o = pop_full(seed, best)
    mm = mood_metrics(K0, tr, o); mm["short_sleep_share"] = float(np.mean(o["short_share"] > 0.5))
    mv = {**check(mm, MT), **{k: ST[k][0] <= mm[k] <= ST[k][1] for k in ST}}
    oos.append(dict(seed=seed, sf=sum(v.values()), anchors=sum(mv.values()), m=m, mm=mm,
                    fails=[(k, round(m[k], 3)) for k in v if not v[k]] + [(k, round(mm[k], 3)) for k in mv if not mv[k]]))
    print("OOS", seed, f"SF {sum(v.values())}/10 anchors {sum(mv.values())}/5", oos[-1]["fails"],
          {k: round(x, 3) for k, x in mm.items()}, flush=True)
json.dump(dict(grid=rows, chosen=best, oos=oos), open("sleep_calib_results.json", "w"), indent=1, default=float)
