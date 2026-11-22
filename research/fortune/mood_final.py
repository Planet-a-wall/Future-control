"""Iteration 12: fix the poverty->depression gradient (Lorant OR ~1.8) and finalise mood calibration."""
import itertools, json
from studies import verdict
from mood_recal import score_mood
from mood_studies import pop, mood_metrics, check, MT, MOOD
J = json.load(open("mood_joint_results.json"))
BASE = J["chosen"]
rows = []
for pov, vol in itertools.product((0.06, 0.08, 0.10, 0.12), (0.40, 0.42)):
    P = {**BASE, "mood_pov": pov, "mood_vol": vol}
    K0, tr, o = pop(1, MOOD, P=P, N=20000); mm = mood_metrics(K0, tr, o); mv = check(mm, MT)
    m = score_mood(P, seed=1); v = verdict(m)
    rows.append(dict(P=dict(mood_pov=pov, mood_vol=vol), mood=mm, mood_n=sum(mv.values()), n=sum(v.values()),
                     fails=[(k, round(m[k], 3)) for k in v if not v[k]]))
    print(rows[-1]["P"], "mood", sum(mv.values()), "/4", {k: round(x, 3) for k, x in mm.items()}, "| SF", sum(v.values()), rows[-1]["fails"], flush=True)
rows.sort(key=lambda r: (-(r["mood_n"] + r["n"]), abs(r["mood"]["dep_OR_poor_vs_rich"] - 1.8)))
best = {**BASE, **rows[0]["P"]}
print("chosen", rows[0]["P"])
oos = []
for seed in (101, 202, 303):
    m = score_mood(best, seed=seed, N_POP=40000, N_COH=6000); v = verdict(m)
    K0, tr, o = pop(seed, MOOD, P=best); mm = mood_metrics(K0, tr, o)
    oos.append(dict(seed=seed, n=sum(v.values()), fails=[(k, round(m[k], 3)) for k in v if not v[k]], m=m, mood=mm,
                    mood_n=sum(check(mm, MT).values())))
    print("OOS", seed, f"SF {sum(v.values())}/10", oos[-1]["fails"], f"mood {oos[-1]['mood_n']}/4",
          {k: round(x, 3) for k, x in mm.items()}, flush=True)
json.dump(dict(grid=rows, chosen=best, oos=oos), open("mood_final_results.json", "w"), indent=1, default=float)
