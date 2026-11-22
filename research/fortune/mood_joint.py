"""Iteration 11: joint calibration (base + mood effects), then out-of-sample at full N."""
import itertools, json
from studies import verdict
from mood_recal import score_mood
from mood_studies import MCAL, pop, mood_metrics, check, MT, MOOD
EFF = dict(a_c=0.15, imp_mult=0.3, spree=0.25, a_f=0.3)
rows = []
for beta, kf, hd in itertools.product((0.03, 0.035, 0.04), (0.45, 0.5), (0.004, 0.006)):
    P = {**MCAL, **EFF, "beta": beta, "kelly_frac": kf, "habit_decay": hd}
    m = score_mood(P, seed=1); v = verdict(m)
    rows.append(dict(P=dict(beta=beta, kelly_frac=kf, habit_decay=hd), n=sum(v.values()),
                     fails=[(k, round(m[k], 3)) for k in v if not v[k]]))
    print(rows[-1], flush=True)
rows.sort(key=lambda r: -r["n"])
best = {**MCAL, **EFF, **rows[0]["P"]}
print("chosen", rows[0])
oos = []
for seed in (101, 202, 303):
    m = score_mood(best, seed=seed, N_POP=40000, N_COH=6000); v = verdict(m)
    K0, tr, o = pop(seed, MOOD, P=best); mm = mood_metrics(K0, tr, o)
    oos.append(dict(seed=seed, n=sum(v.values()), fails=[(k, round(m[k], 3)) for k in v if not v[k]],
                    m=m, mood=mm, mood_pass=check(mm, MT)))
    print("OOS", seed, sum(v.values()), "/10", oos[-1]["fails"], {k: round(x, 3) for k, x in mm.items()}, flush=True)
json.dump(dict(grid=rows, chosen=best, oos=oos), open("mood_joint_results.json", "w"), indent=1, default=float)
