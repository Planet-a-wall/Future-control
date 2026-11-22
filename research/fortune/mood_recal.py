"""Iteration 10: recalibrate mood effect sizes so the 10 stylized facts hold with mood on,
while the 4 psychiatric anchors stay inside their bands."""
import itertools, json, sys
import numpy as np
import studies
from studies import verdict
from model import mood_traits
from mood_studies import MOOD, MCAL, pop, mood_metrics, check, MT

def score_mood(P, seed, N_POP=20000, N_COH=3000):
    orig = studies.traits
    studies.traits = lambda N, r, **kw: {**orig(N, r, **kw), **mood_traits(N, np.random.default_rng(seed + 5), P=P)}
    studies.N_POP, studies.N_COH = N_POP, N_COH
    m = studies.score("mood", MOOD, seed=seed, P=P)
    studies.traits = orig
    return m

if __name__ == "__main__":
    rows = []
    for a_c, imp_mult, spree, a_f in itertools.product((0.15, 0.3), (0.3, 0.6, 1.0), (0.25, 0.5), (0.3, 1.0)):
        P = {**MCAL, "a_c": a_c, "imp_mult": imp_mult, "spree": spree, "a_f": a_f}
        m = score_mood(P, seed=1); v = verdict(m)
        rows.append(dict(P=dict(a_c=a_c, imp_mult=imp_mult, spree=spree, a_f=a_f), n=sum(v.values()),
                         fails=[(k, round(m[k], 3)) for k in v if not v[k]]))
        print(rows[-1]); sys.stdout.flush()
    rows.sort(key=lambda r: -r["n"])
    json.dump(rows, open("mood_recal_results.json", "w"), indent=1)
    print("BEST", rows[:4])
