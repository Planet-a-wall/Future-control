"""Robustness at full population size (N=40000) - every parameter x0.8 / x1.2."""
import json
import studies
from studies import score, verdict
from model import DEFAULT
from validate import CAL, FULL
studies.N_POP, studies.N_COH = 40000, 6000
keys = ["r", "sigma", "mu_max", "kelly_frac", "f_cap", "save_wealth_rate", "cons_income", "habit_decay",
        "beta", "beta_neg", "gamma", "psi", "ego_gain", "goal_med", "lam_loss", "fraud_pi", "fraud_det",
        "y_R", "rho_R", "rescue_p"]
rows = []
for k in keys:
    for s in (0.8, 1.2):
        P = dict(CAL); P[k] = CAL.get(k, DEFAULT[k]) * s
        m = score("rob", FULL, seed=11, P=P); v = verdict(m)
        fails = [x for x in v if not v[x]]
        rows.append(dict(param=k, scale=s, n=sum(v.values()), fails=fails, vals={x: m[x] for x in fails}))
        print(f"{k:16s} x{s}: {sum(v.values())}/10 {[(x, round(m[x],3)) for x in fails]}")
n = [r["n"] for r in rows]
print("mean pass", sum(n)/len(n), "configs with >=9/10:", sum(x >= 9 for x in n), "/", len(n))
json.dump(rows, open("robust_full_results.json", "w"), indent=1, default=float)
