"""
Validation of the calibrated model (found by sweep.py on seed=1, quick N):
  A. out-of-sample: new seeds, full population size
  B. robustness: every numeric parameter perturbed by -20% / +20%
  C. ablation: remove one mechanism at a time
Run: python3 validate.py
"""
import json
import numpy as np
import studies
from studies import score, verdict, TARGETS
from model import DEFAULT

CAL = dict(habit_decay=0.004, mu_max=0.02, sigma=0.10, beta=0.04)
FULL = dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True, merit=True)


def main():
    studies.N_POP, studies.N_COH = 40000, 6000
    out = {}

    print("A. out-of-sample (N=40000, heirs 6000)")
    A = []
    for seed in (101, 202, 303):
        m = score("oos", FULL, seed=seed, P=CAL); v = verdict(m)
        A.append(dict(seed=seed, n=sum(v.values()), fails=[k for k in v if not v[k]], m=m))
        print(f"  seed {seed}: {sum(v.values())}/{len(v)}  fails={[k for k in v if not v[k]]}")
    out["A"] = A

    print("B. robustness (+-20%, quick N)")
    studies.N_POP, studies.N_COH = 20000, 3000
    B = []
    keys = ["r", "sigma", "mu_max", "kelly_frac", "f_cap", "save_wealth_rate", "cons_income", "habit_decay",
            "beta", "beta_neg", "gamma", "psi", "ego_gain", "goal_med", "lam_loss", "fraud_pi", "fraud_det",
            "y_R", "rho_R", "rescue_p"]
    for k in keys:
        for s in (0.8, 1.2):
            P = dict(CAL); P[k] = (CAL.get(k, DEFAULT[k])) * s
            m = score("rob", FULL, seed=7, P=P); v = verdict(m)
            B.append(dict(param=k, scale=s, n=sum(v.values()), fails=[x for x in v if not v[x]]))
            print(f"  {k:16s} x{s}: {sum(v.values())}/{len(v)}  {[x for x in v if not v[x]]}")
    out["B"] = B

    print("C. ablation (quick N)")
    C = []
    for drop in FULL:
        fl = dict(FULL); fl[drop] = False
        m = score("abl", fl, seed=7, P=CAL); v = verdict(m)
        C.append(dict(removed=drop, n=sum(v.values()), fails=[x for x in v if not v[x]],
                      m={x: m[x] for x in TARGETS}))
        print(f"  without {drop:10s}: {sum(v.values())}/{len(v)}  fails={[x for x in v if not v[x]]}")
    m = score("full", FULL, seed=7, P=CAL); v = verdict(m)
    C.append(dict(removed="none", n=sum(v.values()), fails=[x for x in v if not v[x]], m={x: m[x] for x in TARGETS}))
    print(f"  full model: {sum(v.values())}/{len(v)}")
    out["C"] = C
    json.dump(out, open("validation_results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
