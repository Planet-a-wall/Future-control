"""
Iteration 15: multi-generation dynasties on the full model (mind + mood + sleep).

Each generation lives T=40 economic years. At the end the estate (after estate tax
above an exemption) is split among 1-4 children; we follow one representative child
per lineage, so the population size stays constant. Children inherit traits partly
(normal-score correlation h per trait), family reputation partly, and nothing else.

Targets (re-specification of the 'dynastic erosion' fact, measured across generations
as the source literature does):
  D1 parent-child rank correlation of end-of-life wealth   0.20-0.50
  D2 top-1% founders: share of gen-1 heirs ending with < 50% of founder wealth  0.20-0.80
  D3 top-1% founders: share of lineages still top-1% after 3 generations        0.03-0.40
  plus the 10 economic facts measured in generation 1 (inherited start)
Run: python3 dynasty.py
"""
import json
import numpy as np
from scipy.stats import norm
from model import simulate, traits, mood_traits, sleep_traits
from studies import rank, hill_alpha

P = json.load(open("sleep_calib_results.json"))["chosen"]
FLAGS = dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True, merit=True,
             mood=True, sleep=True)
H = dict(v=0.45, q_intr=0.25, phi=0.2, iota=0.25, omega=0.25, ego=0.25, goal=0.3, c0=0.1,
         m_set=0.3, m_vol=0.3, m_tau=0.2, d_sd=0.2, m_eta=0.2, bipolar=0.4, S_base=0.3)
KIDS, KIDS_P = np.array([1, 2, 3, 4]), np.array([0.25, 0.45, 0.20, 0.10])
TAX, EXEMPT = 0.40, 200.0
N, GENS = 40000, 4


def fresh(n, seed):
    t = traits(n, np.random.default_rng(seed))
    t.update(mood_traits(n, np.random.default_rng(seed + 5), P=P))
    t.update(sleep_traits(n, np.random.default_rng(seed + 6), P=P))
    return t


def inherit(parent, seed):
    rng = np.random.default_rng(seed)
    pool = fresh(N, seed + 1)
    child = {}
    for k, xp in parent.items():
        h = H.get(k, 0.0)
        zp = norm.ppf((rank(xp + 1e-12 * rng.random(N)) * (N - 1) + 0.5) / N)
        zc = h * zp + np.sqrt(1 - h * h) * rng.standard_normal(N)
        q = np.sort(pool[k])
        child[k] = q[np.clip((norm.cdf(zc) * N).astype(int), 0, N - 1)]
    return child


def gini(x):
    x = np.sort(np.maximum(x, 0)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def run_dynasty(P_, N_=N, gens_=GENS, tax=TAX, exempt=EXEMPT, verbose=True, seed=2026):
    global P, N
    P, N = P_, N_
    rng = np.random.default_rng(seed)
    tr = fresh(N, 500)
    K0 = 2.0 * np.exp(rng.normal(0, 1.5, N))
    R0 = np.zeros(N)
    founder_final = None; founder_top = None
    gens = []
    lineage_top = []
    for g in range(gens_):
        o = simulate(K0, tr, FLAGS, seed=1000 + g, P=P, R0=R0)
        Kf = np.maximum(o["K"], 0)
        top1 = Kf >= np.quantile(Kf, 0.99)
        rec = dict(gen=g, gini=gini(Kf), alpha=hill_alpha(Kf), top1_share=float(Kf[top1].sum() / Kf.sum()),
                   median=float(np.median(Kf)), bankrupt=float(np.mean(o["n_bankrupt"] > 0)),
                   jailed=float(np.mean(o["jailed_t"] >= 0)), dep=float(np.mean(o["n_dep"] > 0)))
        q0 = rank(K0 + 1e-9 * rng.random(N)); qf = rank(Kf + 1e-9 * rng.random(N))
        rec["mobility_Q1_to_Q5"] = float(np.mean(qf[q0 < 0.2] >= 0.8))
        rec["rank_rank_start_end"] = float(np.polyfit(q0, qf, 1)[0])
        if g == 0:
            founder_final = Kf.copy(); founder_top = top1.copy()
        else:
            rec["D1_parent_child_rank"] = float(np.corrcoef(rank(prev_final), qf)[0, 1])
            if g == 1:
                rec["D2_heir_lost_half_vs_founder"] = float(np.mean(Kf[founder_top] < 0.5 * founder_final[founder_top]))
                rec["heir_lost_half_vs_own_inheritance"] = float(np.mean(Kf[founder_top] < 0.5 * K0[founder_top]))
            rec["founder_lineage_still_top1"] = float(np.mean(top1[founder_top]))
        gens.append(rec)
        if verbose: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        # --- next generation
        prev_final = Kf.copy()
        kids = rng.choice(KIDS, size=N, p=KIDS_P)
        estate = Kf - tax * np.maximum(Kf - exempt, 0)
        K0 = estate / kids
        R0 = 0.5 * np.maximum(o["R"], 0)
        tr = inherit(tr, seed=2000 + g)


    return gens

if __name__ == "__main__":
    gens = run_dynasty(P)
    D3 = gens[-1]["founder_lineage_still_top1"]
    targets = {
        "D1_parent_child_rank": (gens[1]["D1_parent_child_rank"], 0.20, 0.50),
        "D2_heir_lost_half_vs_founder": (gens[1]["D2_heir_lost_half_vs_founder"], 0.20, 0.80),
        "D3_top1_after_3_generations": (D3, 0.03, 0.40),
    }
    print("\nTargets:")
    for k, (v, lo, hi) in targets.items():
        print(f"  {'PASS' if lo <= v <= hi else 'fail'} {k} = {v:.3f}  (target {lo}-{hi})")
    json.dump(dict(gens=gens, targets={k: dict(value=v, lo=lo, hi=hi, pass_=bool(lo <= v <= hi))
                                       for k, (v, lo, hi) in targets.items()},
                   settings=dict(H=H, kids=KIDS.tolist(), kids_p=KIDS_P.tolist(), tax=TAX, exempt=EXEMPT)),
              open("dynasty_results.json", "w"), indent=1, default=float)
