"""
Iterative studies: add one mechanism at a time and score each model version
against stylized facts (SF) from the empirical literature.

Run: python3 studies.py [--quick]
"""
import json
import sys
import numpy as np
from model import simulate, traits, DEFAULT

QUICK = "--quick" in sys.argv
N_POP = 20000 if QUICK else 40000
N_COH = 3000 if QUICK else 6000

VERSIONS = {
    "M1 static mind": dict(),
    "M2 +habits,+goal-drive": dict(ratchet=True, goal_drive=True),
    "M3 +hubris": dict(ratchet=True, goal_drive=True, hubris=True),
    "M4 +loss-risk,+fraud": dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True),
    "M5 +merit": dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True, merit=True),
}

# targets: (low, high, source)
TARGETS = {
    "mobility_Q1_to_Q5": (0.03, 0.15, "Chetty et al. 2014: ~7.5% bottom->top quintile (income, US)"),
    "rank_rank_slope": (0.20, 0.50, "Chetty et al. 2014: rank-rank slope ~0.34 (income)"),
    "pareto_alpha": (1.2, 2.0, "Gabaix 2009; top-wealth tails alpha ~1.2-2"),
    "heirs_lost_half": (0.20, 0.80, "Arnott-Bernstein-Wu 2015 / Bennedsen et al. 2007 (qualitative: dynasties erode)"),
    "great_then_fell": (0.02, 0.30, "rise-and-fall exists but is a minority (qualitative)"),
    "fallers_higher_conf": (0.0, 99, "Malmendier-Tate 2005/08, Billett-Qian 2008: success -> overconfidence -> value-destroying bets"),
    "heirs_bankrupt": (0.0, 0.10, "qualitative: large inherited fortunes rarely reach insolvency within one lifetime"),
    "bankrupt_any": (0.03, 0.20, "qualitative: lifetime insolvency is a minority outcome"),
    "jailed_any": (0.0005, 0.02, "qualitative: white-collar imprisonment is rare"),
    "extr_minus_intr_drop": (0.03, 1.0, "Niemiec-Ryan-Deci 2009 / Kasser-Ryan 1996: extrinsic attainment sustains less than intrinsic (relative)"),
}


def hill_alpha(x, frac=0.01):
    x = np.sort(x[x > 0])[::-1]
    k = max(int(len(x) * frac), 10)
    return float(k / np.sum(np.log(x[:k] / x[k])))


def rank(x):
    return np.argsort(np.argsort(x)) / (len(x) - 1)


def score(name, flags, seed=1, P=None):
    rng = np.random.default_rng(seed)
    # --- population cohort
    tr = traits(N_POP, rng)
    K0 = 2.0 * np.exp(rng.normal(0, 1.5, N_POP))
    out = simulate(K0, tr, flags, seed=seed + 10, P=P)
    Kf = out["K"]
    q0 = rank(K0); qf = rank(Kf + 1e-9 * rng.random(N_POP))
    m = {}
    m["mobility_Q1_to_Q5"] = float(np.mean(qf[q0 < 0.2] >= 0.8))
    m["rank_rank_slope"] = float(np.polyfit(q0, qf, 1)[0])
    m["pareto_alpha"] = hill_alpha(Kf)
    g = out["ever_great"]
    fell = g & (((out["t_first_bankrupt"] > out["t_great"])) | (out["jailed_t"] > out["t_great"]) |
                (out["K"] < 0.1 * out["peak"]))
    m["share_ever_great"] = float(g.mean())
    m["great_then_fell"] = float(fell.sum() / max(g.sum(), 1))
    m["great_jailed"] = float(((out["jailed_t"] > out["t_great"]) & g).sum() / max(g.sum(), 1))
    if fell.sum() > 5 and (g & ~fell).sum() > 5:
        m["fallers_higher_conf"] = float(np.median(out["peak_c"][fell]) - np.median(out["peak_c"][g & ~fell]))
    else:
        m["fallers_higher_conf"] = float("nan")
    ext = (tr["phi"] < 0.3) & ~np.isnan(out["q_after_goal"])
    intr = (tr["phi"] > 0.7) & ~np.isnan(out["q_after_goal"])
    m["extrinsic_drive_drop"] = float(np.mean(out["q_at_goal"][ext] - out["q_after_goal"][ext])) if ext.sum() > 5 else float("nan")
    m["intrinsic_drive_drop"] = float(np.mean(out["q_at_goal"][intr] - out["q_after_goal"][intr])) if intr.sum() > 5 else float("nan")
    m["extr_minus_intr_drop"] = m["extrinsic_drive_drop"] - m["intrinsic_drive_drop"]
    m["bankrupt_any"] = float(np.mean(out["n_bankrupt"] > 0))
    m["jailed_any"] = float(np.mean(out["jailed_t"] >= 0))
    # --- heirs cohort (king / sheikh / billionaire child)
    trh = traits(N_COH, np.random.default_rng(seed + 2))
    Kh0 = np.full(N_COH, 5000.0)
    oh = simulate(Kh0, trh, flags, seed=seed + 20, P=P, R0=20.0)
    m["heirs_lost_half"] = float(np.mean(oh["K"] < 0.5 * Kh0))
    m["heirs_bankrupt"] = float(np.mean(oh["n_bankrupt"] > 0))
    m["heirs_jailed"] = float(np.mean(oh["jailed_t"] >= 0))
    m["heirs_median_final"] = float(np.median(oh["K"]))
    return m


def verdict(m):
    res = {}
    for k, (lo, hi, _) in TARGETS.items():
        v = m.get(k, float("nan"))
        res[k] = bool(lo <= v <= hi) if v == v else False
    return res


if __name__ == "__main__":
    allres = {}
    for name, fl in VERSIONS.items():
        m = score(name, fl)
        v = verdict(m)
        allres[name] = dict(metrics=m, pass_=v, n_pass=sum(v.values()))
        print(f"\n=== {name}: {sum(v.values())}/{len(v)} targets met")
        for k in TARGETS:
            print(f"  {'PASS' if v[k] else 'fail'}  {k:24s} = {m.get(k, float('nan')):8.3f}   target {TARGETS[k][0]}..{TARGETS[k][1]}")
        for k in ("share_ever_great", "great_jailed", "bankrupt_any", "jailed_any", "heirs_bankrupt",
                  "heirs_jailed", "heirs_median_final"):
            print(f"        {k:24s} = {m[k]:8.4f}")
    with open("results_studies.json", "w") as fh:
        json.dump(allres, fh, indent=1)
