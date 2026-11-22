"""
Iteration 18b: do the headline findings survive the final calibration?
Full model: mind + mood + sleep + capital tax + wealth-dependent returns + steep satiation.
Run: python3 final_check.py  -> final_check_results.json
"""
import json
import numpy as np
from model import simulate
from sleep_studies import full_traits, cohort, pop_full, FLAGS

P = json.load(open("final_calibration.json"))["chosen"]
OUT = {}


def fall_mask(o):
    g = o["ever_great"]
    return g, g & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))


# H1 hubris index monotone in post-success fall rate
K0, tr, o = pop_full(42, P, N=80000)
g, fell = fall_mask(o)
Ez = 1 / np.sqrt(2 * np.pi)
from model import DEFAULT
PF = {**DEFAULT, **P}
c_eq = lambda e, i, s: 1 + Ez * (PF["beta"] * (1 + PF["ego_gain"] * e) * (1 + PF["psi"] * s) - PF["beta_neg"]) / (PF["gamma"] * (0.5 + i)) * (1 + PF["psi"] * s)
Hpost = PF["kelly_frac"] * c_eq(tr["ego"], tr["iota"], 1.0)
bins = [0, 1.0, 1.5, 2.0, 2.5, 10]
OUT["H1"] = [dict(lo=a, hi=b, n=int((g & (Hpost >= a) & (Hpost < b)).sum()),
                  p_fall=float(fell[g & (Hpost >= a) & (Hpost < b)].mean()) if (g & (Hpost >= a) & (Hpost < b)).sum() else None)
             for a, b in zip(bins[:-1], bins[1:])]
print("H1", [(b["lo"], b["n"], round(b["p_fall"], 3) if b["p_fall"] is not None else None) for b in OUT["H1"]])

# H4 enjoyment (phi) and integrity (iota) in the population
def by(t, edges):
    r = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (tr[t] >= a) & (tr[t] < b)
        r.append(dict(lo=a, p_great=float(g[s].mean()), p_fall_given_great=float(fell[s & g].sum() / max((s & g).sum(), 1)),
                      p_jail=float(np.mean(o["jailed_t"][s] >= 0)), p_bankrupt=float(np.mean(o["n_bankrupt"][s] > 0))))
    return r
OUT["H4"] = dict(phi=by("phi", [0, .25, .5, .75, 1.01]), iota=by("iota", [0, .4, .6, .8, 1.01]), ego=by("ego", [0, .15, .3, .5, 1.01]))
for k, rows in OUT["H4"].items():
    print("H4", k, [(r["lo"], round(r["p_great"], 3), round(r["p_fall_given_great"], 3), round(r["p_jail"], 4), round(r["p_bankrupt"], 3)) for r in rows])

# H2 birth
OUT["H2"] = []
for lab, k0, r0 in [("poor", 0.1, 0), ("middle", 2.0, 0), ("rich", 50.0, 5), ("heir/royal", 5000.0, 20)]:
    n = 20000
    t2 = full_traits(n, np.random.default_rng(7), 7, P)
    oc = simulate(np.full(n, k0), t2, FLAGS, seed=8, P=P, R0=r0)
    g2, f2 = fall_mask(oc)
    OUT["H2"].append(dict(cohort=lab, p_great=float(g2.mean()), p_fall_given_great=float(f2.sum() / max(g2.sum(), 1)),
                          p_final_ge_100=float(np.mean(oc["K"] >= 100)), p_bankrupt=float(np.mean(oc["n_bankrupt"] > 0)),
                          p_jail=float(np.mean(oc["jailed_t"] >= 0))))
    print("H2", OUT["H2"][-1])

# H3 born poor, mental profiles
prof = {"average": {}, "vision+drive": dict(v=0.85, q_intr=0.9),
        "+high ego": dict(v=0.85, q_intr=0.9, ego=0.7),
        "+high ego +low integrity": dict(v=0.85, q_intr=0.9, ego=0.7, iota=0.2),
        "+enjoys work": dict(v=0.85, q_intr=0.9, phi=0.85),
        "+enjoys +low ego +integrity": dict(v=0.85, q_intr=0.9, phi=0.85, ego=0.1, iota=0.9)}
OUT["H3"] = {}
for pn, ov in prof.items():
    OUT["H3"][pn] = cohort(P, FLAGS, **ov)
    r = OUT["H3"][pn]
    print(f"H3 {pn:28s} great={r['p_great']:.3f} stay={r['p_stay']:.3f} fall|g={r['p_fall_given_great']:.3f} jail={r['p_jail']:.4f}")

# H5 merit: money-only vs honest feedback
OUT["H5"] = {}
for lab, extra in [("money only", {}), ("honest feedback", dict(merit_feedback=True))]:
    OUT["H5"][lab] = []
    for om in (0.0, 0.05, 0.1, 0.2):
        r = cohort(P, {**FLAGS, **extra}, v=0.85, q_intr=0.9, omega=om)
        OUT["H5"][lab].append(dict(omega=om, stay=r["p_stay"], median=r["median_final"]))
    print("H5", lab, [(x["omega"], round(x["stay"], 3), round(x["median"], 1)) for x in OUT["H5"][lab]])

# H6 mood profiles and coping
STRONG = dict(v=0.8, q_intr=0.85)
moods = {"stable": dict(m_vol=0.25, d_sd=0.3, m_tau=6.0, m_set=0.0, bipolar=0.0),
         "typical": dict(m_set=0.0, bipolar=0.0),
         "daily swings": dict(m_vol=0.3, d_sd=1.2, m_tau=2.0, m_set=0.0, bipolar=0.0),
         "depression-prone": dict(m_set=-0.6, m_vol=0.6, bipolar=0.0),
         "elevated-prone": dict(m_set=0.6, m_vol=0.6, bipolar=0.0),
         "bipolar-like": dict(m_vol=1.1, m_tau=4.0, m_set=0.0, bipolar=1.0)}
OUT["H6"] = {}
for mn, mo in moods.items():
    OUT["H6"][mn] = {}
    for cn, cf in {"none": {}, "delay": dict(delay=True), "regulate": dict(regulate=True), "precommit": dict(precommit=True)}.items():
        if mn in ("stable", "depression-prone") and cn != "none":
            continue
        r = cohort(P, {**FLAGS, **cf}, mood_over=mo, **STRONG)
        OUT["H6"][mn][cn] = r
        print(f"H6 {mn:17s} {cn:9s} stay={r['p_stay']:.3f} bankrupt={r['p_bankrupt']:.3f} fall|g={r['p_fall_given_great']:.3f} med={r['median_final']:.1f}")

# H7 sleep
OUT["H7"] = {}
for sn, so in {"good 7.8h": dict(S_base=7.8), "short 5.8h": dict(S_base=5.8)}.items():
    for iname, fl in {"none": {}, "protect sleep": dict(sleep_protect=True), "routine only": dict(delay=True, precommit=True)}.items():
        r = cohort(P, {**FLAGS, **fl}, sleep_over=so, **STRONG)
        OUT["H7"][f"{sn} | {iname}"] = r
        print(f"H7 {sn:10s} {iname:14s} stay={r['p_stay']:.3f} med={r['median_final']:.1f} depM={r['months_dep']:.1f}")

json.dump(OUT, open("final_check_results.json", "w"), indent=1, default=float)
