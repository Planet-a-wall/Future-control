"""
Iterations 7-12: multi-timescale mental states.

S7  calibrate the mood module to psychiatric epidemiology
S8  re-validate the 10 stylized facts with mood switched on
S9  theorem check: confidence volatility costs growth  I*kappa^2*Var(c)  (Jensen),
    and the long-run growth rate does not depend on the timescale of mood
S10 same mood amplitude, different timescales (1, 3, 12, 36 months): ruin and falls
S11 mood profiles x coping strategies
S12 poverty <-> mood feedback
Run: python3 mood_studies.py [S7|S8|...]
"""
import json
import sys
import numpy as np
import studies
from studies import verdict, TARGETS, rank, hill_alpha
from model import simulate, traits, mood_traits, DEFAULT
from validate import CAL, FULL

MOOD = {**FULL, "mood": True}
RUN = (sys.argv[1:] or ["S7", "S8", "S9", "S10", "S11", "S12"]) if __name__ == "__main__" else []
OUT = {}
try:
    OUT = json.load(open("mood_results.json"))
except Exception:
    pass

MT = {  # psychiatric anchors (qualitative bands around published estimates)
    "dep_lifetime": (0.12, 0.25, "Kessler et al. 2005: lifetime MDD ~16.6%"),
    "dep_median_months": (2.0, 6.0, "Spijker et al. 2002: median episode ~3 months"),
    "elev_lifetime": (0.015, 0.06, "Merikangas et al. 2007/2011: bipolar spectrum ~2-4%"),
    "dep_OR_poor_vs_rich": (1.3, 3.0, "Lorant et al. 2003: low SES OR ~1.8 for depression"),
}


def pop(seed, flags, P=None, N=40000, mood_over=None):
    rng = np.random.default_rng(seed)
    tr = traits(N, rng)
    tr.update(mood_traits(N, np.random.default_rng(seed + 5), P=P, **(mood_over or {})))
    K0 = 2.0 * np.exp(rng.normal(0, 1.5, N))
    return K0, tr, simulate(K0, tr, flags, seed=seed + 10, P=P)


def mood_metrics(K0, tr, o):
    m = {}
    dep = o["n_dep"] > 0
    m["dep_lifetime"] = float(dep.mean())
    m["dep_median_months"] = float(np.median(o["dep_durs"])) if len(o["dep_durs"]) else float("nan")
    m["elev_lifetime"] = float(o["ever_elev"].mean())
    q0 = rank(K0)
    poor, rich = q0 < 0.2, q0 >= 0.8
    pp, pr = dep[poor].mean(), dep[rich].mean()
    m["dep_OR_poor_vs_rich"] = float((pp / (1 - pp)) / (pr / (1 - pr)))
    return m


def check(m, T):
    return {k: bool(T[k][0] <= m[k] <= T[k][1]) for k in T}


# ------------------------------------------------------------------ S7
if "S7" in RUN:
    print("S7 calibrating mood module")
    best = None
    rows = []
    import itertools
    for dep_th, elev_th, inertia, vol in itertools.product((-1.9, -2.0, -2.1), (2.4, 2.7, 3.0), (1.0, 2.0, 3.0), (0.38, 0.4)):
                pov, tau = 0.2, 6.0
                P = {**CAL, "dep_th": dep_th, "elev_th": elev_th, "mood_vol": vol, "mood_pov": pov, "mood_tau": tau,
                     "inertia": inertia}
                K0, tr, o = pop(1, MOOD, P=P, N=20000)
                m = mood_metrics(K0, tr, o); v = check(m, MT)
                rows.append(dict(P=dict(dep_th=dep_th, elev_th=elev_th, mood_vol=vol, mood_pov=pov, mood_tau=tau,
                                        inertia=inertia),
                                 m=m, n=sum(v.values())))
                if sum(v.values()) >= 3: print(f"  {rows[-1]['P']}: {sum(v.values())}/4  " +
                      " ".join(f"{k}={m[k]:.3f}" for k in MT))
    rows.sort(key=lambda r: (-r["n"], abs(r["m"]["dep_lifetime"] - 0.17) + abs(r["m"]["dep_median_months"] - 3) / 10))
    OUT["S7"] = dict(rows=rows, chosen=rows[0]["P"])
    print("  chosen:", rows[0]["P"], rows[0]["m"])

MCAL = {**CAL, **OUT.get("S7", {}).get("chosen", {})}
try:   # final joint calibration (iterations 10-12) supersedes S7 when available
    if __name__ == "__main__":
        MCAL = json.load(open("mood_final_results.json"))["chosen"]
        print("using final calibration:", {k: MCAL[k] for k in ("beta", "kelly_frac", "mood_pov", "mood_vol", "a_c", "imp_mult")})
except FileNotFoundError:
    pass

# ------------------------------------------------------------------ S8
if "S8" in RUN:
    print("S8 re-validation with mood on (out-of-sample seeds)")
    res = []
    for seed in (101, 202, 303):
        studies_P = MCAL
        rng = np.random.default_rng(seed)
        # reuse studies.score but with mood traits: monkeypatch traits to include mood
        orig = studies.traits
        studies.traits = lambda N, r, **kw: {**orig(N, r, **kw), **mood_traits(N, np.random.default_rng(seed + 5), P=MCAL)}
        studies.N_POP, studies.N_COH = 40000, 6000
        m = studies.score("mood", MOOD, seed=seed, P=MCAL)
        studies.traits = orig
        v = verdict(m)
        res.append(dict(seed=seed, n=sum(v.values()), fails=[k for k in v if not v[k]], m=m))
        print(f"  seed {seed}: {sum(v.values())}/10 fails={[ (k, round(m[k],3)) for k in v if not v[k]]}")
        K0, tr, o = pop(seed, MOOD, P=MCAL)
        mm = mood_metrics(K0, tr, o)
        print("     mood anchors:", {k: round(x, 3) for k, x in mm.items()}, check(mm, MT))
        res[-1]["mood"] = mm
    OUT["S8"] = res

# ------------------------------------------------------------------ S9
if "S9" in RUN:
    print("S9 Jensen penalty and timescale invariance (single venture, no consumption)")
    kap, mu, sig, r = 0.5, 0.012, 0.10, 0.0
    I = mu ** 2 / (2 * sig ** 2)
    rng = np.random.default_rng(3)
    rows = []
    for cbar, sd in [(2.0, 0.0), (2.0, 0.5), (2.0, 1.0), (2.0, 1.5), (3.0, 1.0)]:
        for tau in (1.0, 3.0, 12.0, 36.0):
            n, T = 4000, 2400
            a = np.exp(-1 / tau)
            x = rng.standard_normal(n); lg = np.zeros(n); ruin = np.zeros(n, bool); W = np.zeros(n)
            for t in range(T):
                x = a * x + np.sqrt(1 - a * a) * rng.standard_normal(n)
                c = np.maximum(cbar + sd * x, 0)
                f = kap * c * mu / sig ** 2
                X = mu + sig * rng.standard_normal(n)
                step = np.log(np.maximum(1 + f * X, 1e-12))
                lg += step; W += step
                ruin |= W < np.log(0.1)        # ever lost 90% from start
            g_sim = lg.mean() / T
            g_th = I * (1 - (kap * cbar - 1) ** 2 - kap ** 2 * sd ** 2)
            rows.append(dict(cbar=cbar, sd=sd, tau=tau, g_sim=float(g_sim), g_th=float(g_th),
                             p_lose90=float(ruin.mean())))
            print(f"  cbar={cbar} sd={sd} tau={tau:>4}: growth sim {g_sim*1200:6.2f}%/yr  theory {g_th*1200:6.2f}%/yr"
                  f"  P(lose 90% within 200 yrs)={ruin.mean():.3f}")
    OUT["S9"] = rows

# ------------------------------------------------------------------ S10
def cohort(over_traits, mood_over, flags, n=20000, seed=9, K0=0.1, P=None):
    tr = traits(n, np.random.default_rng(seed), **over_traits)
    tr.update(mood_traits(n, np.random.default_rng(seed + 5), P=P, **mood_over))
    o = simulate(np.full(n, K0), tr, flags, seed=seed + 1, P=P)
    g = o["ever_great"]
    f = g & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))
    return dict(p_great=float(g.mean()), p_fall_given_great=float(f.sum() / max(g.sum(), 1)),
                p_stay=float((g & ~f).mean()), p_bankrupt=float(np.mean(o["n_bankrupt"] > 0)),
                p_jail=float(np.mean(o["jailed_t"] >= 0)), median_final=float(np.median(o["K"])),
                months_dep=float(np.mean(o["months_dep"])), imp_hi=float(np.mean(o["imp_hi"])))

STRONG = dict(v=0.8, q_intr=0.85)
if "S10" in RUN:
    print("S10 same amplitude, different timescales (born poor, strong mind; no outcome feedback)")
    rows = []
    for tau in (1.0, 3.0, 12.0, 36.0):
        r_ = cohort(STRONG, dict(m_tau=tau, m_vol=0.6, m_set=0.0, m_eta=0.0, d_sd=0.6, bipolar=0.0), MOOD, P=MCAL)
        r_["tau"] = tau; rows.append(r_)
        print(f"  tau={tau:>4} months: " + " ".join(f"{k}={v:.3f}" for k, v in r_.items() if k != "tau"))
    base = cohort(STRONG, dict(m_tau=6.0, m_vol=0.0, m_set=0.0, m_eta=0.0, d_sd=0.0, bipolar=0.0), MOOD, P=MCAL)
    print("  no mood swings:     " + " ".join(f"{k}={v:.3f}" for k, v in base.items()))
    OUT["S10"] = dict(rows=rows, flat=base)

# ------------------------------------------------------------------ S11
if "S11" in RUN:
    print("S11 mood profiles x coping (born poor, strong mind)")
    profiles = {
        "stable": dict(m_vol=0.25, d_sd=0.3, m_tau=6.0, m_set=0.0, bipolar=0.0),
        "typical": dict(m_set=0.0, bipolar=0.0),
        "daily swings": dict(m_vol=0.3, d_sd=1.2, m_tau=2.0, m_set=0.0, bipolar=0.0),
        "long episodes": dict(m_vol=0.7, m_tau=24.0, m_set=0.0, bipolar=0.0),
        "low set-point (depression-prone)": dict(m_set=-0.6, m_vol=0.6, bipolar=0.0),
        "high set-point (elevated-prone)": dict(m_set=0.6, m_vol=0.6, bipolar=0.0),
        "bipolar-like": dict(m_vol=1.1, m_tau=4.0, m_set=0.0, bipolar=1.0),
    }
    coping = {"none": {}, "precommit": dict(precommit=True), "regulate": dict(regulate=True),
              "delay 48h": dict(delay=True), "all three": dict(precommit=True, regulate=True, delay=True)}
    res = {}
    for pn, mo in profiles.items():
        res[pn] = {}
        for cn, cf in coping.items():
            if pn not in ("typical", "long episodes", "bipolar-like", "high set-point (elevated-prone)", "daily swings") and cn != "none":
                continue
            r_ = cohort(STRONG, mo, {**MOOD, **cf}, P=MCAL)
            res[pn][cn] = r_
            print(f"  {pn:34s} {cn:10s} great={r_['p_great']:.3f} stay={r_['p_stay']:.3f} "
                  f"fall|g={r_['p_fall_given_great']:.3f} bankrupt={r_['p_bankrupt']:.3f} jail={r_['p_jail']:.4f} "
                  f"med={r_['median_final']:.1f} depM={r_['months_dep']:.1f}")
    OUT["S11"] = res

# ------------------------------------------------------------------ S12
if "S12" in RUN:
    print("S12 poverty <-> mood loop (population)")
    res = {}
    for label, P in [("feedback on", MCAL), ("no outcome/poverty feedback", {**MCAL, "mood_eta": 0.0, "mood_pov": 0.0})]:
        K0, tr, o = pop(77, MOOD, P=P)
        if label.startswith("no"):
            pass
        q0 = rank(K0); qf = rank(o["K"] + 1e-9 * np.random.default_rng(0).random(len(K0)))
        dep = o["n_dep"] > 0
        res[label] = dict(dep_by_birth_quintile=[float(dep[(q0 >= a) & (q0 < a + 0.2 + 1e-9)].mean()) for a in np.arange(0, 1, 0.2)],
                          mobility_Q1_to_Q5=float(np.mean(qf[q0 < 0.2] >= 0.8)),
                          months_dep_poor=float(o["months_dep"][q0 < 0.2].mean()),
                          months_dep_rich=float(o["months_dep"][q0 >= 0.8].mean()),
                          final_median_poor=float(np.median(o["K"][q0 < 0.2])))
        print(" ", label, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for k, v in res[label].items()})
    OUT["S12"] = res

json.dump(OUT, open("mood_results.json", "w"), indent=1, default=float)
