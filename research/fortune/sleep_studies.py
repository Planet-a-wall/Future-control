"""
Iterations 13-15: sleep and fatigue inside the multi-timescale mind.

S13 re-validation: 10 economic facts + 4 psychiatric anchors + short-sleep prevalence
S14 work intensity: is there an optimal drive once fatigue and sleep debt are real?
S15 profiles and interventions (busy short-sleeper, protected sleep, breaks, routine)
Run: python3 sleep_studies.py [S13|S14|S15]
"""
import json
import sys
import numpy as np
import studies
from studies import verdict, rank
from model import simulate, traits, mood_traits, sleep_traits
from mood_studies import MT, mood_metrics, check

FINAL = json.load(open("mood_final_results.json"))["chosen"]
if __name__ == "__main__":
    try:   # iteration 14 sleep calibration supersedes the mood-only calibration
        FINAL = json.load(open("sleep_calib_results.json"))["chosen"]
        print("using sleep calibration:", {k: FINAL[k] for k in ("sleep_mean", "k_work", "rho_fat")})
    except FileNotFoundError:
        pass
FLAGS = dict(ratchet=True, goal_drive=True, hubris=True, loss_risk=True, fraud=True, merit=True,
             mood=True, sleep=True)
ST = {"short_sleep_share": (0.28, 0.42, "CDC / Liu et al. 2016: ~35% of adults sleep < 7 h")}
RUN = (sys.argv[1:] or ["S13", "S14", "S15"]) if __name__ == "__main__" else []
try:
    OUT = json.load(open("sleep_results.json"))
except Exception:
    OUT = {}


def full_traits(N, r, seed, P, sleep_over=None, mood_over=None, **kw):
    t = traits(N, r, **kw)
    t.update(mood_traits(N, np.random.default_rng(seed + 5), P=P, **(mood_over or {})))
    t.update(sleep_traits(N, np.random.default_rng(seed + 6), P=P, **(sleep_over or {})))
    return t


def score_full(P, seed, flags=FLAGS, N_POP=40000, N_COH=6000):
    orig = studies.traits
    studies.traits = lambda N, r, **kw: full_traits(N, r, seed, P, **kw)
    studies.N_POP, studies.N_COH = N_POP, N_COH
    m = studies.score("full", flags, seed=seed, P=P)
    studies.traits = orig
    return m


def pop_full(seed, P, flags=FLAGS, N=40000):
    rng = np.random.default_rng(seed)
    tr = full_traits(N, rng, seed, P)
    K0 = 2.0 * np.exp(rng.normal(0, 1.5, N))
    return K0, tr, simulate(K0, tr, flags, seed=seed + 10, P=P)


def cohort(P, flags, n=20000, seed=9, K0=0.1, sleep_over=None, mood_over=None, **over):
    tr = full_traits(n, np.random.default_rng(seed), seed, P, sleep_over=sleep_over, mood_over=mood_over, **over)
    o = simulate(np.full(n, K0), tr, flags, seed=seed + 1, P=P)
    g = o["ever_great"]
    f = g & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))
    return dict(p_great=float(g.mean()), p_stay=float((g & ~f).mean()),
                p_fall_given_great=float(f.sum() / max(g.sum(), 1)), p_bankrupt=float(np.mean(o["n_bankrupt"] > 0)),
                p_jail=float(np.mean(o["jailed_t"] >= 0)), median_final=float(np.median(o["K"])),
                debt=float(np.mean(o["debt_mean"])), fatigue=float(np.mean(o["fat_mean"])),
                months_dep=float(np.mean(o["months_dep"])))


if "S13" in RUN:
    print("S13 re-validation with sleep + fatigue")
    res = []
    for seed in (101, 202, 303):
        m = score_full(FINAL, seed); v = verdict(m)
        K0, tr, o = pop_full(seed, FINAL)
        mm = mood_metrics(K0, tr, o); mm["short_sleep_share"] = float(np.mean(o["short_share"] > 0.5))
        mv = {**check(mm, MT), **{k: ST[k][0] <= mm[k] <= ST[k][1] for k in ST}}
        res.append(dict(seed=seed, n=sum(v.values()), fails=[(k, round(m[k], 3)) for k in v if not v[k]],
                        anchors=mm, anchors_pass=mv, m=m))
        print(f"  seed {seed}: SF {sum(v.values())}/10 {res[-1]['fails']} | anchors {sum(mv.values())}/5 "
              + " ".join(f"{k}={x:.3f}" for k, x in mm.items()), flush=True)
    OUT["S13"] = res

if "S14" in RUN:
    print("S14 work intensity (born poor, strong vision, mostly intrinsic)")
    rows = []
    for protect in (False, True):
        for qi in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            r_ = cohort(FINAL, {**FLAGS, "sleep_protect": protect}, v=0.8, q_intr=qi, phi=0.85)
            r_.update(q_intr=qi, protect=protect); rows.append(r_)
            print(f"  protect={protect!s:5} q_intr={qi}: stay={r_['p_stay']:.3f} great={r_['p_great']:.3f} "
                  f"med={r_['median_final']:.1f} bankrupt={r_['p_bankrupt']:.3f} debt={r_['debt']:.2f} "
                  f"fatigue={r_['fatigue']:.3f} depM={r_['months_dep']:.1f}", flush=True)
    OUT["S14"] = rows

if "S15" in RUN:
    print("S15 profiles x interventions (born poor, strong mind)")
    STRONG = dict(v=0.8, q_intr=0.85)
    profiles = {
        "good sleeper (7.8 h)": dict(sleep_over=dict(S_base=7.8)),
        "typical sleeper": dict(),
        "short sleeper (5.8 h)": dict(sleep_over=dict(S_base=5.8)),
        "busy short sleeper": dict(sleep_over=dict(S_base=5.8), mood_over=dict(d_sd=0.9), phi=0.2, q_intr=0.85),
    }
    inter = {"none": {}, "protect sleep": dict(sleep_protect=True), "breaks": dict(breaks=True),
             "routine only (delay+precommit)": dict(delay=True, precommit=True),
             "sleep + breaks + routine": dict(sleep_protect=True, breaks=True, delay=True, precommit=True)}
    res = {}
    for pn, po in profiles.items():
        res[pn] = {}
        for iname, iflags in inter.items():
            kw = dict(STRONG); kw.update({k: v for k, v in po.items() if k not in ("sleep_over", "mood_over")})
            r_ = cohort(FINAL, {**FLAGS, **iflags}, sleep_over=po.get("sleep_over"), mood_over=po.get("mood_over"), **kw)
            res[pn][iname] = r_
            print(f"  {pn:22s} {iname:32s} stay={r_['p_stay']:.3f} great={r_['p_great']:.3f} "
                  f"fall|g={r_['p_fall_given_great']:.3f} bankrupt={r_['p_bankrupt']:.3f} med={r_['median_final']:.1f} "
                  f"debt={r_['debt']:.2f} depM={r_['months_dep']:.1f}", flush=True)
    OUT["S15"] = res

if RUN:
    json.dump(OUT, open("sleep_results.json", "w"), indent=1, default=float)
