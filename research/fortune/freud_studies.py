"""
Iteration 19: hedonism, the law of attraction, and Freud's principles as model profiles.
Born poor (K0=0.1), strong vision unless stated; final calibration; full model.
Run: python3 freud_studies.py -> freud_results.json
"""
import json
import numpy as np
from model import simulate
from sleep_studies import full_traits, FLAGS

P = json.load(open("final_calibration.json"))["chosen"]
BASE = dict(v=0.85, q_intr=0.9)
PROFILES = {
    # name: (trait overrides, mood overrides, extra flags)
    "baseline strong mind": (dict(), {}, {}),
    "pleasure principle (hedonist, id)": (dict(hedon=1.0, iota=0.4), dict(d_sd=0.9), {}),
    "reality principle (strong ego)": (dict(hedon=0.1), {}, dict(regulate=True, precommit=True, delay=True)),
    "strict superego (guilt, anxiety)": (dict(iota=0.95, ego=0.05, phi=0.3), dict(m_set=-0.4), {}),
    "sublimation (drive into work)": (dict(phi=0.9, hedon=0.2), {}, {}),
    "id + narcissism (ego 0.7, low integrity, hedonist)": (dict(ego=0.7, iota=0.2, hedon=0.8), {}, {}),
    "law of attraction only (visualise, no plan)": (dict(c0=1.8, v=0.35, q_intr=0.6), dict(m_set=0.4), {}),
    "attraction + action (MCII / act-as-if)": (dict(c0=1.1), dict(m_set=0.2), dict(delay=True)),
    "repetition compulsion (no learning after ruin)": (dict(ego=0.5), {}, dict(repeat=True)),
}
OUT = {}
for name, (tov, mov, fl) in PROFILES.items():
    n = 20000
    tr = full_traits(n, np.random.default_rng(9), 9, P, mood_over=mov, **{**BASE, **tov})
    o = simulate(np.full(n, 0.1), tr, {**FLAGS, **fl}, seed=10, P=P)
    g = o["ever_great"]
    f = g & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))
    b = o["n_bankrupt"] > 0
    OUT[name] = dict(p_great=float(g.mean()), p_stay=float((g & ~f).mean()),
                     p_fall_given_great=float(f.sum() / max(g.sum(), 1)), p_bankrupt=float(b.mean()),
                     serial_bankruptcies=float(o["n_bankrupt"][b].mean()) if b.any() else 0.0,
                     p_jail=float(np.mean(o["jailed_t"] >= 0)), median_final=float(np.median(o["K"])),
                     months_dep=float(np.mean(o["months_dep"])))
    r = OUT[name]
    print(f"{name:50s} great={r['p_great']:.3f} stay={r['p_stay']:.3f} fall|g={r['p_fall_given_great']:.3f} "
          f"bankrupt={r['p_bankrupt']:.3f} serial={r['serial_bankruptcies']:.2f} jail={r['p_jail']:.4f} "
          f"med={r['median_final']:.1f} depM={r['months_dep']:.1f}", flush=True)
json.dump(OUT, open("freud_results.json", "w"), indent=1)
