"""Iteration 16: capital-income tax + joint calibration of single-generation facts, psychiatric/sleep
anchors and multi-generation facts (incl. stationarity of the wealth distribution)."""
import itertools, json
import numpy as np
from studies import verdict
from sleep_studies import score_full, pop_full, ST
from mood_studies import MT, mood_metrics, check
import dynasty

BASE = json.load(open("sleep_calib_results.json"))["chosen"]


def dyn_targets(g):
    t = {
        "D1_parent_child_rank": (g[1]["D1_parent_child_rank"], 0.25, 0.45),   # Adermon et al. 2018: 0.3-0.4
        "D2_heir_lost_half_vs_founder": (g[1]["D2_heir_lost_half_vs_founder"], 0.20, 0.80),
        "D3_top1_after_3_gen": (g[-1]["founder_lineage_still_top1"], 0.03, 0.40),
        "S1_gini_drift": (g[-1]["gini"] - g[1]["gini"], -0.03, 0.03),
        "S2_top1_share_ratio": (g[-1]["top1_share"] / g[1]["top1_share"], 0.85, 1.15),
    }
    return t, {k: lo <= v <= hi for k, (v, lo, hi) in t.items()}


def evaluate(P, seed=1, N_single=20000, N_dyn=20000, full=False):
    m = score_full(P, seed, N_POP=N_single, N_COH=N_single // 7); v = verdict(m)
    K0, tr, o = pop_full(seed, P, N=N_single)
    mm = mood_metrics(K0, tr, o); mm["short_sleep_share"] = float(np.mean(o["short_share"] > 0.5))
    mv = {**check(mm, MT), **{k: ST[k][0] <= mm[k] <= ST[k][1] for k in ST}}
    g = dynasty.run_dynasty(P, N_=N_dyn, verbose=False, seed=2026 + seed)
    dt, dv = dyn_targets(g)
    fails = [(k, round(m[k], 3)) for k in v if not v[k]] + [(k, round(mm[k], 3)) for k in mv if not mv[k]] \
        + [(k, round(dt[k][0], 3)) for k in dv if not dv[k]]
    return dict(sf=sum(v.values()), anchors=sum(mv.values()), dyn=sum(dv.values()),
                total=sum(v.values()) + sum(mv.values()) + sum(dv.values()), fails=fails,
                m=m, mm=mm, dyn_vals={k: x[0] for k, x in dt.items()}, gens=g)


if __name__ == "__main__":
    rows = []
    import sys
    grid = [dict(cap_tax=ct, r=0.004, ret_prem=pr) for ct, pr in itertools.product((0.15, 0.2, 0.25), (0.001, 0.0015, 0.002))]
    for G in grid:
        P = {**BASE, **G}
        ct, r = G["cap_tax"], G["ret_prem"]
        e = evaluate(P)
        rows.append(dict(P=G, **{k: e[k] for k in ("sf", "anchors", "dyn", "total", "fails", "dyn_vals")}))
        print(f"cap_tax={ct} prem={r}: SF {e['sf']}/10 anchors {e['anchors']}/5 dyn {e['dyn']}/5  fails={e['fails']}", flush=True)
    rows.sort(key=lambda x: -x["total"])
    best = {**BASE, **rows[0]["P"]}
    print("chosen", rows[0]["P"], rows[0]["total"], "/20")
    oos = []
    for seed in (101, 202, 303):
        e = evaluate(best, seed=seed, N_single=40000, N_dyn=40000)
        oos.append({k: e[k] for k in ("sf", "anchors", "dyn", "total", "fails", "dyn_vals", "mm", "gens")} | dict(seed=seed))
        print(f"OOS {seed}: SF {e['sf']}/10 anchors {e['anchors']}/5 dyn {e['dyn']}/5 fails={e['fails']}", flush=True)
        print("   gens gini:", [round(x["gini"], 3) for x in e["gens"]], "top1:", [round(x["top1_share"], 3) for x in e["gens"]],
              "mobility:", [round(x["mobility_Q1_to_Q5"], 3) for x in e["gens"]], flush=True)
    json.dump(dict(grid=rows, chosen=best, oos=oos), open("dynasty_calib_results.json", "w"), indent=1, default=float)
