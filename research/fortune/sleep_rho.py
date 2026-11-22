"""When does an optimal work intensity appear? Sensitivity of S14 to fatigue strength rho_fat."""
import json
from sleep_studies import cohort, FLAGS
FINAL = json.load(open("sleep_calib_results.json"))["chosen"]
res = {}
for rho in (0.08, 0.3, 0.6, 1.0):
    P = {**FINAL, "rho_fat": rho}
    rows = []
    for qi in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r_ = cohort(P, FLAGS, n=12000, v=0.8, q_intr=qi, phi=0.85)
        rows.append(dict(q_intr=qi, stay=r_["p_stay"], med=r_["median_final"], fat=r_["fatigue"]))
    best = max(rows, key=lambda r: r["stay"])
    res[str(rho)] = dict(rows=rows, best_q=best["q_intr"])
    print(f"rho_fat={rho}: best q_intr={best['q_intr']}  stay by q: " + " ".join(f"{r['stay']:.3f}" for r in rows)
          + "  fatigue at q=1: %.2f" % rows[-1]["fat"], flush=True)
json.dump(res, open("sleep_rho_results.json", "w"), indent=1)
