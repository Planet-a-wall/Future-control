"""
DRAFT - written but never executed (the run was stopped by the user); kept for a later real-data study.

U2 (ignition / three-polarizer law) - design analysis before collecting real data.
Experiment: participants with one personal goal are randomised to 0, 1, 3 or 8 planning steps
(n = steps + 1 measurements). Outcome: acted on the goal within 14 days (yes/no).
QIT:  p(n) = eps + (1-eps) * E_theta[cos^{2n}(theta/n)],  theta ~ N(theta0, sd) per person
Rivals:
  R1 saturating exponential  p(n) = eps + (pmax-eps)(1 - exp(-lam (n-1)))   (classical 'more planning helps')
  R2 logistic in n           p(n) = 1/(1+exp(-(a + b n)))
  R0 no effect               p(n) = const
Question: with N per arm, how often does the analysis pick the true model (Delta AIC > 4)?
"""
import json
import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(7)
STEPS = np.array([0, 1, 3, 8]); NN = STEPS + 1

def p_qit(n, th0, eps, sd=0.25, m=400):
    th = np.clip(th0 + sd * np.random.default_rng(0).standard_normal(m), 0, np.pi / 2)
    return eps + (1 - eps) * np.mean(np.cos(th[None, :] / n[:, None]) ** (2 * n[:, None]), axis=1)
def p_r1(n, eps, pmax, lam): return eps + (pmax - eps) * (1 - np.exp(-lam * (n - 1)))
def p_r2(n, a, b): return 1 / (1 + np.exp(-(a + b * n)))

def nll(p, y, N):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.sum(y * np.log(p) + (N - y) * np.log(1 - p))
def fit_all(y, N):
    out = {}
    best = min((minimize(lambda x: nll(p_qit(NN, x[0], 1 / (1 + np.exp(-x[1]))), y, N), [t, -2], method="Nelder-Mead")
                for t in (0.8, 1.2, 1.5)), key=lambda r: r.fun)
    out["QIT"] = 2 * best.fun + 2 * 2
    best = min((minimize(lambda x: nll(p_r1(NN, 1 / (1 + np.exp(-x[0])), 1 / (1 + np.exp(-x[1])), np.exp(x[2])), y, N),
                         [-2, 1, l], method="Nelder-Mead") for l in (-2, -1, 0)), key=lambda r: r.fun)
    out["R1"] = 2 * best.fun + 2 * 3
    r = minimize(lambda x: nll(p_r2(NN, x[0], x[1]), y, N), [-1, 0.3], method="Nelder-Mead"); out["R2"] = 2 * r.fun + 2 * 2
    pc = y.sum() / N.sum(); out["R0"] = 2 * nll(np.full(4, pc), y, N) + 2
    return out

def power(truth, Narm, reps=300):
    win = 0; wrong = 0
    for _ in range(reps):
        if truth == "QIT": p = p_qit(NN, 1.35, 0.05)        # dream ~77 deg from action, 5% act anyway
        else: p = p_r1(NN, 0.10, 0.75, 0.35)                 # classical: smooth saturating benefit
        N = np.full(4, Narm); y = rng.binomial(N, p)
        a = fit_all(y, N); s = sorted(a, key=a.get)
        if s[0] == truth and a[s[1]] - a[s[0]] > 4: win += 1
        if s[0] != truth and a[s[1]] - a[s[0]] > 4 and s[0] in ("QIT", "R1"): wrong += 1
    return win / reps, wrong / reps

res = dict(true_curves=dict(QIT=p_qit(NN, 1.35, 0.05).tolist(), R1=p_r1(NN, 0.10, 0.75, 0.35).tolist()), rows=[])
print("true p(n) for steps 0,1,3,8  QIT:", np.round(res["true_curves"]["QIT"], 3), " R1:", np.round(res["true_curves"]["R1"], 3))
for Narm in (50, 100, 200, 400):
    pq, fq = power("QIT", Narm); pr, fr = power("R1", Narm)
    res["rows"].append(dict(N_per_arm=Narm, power_QIT=pq, wrong_when_QIT=fq, power_R1=pr, wrong_when_R1=fr))
    print(f"N/arm={Narm}: QIT true -> correct {pq:.2f} (wrong {fq:.2f}) | R1 true -> correct {pr:.2f} (QIT falsely wins {fr:.2f})", flush=True)
json.dump(res, open("u2_design_results.json", "w"), indent=1)
