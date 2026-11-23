"""
Reactor of Fortune: chain reactions (branching processes) as the shared mathematics of
nuclear fission, neuronal avalanches and life's action-opportunity cascades.
Superego = control rods; deliberation delay = delayed-neutron fraction.

A  Galton-Watson: survival, growth and avalanche-size entropy vs multiplication factor k
B  point kinetics with delayed neutrons: period vs reactivity; prompt-critical boundary rho = beta
C  life reactor: hubris inserts reactivity after success; compare superego/ego control policies
Run: python3 reactor.py -> reactor_results.json, figs/*.png
"""
import json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
OUT = {}

# ---------------- A. branching process
A = []
for k in (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5):
    ext = 1.0 if k <= 1 else brentq(lambda q: q - np.exp(k * (q - 1)), 0, 1 - 1e-12)
    sizes = []
    for _ in range(20000):
        n, s = 1, 1
        while n and s < 10000:
            n = rng.poisson(k * n); s += n
        sizes.append(s)
    sizes = np.array(sizes)
    finite = sizes[sizes < 10000]
    h, _ = np.histogram(np.log10(finite), bins=40, range=(0, 4))
    p = h[h > 0] / h.sum()
    A.append(dict(k=k, p_survive=1 - ext, mean_finite_size=float(finite.mean()),
                  p_explode=float(np.mean(sizes >= 10000)), entropy_bits=float(-(p * np.log2(p)).sum())))
    print("A", {x: round(y, 3) for x, y in A[-1].items()})
OUT["A"] = A

# ---------------- B. point kinetics, one delayed group (U-235-like constants)
beta, lam, Lam = 0.0065, 0.08, 1e-4           # delayed fraction, decay const (1/s), generation time (s)
def period(rho):
    # inhour (one group): rho = w*Lam + beta*w/(w+lam); largest root w -> period 1/w
    w = brentq(lambda w: w * Lam + beta * w / (w + lam) - rho, 1e-9, 1e5)
    return 1 / w
B = [dict(rho=r, rho_over_beta=r / beta, period_s=period(r)) for r in (0.0005, 0.001, 0.003, 0.005, 0.0065, 0.008, 0.01)]
for b in B:
    print(f"B rho/beta={b['rho_over_beta']:.2f}: period = {b['period_s']:.4g} s")
OUT["B"] = dict(beta=beta, rows=B)

# ---------------- C. life reactor with hubris and control policies
def life(policy, n=4000, T=400, gain=0.4, delay=1, p_imp=None, seed=0):
    """Life reactor (faithful to reactor control).
    k_total = k0*(1+h) + impulse - u.
      h      hubris, inflated by success (self-attribution); a strict superego damps it (guilt)
      impulse  random reactivity spikes from impulsive decisions (prob p_imp per step, +0.15),
               unseen by the controller in advance  (the 'prompt' part)
      u      control rods. id: none; strict superego: fixed rods; ego: integral control on the total
             growth observed 'delay' steps ago, aiming at growth to a safe level then holding (k=1).
    Meltdown A > C; fizzle A = 0."""
    r = np.random.default_rng(seed)
    C, A_safe, k0 = 2000.0, 800.0, 1.04
    if p_imp is None:
        p_imp = 0.3 if "impulsive" in policy or policy.startswith("id") else 0.05
    A = np.full(n, 100.0); h = np.zeros(n); out = np.zeros(n); u = np.zeros(n)
    melt = np.zeros(n, bool); dead = np.zeros(n, bool)
    histg = [np.ones(n)] * (delay + 1); histA = [A.copy()] * (delay + 1)
    hub_rate = 0.003 if policy.startswith("strict") else 0.01
    for t in range(T):
        impulse = 0.15 * (r.random(n) < p_imp)
        if policy.startswith("id"):
            u = np.zeros(n)
        elif policy.startswith("strict"):
            u = np.full(n, 0.07)
        else:
            g_seen, A_seen = histg[-1 - delay], histA[-1 - delay]
            g_target = 1 + 0.03 * np.clip(1 - A_seen / A_safe, -1, 1)
            u = np.maximum(0, u + gain * (g_seen - g_target))
        k = np.clip(k0 * (1 + h) + impulse - u, 0, 3)
        A_new = r.poisson(k * A).astype(float)
        A_new[dead | melt] = 0
        g = np.where(A > 0, A_new / np.maximum(A, 1), 1)
        h = np.clip(h + hub_rate * (g > 1) - 0.005 * h, 0, 1)
        new_melt = (A_new > C) & ~melt
        melt |= new_melt; A_new[melt] = 0
        dead |= (A_new == 0) & ~melt
        out += A_new; A = A_new
        histg.append(g); histA.append(A.copy())
    return dict(output=float(np.median(out)), p_meltdown=float(melt.mean()), p_fizzle=float(dead.mean()),
                p_sustained=float(np.mean(~melt & ~dead)))
C_rows = {}
for pol in ("id (no rods)", "strict superego (rods always in)", "ego: adaptive rods, impulsive", "ego: adaptive rods, deliberate"):
    C_rows[pol] = life(pol)
    print(f"C {pol:34s}", {x: round(y, 3) for x, y in C_rows[pol].items()})
sweep = []
for p_imp in (0.05, 0.15, 0.3):
    for d in (1, 4, 10):
        r_ = life("ego: adaptive rods, deliberate", p_imp=p_imp, delay=d)
        sweep.append(dict(p_imp=p_imp, delay=d, **r_))
        print(f"C sweep impulsive-share={p_imp} feedback-delay={d}:", {x: round(y, 3) for x, y in r_.items()})
OUT["C"] = dict(policies=C_rows, sweep=sweep)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ks = [a["k"] for a in A]
ax[0].plot(ks, [a["entropy_bits"] for a in A], "o-"); ax[0].axvline(1, ls="--", c="r")
ax[0].set(xlabel="multiplication factor k", ylabel="entropy of cascade sizes (bits)", title="A: richest repertoire at criticality")
ax[1].semilogy([b["rho_over_beta"] for b in B], [b["period_s"] for b in B], "o-"); ax[1].axvline(1, ls="--", c="r")
ax[1].set(xlabel="reactivity / delayed fraction", ylabel="period (s)", title="B: prompt-critical cliff at rho = beta")
pols = list(C_rows); x = np.arange(len(pols))
ax[2].bar(x - 0.2, [C_rows[p]["p_sustained"] for p in pols], 0.4, label="sustained")
ax[2].bar(x + 0.2, [C_rows[p]["p_meltdown"] for p in pols], 0.4, label="meltdown")
ax[2].set_xticks(x); ax[2].set_xticklabels(["id", "strict\nsuperego", "ego\nimpulsive", "ego\ndeliberate"])
ax[2].set(title="C: control policies"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/reactor.png", dpi=130)
json.dump(OUT, open("reactor_results.json", "w"), indent=1, default=float)
