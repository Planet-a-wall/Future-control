"""
Studies E0-E6 on the calibrated model (answers the research questions).

E0  Kelly hubris threshold: growth of log-capital vs confidence c (fixed c)
E1  hubris equilibrium formula -> predicts who falls after success
E2  birth endowment (king / heir / middle / poor), identical trait distributions
E3  low birth + strong mind: which mental profile rises and stays up
E4  why the successful fall: mechanism decomposition and timing
E5  merit (giving, integrity) and enjoyment (intrinsic share) effects
Run: python3 predictions.py   -> figs/*.png, predictions_results.json
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import simulate, traits, DEFAULT
from validate import CAL, FULL

os.makedirs("figs", exist_ok=True)
P = {**DEFAULT, **CAL}
OUT = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- E0
# fixed confidence, single venture, no consumption: time-average log growth
kappa, mu, sig = P["kelly_frac"], 0.012, P["sigma"]
cs = np.linspace(0.5, 7, 27)
rng = np.random.default_rng(0)
g_sim, g_th = [], []
for c in cs:
    f = min(kappa * c * mu / sig**2, 50)
    X = P["r"] + mu + sig * rng.standard_normal(2_000_000)
    lr = np.log(np.maximum(1 + P["r"] + f * (X - P["r"]), 1e-300))
    g_sim.append(lr.mean()); g_th.append(P["r"] + f * mu - f**2 * sig**2 / 2)
c_star = 2 / kappa
OUT["E0"] = dict(c_star=c_star, cs=cs.tolist(), g_sim=g_sim, g_th=g_th,
                 zero_cross_sim=float(np.interp(0, -np.array(g_sim) + P["r"] * 0, cs)) if False else None)
# where excess growth (g - r) crosses zero
ex = np.array(g_sim) - P["r"]
i = np.where(np.diff(np.sign(ex)) < 0)[0][0]
OUT["E0"]["excess_zero_sim"] = float(cs[i] + (cs[i + 1] - cs[i]) * ex[i] / (ex[i] - ex[i + 1]))
print("E0 Kelly: theory c*=%.2f, simulated excess-growth zero at c=%.2f" % (c_star, OUT["E0"]["excess_zero_sim"]))

# ---------------------------------------------------------------- base population
N = 80000
rng = np.random.default_rng(42)
tr = traits(N, rng)
K0 = 2.0 * np.exp(rng.normal(0, 1.5, N))
o = simulate(K0, tr, FULL, seed=43, P=CAL)
great = o["ever_great"]
fell = great & ((o["t_first_bankrupt"] > o["t_great"]) | (o["jailed_t"] > o["t_great"]) | (o["K"] < 0.1 * o["peak"]))
jail_after = great & (o["jailed_t"] > o["t_great"])

# ---------------------------------------------------------------- E1 hubris law
def c_eq(e, iota, s):
    Ez = 1 / np.sqrt(2 * np.pi)
    b = P["beta"] * (1 + P["ego_gain"] * e) * (1 + P["psi"] * s)
    g = P["gamma"] * (0.5 + iota) / (1 + P["psi"] * s)
    return 1 + Ez * (b - P["beta_neg"]) / g

H_pre = kappa * c_eq(tr["ego"], tr["iota"], 0.0)       # before success (status ~ 0)
H_post = kappa * c_eq(tr["ego"], tr["iota"], 1.0)      # after success (status saturated)
bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 10]
E1 = []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = great & (H_post >= lo) & (H_post < hi)
    E1.append(dict(lo=lo, hi=hi, n=int(sel.sum()), p_fall=float(fell[sel].mean()) if sel.sum() else None,
                   p_jail=float(jail_after[sel].mean()) if sel.sum() else None))
OUT["E1"] = dict(bins=E1, share_pre_over2=float(np.mean(H_pre > 2)), share_post_over2=float(np.mean(H_post > 2)))
# critical ego as function of integrity (s=1): kappa*c_eq = 2
Ez = 1 / np.sqrt(2 * np.pi)
iot = np.linspace(0, 1, 11)
need = (2 / kappa - 1) * P["gamma"] * (0.5 + iot) / (1 + P["psi"]) / Ez   # = b - beta_neg
e_crit = ((need + P["beta_neg"]) / (P["beta"] * (1 + P["psi"])) - 1) / P["ego_gain"]
OUT["E1"]["e_crit"] = dict(iota=iot.tolist(), e_crit=e_crit.tolist())
print("E1 fall rate among the successful by predicted post-success hubris index H=kappa*c_eq:")
for b in E1:
    print(f"   H in [{b['lo']},{b['hi']}): n={b['n']:6d}  P(fall)={b['p_fall']}  P(jail)={b['p_jail']}")
print(f"   share with H>2 before success {OUT['E1']['share_pre_over2']:.3f}, after success {OUT['E1']['share_post_over2']:.3f}")

# ---------------------------------------------------------------- E2 birth endowment
E2 = []
for label, k0, r0 in [("poor (0.1)", 0.1, 0.0), ("middle (2)", 2.0, 0.0), ("rich (50)", 50.0, 5.0),
                      ("billionaire heir / royal (5000)", 5000.0, 20.0)]:
    n = 20000
    t2 = traits(n, np.random.default_rng(7))       # identical trait draws for every cohort
    oc = simulate(np.full(n, k0), t2, FULL, seed=8, P=CAL, R0=r0)
    g2 = oc["ever_great"]
    f2 = g2 & ((oc["t_first_bankrupt"] > oc["t_great"]) | (oc["jailed_t"] > oc["t_great"]) | (oc["K"] < 0.1 * oc["peak"]))
    E2.append(dict(cohort=label, K0=k0, p_great=float(g2.mean()), p_fall_given_great=float(f2.sum() / max(g2.sum(), 1)),
                   median_final=float(np.median(oc["K"])), p_final_ge_100=float(np.mean(oc["K"] >= 100)),
                   p_bankrupt=float(np.mean(oc["n_bankrupt"] > 0)), p_jail=float(np.mean(oc["jailed_t"] >= 0))))
    print("E2", E2[-1])
OUT["E2"] = E2

# ---------------------------------------------------------------- E3 low birth, strong mind
E3 = []
n = 20000
profiles = {
    "average mind": dict(),
    "high vision + drive": dict(v=0.85, q_intr=0.9),
    "high vision + drive + high ego": dict(v=0.85, q_intr=0.9, ego=0.7),
    "high vision + drive + high ego + low integrity": dict(v=0.85, q_intr=0.9, ego=0.7, iota=0.2),
    "high vision + drive + enjoys work (phi=0.85)": dict(v=0.85, q_intr=0.9, phi=0.85),
    "high vision + drive + enjoys + low ego + integrity": dict(v=0.85, q_intr=0.9, phi=0.85, ego=0.1, iota=0.9),
    "... + gives (omega=0.15)": dict(v=0.85, q_intr=0.9, phi=0.85, ego=0.1, iota=0.9, omega=0.15),
}
for name, over in profiles.items():
    t3 = traits(n, np.random.default_rng(9), **over)
    oc = simulate(np.full(n, 0.1), t3, FULL, seed=10, P=CAL)
    g3 = oc["ever_great"]
    f3 = g3 & ((oc["t_first_bankrupt"] > oc["t_great"]) | (oc["jailed_t"] > oc["t_great"]) | (oc["K"] < 0.1 * oc["peak"]))
    tg = oc["t_great"][g3] / 12
    E3.append(dict(profile=name, p_great=float(g3.mean()), p_fall_given_great=float(f3.sum() / max(g3.sum(), 1)),
                   p_great_and_stay=float((g3 & ~f3).mean()), median_years_to_great=float(np.median(tg)) if g3.sum() else None,
                   p_jail=float(np.mean(oc["jailed_t"] >= 0)), median_final=float(np.median(oc["K"]))))
    print("E3", E3[-1])
OUT["E3"] = E3

# ---------------------------------------------------------------- E4 why the successful fall
self_made = great & (K0 < 20)
f_sm = fell & self_made
mech = dict(
    jailed=float(np.mean(o["jailed_t"][f_sm] >= 0)),
    bankrupt=float(np.mean((o["t_first_bankrupt"][f_sm] > o["t_great"][f_sm]) & (o["jailed_t"][f_sm] < 0))),
    eroded_90pct=float(np.mean((o["t_first_bankrupt"][f_sm] < 0) & (o["jailed_t"][f_sm] < 0))),
)
fall_t = np.where(o["jailed_t"] >= 0, o["jailed_t"], o["t_first_bankrupt"])
dt_years = (fall_t[f_sm & (fall_t > 0)] - o["t_great"][f_sm & (fall_t > 0)]) / 12
OUT["E4"] = dict(n_self_made_great=int(self_made.sum()), p_fall=float(f_sm.sum() / max(self_made.sum(), 1)),
                 mechanisms=mech, years_great_to_fall_median=float(np.median(dt_years)),
                 years_great_to_fall_iqr=[float(np.percentile(dt_years, 25)), float(np.percentile(dt_years, 75))],
                 peak_c_fallers=float(np.median(o["peak_c"][f_sm])), peak_c_stayers=float(np.median(o["peak_c"][self_made & ~fell])),
                 max_lev_fallers=float(np.median(o["max_lev"][f_sm])), max_lev_stayers=float(np.median(o["max_lev"][self_made & ~fell])))
print("E4", OUT["E4"])

# ---------------------------------------------------------------- E5 merit and enjoyment (population)
def by(trait, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tr[trait] >= lo) & (tr[trait] < hi)
        g5 = great & sel
        rows.append(dict(lo=lo, hi=hi, n=int(sel.sum()), p_great=float(great[sel].mean()),
                         p_fall_given_great=float(fell[g5].sum() / max(g5.sum(), 1)),
                         p_jail=float(np.mean(o["jailed_t"][sel] >= 0)),
                         p_bankrupt=float(np.mean(o["n_bankrupt"][sel] > 0))))
    return rows
OUT["E5"] = dict(phi=by("phi", [0, 0.25, 0.5, 0.75, 1.0001]), iota=by("iota", [0, 0.4, 0.6, 0.8, 1.0001]),
                 omega=by("omega", [0, 0.03, 0.08, 0.15, 1.0001]), ego=by("ego", [0, 0.15, 0.3, 0.5, 1.0001]),
                 v=by("v", [0, 0.2, 0.4, 0.6, 1.0001]))
for k, rows in OUT["E5"].items():
    print("E5", k, [(r["lo"], round(r["p_great"], 3), round(r["p_fall_given_great"], 3), round(r["p_jail"], 4)) for r in rows])

# ---------------------------------------------------------------- figures (PNG previews)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(cs, np.array(g_th) * 1200, label="theory r + f mu - f^2 sigma^2/2")
ax[0].plot(cs, np.array(g_sim) * 1200, "o", ms=3, label="simulation")
ax[0].axhline(P["r"] * 1200, ls=":", c="gray"); ax[0].axvline(c_star, ls="--", c="r", label=f"c* = 2/kappa = {c_star:.0f}")
ax[0].set(xlabel="confidence c (perceived / true edge)", ylabel="log-growth %/yr", title="E0: hubris threshold (Kelly)",
          ylim=(-25, 15))
ax[0].legend(fontsize=7)
xs = [f"[{b['lo']},{b['hi']})" for b in E1]
ax[1].bar(xs, [b["p_fall"] or 0 for b in E1], color="#C23B3B")
ax[1].set(xlabel="predicted post-success hubris index H = kappa * c_eq", ylabel="P(fall | became great)",
          title="E1: formula predicts who falls")
fig.tight_layout(); fig.savefig("figs/e0_e1_hubris.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
lab = [r["cohort"].split(" (")[0] for r in E2]
ax[0].bar(lab, [r["p_final_ge_100"] for r in E2], color="#2F6FB2", label="final >= 100")
ax[0].bar(lab, [r["p_great"] * r["p_fall_given_great"] for r in E2], color="#C23B3B", alpha=0.7, label="became great then fell")
ax[0].set(title="E2: same minds, different births", ylabel="share of cohort"); ax[0].legend(fontsize=7)
ax[0].tick_params(axis="x", labelsize=7)
lab3 = [str(i) for i in range(len(E3))]
ax[1].bar(lab3, [r["p_great_and_stay"] for r in E3], color="#3A9A5B", label="great and stayed")
ax[1].bar(lab3, [r["p_great"] - r["p_great_and_stay"] for r in E3], bottom=[r["p_great_and_stay"] for r in E3],
          color="#C23B3B", label="great then fell")
ax[1].set(title="E3: born poor (K0=0.1), mental profiles 0-6", ylabel="share of cohort"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/e2_e3_birth_mind.png", dpi=130); plt.close(fig)

json.dump(OUT, open("predictions_results.json", "w"), indent=1, default=float)
