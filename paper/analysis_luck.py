"""
CIOT chapter 10: opportunity, timing and coincidence.

1. Opportunity funnel as a thinned Poisson process with transient windows
   (Theorem 10.1) - Monte Carlo check and profile comparison.
2. Optimal attention breadth (Proposition 10.3).
3. Coincidences expected by chance (law of truly large numbers).
4. Sample size for a behaviour-independent psi test of "luck".

Run: python3 analysis_luck.py -> figures/f10_*.svg, results_luck.json
Profile values are ILLUSTRATIVE (directions from Chapters 8-10 literature).
"""
import json
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
rng = np.random.default_rng(20261124)
plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.titlesize": 10.5,
    "legend.frameon": False, "svg.fonttype": "path",
})
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = ("#2F6FB2", "#D9822B", "#3A9A5B", "#C23B3B",
                                               "#6B7280", "#7A4FB5", "#1F9E9A")

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)

R = {}
# ------------------------------------------------------------ 1. funnel
L0, wbar, alpha, top = 1.0, 2.0, 1.5, 0.01      # 1 potential opportunity/day, window mean 2 d
v_top = top ** (-1 / alpha)                      # Pareto(x_m=1): P(V>v)=v^-alpha
PROF = [  # code, name, E, d, r, ell, k
    ("P0", "Typical", 1.0, 0.30, 0.40, 3.0, 0.30),
    ("P1", "Low distraction", 1.0, 0.20, 0.50, 2.0, 0.35),
    ("P2", "Focused", 0.8, 0.15, 0.60, 1.5, 0.45),
    ("P3", "Calm", 1.0, 0.45, 0.45, 2.5, 0.30),
    ("P4", "Sleep-restricted", 0.9, 0.20, 0.25, 5.0, 0.20),
    ("P5", "Busy", 1.6, 0.15, 0.30, 6.0, 0.25),
    ("P6", "Positive fantasy only", 1.0, 0.35, 0.30, 4.0, 0.15),
    ("P7", "Open + prepared", 1.4, 0.45, 0.65, 1.5, 0.40),
]
def rate(E, d, r, ell, k):
    return L0 * E * d * r * (wbar / (wbar + ell)) * k

def mc_rate(E, d, r, ell, k, T=3650.0, runs=200):
    n_tot = 0
    for _ in range(runs):
        n = rng.poisson(L0 * E * T)
        noticed = rng.random(n) < d
        recog = rng.random(n) < r
        lat = rng.exponential(ell, n); win = rng.exponential(wbar, n)
        conv = rng.random(n) < k
        n_tot += np.sum(noticed & recog & (lat < win) & conv)
    return n_tot / (T * runs)

R["funnel"] = dict(L0=L0, wbar=wbar, alpha=alpha, v_top=v_top, rows=[])
for code, name, E, d, r, ell, k in PROF:
    lam = rate(E, d, r, ell, k); lam_mc = mc_rate(E, d, r, ell, k)
    yr = 365 * lam
    p_big = 1 - np.exp(-yr * top)
    R["funnel"]["rows"].append(dict(code=code, name=name, E=E, d=d, r=r, ell=ell, k=k,
                                    timing=wbar / (wbar + ell), rate=lam, rate_mc=lam_mc,
                                    per_year=yr, p_big_year=p_big,
                                    p_big_5y=1 - np.exp(-5 * yr * top)))
    print(f"{code} rate={lam:.4f} mc={lam_mc:.4f} /yr={yr:.1f} P(big,1y)={p_big:.3f}")

rows = R["funnel"]["rows"]
fig, ax = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
y = np.arange(len(rows))[::-1]
factors = [("E", "exposure E", "#93C5FD"), ("d", "notice d", "#60A5FA"), ("r", "recognise r", "#3B82F6"),
           ("timing", r"in time $\bar w/(\bar w+\ell)$", "#1D4ED8"), ("k", "convert k", "#1E3A8A")]
base = rows[0]
for j, (key, lab, c) in enumerate(factors):
    vals = [np.log2(rr[key] / base[key]) for rr in rows]
    ax[0].barh(y + (2 - j) * 0.15, vals, height=0.15, color=c, label=lab)
ax[0].axvline(0, c="k", lw=0.7)
ax[0].set_yticks(y); ax[0].set_yticklabels([f"{rr['code']} {rr['name']}" for rr in rows], fontsize=8)
ax[0].set(xlabel=r"$\log_2$(factor / typical)", title="Funnel factors relative to typical")
ax[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.18))
ax[1].barh(y, [rr["per_year"] for rr in rows], color=[GRAY, TEAL, BLUE, GREEN, RED, ORANGE, PURPLE, "#0F766E"])
for yy, rr in zip(y, rows):
    ax[1].text(rr["per_year"] + 0.5, yy, f"{rr['per_year']:.1f}  ({100*rr['p_big_year']:.0f}%)",
               va="center", fontsize=7.5)
ax[1].set_yticks(y); ax[1].set_yticklabels([])
ax[1].set(xlabel="captured opportunities per year", title="Result (P of a top-1% break in 1 yr)",
          xlim=(0, max(rr["per_year"] for rr in rows) * 1.45))
save(fig, "f10_1_funnel")

# Monte Carlo validation scatter
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.scatter([rr["rate"] for rr in rows], [rr["rate_mc"] for rr in rows], c=BLUE, s=24, zorder=3)
m = max(rr["rate"] for rr in rows) * 1.1
ax.plot([0, m], [0, m], c=RED, lw=1, label="Theorem 10.1")
ax.set(xlabel=r"$\Lambda_{\rm eff}$ formula (per day)", ylabel="Monte Carlo (per day)",
       title="Thinning theorem: formula vs simulation")
ax.legend(fontsize=8); save(fig, "f10_2_mc")

# ------------------------------------------------------------ 2. optimal breadth
G, gam, c = 1.0, 0.5, 4.0
betas = np.linspace(0, 1, 400)
R["breadth"] = dict(G=G, gamma=gam, c=c, rows=[])
fig, ax = plt.subplots(figsize=(6.4, 3.1))
for V, col in [(0.1, GRAY), (0.2, ORANGE), (0.5, BLUE), (1.0, GREEN), (2.0, PURPLE)]:
    J = G * (1 - gam * betas) + V * (1 - np.exp(-c * betas))
    bstar = np.log(c * V / (gam * G)) / c if c * V > gam * G else 0.0
    bstar = min(max(bstar, 0.0), 1.0)
    R["breadth"]["rows"].append(dict(V=V, beta_star=float(bstar), beta_num=float(betas[np.argmax(J)])))
    ax.plot(betas, J, c=col, lw=1.5, label=f"$V={V}$")
    ax.scatter(bstar, G * (1 - gam * bstar) + V * (1 - np.exp(-c * bstar)), c=col, s=22, zorder=3)
ax.set(xlabel=r"attention breadth $\beta$ (0 = narrow focus, 1 = fully open)",
       ylabel=r"value rate $J(\beta)$", title=r"Proposition 10.3: optimal breadth $\beta^*$ grows with serendipity value $V$")
ax.title.set_fontsize(9.5); ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
save(fig, "f10_3_breadth")

# ------------------------------------------------------------ 3. coincidences
waking_min = 16 * 60
def coinc_per_year(n_think, p_call, window_min):
    return 365 * n_think * p_call * (window_min / waking_min)
# Monte Carlo for n_think=20, p_call=1/100, window=30 min
def mc_coinc(n_think=20, p_call=0.01, window=30, days=365, runs=20000):
    tot = 0
    for _ in range(runs):
        # each day, n_think thoughts of distinct acquaintances; each thought-of person calls w.p. p_call
        calls = rng.random((days, n_think)) < p_call
        t_th = rng.uniform(0, waking_min, (days, n_think))
        t_call = rng.uniform(0, waking_min, (days, n_think))
        hit = calls & (t_call > t_th) & (t_call - t_th <= window)
        tot += hit.sum()
    return tot / runs
an = coinc_per_year(20, 0.01, 30)
# exact expectation accounting for boundary effect (call must fall in [t, t+w] within the day)
w = 30 / waking_min
an_exact = 365 * 20 * 0.01 * (w - w**2 / 2)
mc = mc_coinc()
R["coinc"] = dict(example=dict(n_think=20, p_call=0.01, window_min=30, approx=an, exact=an_exact, mc=mc),
                  littlewood_days=1e6 / (8 * 3600),
                  miracles_per_year=8 * 3600 * 365 / 1e6,
                  birthday_23=float(1 - np.prod([(365 - i) / 365 for i in range(23)])))
print("coinc", an, an_exact, mc, R["coinc"]["littlewood_days"], R["coinc"]["birthday_23"])
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
nt = np.arange(1, 61)
for pc, col, lab in [(1 / 30, RED, "calls you monthly"), (1 / 100, BLUE, "every ~3 months"),
                     (1 / 365, GRAY, "yearly")]:
    ax[0].plot(nt, [365 * n_ * pc * (w - w**2 / 2) for n_ in nt], c=col, lw=1.5, label=lab)
ax[0].scatter([20], [mc], c="k", s=20, zorder=3, label="Monte Carlo")
ax[0].set(xlabel="acquaintances thought of per day", ylabel="'thought of them, then they called'\nper year",
          title="Expected coincidences by chance (30-min window)")
ax[0].title.set_fontsize(9); ax[0].legend(fontsize=7)
N_ev = np.logspace(3, 7, 200)
ax[1].semilogx(N_ev, 1 - np.exp(-N_ev / 1e6), c=PURPLE, lw=1.6)
ax[1].axvline(8 * 3600 * 35, c=GRAY, ls="--", lw=0.9)
ax[1].text(8 * 3600 * 35 * 1.15, 0.2, "35 days of\n1 event/s, 8 h/day", fontsize=7.5)
ax[1].set(xlabel="number of events experienced", ylabel="P(at least one 1-in-a-million)",
          title="Littlewood's law"); ax[1].title.set_fontsize(9)
save(fig, "f10_4_coincidence")

# ------------------------------------------------------------ 4. psi-luck test sample size
# Outcome: win/lose in a QRNG lottery with p0 = 0.2; psi predicts p0 + delta in intention condition.
p0 = 0.2
z = norm.ppf(0.975) + norm.ppf(0.9)
R["psi_luck"] = {}
for dlt in [0.02, 0.01, 0.005]:
    p1 = p0 + dlt; pb = (p0 + p1) / 2
    n = ((norm.ppf(0.975) * np.sqrt(2 * pb * (1 - pb)) + norm.ppf(0.9) * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / dlt) ** 2
    R["psi_luck"][str(dlt)] = float(n)
print(R["psi_luck"])

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
with open(os.path.join(HERE, "results_luck.json"), "w") as fh:
    json.dump(clean(R), fh, indent=1)
print(json.dumps(clean(R["breadth"]), indent=0))
