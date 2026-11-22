"""
CIOT research proposal - all numerical results and vector figures.

Run:  python3 analysis.py
Writes figures/*.svg and results.json (every number quoted in the chapters).
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
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(20261122)
R = {}

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = ("#2F6FB2", "#D9822B", "#3A9A5B",
                                         "#C23B3B", "#6B7280", "#7A4FB5")
plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.titlesize": 10.5,
    "legend.frameon": False, "svg.fonttype": "path", "figure.dpi": 150,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".svg"))
    plt.close(fig)

def H_gauss(v):
    return 0.5 * np.log(2 * np.pi * np.e * v)

def H2(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

# ========================================================== Chapter 3: OU (v1)
aL, sigma, g, x0 = 1 / 11, 1.0, 10.0, 0.0
dt, days, M = 0.05, 365, 6000
steps = int(days / dt); t = np.arange(1, steps + 1) * dt
Cs = [0.25, 0.5, 1.0]; cols = [ORANGE, BLUE, GREEN]
R["ch3"] = {"params": dict(aL=aL, sigma=sigma, g=g, x0=x0, dt=dt, days=days, paths=M),
            "rows": []}
fig1, ax1 = plt.subplots(figsize=(6.4, 3.4))
fig2, ax2 = plt.subplots(figsize=(6.4, 3.2))
finals = {}
for C, c in zip(Cs, cols):
    k = aL * C
    x = np.full(M, x0); mean = np.empty(steps); sd = np.empty(steps)
    for i in range(steps):
        x = x + k * (g - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(M)
        mean[i] = x.mean(); sd[i] = x.std()
    finals[C] = x.copy()
    var_th = sigma**2 / (2 * k) * (1 - np.exp(-2 * k * days))
    tau_sim = t[np.argmax(mean >= g * (1 - np.exp(-1)))]
    t95 = t[np.argmax(mean >= 0.95 * g)]
    R["ch3"]["rows"].append(dict(C=C, k=k, var_th=var_th, var_sim=float(x.var()),
                                 H_th=H_gauss(var_th), H_sim=H_gauss(x.var()),
                                 tau_th=1 / k, tau_sim=float(tau_sim),
                                 t95_th=np.log(20) / k, t95_sim=float(t95),
                                 mean_sim=float(x.mean())))
    ax1.plot(t, mean, c=c, lw=1.6, label=f"$C={C}$")
    ax1.fill_between(t, mean - sd, mean + sd, color=c, alpha=0.12, lw=0)
    ax1.plot(t, g + (x0 - g) * np.exp(-k * t), c="k", lw=0.7, ls=":")
    ax2.hist(x, bins=70, density=True, color=c, alpha=0.35)
    xs = np.linspace(g - 16, g + 16, 400)
    ax2.plot(xs, norm.pdf(xs, g, np.sqrt(var_th)), c=c, lw=1.4, label=f"$C={C}$ (theory)")
ax1.axhline(g, c=GRAY, lw=0.8, ls="--")
ax1.set(xlabel="time $t$ (days)", ylabel=r"$x(t)$",
        title=r"Mean $\pm$ 1 s.d. of 6000 paths; dotted = Theorem 3.1")
ax1.legend(loc="lower right")
ax2.set(xlabel=r"$x(365)$", ylabel="density",
        title="Outcome distribution at day 365: histogram (simulation) vs Gaussian (theory)")
ax2.legend()
save(fig1, "f3_1_mean"); save(fig2, "f3_2_dist")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
cg = np.linspace(0.05, 1, 200)
ax.plot(cg, H_gauss(sigma**2 / (2 * aL * cg)), c=BLUE, lw=1.6, label="Theorem 3.3")
ax.scatter([r["C"] for r in R["ch3"]["rows"]], [r["H_sim"] for r in R["ch3"]["rows"]],
           c=RED, zorder=3, label="simulation")
ax.set(xlabel="continuity $C$", ylabel=r"$H_f$ (nats)", title="Future entropy vs continuity")
ax.legend(); save(fig, "f3_3_entropy")
R["ch3"]["dH_double"] = -0.5 * np.log(2)

# v1 after stopping (no persistence)
Cgrid = np.linspace(0, 1, 41)
mu = 1 / 30
v1_after = []
for C in Cgrid:
    x = np.full(1500, x0)
    for i in range(int(120 / dt)):
        x = x + aL * C * (g - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(x.size)
    for i in range(int(120 / dt)):
        x = x - mu * x * dt + sigma * np.sqrt(dt) * rng.standard_normal(x.size)
    v1_after.append(float(x.mean() / g))
R["ch3"]["v1_after_stop"] = dict(mu=mu, C=Cgrid.tolist(), frac=v1_after,
                                 analytic_max=float(np.exp(-mu * 120)))

# ========================================================== Chapter 4: bistable (v2)
a_in, b, K, n, lam = 0.09, 0.2, 1.0, 4, 0.1
f = lambda h: b * h**n / (K**n + h**n) - lam * h
hh = np.linspace(0, 3, 300001)
low = hh <= 1.0
i_max = np.argmax(-f(hh[low]))
C_star = float(-f(hh[low])[i_max] / a_in); h_fold = float(hh[low][i_max])
# fixed points at C=0
sgn = np.sign(f(hh[1:])); roots = hh[1:][np.where(np.diff(sgn) != 0)[0]]
R["ch4"] = dict(params=dict(a=a_in, b=b, K=K, n=n, lam=lam), C_star=C_star,
                h_fold=h_fold, roots_C0=[0.0] + roots.tolist())

# bifurcation diagram
fig, ax = plt.subplots(figsize=(6.4, 3.4))
cb = np.linspace(0, 1, 401)
st_x, st_y, un_x, un_y = [], [], [], []
for C in cb:
    F = a_in * C + f(hh)
    idx = np.where(np.diff(np.sign(F)) != 0)[0]
    for j in idx:
        h0 = hh[j]
        dF = (F[j + 1] - F[j]) / (hh[j + 1] - hh[j])
        (st_x if dF < 0 else un_x).append(C); (st_y if dF < 0 else un_y).append(h0)
    if C == 0:
        st_x.append(0); st_y.append(0)
ax.plot(st_x, st_y, ".", c=BLUE, ms=2.2, label="stable equilibrium")
ax.plot(un_x, un_y, ".", c=RED, ms=1.6, label="unstable equilibrium")
ax.axvline(C_star, c=GRAY, ls="--", lw=0.9)
ax.annotate("", xy=(C_star, 1.62), xytext=(C_star, 0.55),
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
ax.annotate("", xy=(0.03, 2.25), xytext=(0.33, 2.55),
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
ax.text(C_star + 0.02, 1.0, rf"$C^*={C_star:.3f}$", color=PURPLE)
ax.text(0.03, 2.62, "remove input: stays on upper branch (hysteresis)", color=PURPLE, fontsize=8.5)
ax.set_ylim(-0.1, 3.0)
ax.set(xlabel="continuity $C$", ylabel=r"equilibrium $h^\ast$",
       title="Saddle-node bifurcation of the structure equation (4.1)")
ax.legend(loc="lower right"); save(fig, "f4_1_bifurcation")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
hp = np.linspace(0, 2.6, 600)
for C, c in [(0, GRAY), (0.3, ORANGE), (C_star, PURPLE), (0.6, GREEN)]:
    ax.plot(hp, a_in * C + f(hp), c=c, lw=1.5, label=rf"$C={C:.3f}$" if C == C_star else f"$C={C}$")
ax.axhline(0, c="k", lw=0.7)
ax.set(xlabel="$h$", ylabel=r"$\dot h$", title=r"Phase lines $\dot h = aC + F(h)$")
ax.legend(ncol=2); save(fig, "f4_2_phase")

persist, Cg2 = [], np.linspace(0, 1, 41)
for C in Cg2:
    h = np.zeros(2000)
    for i in range(int(120 / dt)):
        h = np.maximum(h + (a_in * C + f(h)) * dt + 0.03 * np.sqrt(dt) * rng.standard_normal(h.size), 0)
    for i in range(int(120 / dt)):
        h = np.maximum(h + f(h) * dt + 0.03 * np.sqrt(dt) * rng.standard_normal(h.size), 0)
    persist.append(float((h > 1).mean()))
R["ch4"]["persist"] = dict(C=Cg2.tolist(), p=persist, noise=0.03)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(Cg2, persist, "s-", c=ORANGE, ms=3.5, label="v2: P(structure persists 120 d after stop)")
ax.plot(Cgrid, v1_after, "o-", c=BLUE, ms=3, label=r"v1: residual $E[x]/g$ 120 d after stop")
ax.axvline(C_star, c=PURPLE, ls="--", lw=1, label=rf"$C^*={C_star:.3f}$ (Theorem 4.2)")
ax.set(xlabel="continuity $C$ during 120-day training", ylabel="value",
       title="Threshold test: linear model v1 vs bistable model v2")
ax.legend(fontsize=8.5); save(fig, "f4_3_threshold")

# C* sensitivity heatmap
lams = np.linspace(0.06, 0.14, 81); bs = np.linspace(0.12, 0.30, 91)
hl = np.linspace(0, 1, 4001)
Cmap = np.array([[np.max(lm * hl - bb * hl**n / (K**n + hl**n)) / a_in for lm in lams] for bb in bs])
# hysteresis existence at C=0: need max_h F(h)/h > 0 for h>0  <=>  max_h b h^{n-1}/(K^n+h^n) > lam
hm = np.linspace(1e-4, 5, 20001)
hyst = np.array([[np.max(bb * hm**(n - 1) / (K**n + hm**n)) > lm for lm in lams] for bb in bs])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
im = ax.pcolormesh(lams, bs, np.clip(Cmap, 0, 1.5), cmap="viridis", shading="auto")
ax.contour(lams, bs, hyst.astype(float), levels=[0.5], colors="w", linewidths=1.2)
ax.plot([lam], [b], "*", c=RED, ms=11)
cb_ = fig.colorbar(im, ax=ax); cb_.set_label(r"$C^*$ (clipped at 1.5)")
ax.set(xlabel=r"decay rate $\lambda$", ylabel=r"self-reinforcement $b$",
       title=r"Sensitivity of $C^*$; white curve = hysteresis boundary; star = baseline")
ax.grid(False); save(fig, "f4_4_sensitivity")
R["ch4"]["sens"] = dict(Cstar_range=[float(Cmap.min()), float(Cmap.max())])

# ========================================================== Chapter 5: info-thermo
N = 16
def hN(r):
    p0 = r + (1 - r) / N; pr = (1 - r) / N
    return float(-(p0 * np.log2(p0)) - ((N - 1) * pr * np.log2(pr) if pr > 0 else 0))
phi = lambda r: np.log2(N) - hN(r)
rows = []
for p in [0.2, 0.5, 0.8, 1.0]:
    PyGx = np.full((N, N), (1 - p) / N) + np.eye(N) * p
    Pxy = PyGx / N
    I = H2(Pxy.sum(0)) - sum(H2(PyGx[x]) / N for x in range(N))
    for q in [0.25, 0.5, 0.75, 1.0]:
        Pa = np.zeros(N)
        for x in range(N):
            for y in range(N):
                Pa[(x - y) % N] += Pxy[x, y] * q; Pa[x] += Pxy[x, y] * (1 - q)
        dH = np.log2(N) - H2(Pa)
        rows.append(dict(p=p, q=q, I=I, dH=dH, eta_exact=dH / I, eta_formula=phi(q * p) / phi(p)))
R["ch5"] = dict(N=N, rows=rows)
fig, ax = plt.subplots(figsize=(6.4, 3.3))
qg = np.linspace(0, 1, 200)
for p, c in zip([0.2, 0.5, 0.8, 1.0], [ORANGE, BLUE, GREEN, PURPLE]):
    ax.plot(qg, [phi(q * p) / phi(p) for q in qg], c=c, lw=1.5, label=f"$p={p}$")
    ax.scatter([r["q"] for r in rows if r["p"] == p], [r["eta_exact"] for r in rows if r["p"] == p],
               c=c, s=16, zorder=3)
ax.plot([0, 1], [0, 1], c=GRAY, ls="--", lw=1, label=r"$\eta=q$ (linear)")
ax.set(xlabel="continuity $q$", ylabel=r"efficiency $\eta=\Delta H/I$",
       title=r"Theorem 5.3: $\eta(q,p)=\varphi(qp)/\varphi(p)\leq q$ (lines) vs exact (dots)")
ax.legend(ncol=2, fontsize=8.5); save(fig, "f5_1_efficiency")
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.scatter([r["I"] for r in rows], [r["dH"] for r in rows], c=[r["q"] for r in rows],
           cmap="viridis", s=26, zorder=3)
mx = max(r["I"] for r in rows) * 1.05
ax.plot([0, mx], [0, mx], c=RED, lw=1.2, label=r"bound $\Delta H = I$ (Touchette-Lloyd)")
ax.fill_between([0, mx], [0, mx], [mx, mx], color=RED, alpha=0.06, label="forbidden region")
ax.set(xlabel="mutual information $I$ (bits)", ylabel=r"entropy reduction $\Delta H$ (bits)",
       title="All 16 controller configurations lie on or below the bound")
ax.legend(fontsize=8.5); save(fig, "f5_2_bound")
kB, Tb = 1.380649e-23, 310.0
E_bit = kB * Tb * np.log(2)
R["ch5"]["landauer"] = dict(E_bit=E_bit, E_day=20 * 86400, bits_day=20 * 86400 / E_bit,
                            goal_bits=np.log2(1e9), goal_cost=np.log2(1e9) * E_bit,
                            ratio=20 * 86400 / (np.log2(1e9) * E_bit))

# ========================================================== Chapter 6: Zeno & refresh
Om = 1.0; T = np.pi / (2 * Om); ns = np.arange(1, 61)
Pz = np.cos(Om * T / ns) ** (2 * ns); Pz_app = np.exp(-(Om * T) ** 2 / ns)
lam_f, theta, Hz = 1 / 7, 0.3, 180
Trs = np.arange(1, 22)
R["ch6"] = dict(zeno=[dict(n=int(k), P=float(Pz[k - 1]), approx=float(Pz_app[k - 1]))
                      for k in (1, 2, 5, 10, 20, 50)],
                lam=lam_f, theta=theta, horizon=Hz, T_det=float(np.log(1 / theta) / lam_f),
                refresh={})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(ns, Pz, "o", c=BLUE, ms=3, label=r"exact $\cos^{2n}(\Omega T/n)$")
ax[0].plot(ns, Pz_app, c=ORANGE, lw=1.3, label=r"$\exp(-\Omega^2T^2/n)$")
ax[0].set(xlabel="number of measurements $n$", ylabel="survival probability",
          title="Quantum Zeno (Prop. 6.2)"); ax[0].legend(fontsize=8)
for s, c in [(0.25, GREEN), (0.5, PURPLE)]:
    sim, th = [], []
    for Tr in Trs:
        k = int(Hz // Tr)
        L = lam_f * rng.lognormal(0, s, size=(20000, k))
        sim.append(float(np.all(np.exp(-L * Tr) > theta, axis=1).mean()))
        lstar = np.log(1 / theta) / Tr
        th.append(float(norm.cdf(np.log(lstar / lam_f) / s) ** k))
    R["ch6"]["refresh"][str(s)] = dict(Tr=Trs.tolist(), sim=sim, theory=th)
    ax[1].plot(Trs, th, c=c, lw=1.4, label=rf"theory $s={s}$")
    ax[1].plot(Trs, sim, "o", c=c, ms=3)
ax[1].axvline(np.log(1 / theta) / lam_f, c=RED, ls="--", lw=1, label=r"$T^*_{det}$")
ax[1].set(xlabel=r"refresh interval $T_r$ (days)", ylabel="P(survive 180 d)",
          title="Classical refresh (Theorem 6.3)"); ax[1].legend(fontsize=7.5)
save(fig, "f6_1_zeno_refresh")
# 50% survival interval
for s in ("0.25", "0.5"):
    d = R["ch6"]["refresh"][s]
    Tg = np.linspace(0.5, 21, 4000)
    k = np.floor(Hz / Tg)
    P = norm.cdf(np.log(np.log(1 / theta) / Tg / lam_f) / float(s)) ** k
    R["ch6"]["refresh"][s]["T50"] = float(Tg[np.argmax(P < 0.5)])
# psi power
d_eff = [1e-3, 1e-4, 1e-5]
z = norm.ppf(0.975) + norm.ppf(0.8)
R["ch6"]["psi_N"] = {str(d): float(z**2 / (4 * d**2)) for d in d_eff}
R["ch6"]["decoherence"] = dict(neuron_s=1e-3, tegmark_low=1e-20, tegmark_high=1e-13)

# ========================================================== Chapter 7: combined model
kap = aL
def combined(C, train=120, total=300, Mp=1500, noise_h=0.03, rec=False):
    x = np.full(Mp, x0); h = np.zeros(Mp)
    nst = int(total / dt); xs = np.empty(nst); hs = np.empty(nst)
    for i in range(nst):
        Ct = C if i * dt < train else 0.0
        h = np.maximum(h + (a_in * Ct + f(h)) * dt + noise_h * np.sqrt(dt) * rng.standard_normal(Mp), 0)
        x = x + (kap * (Ct + h) * (g - x) - mu * (x - x0)) * dt + sigma * np.sqrt(dt) * rng.standard_normal(Mp)
        xs[i] = x.mean(); hs[i] = h.mean()
    return xs, hs, x
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
tt = np.arange(1, int(300 / dt) + 1) * dt
R["ch7"] = dict(kappa=kap, mu=mu, traj={})
for C, c in [(0.3, ORANGE), (0.4, GRAY), (0.5, BLUE), (0.7, GREEN)]:
    xs, hs, xf = combined(C)
    ax[0].plot(tt, xs, c=c, lw=1.4, label=f"$C={C}$"); ax[1].plot(tt, hs, c=c, lw=1.4)
    R["ch7"]["traj"][str(C)] = dict(x_120=float(xs[int(120 / dt) - 1]), x_300=float(xs[-1]),
                                    h_300=float(hs[-1]))
for a_ in ax:
    a_.axvspan(0, 120, color="#DBEAFE", alpha=0.5, lw=0)
ax[0].set(xlabel="day", ylabel=r"$E[x(t)]$", title="Outcome (training = shaded)")
ax[1].set(xlabel="day", ylabel=r"$E[h(t)]$", title="Accumulated structure")
ax[0].legend(fontsize=8); save(fig, "f7_1_combined")
Cg3 = np.linspace(0, 1, 26); xend = []
for C in Cg3:
    xs, hs, xf = combined(C, Mp=800)
    xend.append(float(xs[-1] / g))
R["ch7"]["x300"] = dict(C=Cg3.tolist(), frac=xend)
hH = R["ch4"]["roots_C0"][-1]
xstar_hi = (kap * hH * g + mu * x0) / (kap * hH + mu)
R["ch7"]["x_star_high"] = float(xstar_hi)
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.plot(Cg3, xend, "o-", c=BLUE, ms=3.5, label="v2 combined: $E[x(300)]/g$")
ax.plot(Cgrid, v1_after, c=GRAY, lw=1, label="v1 (for comparison)")
ax.axvline(C_star, c=PURPLE, ls="--", lw=1)
ax.axhline(xstar_hi / g, c=GREEN, ls=":", lw=1, label=r"predicted plateau $x^\ast_{high}/g$")
ax.set(xlabel="continuity $C$ during training (days 0-120)", ylabel="fraction of goal",
       title="Outcome 180 days after training stops")
ax.legend(fontsize=8.5); save(fig, "f7_2_outcome")

# Study B power (two proportions)
def n_two_prop(p1, p2, a=0.05, pw=0.8):
    pb = (p1 + p2) / 2
    za, zb = norm.ppf(1 - a / 2), norm.ppf(pw)
    return float(((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2)
R["ch7"]["power"] = {f"{p1}-{p2}": n_two_prop(p1, p2) for p1, p2 in [(0.6, 0.35), (0.6, 0.45), (0.7, 0.4)]}

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(clean(R), fh, indent=1)
print(json.dumps(clean({k: v for k, v in R.items()}), indent=1)[:200])
print("done")
