"""
CIOT supplementary chapters 8-9: person profiles x routine.

Each profile is a parameter vector of the canonical model (S) plus the
refresh model (Theorem 6.3) and the controller model (Theorem 5.3).
Values are ILLUSTRATIVE: directions come from the literature reviewed in
Chapter 8, magnitudes are assumptions to be estimated in Study C.

Run:  python3 analysis_supp.py   -> figures/f8_*.svg, f9_*.svg, results_supp.json
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
rng = np.random.default_rng(20261123)
plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.titlesize": 10.5,
    "legend.frameon": False, "svg.fonttype": "path",
})
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, BROWN = (
    "#2F6FB2", "#D9822B", "#3A9A5B", "#C23B3B", "#6B7280", "#7A4FB5", "#1F9E9A", "#8C5A3C")

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name + ".svg")); plt.close(fig)

# ------------------------------------------------------------------ profiles
BASE = dict(q=0.45, p=0.80, a=0.090, b=0.200, lam=0.100, sigma=1.00, nu=0.030,
            lam_f=1 / 7, s=0.35)
PROFILES = [
    ("P0", "Typical", dict()),
    ("P1", "Low distraction", dict(q=0.55, sigma=0.85, s=0.30)),
    ("P2", "Focused (deep work)", dict(q=0.65, p=0.90, a=0.100, sigma=0.90)),
    ("P3", "Calm (meditation practice)", dict(q=0.55, p=0.85, lam=0.095, sigma=0.80, nu=0.025, s=0.25)),
    ("P4", "Sleep-restricted", dict(q=0.38, p=0.70, a=0.075, lam=0.110, sigma=1.20, nu=0.035,
                                   lam_f=1 / 5, s=0.50)),
    ("P5", "Busy / frequent switching", dict(q=0.35, p=0.75, a=0.085, lam=0.105, sigma=1.15,
                                            lam_f=1 / 6, s=0.45)),
    ("P6", "Positive fantasy only", dict(q=0.50, p=0.70, a=0.060)),
]
# routine = if-then plans tied to a fixed cue + daily review + protected sleep window
ROUTINE = dict(b_mult=1.20, lam_mult=0.90, s_mult=0.85, Tr=1)
NO_ROUTINE_Tr = 4
K, n, kap, mu, g, x0 = 1.0, 4, 1 / 11, 1 / 30, 10.0, 0.0
theta, Hz, N = 0.3, 180, 16
dt = 0.05

def params(over, routine):
    P = dict(BASE); P.update(over)
    if routine:
        P["b"] *= ROUTINE["b_mult"]; P["lam"] *= ROUTINE["lam_mult"]; P["s"] *= ROUTINE["s_mult"]
        P["Tr"] = ROUTINE["Tr"]
    else:
        P["Tr"] = NO_ROUTINE_Tr
    return P

def C_star(P):
    h = np.linspace(0, 1.2, 120001)
    G = P["lam"] * h - P["b"] * h**n / (K**n + h**n)
    # fold = first local max of G
    i = np.argmax(np.diff(G) < 0)
    return float(G[: i + 1].max() / P["a"]), float(h[i])

def hyst(P):
    hm = np.linspace(1e-4, 5, 50001)
    return bool(np.max(P["b"] * hm**(n - 1) / (K**n + hm**n)) > P["lam"])

def phi(r):
    p0 = r + (1 - r) / N; pr = (1 - r) / N
    return float(np.log2(N) + p0 * np.log2(p0) + ((N - 1) * pr * np.log2(pr) if pr > 0 else 0))

def survive(P):
    k = int(Hz // P["Tr"])
    lstar = np.log(1 / theta) / P["Tr"]
    return float(norm.cdf(np.log(lstar / P["lam_f"]) / P["s"]) ** k)

def simulate(P, train=120, total=300, M=2000, keep=False):
    F = lambda h: P["b"] * h**n / (K**n + h**n) - P["lam"] * h
    h = np.zeros(M); x = np.full(M, x0)
    steps = int(total / dt); xs = np.empty(steps) if keep else None
    for i in range(steps):
        C = P["q"] if i * dt < train else 0.0
        h = np.maximum(h + (P["a"] * C + F(h)) * dt + P["nu"] * np.sqrt(dt) * rng.standard_normal(M), 0)
        x = x + (kap * (C + h) * (g - x) - mu * (x - x0)) * dt + P["sigma"] * np.sqrt(dt) * rng.standard_normal(M)
        if keep: xs[i] = x.mean()
    return h, x, xs

R = {"base": BASE, "routine": ROUTINE, "no_routine_Tr": NO_ROUTINE_Tr, "profiles": []}
traj = {}
for code, name, over in PROFILES:
    for routine in (False, True):
        P = params(over, routine)
        cs, hf = C_star(P)
        h, x, xs = simulate(P, keep=code in ("P0", "P2", "P4", "P5", "P6"))
        if xs is not None: traj[(code, routine)] = xs
        row = dict(code=code, name=name, routine=routine, q=P["q"], p=P["p"], a=P["a"], b=P["b"],
                   lam=P["lam"], sigma=P["sigma"], nu=P["nu"], lam_f=P["lam_f"], s=P["s"], Tr=P["Tr"],
                   C_star=cs, margin=P["q"] - cs, hysteresis=hyst(P),
                   persist=float((h > 1).mean()), x300=float(x.mean() / g), sd300=float(x.std()),
                   dH=phi(P["q"] * P["p"]), eta=phi(P["q"] * P["p"]) / phi(P["p"]),
                   survive=survive(P))
        R["profiles"].append(row)
        print(f"{code} {'R' if routine else '-'} C*={cs:.3f} q={P['q']:.2f} margin={P['q']-cs:+.3f} "
              f"hyst={row['hysteresis']} persist={row['persist']:.3f} x300={row['x300']:.3f} "
              f"eta={row['eta']:.3f} dH={row['dH']:.3f} surv={row['survive']:.3f}")

# second-level interventions for the profiles that a routine alone cannot rescue
EXTRA = [
    ("P4+", "Sleep-restricted + routine + sleep partly restored",
     dict(q=0.42, p=0.75, a=0.083, lam=0.105, sigma=1.10, nu=0.032, lam_f=1 / 6, s=0.42)),
    ("P5+", "Busy + routine + time-blocking", dict(q=0.45, p=0.75, a=0.085, lam=0.105, sigma=1.15,
                                                  lam_f=1 / 6, s=0.45)),
    ("P6+", "Positive fantasy -> MCII + routine", dict(q=0.50, p=0.80, a=0.085)),
]
R["extra"] = []
for code, name, over in EXTRA:
    P = params(over, True)
    cs, _ = C_star(P)
    h, x, _ = simulate(P)
    row = dict(code=code, name=name, q=P["q"], p=P["p"], a=P["a"], b=P["b"], lam=P["lam"], C_star=cs,
               margin=P["q"] - cs, persist=float((h > 1).mean()), x300=float(x.mean() / g),
               eta=phi(P["q"] * P["p"]) / phi(P["p"]), survive=survive(P))
    R["extra"].append(row)
    print(f"{code} C*={cs:.3f} margin={P['q']-cs:+.3f} persist={row['persist']:.3f} "
          f"x300={row['x300']:.3f} eta={row['eta']:.3f} surv={row['survive']:.3f}")

# ------------------------------------------------------------------ figures
rows = R["profiles"]
labels = [f"{r['code']} {r['name']}" for r in rows if not r["routine"]]
nr = [r for r in rows if not r["routine"]]; wr = [r for r in rows if r["routine"]]
yy = np.arange(len(nr))[::-1]

fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.6), sharey=True)
for j, (key, title, xl) in enumerate([("persist", "P(structure persists\n120 d after stop)", (0, 1)),
                                      ("x300", "Outcome kept at day 300\n(fraction of goal)", (0, 1)),
                                      ("survive", "P(intention survives 180 d)\nrefresh: 4 d vs daily", (0, 1))]):
    a_ = ax[j]
    for y, r0, r1 in zip(yy, nr, wr):
        a_.plot([r0[key], r1[key]], [y, y], c="#CBD5E1", lw=2.2, zorder=1)
        a_.scatter(r0[key], y, c=GRAY, s=28, zorder=2, label="no routine" if y == yy[0] else None)
        a_.scatter(r1[key], y, c=GREEN, s=28, zorder=3, label="with routine" if y == yy[0] else None)
    a_.set(title=title, xlim=(-0.04, 1.04)); a_.title.set_fontsize(9)
ax[0].set_yticks(yy); ax[0].set_yticklabels(labels, fontsize=8)
ax[2].legend(loc="lower right", fontsize=7.5)
save(fig, "f9_1_profiles")

fig, ax = plt.subplots(figsize=(6.6, 3.7))
tt = np.arange(1, int(300 / dt) + 1) * dt
cols = {"P0": GRAY, "P2": BLUE, "P4": RED, "P5": ORANGE, "P6": PURPLE}
names = {c: nm for c, nm, _ in PROFILES}
for (code, routine), xs in traj.items():
    ax.plot(tt, xs / g, c=cols[code], lw=1.4 if routine else 1.1, ls="-" if routine else "--",
            label=f"{code} {names[code]}" if routine else None)
ax.axvspan(0, 120, color="#DBEAFE", alpha=0.5, lw=0)
ax.set(xlabel="day", ylabel=r"$E[x(t)]/g$",
       title="Outcome trajectories: dashed = no routine, solid = with routine (training shaded)")
ax.title.set_fontsize(9.5); ax.set_ylim(0, 1.0)
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "f9_2_trajectories")

# conceptual inverted-U (Yerkes-Dodson / LC-NE adaptive gain)
fig, ax = plt.subplots(figsize=(6.6, 3.1))
A = np.linspace(0, 1, 400)
ax.plot(A, np.exp(-(A - 0.45)**2 / (2 * 0.17**2)), c=BLUE, lw=1.8, label="complex / learning task")
ax.plot(A, np.exp(-(A - 0.62)**2 / (2 * 0.22**2)), c=ORANGE, lw=1.4, ls="--", label="simple / routine task")
pts = [("sleep-restricted", 0.12, RED), ("calm", 0.32, TEAL), ("low distraction", 0.42, GREEN),
       ("focused", 0.50, BLUE), ("busy / stressed", 0.82, BROWN)]
offs = {"sleep-restricted": (6, -12), "calm": (-46, 4), "low distraction": (-92, -2),
        "focused": (8, 4), "busy / stressed": (6, 6)}
for lab, a0, c in pts:
    yv = np.exp(-(a0 - 0.45)**2 / (2 * 0.17**2))
    ax.scatter(a0, yv, c=c, s=34, zorder=3)
    ax.annotate(lab, (a0, yv), xytext=offs[lab], textcoords="offset points", fontsize=8, color=c)
ax.set(xlabel="arousal (tonic LC-NE, catecholamines; conceptual axis)", ylabel="relative effectiveness",
       title="Conceptual inverted-U: the best state depends on the task", ylim=(0, 1.3))
ax.legend(fontsize=8, loc="upper left"); save(fig, "f8_1_inverted_u")

# daily cycle mapped to CIOT parameters (example, early-to-intermediate chronotype)
fig, ax = plt.subplots(figsize=(7.4, 3.3))
blocks = [(0, 6.5, "#1E3A5F", "sleep: slow waves + spindles\n(a up, lambda down, recovery)"),
          (6.5, 7.2, "#9CA3AF", "goal review\n(refresh)"),
          (7.2, 9.0, "#2F6FB2", "focus block\n(q, p up)"), (9.0, 9.3, "#A7F3D0", "break"),
          (9.3, 11.0, "#2F6FB2", "focus block"), (11.0, 12.5, "#E5E7EB", "admin /\nshallow"),
          (12.5, 13.5, "#A7F3D0", "rest / walk\n(recovery)"), (13.5, 16.0, "#93C5FD", "shallow /\nsocial"),
          (16.0, 17.0, "#FCD34D", "exercise"), (17.0, 21.5, "#E5E7EB", "free / family"),
          (21.5, 22.5, "#C4B5FD", "if-then plan\n(b up)"), (22.5, 24, "#1E3A5F", "sleep")]
rows_y = [-0.75, -1.55]
k = 0
for s0, s1, c, lab in blocks:
    ax.barh(0, s1 - s0, left=s0, color=c, edgecolor="white", height=0.7)
    mid = (s0 + s1) / 2
    if s1 - s0 >= 3:
        ax.text(mid, 0, lab, ha="center", va="center", fontsize=7.2,
                color="white" if c in ("#1E3A5F", "#2F6FB2") else "#1B1F27")
    else:
        yl = rows_y[k % 2]; k += 1
        ax.plot([mid, mid], [-0.36, yl + 0.22], c="#9CA3AF", lw=0.6)
        ax.text(mid, yl, lab, ha="center", va="top", fontsize=7.0, color="#1B1F27")
ax.set(xlim=(0, 24), ylim=(-2.4, 0.5), yticks=[], xlabel="hour of day (example; shift with chronotype)")
ax.set_xticks(range(0, 25, 2)); ax.grid(False); ax.spines["left"].set_visible(False)
ax.set_title("Example daily cycle: each phase targets a different CIOT parameter", fontsize=9.5)
save(fig, "f9_3_daily_cycle")

with open(os.path.join(HERE, "results_supp.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print("done")
