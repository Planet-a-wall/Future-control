"""Vector figures for the CIOT-U handbook, built from the verified result files."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H)
F = os.path.join(H, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E5E7EB", "legend.frameon": False, "svg.fonttype": "path"})
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = "#2F6FB2", "#D9822B", "#3A9A5B", "#C23B3B", "#6B7280", "#7A4FB5", "#1F9E9A"
def save(fig, n): fig.tight_layout(); fig.savefig(os.path.join(F, n + ".svg")); plt.close(fig)
load = lambda p: json.load(open(os.path.join(R, p)))

# h2: ignition - transfer curves and n_min heatmap
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2))
for deg, c in [(90, RED), (70, ORANGE), (45, BLUE), (20, GREEN)]:
    th = np.radians(deg); n = np.arange(1, 12)
    ax[0].plot(n - 1, np.cos(th / n) ** (2 * n), "o-", ms=3, c=c, label=f"{deg}°")
ax[0].set(xlabel="planning / reality-check steps", ylabel="P(dream becomes action)  M",
          title="Dream -> action transfer"); ax[0].legend(title="dream-action angle", fontsize=7.5)
ig = load("unified/ignition_results.json")["base_n_min"]
degs = list(ig.keys()); Bs = list(ig[degs[0]].keys())
Z = np.array([[ig[d][b] for b in Bs] for d in degs], float)
im = ax[1].imshow(Z, cmap="YlOrRd", aspect="auto")
for i in range(len(degs)):
    for j in range(len(Bs)):
        ax[1].text(j, i, int(Z[i, j]), ha="center", va="center", fontsize=8)
ax[1].set_xticks(range(len(Bs))); ax[1].set_xticklabels(Bs); ax[1].set_yticks(range(len(degs)))
ax[1].set_yticklabels([f"{d}°" for d in degs]); ax[1].grid(False)
ax[1].set(xlabel="environment multiplier B", ylabel="dream-action angle", title="Minimum n to ignite")
save(fig, "h2_ignition")

# h3: modern profiles x interventions
P = load("unified/ignition_results.json")["profiles"]
names = list(P); ivs = list(P[names[0]])
fig, ax = plt.subplots(figsize=(7.4, 3.4))
w = 0.13; cols = [GRAY, TEAL, BLUE, PURPLE, ORANGE, GREEN]
for j, (iv, c) in enumerate(zip(ivs, cols)):
    ax.bar(np.arange(len(names)) + (j - 2.5) * w, [P[n][iv]["k_with_3_step_plan"] for n in names], w, color=c, label=iv)
ax.axhline(1, c=RED, ls="--", lw=1); ax.text(len(names) - 0.6, 1.03, "ignition k = 1", color=RED, fontsize=7.5)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(", ", "\n").replace(" (", "\n(") for n in names], fontsize=7)
ax.set(ylabel="chain factor k (3-step plan)", title="Modern-life profiles: what makes life ignite (illustrative)")
ax.legend(fontsize=6.5, ncol=3, loc="upper left"); save(fig, "h3_profiles")

# h4: mood and sleep (final calibration)
fc = load("research/fortune/final_check_results.json")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
m = fc["H6"]; labs = ["stable", "typical", "daily swings", "depression-prone", "elevated-prone", "bipolar-like"]
ax[0].bar(range(6), [m[l]["none"]["p_bankrupt"] for l in labs], color=[GREEN, GRAY, ORANGE, BLUE, RED, PURPLE])
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(["stable", "typical", "daily\nswings", "depr.\nprone", "elevated\nprone", "bipolar\nlike"], fontsize=7)
ax[0].set(ylabel="P(bankruptcy)", title="Mood profile -> ruin risk")
s = fc["H7"]; keys = list(s)
ax[1].bar(range(len(keys)), [s[k]["p_stay"] for k in keys], color=[GREEN, GREEN, GREEN, RED, BLUE, ORANGE])
ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([k.replace(" | ", "\n") for k in keys], fontsize=6.5)
ax[1].set(ylabel="P(success and stayed)", title="Sleep: only sleep fixes sleep")
save(fig, "h4_mood_sleep")

# h5: Kelly growth = information - KL; criticality boundary
kc = np.linspace(0, 3, 300)
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.plot(kc, 1 - (kc - 1) ** 2, c=BLUE, lw=1.8, label="growth / I  = 1 - (kappa c - 1)^2")
ax.axhline(0, c="k", lw=0.7); ax.axvline(2, c=RED, ls="--"); ax.axvline(1, c=GREEN, ls=":")
ax.fill_between(kc, -2, 1.2, where=kc > 2, color=RED, alpha=0.07)
ax.text(2.05, 0.7, "k < 1 after 2: meltdown zone\n(hubris threshold = criticality)", color=RED, fontsize=7.5)
ax.text(1.03, -1.1, "calibrated\n(best)", color=GREEN, fontsize=7.5)
ax.set(xlabel="kappa x c  (bet size relative to Kelly, from confidence)", ylabel="relative growth", ylim=(-1.6, 1.2),
       title="One boundary: Kelly hubris threshold = chain-reaction criticality (U1)")
ax.legend(fontsize=7.5, loc="lower left"); save(fig, "h5_kelly")

# h6: hubris index and traits
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
h1 = [b for b in fc["H1"] if b["p_fall"] is not None and b["n"] >= 20]
ax[0].bar([f"{b['lo']}-{b['hi']}" for b in h1], [b["p_fall"] for b in h1], color=RED)
ax[0].set(xlabel="post-success hubris index H", ylabel="P(fall | became great)", title="U4: who falls after success")
h4 = fc["H4"]
for key, c, lab in [("ego", RED, "ego"), ("iota", GREEN, "integrity"), ("phi", PURPLE, "enjoyment")]:
    ax[1].plot(range(4), [r["p_fall_given_great"] for r in h4[key]], "o-", c=c, label=lab)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["low", "", "", "high"])
ax[1].set(ylabel="P(fall | great)", title="Traits after success"); ax[1].legend(fontsize=7.5)
save(fig, "h6_hubris")

# h8: generations
fcal = load("research/fortune/final_calibration.json")["oos"][0]["gens"]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
g = range(len(fcal))
ax.plot(g, [x["gini"] for x in fcal], "o-", c=BLUE, label="Gini")
ax.plot(g, [x["top1_share"] for x in fcal], "s-", c=ORANGE, label="top 1% share")
ax.plot(g, [x["mobility_Q1_to_Q5"] for x in fcal], "^-", c=GREEN, label="bottom->top mobility")
ax.set_xticks(list(g)); ax.set_xticklabels([f"gen {i}" for i in g])
ax.set(title="Four generations: stable distribution, ~12% of top dynasties remain after 3 generations", ylim=(0, 0.8))
ax.title.set_fontsize(8.5); ax.legend(fontsize=7.5, ncol=3); save(fig, "h8_generations")
print(sorted(os.listdir(F)))
