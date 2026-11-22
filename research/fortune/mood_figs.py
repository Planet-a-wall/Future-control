import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("mood_results.json"))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1, 1.5]))
rows = R["S10"]["rows"]; flat = R["S10"]["flat"]
lab = ["no swings"] + [f"tau={int(r['tau'])} mo" for r in rows]
med = [flat["median_final"]] + [r["median_final"] for r in rows]
dep = [flat["months_dep"]] + [r["months_dep"] for r in rows]
imp = [flat["imp_hi"]] + [r["imp_hi"] for r in rows]
ax[0].bar(lab, med, color=["#3A9A5B"] + ["#2F6FB2"] * 4)
for i, (d, m_) in enumerate(zip(dep, med)):
    ax[0].text(i, m_ + 0.4, f"dep {d:.0f} mo\nimpulse {imp[i]:.0f}", ha="center", fontsize=7)
ax[0].set(title="S10: same mood amplitude, different timescales", ylabel="median final wealth (born poor, strong mind)"); ax[0].tick_params(axis="x", rotation=20)
S = R["S11"]; profs = list(S.keys()); cop = ["none", "precommit", "regulate", "delay 48h", "all three"]
M = np.full((len(profs), len(cop)), np.nan)
for i, p in enumerate(profs):
    for j, c in enumerate(cop):
        if c in S[p]: M[i, j] = S[p][c]["p_stay"]
im = ax[1].imshow(M, cmap="viridis", aspect="auto")
for i in range(len(profs)):
    for j in range(len(cop)):
        if not np.isnan(M[i, j]):
            ax[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.2 else "k", fontsize=8)
ax[1].set_xticks(range(len(cop))); ax[1].set_xticklabels(cop)
ax[1].set_yticks(range(len(profs))); ax[1].set_yticklabels(profs)
ax[1].set_title("S11: P(became great and stayed) by mood profile x coping")
fig.colorbar(im, ax=ax[1], fraction=0.03)
fig.tight_layout(); fig.savefig("figs/mood_s10_s11.png", dpi=130)
