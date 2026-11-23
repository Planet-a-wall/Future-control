"""
Quantum-like Intention Theory (QIT): quantum *probability* (not quantum brain physics) for
how imagined goals turn into action. Tests:
 T1 'three-polarizer' intention transfer: vision -> (k reality/plan steps) -> action
 T2 classical latent-variable model cannot raise P(action|vision) by asking intermediate questions
 T3 order effects + QQ equality (Wang et al. 2014) for quantum vs classical Markov models
 T4 interference phase from the disjunction effect (Tversky & Shafir 1992)
 T5 Zeno-like intention keeping: -ln P vs 1/n is linear (quantum) vs threshold (classical, Thm 6.3)
 T6 embed T1 into the fortune model (born poor, strong vision): action conversion multiplies drive
Run: python3 qciot.py -> qciot_results.json, figs/*.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
OUT = {}
rng = np.random.default_rng(0)
ket = lambda a: np.array([np.cos(a), np.sin(a)])
proj = lambda a: np.outer(ket(a), ket(a))

# ---------------- T1 intention transfer through k intermediate steps
def transfer(theta, k):
    """vision at 0, action at theta, k equally spaced intermediate (reality/plan) projectors."""
    if k == 0:
        return np.cos(theta) ** 2
    return np.cos(theta / (k + 1)) ** (2 * (k + 1))
T1 = {}
for deg in (90, 70, 45, 20):
    th = np.radians(deg)
    T1[deg] = [float(transfer(th, k)) for k in range(0, 11)]
    print(f"T1 theta={deg:3d} deg: P(action) with k=0..5 steps:", [round(x, 3) for x in T1[deg][:6]], "k=10:", round(T1[deg][10], 3))
OUT["T1"] = T1

# ---------------- T2 classical latent variable: asking R just reveals; marginal unchanged
# classical: hidden variable r; P(A|V) = sum_r P(r|V) P(A|r) whether or not R is asked
pRV = np.array([0.5, 0.5]); pAr = np.array([0.5, 0.5])
OUT["T2"] = dict(classical_without_asking=float(pRV @ pAr), classical_with_asking=float(pRV @ pAr),
                 quantum_without=float(transfer(np.pi / 2, 0)), quantum_with=float(transfer(np.pi / 2, 1)))
print("T2", OUT["T2"])

# ---------------- T3 order effects and QQ equality
def seq_prob(psi, a, b, ya, yb):
    Pa = proj(a) if ya else np.eye(2) - proj(a)
    Pb = proj(b) if yb else np.eye(2) - proj(b)
    v = Pb @ Pa @ psi
    return float(v @ v)
qq_q, ord_q = [], []
for _ in range(2000):
    psi = ket(rng.uniform(0, np.pi)); a, b = rng.uniform(0, np.pi, 2)
    AB = {(x, y): seq_prob(psi, a, b, x, y) for x in (1, 0) for y in (1, 0)}
    BA = {(x, y): seq_prob(psi, b, a, y, x) for x in (1, 0) for y in (1, 0)}   # keys (A answer, B answer)
    ord_q.append(abs(AB[(1, 1)] + AB[(0, 1)] - (BA[(1, 1)] + BA[(0, 1)])))   # P(B=yes) depends on order
    qq_q.append((AB[(1, 0)] + AB[(0, 1)]) - (BA[(1, 0)] + BA[(0, 1)]))
# classical Markov model with arbitrary transition kernels: generically violates QQ
qq_c = []
for _ in range(2000):
    pA = rng.uniform(); pB_A = rng.uniform(size=2); pB = rng.uniform(); pA_B = rng.uniform(size=2)
    AB_mix = pA * (1 - pB_A[0]) + (1 - pA) * pB_A[1]
    BA_mix = pB * (1 - pA_B[0]) + (1 - pB) * pA_B[1]
    qq_c.append(AB_mix - BA_mix)
OUT["T3"] = dict(mean_abs_order_effect_quantum=float(np.mean(ord_q)), qq_quantum_max_abs=float(np.max(np.abs(qq_q))),
                 qq_classical_mean_abs=float(np.mean(np.abs(qq_c))))
print("T3", OUT["T3"])

# ---------------- T4 interference from the disjunction effect
p_win, p_lose, p_unknown = 0.69, 0.59, 0.36       # Tversky & Shafir 1992 two-stage gamble
classical = 0.5 * p_win + 0.5 * p_lose
amp = 2 * np.sqrt(0.5 * p_win) * np.sqrt(0.5 * p_lose)
cos_phi = (p_unknown - classical) / amp
OUT["T4"] = dict(classical_prediction=classical, observed=p_unknown, interference=p_unknown - classical,
                 cos_phi=float(cos_phi), phi_deg=float(np.degrees(np.arccos(cos_phi))))
print("T4", OUT["T4"])

# ---------------- T5 Zeno-like keeping of an intention
ns = np.arange(1, 41); Om_T = np.pi / 2
P_q = np.cos(Om_T / ns) ** (2 * ns)
OUT["T5"] = dict(n=ns.tolist(), P=P_q.tolist(), slope_lnP_vs_inv_n=float(np.polyfit(1 / ns[4:], -np.log(P_q[4:]), 1)[0]),
                 theory_slope=float(Om_T ** 2))
print("T5 slope of -lnP vs 1/n:", round(OUT["T5"]["slope_lnP_vs_inv_n"], 3), "theory (Omega T)^2 =", round(Om_T ** 2, 3))

# ---------------- T6 fortune embedding
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fortune"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fortune"))
from sleep_studies import cohort, FLAGS
P = json.load(open("final_calibration.json"))["chosen"]
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))
T6 = []
for deg in (80, 45):
    for k in (0, 1, 3, 8):
        conv = transfer(np.radians(deg), k)
        r = cohort(P, FLAGS, n=15000, v=0.85, q_intr=float(0.95 * conv), phi=0.85)
        T6.append(dict(theta=deg, steps=k, conversion=float(conv), p_great=r["p_great"], p_stay=r["p_stay"],
                       median=r["median_final"]))
        print(f"T6 theta={deg} steps={k}: conversion={conv:.3f} great={r['p_great']:.3f} stay={r['p_stay']:.3f} med={r['median_final']:.1f}", flush=True)
OUT["T6"] = T6

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for deg, v in T1.items():
    ax[0].plot(range(11), v, "o-", ms=3, label=f"vision-action angle {deg} deg")
ax[0].set(xlabel="number of intermediate reality / plan steps k", ylabel="P(intention becomes action)",
          title="QIT: 'three-polarizer' law of intention"); ax[0].legend(fontsize=7)
ax[1].plot(1 / ns, -np.log(P_q), "o", ms=3, label="quantum-like keeping")
ax[1].plot(1 / ns, Om_T ** 2 / ns, label="slope (Omega T)^2")
ax[1].set(xlabel="1 / number of reviews", ylabel="-ln P(intention kept)", title="Zeno-like signature: linear in 1/n")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figs/qit_t1_t5.png", dpi=130)
json.dump(OUT, open("qciot_results.json", "w"), indent=1, default=float)
