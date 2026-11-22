"""
CIOT (Cumulative Intentional Order Theory) - numerical tests.

Test 1  : OU model v1 - variance / entropy predictions vs simulation
Test 2  : OU model v1 - convergence time tau vs simulation (+ Lally 2009 calibration)
Test 3  : threshold claim - v1 (no threshold) vs v2 bistable habit model
Test 4  : Touchette-Lloyd bound  dH <= I(X;Y)  with attention-lapse controller
Test 5  : Quantum Zeno effect vs classical "refresh" analogue
Test 6  : Landauer cost of intention vs brain energy budget
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

def H_gauss(var):
    return 0.5 * np.log(2 * np.pi * np.e * var)

def H_disc(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

print("=" * 70)
# ---------------------------------------------------------------- Test 1+2
aL, sigma, g, x0 = 1 / 11, 1.0, 10.0, 0.0       # aL per day -> C=0.5 gives tau=22 d
dt, days, npaths = 0.05, 365, 4000
steps = int(days / dt)
Cs = [0.25, 0.5, 1.0]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
print("TEST 1+2: OU model  dx = a*C*L*(g-x)dt + sigma dW")
print(f"{'C':>5} {'Var theory':>11} {'Var sim':>9} {'H theory':>9} {'H sim':>7}"
      f" {'tau theory':>11} {'tau sim':>8}")
for C in Cs:
    k = aL * C
    x = np.full(npaths, x0)
    mean_traj = np.empty(steps)
    for i in range(steps):
        x = x + k * (g - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(npaths)
        mean_traj[i] = x.mean()
    var_sim = x.var()
    var_th = sigma**2 / (2 * k)
    t = np.arange(steps) * dt
    # tau = time where mean reaches (1 - 1/e) of g
    tau_sim = t[np.argmax(mean_traj >= g * (1 - np.exp(-1)))]
    print(f"{C:5.2f} {var_th:11.2f} {var_sim:9.2f} {H_gauss(var_th):9.3f}"
          f" {H_gauss(var_sim):7.3f} {1/k:11.1f} {tau_sim:8.1f}")
    ax[0].plot(t, mean_traj, label=f"C={C}")
    ax[1].hist(x, bins=60, density=True, alpha=0.5, label=f"C={C}")
ax[0].axhline(g, ls="--", c="k", lw=0.8)
ax[0].set(title="Mean trajectory toward goal g", xlabel="day", ylabel="x")
ax[1].set(title="Outcome distribution at day 365", xlabel="x")
for a in ax: a.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/test1_ou.png", dpi=120); plt.close(fig)
print(f"Lally calibration: tau=22 d -> 95% at {3*22} d (Lally median = 66 d)")
print("dH when C doubles (theory): -0.5*ln2 =", round(-0.5 * np.log(2), 3), "nats")

print("=" * 70)
# ---------------------------------------------------------------- Test 3
print("TEST 3: Threshold claim")
# v1: train with C for 120 d, then C=0 (pure noise + no pull) for 120 d
Cgrid = np.linspace(0, 1, 41)
v1_final = []
for C in Cgrid:
    k = aL * C
    x = np.full(800, x0)
    for i in range(int(120 / dt)):
        x = x + k * (g - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(x.size)
    lam_v1 = 1 / 30                               # structure decays without input
    for i in range(int(120 / dt)):
        x = x - lam_v1 * x * dt + sigma * np.sqrt(dt) * rng.standard_normal(x.size)
    v1_final.append(x.mean())

# v2: habit/structure h with self-reinforcement (bistable)
a_in, b, K, n, lam = 0.09, 0.2, 1.0, 4, 0.1
f = lambda h: b * h**n / (K**n + h**n) - lam * h
hh = np.linspace(0, 1, 10001)
C_star = np.max(-f(hh)) / a_in
v2_final, v2_persist = [], []
for C in Cgrid:
    h = np.zeros(800)
    for i in range(int(120 / dt)):
        h = np.maximum(h + (a_in * C + f(h)) * dt
                       + 0.03 * np.sqrt(dt) * rng.standard_normal(h.size), 0)
    for i in range(int(120 / dt)):
        h = np.maximum(h + f(h) * dt + 0.03 * np.sqrt(dt) * rng.standard_normal(h.size), 0)
    v2_final.append(h.mean()); v2_persist.append((h > 1).mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(Cgrid, np.array(v1_final) / g, "o-", label="v1 (OU + decay)")
ax[0].plot(Cgrid, v2_persist, "s-", label="v2 bistable: P(habit persists)")
ax[0].axvline(C_star, ls="--", c="r", label=f"C* theory = {C_star:.2f}")
ax[0].set(title="Result 120 d after stopping", xlabel="continuity C during 120 d training")
ax[0].legend()
ax[1].plot(hh * 3, f(hh * 3), label="f(h): self-reinforcement - decay")
ax[1].axhline(0, c="k", lw=0.8)
ax[1].set(title="v2 phase line (C=0): stable 0, unstable 1, stable ~1.85", xlabel="h")
fig.tight_layout(); fig.savefig(f"{OUT}/test3_threshold.png", dpi=120); plt.close(fig)
print(f"v1 residual after stop (fraction of g) at C=0.25/0.5/1: "
      f"{v1_final[10]/g:.3f} / {v1_final[20]/g:.3f} / {v1_final[40]/g:.3f}  -> smooth, no threshold")
print(f"v2 analytic critical continuity C* = {C_star:.3f}")
for C, p in zip(Cgrid[::5], v2_persist[::5]):
    print(f"   C={C:.3f}  P(habit persists 120 d after stop) = {p:.3f}")

print("=" * 70)
# ---------------------------------------------------------------- Test 4
print("TEST 4: Touchette-Lloyd bound  H(X)-H(X') <= I(X;Y)   (N=16 states, bits)")
N = 16
rows = []
for p in [0.2, 0.5, 0.8, 1.0]:
    # channel: Y = X w.p. p, else uniform
    PyGx = np.full((N, N), (1 - p) / N) + np.eye(N) * p
    Pxy = PyGx / N
    Py = Pxy.sum(0)
    I = H_disc(Py) - sum(H_disc(PyGx[x]) / N for x in range(N))
    for q in [0.25, 0.5, 1.0]:                    # q = continuity (acts on info w.p. q)
        Px_after = np.zeros(N)
        for x in range(N):
            for y in range(N):
                Px_after[(x - y) % N] += Pxy[x, y] * q
                Px_after[x] += Pxy[x, y] * (1 - q)
        dH = np.log2(N) - H_disc(Px_after)
        rows.append((p, q, I, dH))
        print(f"  obs accuracy p={p:.1f}  continuity q={q:.2f}  I={I:.3f}  dH={dH:.3f}"
              f"  bound holds: {dH <= I + 1e-12}  efficiency dH/I={dH/I if I else 0:.2f}")

print("=" * 70)
# ---------------------------------------------------------------- Test 5
print("TEST 5: Quantum Zeno vs classical refresh analogue")
Om = 1.0; T = np.pi / (2 * Om)                    # without measurement -> fully flipped
ns = np.arange(1, 51)
P_zeno = np.cos(Om * T / ns) ** (2 * ns)
lam_f, theta, horizon = 1 / 7, 0.3, 180           # forgetting 1/7 per day, recall threshold
Trs = np.arange(1, 22)
surv = []
for Tr in Trs:
    k = int(horizon // Tr)
    lam_draw = lam_f * rng.lognormal(0, 0.5, size=(5000, k))
    ok = np.all(np.exp(-lam_draw * Tr) > theta, axis=1)
    surv.append(ok.mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ns, P_zeno, "o-"); ax[0].set(title="Quantum Zeno: survival vs # measurements",
                                         xlabel="n measurements", ylabel="P(stay)")
ax[1].plot(Trs, surv, "s-"); ax[1].axvline(np.log(1/theta)/lam_f, ls="--", c="r",
                                            label="T* = ln(1/θ)/λ")
ax[1].set(title="Classical: P(intention survives 180 d)", xlabel="refresh interval (days)")
ax[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/test5_zeno.png", dpi=120); plt.close(fig)
for n_ in [1, 2, 5, 10, 50]:
    print(f"  Zeno n={n_:2d}  P(stay)={P_zeno[n_-1]:.3f}")
print(f"  classical T* = {np.log(1/theta)/lam_f:.1f} d")
for Tr, s in zip(Trs, surv):
    if Tr in (1, 3, 5, 7, 9, 12, 15, 21):
        print(f"  refresh every {Tr:2d} d -> P(survive 180 d) = {s:.3f}")

print("=" * 70)
# ---------------------------------------------------------------- Test 6
print("TEST 6: Landauer")
kB, Tbody = 1.380649e-23, 310
E_bit = kB * Tbody * np.log(2)
E_day = 20 * 86400
print(f"  kT ln2 at 310 K = {E_bit:.2e} J/bit")
print(f"  brain energy/day = {E_day:.2e} J  -> Landauer max = {E_day/E_bit:.2e} bits/day")
print(f"  bits to specify 1 goal out of 1e9 options = {np.log2(1e9):.1f} bits"
      f" -> cost {np.log2(1e9)*E_bit:.1e} J")
