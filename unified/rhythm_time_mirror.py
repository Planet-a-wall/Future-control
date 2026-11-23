"""
Three new layers of CIOT-U.
 RHYTHM (law of vibration, scientific part): coupled oscillators (Kuramoto).
   R1 synchrony threshold K_c = 2*gamma (Lorentzian spread gamma);  R2 r = sqrt(1 - K_c/K)
 TIME (law of time): delayed self-correction is stable iff gain*delay < pi/2
   T1 continuous delay equation;  T2 consistency with the life-reactor results (gain 0.4: d=1 stable, d>=4 unstable)
 MIRROR (law of mirror, scientific part): others reflect your behaviour, with distortion
   M1 flattery bias delta shifts equilibrium confidence by beta*delta/gamma
   M2 two-person emotional mirror loop amplifies by 1/(1-rho^2), stable iff |rho|<1
   M3 fortune model: more distorted mirror after success (psi) -> more falls
Functions are imported by verify_all.py; running this file prints the results and writes figures.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- RHYTHM
def kuramoto_r(K, gamma=0.5, N=2000, T=60.0, dt=0.02, seed=0):
    rng = np.random.default_rng(seed)
    w = gamma * np.tan(np.pi * (rng.random(N) - 0.5))          # Lorentzian natural frequencies
    w = np.clip(w, -50, 50)
    th = rng.uniform(0, 2 * np.pi, N); rs = []
    for i in range(int(T / dt)):
        z = np.mean(np.exp(1j * th))
        th = th + dt * (w + K * np.abs(z) * np.sin(np.angle(z) - th))
        if i * dt > T / 2: rs.append(np.abs(z))
    return float(np.mean(rs))

# ---------------------------------------------------------------- TIME
def delay_stable(gain, d, T=200.0, dt=0.01):
    """x'(t) = -gain * x(t-d). Returns True if the deviation decays."""
    n_d = max(1, int(round(d / dt))); x = np.ones(n_d + 1); hist = list(x)
    for _ in range(int(T / dt)):
        hist.append(hist[-1] - dt * gain * hist[-1 - n_d])
    tail = np.abs(np.array(hist[-int(20 / dt):]))
    return bool(tail.max() < 0.5)

# ---------------------------------------------------------------- MIRROR
def confidence_eq(delta, beta=0.04, gamma=0.02, T=20000, seed=0):
    """c' = gamma(1-c) + beta*(signal), signal = true outcome sign-weighted + flattery delta."""
    rng = np.random.default_rng(seed); c = 1.0; cs = []
    for t in range(T):
        z = rng.standard_normal()
        c = c + gamma * (1 - c) + beta * (0.0 * z + delta) + 0.01 * z
        if t > T // 2: cs.append(c)
    return float(np.mean(cs))

def mirror_loop(a, rho, steps=400):
    m1 = m2 = 0.0
    for _ in range(steps):
        m1, m2 = a + rho * m2, rho * m1
        if abs(m1) > 1e6: return float("inf")
    return m1

def fortune_psi(psis=(0.0, 0.5, 1.0, 1.5), n=12000):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "research", "fortune"))
    cwd = os.getcwd(); os.chdir(os.path.join(os.path.dirname(HERE), "research", "fortune"))
    from sleep_studies import cohort, FLAGS
    P = json.load(open("final_calibration.json"))["chosen"]
    out = {}
    for psi in psis:
        r = cohort({**P, "psi": psi}, FLAGS, n=n, v=0.85, q_intr=0.9)
        out[psi] = dict(p_great=r["p_great"], p_fall_given_great=r["p_fall_given_great"], p_stay=r["p_stay"])
    os.chdir(cwd)
    return out

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    OUT = {}
    Ks = [0.4, 0.8, 0.9, 1.1, 1.3, 1.6, 2.0, 3.0]
    OUT["R"] = {K: kuramoto_r(K) for K in Ks}
    print("R  r(K), gamma=0.5 (K_c=1.0):", {K: round(v, 3) for K, v in OUT["R"].items()})
    gains = np.linspace(0.1, 1.0, 10); ds = np.array([0.5, 1, 2, 3, 4, 6])
    OUT["T"] = [[delay_stable(g, d) for d in ds] for g in gains]
    OUT["M1"] = {d: confidence_eq(d) for d in (0.0, 0.1, 0.2, 0.4)}
    print("M1 c_eq vs flattery:", {k: round(v, 3) for k, v in OUT["M1"].items()}, " theory shift = beta*delta/gamma = 2*delta")
    OUT["M2"] = {rho: mirror_loop(1.0, rho) for rho in (0.0, 0.3, 0.6, 0.9, 1.1)}
    print("M2 loop:", OUT["M2"])
    OUT["M3"] = fortune_psi()
    print("M3 fortune psi:", OUT["M3"])
    json.dump(OUT, open("rtm_results.json", "w"), indent=1, default=str)
    figdir = os.path.join(os.path.dirname(HERE), "handbook", "figures")
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path",
                         "axes.grid": True, "grid.color": "#E5E7EB"})
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    kk = np.linspace(0.3, 3, 200)
    ax[0].plot(kk, np.sqrt(np.clip(1 - 1.0 / kk, 0, None)), c="#2F6FB2", label="theory sqrt(1-Kc/K)")
    ax[0].plot(Ks, [OUT["R"][K] for K in Ks], "o", c="#C23B3B", label="simulation")
    ax[0].axvline(1.0, ls="--", c="gray"); ax[0].set(xlabel="coupling K (routines, shared rhythm)", ylabel="synchrony r",
                                                   title="Rhythm: sync needs K > 2*gamma"); ax[0].legend(fontsize=7)
    G, D = np.meshgrid(ds, gains)
    ax[1].imshow(np.array(OUT["T"], float), origin="lower", aspect="auto", cmap="RdYlGn",
                 extent=[ds[0], ds[-1], gains[0], gains[-1]])
    dd = np.linspace(0.5, 6, 100); ax[1].plot(dd, np.pi / 2 / dd, "k--", label="gain*delay = pi/2")
    ax[1].set(xlim=(0.5, 6), ylim=(0.1, 1.0), xlabel="awareness delay d", ylabel="correction gain",
              title="Time: stable (green) iff gain*d < pi/2"); ax[1].legend(fontsize=7); ax[1].grid(False)
    ps = list(OUT["M3"])
    ax[2].plot(ps, [OUT["M3"][p]["p_fall_given_great"] for p in ps], "o-", c="#C23B3B", label="P(fall | great)")
    ax[2].plot(ps, [OUT["M3"][p]["p_stay"] for p in ps], "s-", c="#3A9A5B", label="P(great and stayed)")
    ax[2].set(xlabel="mirror distortion after success (psi)", title="Mirror: distorted mirrors cause falls"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "h9_rtm.svg")); fig.savefig(os.path.join(HERE, "figs", "rtm.png"), dpi=120)
