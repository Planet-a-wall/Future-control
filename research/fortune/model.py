"""
Fortune dynamics (CIOT v3) - agent-based model, monthly steps.

Each agent has wealth/status capital K (units: one median annual income),
traits drawn at birth, and mental state variables that evolve:

  v      vision / information quality about the world (0-1)
  q      drive = phi * q_intr + (1-phi) * q_extr(K / goal)
  phi    intrinsic share of motivation ("enjoyment of the work itself")
  c      confidence multiplier on perceived edge (1 = calibrated)
  iota   integrity (0-1)
  omega  giving rate -> reputation capital R ("merit" built in this life)

Investment rule (fractional Kelly with perceived edge):
  mu    = mu_max * q * v  (+ merit opportunity premium)
  f     = kelly_frac * c * mu / sigma^2        (capped)
  K'    = K + K*(r + f*(X - r)) + income - consumption - giving

Feature flags switch mechanisms on/off so each study can add one at a time:
  ratchet    consumption habit (lifestyle inflation is sticky downward)
  goal_drive extrinsic drive collapses once K passes the personal goal
  hubris     self-attribution: wins raise c more than losses lower it;
             status weakens correction
  loss_risk  drawdown from peak -> risk seeking (prospect theory)
  fraud      drawdown + low integrity -> fraud -> detection -> jail
  merit      reputation capital: opportunity premium + rescue near ruin
"""
import numpy as np

DEFAULT = dict(
    T=480, r=0.0033, sigma=0.10, mu_max=0.03, kelly_frac=0.5, f_cap=2.5,
    income_base=0.5, save_wealth_rate=0.003, cons_income=0.85, habit_decay=0.004,
    beta=0.04, beta_neg=0.03, gamma=0.02, psi=0.5, K_med=2.0, ego_gain=3.0, c_cap=6.0,
    goal_med=30.0, goal_sd=0.8, q_extr_free=0.7,
    lam_loss=1.5, fraud_pi=0.01, fraud_det=0.02,
    y_R=0.15, R_half=5.0, fb_gain=2.0, rho_R=0.01, rescue_p=0.6,
    great=100.0,
)
ALL_OFF = dict(ratchet=False, goal_drive=False, hubris=False, loss_risk=False,
               fraud=False, merit=False, merit_feedback=False)


def traits(N, rng, **over):
    t = dict(
        v=rng.beta(2, 3, N), q_intr=rng.beta(2, 2, N), phi=rng.beta(2, 2, N),
        iota=rng.beta(4, 2, N), omega=rng.beta(1.2, 12, N), ego=rng.beta(1.5, 4, N),
        c0=np.exp(rng.normal(0, 0.2, N)), goal=None,
    )
    t["goal"] = DEFAULT["goal_med"] * np.exp(rng.normal(0, DEFAULT["goal_sd"], N))
    for k, val in over.items():
        t[k] = np.broadcast_to(np.asarray(val, float), (N,)).copy() if np.ndim(val) == 0 else np.asarray(val, float)
    return t


def simulate(K0, tr, flags, seed=0, R0=0.0, P=None, windfall=None, record=False):
    P = {**DEFAULT, **(P or {})}
    fl = {**ALL_OFF, **flags}
    rng = np.random.default_rng(seed)
    N = len(K0)
    K = np.asarray(K0, float).copy()
    R = np.broadcast_to(np.asarray(R0, float), (N,)).copy()
    c = tr["c0"].copy()
    peak = K.copy()
    income = (P["income_base"] + tr["v"]) / 12.0
    H = np.zeros(N)
    alive = np.ones(N, bool)          # not in jail
    exposed = np.zeros(N, bool)
    n_bankrupt = np.zeros(N, int); t_first_bankrupt = np.full(N, -1)
    jailed_t = np.full(N, -1)
    ever_great = np.zeros(N, bool); t_great = np.full(N, -1)
    rescued = np.zeros(N, int); frauds = np.zeros(N, int)
    peak_c = c.copy(); max_lev = np.zeros(N)
    q_at_goal = np.full(N, np.nan); q_after_goal = np.full(N, np.nan); t_goal = np.full(N, -1)
    lev_hist, ret_hist = [], []
    traj = [] if record else None
    np.seterr(over="ignore", invalid="ignore", divide="ignore")
    for t in range(P["T"]):
        if windfall is not None and t == 0:
            K = K + windfall
            peak = np.maximum(peak, K)
        # --- drive
        if fl["goal_drive"]:
            q_extr = 1.0 / (1.0 + np.exp(-3.0 * (1.0 - np.maximum(K, 0) / tr["goal"])))
        else:
            q_extr = np.full(N, P["q_extr_free"])
        q = tr["phi"] * tr["q_intr"] + (1 - tr["phi"]) * q_extr
        # record drive around goal attainment
        hit = (t_goal < 0) & (K >= tr["goal"])
        t_goal[hit] = t; q_at_goal[hit] = q[hit]
        later = (t_goal >= 0) & (t == t_goal + 60)
        q_after_goal[later] = q[later]
        # --- edge and leverage
        mu = P["mu_max"] * q * tr["v"]
        f = P["kelly_frac"] * c * mu / P["sigma"] ** 2
        dd = np.where(peak > 5.0, 1.0 - np.maximum(K, 0) / peak, 0.0)
        if fl["loss_risk"]:
            f = f * (1.0 + P["lam_loss"] * dd * (1 - tr["phi"]))
        f = np.clip(f, 0, P["f_cap"])
        f = np.where(K > 0, f, 0.0)
        # --- returns
        z = rng.standard_normal(N)
        X = P["r"] + mu + P["sigma"] * z
        invest = K * (P["r"] + f * (X - P["r"]))
        net_income = income + (P["y_R"] * R / (R + P["R_half"]) / 12.0 if fl["merit"] else 0.0)
        cons = P["cons_income"] * income + P["save_wealth_rate"] * np.maximum(K, 0)
        if fl["ratchet"]:
            # habit: spending can rise at once but falls back only at rate habit_decay
            H = np.maximum(cons, H * (1 - P["habit_decay"]))
            cons = H
        give = tr["omega"] * (income + P["save_wealth_rate"] * np.maximum(K, 0)) if fl["merit"] else 0.0
        K = np.where(alive, K + invest + net_income - cons - give, 0.0)
        if fl["merit"]:
            R = np.where(alive, R + give - P["rho_R"] * R, 0.0)
        # --- confidence (self-attribution)
        if fl["hubris"]:
            status = np.maximum(0, np.log(np.maximum(K, 1e-9) / P["K_med"]))
            s_sat = 1.0 - np.exp(-status / 3.0)          # power effect saturates
            b_eff = P["beta"] * (1 + P["ego_gain"] * tr["ego"]) * (1 + P["psi"] * s_sat)
            g_eff = P["gamma"] * (0.5 + tr["iota"]) / (1 + P["psi"] * s_sat)
            if fl.get("merit_feedback"):
                # hypothesis H-merit: reputation buys honest feedback that corrects confidence
                g_eff = g_eff * (1 + P["fb_gain"] * R / (R + P["R_half"]))
            sig = np.where(f > 0, np.where(z > 0, b_eff * z, P["beta_neg"] * z), 0.0)
            c = np.clip(c + sig + g_eff * (1 - c), 0.3, P["c_cap"])
        # --- fraud and jail
        if fl["fraud"]:
            pf = P["fraud_pi"] * (1 - tr["iota"]) ** 2 * (1 - tr["phi"]) * np.maximum(0, dd - 0.3)
            new = alive & ~exposed & (peak > 5.0) & (rng.random(N) < pf)
            K = np.where(new, K + 0.3 * (peak - K), K)
            exposed |= new; frauds += new
            det = alive & exposed & (rng.random(N) < P["fraud_det"])
            jailed_t[det] = t; alive &= ~det
            K = np.where(det, 0.0, K); R = np.where(det, 0.0, R)
        # --- ruin, rescue
        ruin = alive & (K < 0)
        if fl["merit"]:
            res = ruin & (rng.random(N) < P["rescue_p"] * R / (R + P["R_half"]))
            K = np.where(res, 0.5, K); R = np.where(res, R * 0.5, R); rescued += res
            ruin &= ~res
        first = ruin & (t_first_bankrupt < 0)
        t_first_bankrupt[first] = t
        n_bankrupt += ruin
        K = np.where(ruin, 0.0, K)
        c = np.where(ruin, 1.0, c); H = np.where(ruin, P["cons_income"] * income, H)
        peak = np.where(ruin, 0.0, np.maximum(peak, K))
        g_now = alive & (K >= P["great"]) & ~ever_great
        t_great[g_now] = t; ever_great |= g_now
        peak_c = np.maximum(peak_c, c); max_lev = np.maximum(max_lev, f)
        if t % 12 == 0:
            lev_hist.append(f.copy()); ret_hist.append(K.copy())
        if record:
            traj.append(K.copy())
    return dict(K=K, peak=peak, alive=alive, jailed_t=jailed_t, n_bankrupt=n_bankrupt,
                t_first_bankrupt=t_first_bankrupt, ever_great=ever_great, t_great=t_great,
                rescued=rescued, frauds=frauds, peak_c=peak_c, max_lev=max_lev, c=c, R=R,
                q_at_goal=q_at_goal, q_after_goal=q_after_goal, t_goal=t_goal,
                lev=np.array(lev_hist), wealth_yr=np.array(ret_hist),
                traj=np.array(traj) if record else None)
